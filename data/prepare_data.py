"""Build the aligned English-Spanish XED dataset and the fixed 70/20/10 split.

Steps
-----
1. Read the Spanish projections (``Projections/es-projections.tsv``): Spanish text + labels.
2. Read the alignment metadata (``subtitle-retrieval/students/pairs-es.txt``): for every
   aligned subtitle pair it gives the English file, Spanish file, English line ids, Spanish
   line ids, English text and Spanish text. The four id fields form the alignment ID.
3. Link every projection row to its pair. Both files are in the same order, so we walk
   through them sequentially and match on the Spanish text (robust to repeated short lines
   such as "¡No!", which would be ambiguous with a plain dictionary lookup).
4. Keep only pairs whose source file is English (``en/...``). In ``pairs-es.txt`` the
   source side of many pairs is a Danish subtitle file (``da/...``), so its "English"
   text is Danish; such pairs cannot form a directly aligned English-Spanish example.
   (An English line could be reached through the Danish-English alignments in
   ``pairs-en.txt``, but that two-step alignment is noisier, so it is not used.)
   ``--keep-non-english-source`` disables this filter.
   Labels are projected from the English annotation, so the English line and the Spanish
   line of a pair share the same label set.
5. Group exact duplicates (same English text OR same Spanish text, after whitespace
   normalisation) with union-find, so that a duplicate can never cross partitions.
6. Multi-label stratified 70/20/10 split over these groups (iterative stratification),
   with a fixed seed. Because we split alignment IDs, EN and ES of a pair always share a
   partition.

Outputs
-------
- data/processed/aligned_en_es.jsonl   (full table, NOT committed: contains XED text)
- data/processed/{train,dev,test}.jsonl
- splits/{train,dev,test}_ids.txt      (alignment IDs only, committed)
- splits/stats.json                    (counts for the report, committed)

Usage::

    uv run python data/prepare_data.py --xed data/raw/XED --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LABELS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]
# XED numeric codes: alphabetical order, 1-based (0 = neutral, not used here).
CODE2LABEL = {str(i + 1): name for i, name in enumerate(LABELS)}


# --------------------------------------------------------------------------- reading
def unquote_csv_field(text: str) -> str:
    """Undo spreadsheet-style quoting found in some XED lines: "a ""b"" c" -> a "b" c."""
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')
    return text


def read_projections(path: Path) -> list[tuple[str, list[str]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        text, _, codes = line.rpartition("\t")
        labels = sorted(
            {
                CODE2LABEL[c.strip()]
                for c in codes.split(",")
                if c.strip() in CODE2LABEL
            },
            key=LABELS.index,
        )
        rows.append((text, labels))
    return rows


def read_pairs(path: Path) -> list[list[str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) == 6:
            pairs.append(fields)
    return pairs


def alignment_id(pair: list[str]) -> str:
    """Stable short ID from (en_file, es_file, en_line_ids, es_line_ids)."""
    key = "|".join(pair[:4])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- linking
def link(projections, pairs):
    """Sequentially match projection rows to pair rows on the Spanish text."""
    records, unmatched = [], []
    j = 0
    for raw_text, labels in projections:
        candidates = {raw_text, unquote_csv_field(raw_text)}
        k = j
        while k < len(pairs) and pairs[k][5] not in candidates:
            k += 1
        if k == len(pairs) or not labels:
            unmatched.append(raw_text)
            continue
        p = pairs[k]
        records.append(
            {
                "align_id": alignment_id(p),
                "en_file": p[0],
                "es_file": p[1],
                "en_line_ids": p[2],
                "es_line_ids": p[3],
                "text_en": p[4].strip(),
                "text_es": p[5].strip(),
                "labels": labels,
            }
        )
        j = k + 1
    return records, unmatched


# --------------------------------------------------------------------------- dedup
def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def duplicate_groups(records) -> list[int]:
    """Union-find: records sharing an exact (normalised) EN or ES text share a group."""
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_seen: dict[tuple[str, str], int] = {}
    for i, r in enumerate(records):
        for lang in ("en", "es"):
            key = (lang, norm(r[f"text_{lang}"]))
            if key in first_seen:
                parent[find(i)] = find(first_seen[key])
            else:
                first_seen[key] = i
    return [find(i) for i in range(len(records))]


# --------------------------------------------------------------------------- split
def binarize(label_lists) -> np.ndarray:
    y = np.zeros((len(label_lists), len(LABELS)), dtype=int)
    for i, labs in enumerate(label_lists):
        for lab in labs:
            y[i, LABELS.index(lab)] = 1
    return y


def stratified_split(records, groups, seed: int):
    """70/20/10 multi-label stratified split over duplicate groups."""
    group_ids = sorted(set(groups))
    g_index = {g: i for i, g in enumerate(group_ids)}
    g_labels = [set() for _ in group_ids]
    for r, g in zip(records, groups):
        g_labels[g_index[g]].update(r["labels"])
    y = binarize([sorted(s) for s in g_labels])
    x = np.zeros((len(group_ids), 1))

    # Step 1: 70% train vs 30% rest.
    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(s1.split(x, y))
    # Step 2: rest -> 2/3 dev (20%), 1/3 test (10%).
    s2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=seed
    )
    dev_sub, test_sub = next(s2.split(x[rest_idx], y[rest_idx]))

    part_of_group = {}
    for name, idx in (
        ("train", train_idx),
        ("dev", rest_idx[dev_sub]),
        ("test", rest_idx[test_sub]),
    ):
        for i in idx:
            part_of_group[group_ids[i]] = name
    return [part_of_group[g] for g in groups]


# --------------------------------------------------------------------------- stats
def stats(records, parts, n_unmatched, n_non_english, group_sizes, seed):
    out = {
        "seed": seed,
        "n_aligned_pairs": len(records),
        "n_unmatched_projection_rows": n_unmatched,
        "n_dropped_non_english_source": n_non_english,
        "n_duplicate_groups": len(group_sizes),
        "n_records_in_multi_record_groups": sum(
            c for c in group_sizes.values() if c > 1
        ),
    }
    for name in ("train", "dev", "test"):
        sub = [r for r, p in zip(records, parts) if p == name]
        label_counts = Counter(lab for r in sub for lab in r["labels"])
        n_labels = Counter(min(len(r["labels"]), 4) for r in sub)
        out[name] = {
            "n": len(sub),
            "share": round(len(sub) / len(records), 4),
            "label_counts": {lab: label_counts[lab] for lab in LABELS},
            "label_share": {
                lab: round(label_counts[lab] / max(len(sub), 1), 4) for lab in LABELS
            },
            "labels_per_line": {
                ("4+" if k == 4 else str(k)): n_labels[k] for k in sorted(n_labels)
            },
        }
    return out


def check_no_leakage(records, parts):
    for lang in ("en", "es"):
        seen: dict[str, str] = {}
        for r, p in zip(records, parts):
            key = norm(r[f"text_{lang}"])
            if seen.setdefault(key, p) != p:
                raise AssertionError(
                    f"Duplicate {lang} text crosses partitions: {key!r}"
                )
    ids = Counter(r["align_id"] for r in records)
    assert max(ids.values()) == 1, "alignment IDs are not unique"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xed", type=Path, default=Path("data/raw/XED"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-non-english-source", action="store_true")
    args = ap.parse_args()

    projections = read_projections(args.xed / "Projections" / "es-projections.tsv")
    pairs = read_pairs(args.xed / "subtitle-retrieval" / "students" / "pairs-es.txt")
    records, unmatched = link(projections, pairs)
    n_linked = len(records)
    if not args.keep_non_english_source:
        records = [r for r in records if r["en_file"].startswith("en/")]
    n_non_english = n_linked - len(records)

    groups = duplicate_groups(records)
    group_sizes = Counter(groups)
    parts = stratified_split(records, groups, args.seed)
    check_no_leakage(records, parts)

    args.out.mkdir(parents=True, exist_ok=True)
    args.splits.mkdir(parents=True, exist_ok=True)
    for r, p, g in zip(records, parts, groups):
        r["split"] = p
        r["dup_group"] = g
    with open(args.out / "aligned_en_es.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("train", "dev", "test"):
        sub = [r for r in records if r["split"] == name]
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (args.splits / f"{name}_ids.txt").write_text(
            "\n".join(r["align_id"] for r in sub) + "\n", encoding="utf-8"
        )

    st = stats(records, parts, len(unmatched), n_non_english, group_sizes, args.seed)
    (args.splits / "stats.json").write_text(json.dumps(st, indent=2), encoding="utf-8")

    print(
        f"Projection rows: {len(projections)} | linked: {n_linked} "
        f"| dropped: {len(unmatched)} | non-English source: {n_non_english} "
        f"| kept: {len(records)}"
    )
    print(
        f"Duplicate groups: {st['n_duplicate_groups']} "
        f"(records in multi-record groups: {st['n_records_in_multi_record_groups']})"
    )
    for name in ("train", "dev", "test"):
        print(f"{name:5s}: {st[name]['n']:6d} ({st[name]['share']:.1%})")
    print("No leakage: OK")


if __name__ == "__main__":
    main()
