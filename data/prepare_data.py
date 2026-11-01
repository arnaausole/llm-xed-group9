"""Build the English-Spanish dataset from XED and make the fixed train/dev/test split.

Steps:
1. Read the Spanish lines with their emotions (Projections/es-projections.tsv).
2. Read the alignments (subtitle-retrieval/students/pairs-es.txt). Each row has 6 columns:
   source file, Spanish file, source line ids, Spanish line ids, source text, Spanish text.
3. Find the alignment row of each Spanish line. Both files are in the same order, so we
   go through them together and match on the Spanish text.
4. Keep only pairs whose source file is English ("en/..."). Many sources in pairs-es.txt
   are Danish files ("da/..."), so their text is Danish, not English. We drop those.
   The English and Spanish line share the same emotions (the Spanish emotions were
   copied from the English annotation).
5. Find duplicates (lines with the same English or the same Spanish text), so that we
   can keep all copies of a line in the same part of the split.
6. Split 70% train / 20% dev / 10% test with seed 42, keeping the emotions balanced
   between the three parts (multi-label stratification).

Output:
  data/processed/*.jsonl   the data with the subtitle text (not in git)
  splits/*_ids.txt         only the IDs of each part (in git)
  splits/stats.json        counts for the report (in git)

How to run it:
  uv run python data/prepare_data.py --xed data/raw/XED --seed 42
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

# In XED the emotions are numbers: 1 = anger, 2 = anticipation, ... (alphabetical order)
CODE_TO_LABEL = {}
for number, label in enumerate(LABELS, start=1):
    CODE_TO_LABEL[str(number)] = label


def read_projections(path):
    """Read the Spanish lines. Each line looks like: text <TAB> 1, 5"""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        text, _, codes = line.rpartition("\t")
        labels = []
        for code in codes.split(","):
            code = code.strip()
            if code in CODE_TO_LABEL and CODE_TO_LABEL[code] not in labels:
                labels.append(CODE_TO_LABEL[code])
        labels.sort(key=LABELS.index)
        rows.append((text, labels))
    return rows


def read_pairs(path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.split("\t")
        if len(columns) == 6:
            pairs.append(columns)
    return pairs


def remove_quotes(text):
    """Some XED lines have extra quotes, like in a spreadsheet: "a ""b"" c" -> a "b" c"""
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')
    return text


def make_id(pair):
    """A short ID made from the files and line numbers of the pair (always the same)."""
    key = "|".join(pair[:4])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def link(projections, pairs):
    """Find the alignment row of every Spanish line.

    We do not use a dictionary because short lines like "¡No!" appear many times.
    Instead we walk through both files at the same time: for each Spanish line we look
    forward from the last match until we find the same text.
    """
    records = []
    n_not_found = 0
    position = 0  # where the previous match was in pairs
    for text, labels in projections:
        k = position
        while k < len(pairs) and pairs[k][5] != text and pairs[k][5] != remove_quotes(text):
            k += 1
        if k == len(pairs) or len(labels) == 0:
            n_not_found += 1
            continue
        pair = pairs[k]
        records.append(
            {
                "align_id": make_id(pair),
                "en_file": pair[0],
                "es_file": pair[1],
                "en_line_ids": pair[2],
                "es_line_ids": pair[3],
                "text_en": pair[4].strip(),
                "text_es": pair[5].strip(),
                "labels": labels,
            }
        )
        position = k + 1
    return records, n_not_found


def normalize(text):
    """Remove extra spaces, so that "Hi  there " and "Hi there" count as the same text."""
    return re.sub(r"\s+", " ", text).strip()


def duplicate_groups(records):
    """Give every record a group number. Records with the same text get the same group.

    Duplicates can form chains: A has the same English text as B, and B has the same
    Spanish text as C, so A, B and C must all be together. We use "union-find" for this:
    every record points to a "parent" record, and the record at the top of the chain
    is the group number.
    """
    parent = list(range(len(records)))  # at the start every record is its own group

    def find_top(i):
        while parent[i] != i:
            i = parent[i]
        return i

    first_record_with_text = {}
    for i, record in enumerate(records):
        for lang in ["en", "es"]:
            key = (lang, normalize(record[f"text_{lang}"]))
            if key in first_record_with_text:
                # same text as an earlier record: join the two groups
                parent[find_top(i)] = find_top(first_record_with_text[key])
            else:
                first_record_with_text[key] = i

    groups = []
    for i in range(len(records)):
        groups.append(find_top(i))
    return groups


def split_groups(records, groups, seed):
    """Split the groups 70/20/10 and return "train", "dev" or "test" for each record."""
    group_ids = sorted(set(groups))
    row_of_group = {}
    for row, group in enumerate(group_ids):
        row_of_group[group] = row

    # y has one row per group and one column per emotion (1 if the group has it)
    y = np.zeros((len(group_ids), len(LABELS)), dtype=int)
    for record, group in zip(records, groups):
        for label in record["labels"]:
            y[row_of_group[group], LABELS.index(label)] = 1
    x = np.zeros((len(group_ids), 1))  # the splitter needs an x, but it is not used

    # First 70% train and 30% rest
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_rows, rest_rows = next(splitter.split(x, y))
    # Then the rest: 2/3 dev (20% of all) and 1/3 test (10% of all)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 / 3, random_state=seed)
    dev_part, test_part = next(splitter.split(x[rest_rows], y[rest_rows]))
    dev_rows = rest_rows[dev_part]
    test_rows = rest_rows[test_part]

    split_of_group = {}
    for row in train_rows:
        split_of_group[group_ids[row]] = "train"
    for row in dev_rows:
        split_of_group[group_ids[row]] = "dev"
    for row in test_rows:
        split_of_group[group_ids[row]] = "test"

    splits = []
    for group in groups:
        splits.append(split_of_group[group])
    return splits


def check_no_leakage(records):
    """Stop with an error if the same text is in two different splits."""
    for lang in ["en", "es"]:
        split_of_text = {}
        for record in records:
            text = normalize(record[f"text_{lang}"])
            if text in split_of_text and split_of_text[text] != record["split"]:
                raise ValueError(f"This line is in two splits: {text!r}")
            split_of_text[text] = record["split"]
    ids = [record["align_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Two records have the same alignment ID")


def make_stats(records, n_not_found, n_danish, groups, seed):
    group_sizes = Counter(groups)
    stats = {
        "seed": seed,
        "n_aligned_pairs": len(records),
        "n_unmatched_projection_rows": n_not_found,
        "n_dropped_non_english_source": n_danish,
        "n_duplicate_groups": len(group_sizes),
        "n_records_in_multi_record_groups": sum(size for size in group_sizes.values() if size > 1),
    }
    for split in ["train", "dev", "test"]:
        part = [record for record in records if record["split"] == split]
        label_counts = Counter()
        labels_per_line = Counter()
        for record in part:
            label_counts.update(record["labels"])
            labels_per_line[min(len(record["labels"]), 4)] += 1  # 4 means "4 or more"
        stats[split] = {
            "n": len(part),
            "share": round(len(part) / len(records), 4),
            "label_counts": {label: label_counts[label] for label in LABELS},
            "label_share": {label: round(label_counts[label] / len(part), 4) for label in LABELS},
            "labels_per_line": {
                ("4+" if k == 4 else str(k)): labels_per_line[k] for k in sorted(labels_per_line)
            },
        }
    return stats


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xed", type=Path, default=Path("data/raw/XED"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Steps 1-3: read and link
    projections = read_projections(args.xed / "Projections" / "es-projections.tsv")
    pairs = read_pairs(args.xed / "subtitle-retrieval" / "students" / "pairs-es.txt")
    records, n_not_found = link(projections, pairs)
    n_linked = len(records)

    # Step 4: keep only pairs with an English source
    records = [record for record in records if record["en_file"].startswith("en/")]
    n_danish = n_linked - len(records)

    # Steps 5-6: duplicates and split
    groups = duplicate_groups(records)
    splits = split_groups(records, groups, args.seed)
    for record, split, group in zip(records, splits, groups):
        record["split"] = split
        record["dup_group"] = group
    check_no_leakage(records)

    # Save
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    write_jsonl("data/processed/aligned_en_es.jsonl", records)
    for split in ["train", "dev", "test"]:
        part = [record for record in records if record["split"] == split]
        write_jsonl(f"data/processed/{split}.jsonl", part)
        with open(f"splits/{split}_ids.txt", "w") as f:
            for record in part:
                f.write(record["align_id"] + "\n")

    stats = make_stats(records, n_not_found, n_danish, groups, args.seed)
    Path("splits/stats.json").write_text(json.dumps(stats, indent=2))

    print(f"Spanish lines: {len(projections)} | linked: {n_linked} | not linked: {n_not_found}")
    print(f"Danish source (dropped): {n_danish} | kept: {len(records)}")
    print(f"Duplicate groups: {stats['n_duplicate_groups']}")
    for split in ["train", "dev", "test"]:
        print(f"{split}: {stats[split]['n']} ({stats[split]['share']:.1%})")
    print("No leakage: OK")


if __name__ == "__main__":
    main()
