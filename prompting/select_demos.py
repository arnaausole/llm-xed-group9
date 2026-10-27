"""Select the five fixed five-shot demonstrations from the training set.

Procedure (deterministic given the seed)
----------------------------------------
1. Candidates: training pairs whose source side is an English subtitle file (in
   ``pairs-es.txt`` part of the source side is Danish), with one or two labels, whose
   English and Spanish lines are both 15-120 characters long (long enough to carry
   meaning, short enough to keep the prompt compact), and whose duplicate group has a
   single member.
2. Shuffle the candidates with the seed.
3. Greedily pick five: each step takes the first candidate (in shuffled order) that adds
   the most not-yet-covered labels, so that the five demonstrations together cover as
   many of the eight labels as possible.

The same alignment IDs are used for every model; English evaluation shows the English
lines, Spanish evaluation shows their Spanish counterparts.

Usage::

    uv run python prompting/select_demos.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("splits/fewshot_ids.txt"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = [json.loads(line) for line in args.train.open(encoding="utf-8")]
    group_size = Counter(r["dup_group"] for r in records)
    candidates = [
        r
        for r in records
        if r["en_file"].startswith("en/")
        and 1 <= len(r["labels"]) <= 2
        and all(15 <= len(r[f"text_{lang}"]) <= 120 for lang in ("en", "es"))
        and group_size[r["dup_group"]] == 1
    ]
    random.Random(args.seed).shuffle(candidates)

    chosen, covered = [], set()
    for _ in range(args.k):
        best = max(
            (r for r in candidates if r not in chosen),
            key=lambda r: len(set(r["labels"]) - covered),
        )
        chosen.append(best)
        covered.update(best["labels"])

    args.out.write_text(
        "\n".join(r["align_id"] for r in chosen) + "\n", encoding="utf-8"
    )
    print(f"{len(candidates)} candidates, labels covered: {len(covered)}/8")
    for r in chosen:
        print(
            f"{r['align_id']}  {r['labels']}\n  EN: {r['text_en']}\n  ES: {r['text_es']}"
        )


if __name__ == "__main__":
    main()
