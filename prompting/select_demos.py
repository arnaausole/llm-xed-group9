"""Choose the five examples used in five-shot prompting.

Rules:
  - they come from the training set
  - they have 1 or 2 emotions
  - the English and the Spanish line are both 15-120 characters long
  - the line has no duplicates in the data
We shuffle these candidates (seed 42) and pick five, one by one. Each time we take the
first candidate that adds the most emotions we do not have yet, so the five examples
show as many of the eight emotions as possible.

The same five IDs are used for all models (English lines for English, Spanish for Spanish).

How to run it:
  uv run python prompting/select_demos.py --seed 42
"""

import argparse
import json
import random
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    with open("data/processed/train.jsonl", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # how many lines are in each duplicate group
    group_size = Counter(record["dup_group"] for record in records)

    candidates = []
    for record in records:
        if not 1 <= len(record["labels"]) <= 2:
            continue
        if not 15 <= len(record["text_en"]) <= 120:
            continue
        if not 15 <= len(record["text_es"]) <= 120:
            continue
        if group_size[record["dup_group"]] != 1:
            continue
        candidates.append(record)
    random.Random(args.seed).shuffle(candidates)

    chosen = []
    covered = set()  # emotions we already have
    for _ in range(5):
        best = None
        best_new = -1
        for record in candidates:
            if record in chosen:
                continue
            new = len(set(record["labels"]) - covered)  # emotions this one would add
            if new > best_new:
                best = record
                best_new = new
        chosen.append(best)
        covered.update(best["labels"])

    with open("splits/fewshot_ids.txt", "w") as f:
        for record in chosen:
            f.write(record["align_id"] + "\n")

    print(f"{len(candidates)} candidates, emotions covered: {len(covered)}/8")
    for record in chosen:
        print(record["align_id"], record["labels"])
        print("  EN:", record["text_en"])
        print("  ES:", record["text_es"])


if __name__ == "__main__":
    main()
