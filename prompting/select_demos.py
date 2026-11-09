"""Pick the five examples for five-shot prompting (seed 42).

Candidates are train lines with 1-2 emotions, EN and ES text of 15-120 characters and no
duplicates. We shuffle them and greedily take the one that adds the most new emotions.
The same 5 ids are used for every model and language.
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
