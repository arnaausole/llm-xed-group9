"""Find test lines where QLoRA is much better or much worse than five-shot
(Jaccard difference of 0.5 or more), for the qualitative analysis.

We print the EN and the ES line, because some errors come from bad alignments. Lines where
one side is more than twice as long are marked "check alignment" (just a rough hint).
Output: results/analysis/examples_test.md
"""

import json
import random
from pathlib import Path

N_EXAMPLES = 8  # examples shown per group
SEED = 42

# (language, prompting system, QLoRA system)
COMPARISONS = [
    ("en", "prompting/qwen3-8b_five-shot", "peft/qwen3-8b_qlora_en"),
    ("es", "prompting/qwen3-8b_five-shot", "peft/qwen3-8b_qlora_es"),
    ("es", "prompting/qwen3-8b_five-shot", "peft/qwen3-8b_qlora_en"),
]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def jaccard(gold, pred):
    gold, pred = set(gold), set(pred)
    if len(gold | pred) == 0:
        return 0.0
    return len(gold & pred) / len(gold | pred)


def check_alignment(text_en, text_es):
    longer = max(len(text_en), len(text_es))
    shorter = max(min(len(text_en), len(text_es)), 1)
    if longer / shorter > 2:
        return "check alignment"
    return ""


def show(example):
    """One example as Markdown."""
    return (
        f"- `{example['id']}` {example['flag']}\n"
        f"  - EN: {example['text_en']}\n"
        f"  - ES: {example['text_es']}\n"
        f"  - gold: {example['gold']} | prompting: {example['prompting']} | QLoRA: {example['qlora']}\n"
    )


def main():
    # The subtitle text is not in the predictions, so we take it from the test data.
    test = {}
    for record in read_jsonl("data/processed/test.jsonl"):
        test[record["align_id"]] = record

    text = "# Test examples: QLoRA vs prompting\n\nMade by analysis/find_examples.py.\n"
    for lang, prompting_system, qlora_system in COMPARISONS:
        prompting = read_jsonl(f"results/{prompting_system}/test/predictions_{lang}.jsonl")
        qlora = read_jsonl(f"results/{qlora_system}/test/predictions_{lang}.jsonl")

        improved, worsened = [], []
        n_same = 0
        for p, q in zip(prompting, qlora):
            record = test[p["align_id"]]
            example = {
                "id": p["align_id"],
                "text_en": record["text_en"],
                "text_es": record["text_es"],
                "gold": p["gold"],
                "prompting": p["pred"],
                "qlora": q["pred"],
                "flag": check_alignment(record["text_en"], record["text_es"]),
            }
            difference = jaccard(q["gold"], q["pred"]) - jaccard(p["gold"], p["pred"])
            if difference >= 0.5:
                improved.append(example)
            elif difference <= -0.5:
                worsened.append(example)
            elif p["pred"] == q["pred"]:
                n_same += 1

        prompting_name = prompting_system.split("/")[1]
        qlora_name = qlora_system.split("/")[1]
        text += f"\n## {lang.upper()}: {qlora_name} vs {prompting_name}\n\n"
        text += f"Out of {len(prompting)} lines: {len(improved)} improved, {len(worsened)} worsened, "
        text += f"{n_same} with exactly the same prediction.\n"

        rng = random.Random(SEED)
        for title, examples in [("Improved by QLoRA", improved), ("Worsened by QLoRA", worsened)]:
            text += f"\n### {title} ({N_EXAMPLES} random examples)\n\n"
            for example in rng.sample(examples, min(N_EXAMPLES, len(examples))):
                text += show(example)

    out = Path("results/analysis/examples_test.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
