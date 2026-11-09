"""Evaluate a trained adapter on dev or test.

The adapter is put on the normal bf16 model and gets the zero-shot prompt, so prompt,
parser and metrics are the same as in zero-shot prompting.
"""

import argparse
import json
import sys
from pathlib import Path

import transformers
import yaml

from peft import PeftModel

# Let Python find our own folders. This comes after the imports above because our
# folder is also called "peft" and would hide the real peft library.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting.run_prompting import environment_info, evaluate_languages, load_model, load_split, model_commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    transformers.set_seed(cfg["seed"])
    records = load_split(args.split)

    out_dir = Path("results/peft") / cfg["experiment"] / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # The base model + the trained adapter
    train_dir = Path("outputs/peft") / cfg["experiment"]
    tokenizer, model = load_model(cfg)
    model = PeftModel.from_pretrained(model, str(train_dir / "adapter"))
    model.eval()

    # No five-shot examples: the trained model gets the zero-shot prompt
    runtime = evaluate_languages(tokenizer, model, records, cfg["eval_languages"], cfg, [], out_dir)

    train_info = json.loads((train_dir / "train_info.json").read_text())
    info = {
        "config": cfg,
        "config_file": args.config,
        "split": args.split,
        "n_examples": len(records),
        "adapter": str(train_dir / "adapter"),
        "best_checkpoint": train_info["best_checkpoint"],
        "model_commit": model_commit(cfg),
        "runtime_seconds": runtime,
    }
    info.update(environment_info())
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
