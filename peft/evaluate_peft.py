"""Evaluate a trained QLoRA adapter with the shared prompting evaluation.

The adapter is applied to the base model loaded exactly as in prompting (same dtype, no
quantization), and queried with the same zero-shot prompt it was trained on, so that
PEFT and prompting differ only by the adapter.

Outputs go to results/peft/<experiment>/<split>[_limit<N>]/ with the same files as
prompting/run_prompting.py.

Usage::

    uv run python peft/evaluate_peft.py --config configs/peft/qwen3-8b_qlora_en.yaml
    uv run python peft/evaluate_peft.py --config ... --split test   # final evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import transformers
import yaml

from peft import PeftModel

# Imported after the libraries: this folder is also called "peft".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting.run_prompting import (
    environment_info,
    evaluate_languages,
    load_model,
    load_split,
    model_commit,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, help="only the first N examples (smoke test)")
    ap.add_argument("--model", help="override the model name (smoke test only)")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs/peft"))
    ap.add_argument("--results", type=Path, default=Path("results/peft"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.model:
        cfg["model"]["name"] = args.model
        cfg["experiment"] += "_override-" + args.model.replace("/", "_")
    transformers.set_seed(cfg["seed"])

    records = load_split(args.data, args.splits, args.split)
    if args.limit:
        records = records[: args.limit]
    run_name = args.split + (f"_limit{args.limit}" if args.limit else "")
    out_dir = args.results / cfg["experiment"] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = args.outputs / cfg["experiment"] / "adapter"
    tok, model = load_model(cfg)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    _, runtime = evaluate_languages(
        tok, model, records, cfg["eval_languages"], cfg, [], out_dir
    )
    train_info = json.loads(
        (args.outputs / cfg["experiment"] / "train_info.json").read_text()
    )
    info = {
        "config": cfg,
        "config_file": str(args.config),
        "split": args.split,
        "limit": args.limit,
        "n_examples": len(records),
        "adapter": str(adapter_dir),
        "best_checkpoint": train_info["best_checkpoint"],
        "model_commit": model_commit(cfg["model"]),
        "runtime_seconds": runtime,
        **environment_info(model),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
