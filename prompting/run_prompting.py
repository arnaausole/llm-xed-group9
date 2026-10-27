"""Zero-shot and five-shot prompting (project brief §3, Phase 2).

For every language in the config, each subtitle of the chosen split is sent to the model
with the shared English instruction (prompting/prompts.py), decoded greedily, parsed with
eval/metrics.py and scored with the shared metrics.

Outputs (in results/prompting/<experiment>/<split>[_limit<N>]/)
---------------------------------------------------------------
- predictions_<lang>.jsonl   one line per example: id, gold, raw output, parsed labels
- prompt_example_<lang>.txt  the complete rendered prompt of the first example
- metrics.json               metrics per language
- run_info.json              config, model commit, library versions, GPU, runtime

Usage::

    uv run python prompting/run_prompting.py --config configs/prompting/qwen3-8b_zero-shot.yaml
    # quick check on 50 dev lines
    uv run python prompting/run_prompting.py --config ... --limit 50
    # final evaluation, only once the prompt is frozen
    uv run python prompting/run_prompting.py --config ... --split test
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import compute_metrics, parse_output
from prompting.prompts import build_messages


# --------------------------------------------------------------------------- data
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_split(data_dir: Path, splits_dir: Path, split: str) -> list[dict]:
    """Load a processed split and check it matches the committed alignment IDs."""
    records = read_jsonl(data_dir / f"{split}.jsonl")
    ids = (splits_dir / f"{split}_ids.txt").read_text(encoding="utf-8").split()
    if [r["align_id"] for r in records] != ids:
        raise ValueError(
            f"{split}.jsonl does not match splits/{split}_ids.txt; "
            "re-run data/prepare_data.py with the documented seed"
        )
    return records


def load_demos(data_dir: Path, demos_file: Path) -> list[dict]:
    train = {r["align_id"]: r for r in read_jsonl(data_dir / "train.jsonl")}
    ids = demos_file.read_text(encoding="utf-8").split()
    missing = [i for i in ids if i not in train]
    if missing:
        raise ValueError(f"Demonstrations not in the training set: {missing}")
    return [train[i] for i in ids]


# --------------------------------------------------------------------------- model
def load_model(cfg: dict):
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    tok.padding_side = "left"  # decoder-only models must be left-padded for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        m["name"],
        revision=m["revision"],
        dtype=getattr(torch, m["dtype"]),
        device_map="auto",
    )
    model.eval()
    return tok, model


def model_commit(m: dict) -> str | None:
    """Resolved Hugging Face commit of the model revision, for reproducibility."""
    try:
        return model_info(m["name"], revision=m["revision"]).sha
    except Exception:  # noqa: BLE001  (offline: record nothing rather than fail)
        return None


@torch.no_grad()
def generate(tok, model, prompts: list[str], cfg: dict) -> list[str]:
    dec = cfg["decoding"]
    outputs = []
    for start in range(0, len(prompts), cfg["batch_size"]):
        batch = prompts[start : start + cfg["batch_size"]]
        # The chat template already inserts the special tokens (e.g. BOS).
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        out = model.generate(
            **enc,
            do_sample=dec["do_sample"],
            max_new_tokens=dec["max_new_tokens"],
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        outputs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)
    return outputs


# --------------------------------------------------------------------------- evaluation
def evaluate_languages(
    tok, model, records, languages, cfg, demos, out_dir: Path
) -> tuple[dict, dict]:
    """Generate, parse and score every language; also used by peft/evaluate_peft.py."""
    template_kwargs = cfg.get("chat_template_kwargs") or {}
    all_metrics, runtime = {}, {}
    for lang in languages:
        print(f"[{cfg['experiment']}] {lang}: {len(records)} examples")
        lang_demos = [(d[f"text_{lang}"], d["labels"]) for d in demos]
        prompts = [
            tok.apply_chat_template(
                build_messages(r[f"text_{lang}"], cfg["prompt"]["variant"], lang_demos),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            for r in records
        ]
        (out_dir / f"prompt_example_{lang}.txt").write_text(
            prompts[0], encoding="utf-8"
        )

        t0 = time.perf_counter()
        raw = generate(tok, model, prompts, cfg)
        runtime[lang] = round(time.perf_counter() - t0, 1)

        parsed = [parse_output(o) for o in raw]
        preds = [labels for labels, _ in parsed]
        malformed = [bad for _, bad in parsed]
        gold = [r["labels"] for r in records]

        with (out_dir / f"predictions_{lang}.jsonl").open("w", encoding="utf-8") as f:
            for r, o, pr, bad in zip(records, raw, preds, malformed):
                row = {
                    "align_id": r["align_id"],
                    "text": r[f"text_{lang}"],
                    "gold": r["labels"],
                    "raw_output": o,
                    "pred": pr,
                    "malformed": bad,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        all_metrics[lang] = compute_metrics(gold, preds, malformed)
        m = all_metrics[lang]
        print(
            f"  micro-F1 {m['micro_f1']:.3f} | Jaccard {m['jaccard_samples']:.3f} "
            f"| macro-F1 {m['macro_f1']:.3f} | malformed {m['malformed_rate']:.1%}"
        )

    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    return all_metrics, runtime


def environment_info(model) -> dict:
    return {
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else str(model.device),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, help="only the first N examples (smoke test)")
    ap.add_argument("--model", help="override the model name (smoke test only)")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--results", type=Path, default=Path("results/prompting"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.model:
        cfg["model"]["name"] = args.model
        cfg["experiment"] += "_override-" + args.model.replace("/", "_")
    transformers.set_seed(cfg["seed"])

    records = load_split(args.data, args.splits, args.split)
    if args.limit:
        records = records[: args.limit]
    p = cfg["prompt"]
    demos = load_demos(args.data, Path(p["demos_file"])) if p["n_shots"] else []
    if len(demos) != p["n_shots"]:
        raise ValueError(f"Expected {p['n_shots']} demonstrations, got {len(demos)}")

    run_name = args.split + (f"_limit{args.limit}" if args.limit else "")
    out_dir = args.results / cfg["experiment"] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(cfg)
    _, runtime = evaluate_languages(
        tok, model, records, cfg["languages"], cfg, demos, out_dir
    )
    info = {
        "config": cfg,
        "config_file": str(args.config),
        "split": args.split,
        "limit": args.limit,
        "n_examples": len(records),
        "demo_ids": [d["align_id"] for d in demos],
        "model_commit": model_commit(cfg["model"]),
        "runtime_seconds": runtime,
        **environment_info(model),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
