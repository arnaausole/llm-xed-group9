"""QLoRA fine-tuning of the primary model (project brief §3, Phase 1).

Training examples use exactly the zero-shot prompt of prompting/prompts.py, rendered with
the model's chat template as at inference time; the target is the gold JSON label list.
The loss is computed on the answer only (TRL prompt-completion format).

Conditions (one config each in configs/peft/): English-only, Spanish-only and bilingual
(English + Spanish training examples together).

Checkpoint selection: one checkpoint per epoch; the one with the lowest loss on the
development set (same languages as training) is kept as the final adapter.

Outputs (outputs/peft/<experiment>/, not committed)
---------------------------------------------------
- checkpoint-*/      per-epoch checkpoints
- adapter/           the selected LoRA adapter
- train_info.json    config, selected checkpoint, loss history, trainable parameters,
                     runtime, peak GPU memory, library versions

Usage::

    uv run python peft/train_qlora.py --config configs/peft/qwen3-8b_qlora_en.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
import trl
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import peft
from peft import LoraConfig

# Imported after the libraries: this folder is also called "peft".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting.prompts import answer_turn, build_messages
from prompting.run_prompting import environment_info, load_split, model_commit


def build_dataset(records, languages, tok, cfg) -> Dataset:
    template_kwargs = cfg.get("chat_template_kwargs") or {}
    rows = []
    for lang in languages:
        for r in records:
            prompt = tok.apply_chat_template(
                build_messages(r[f"text_{lang}"], cfg["prompt"]["variant"]),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            # TRL appends the EOS token (<|im_end|> for Qwen3) to the completion.
            rows.append({"prompt": prompt, "completion": answer_turn(r["labels"])})
    return Dataset.from_list(rows)


def load_model(cfg: dict, use_4bit: bool):
    m, q = cfg["model"], cfg["quantization"]
    bnb = None
    if use_4bit and q["load_in_4bit"]:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        )
    return AutoModelForCausalLM.from_pretrained(
        m["name"],
        revision=m["revision"],
        dtype=getattr(torch, m["dtype"]),
        quantization_config=bnb,
        device_map="auto",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs/peft"))
    # Smoke-test options (e.g. on a laptop without bitsandbytes); never for reported runs.
    ap.add_argument("--model", help="override the model name")
    ap.add_argument("--limit", type=int, help="only the first N train/dev examples")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.model:
        cfg["model"]["name"] = args.model
        cfg["experiment"] += "_override-" + args.model.replace("/", "_")
    if args.no_4bit:  # the paged 8-bit optimizer also needs bitsandbytes
        cfg["training"]["optim"] = "adamw_torch"
    transformers.set_seed(cfg["seed"])
    out_dir = args.outputs / cfg["experiment"]

    train = load_split(args.data, args.splits, "train")
    dev = load_split(args.data, args.splits, "dev")
    if args.limit:
        train, dev = train[: args.limit], dev[: args.limit]

    tok = AutoTokenizer.from_pretrained(
        cfg["model"]["name"], revision=cfg["model"]["revision"]
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    langs = cfg["train_languages"]
    train_ds = build_dataset(train, langs, tok, cfg)
    dev_ds = build_dataset(dev, langs, tok, cfg)
    print(f"[{cfg['experiment']}] train {len(train_ds)} | dev {len(dev_ds)} ({langs})")

    model = load_model(cfg, use_4bit=not args.no_4bit)
    dtype = cfg["model"]["dtype"]
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        **cfg["training"],
        max_steps=args.max_steps,
        completion_only_loss=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        bf16=dtype == "bfloat16" and torch.cuda.is_available(),
        fp16=dtype == "float16" and torch.cuda.is_available(),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    runtime = round(time.perf_counter() - t0, 1)

    trainer.save_model(
        str(out_dir / "adapter")
    )  # the best checkpoint (lowest dev loss)
    info = {
        "config": cfg,
        "config_file": str(args.config),
        "smoke_test": {
            "limit": args.limit,
            "max_steps": args.max_steps,
            "no_4bit": args.no_4bit,
        },
        "n_train": len(train_ds),
        "n_dev": len(dev_ds),
        "model_commit": model_commit(cfg["model"]),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_loss": trainer.state.best_metric,
        "train_runtime_seconds": runtime,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        if torch.cuda.is_available()
        else None,
        "log_history": trainer.state.log_history,
        **environment_info(trainer.model),
    }
    info["versions"].update({"peft": peft.__version__, "trl": trl.__version__})
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(
        f"Best checkpoint: {info['best_checkpoint']} (dev loss {info['best_dev_loss']})"
    )
    print(f"Adapter saved to {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
