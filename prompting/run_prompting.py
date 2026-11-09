"""Zero-shot and five-shot prompting.

Saves predictions, metrics and run info in results/prompting/<experiment>/<split>/.
Example:
  uv run python prompting/run_prompting.py --config configs/prompting/qwen3-8b_zero-shot.yaml --limit 50
"""

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

# Let Python find our own folders (eval/ and prompting/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import compute_metrics, parse_output
from prompting.prompts import build_messages


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_split(split):
    # check that the data is the same as the ids in splits/
    records = read_jsonl(f"data/processed/{split}.jsonl")
    saved_ids = Path(f"splits/{split}_ids.txt").read_text().split()
    ids = [record["align_id"] for record in records]
    if ids != saved_ids:
        raise ValueError(f"{split}.jsonl is not the saved split, run data/prepare_data.py again")
    return records


def load_demos(demos_file):
    train = {}
    for record in read_jsonl("data/processed/train.jsonl"):
        train[record["align_id"]] = record
    demos = []
    for align_id in Path(demos_file).read_text().split():
        demos.append(train[align_id])
    return demos


def load_model(cfg):
    name = cfg["model"]["name"]
    revision = cfg["model"]["revision"]  # exact version of the model
    dtype = getattr(torch, cfg["model"]["dtype"])  # "bfloat16" -> torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    # To give the model several prompts at once, the shorter ones are filled up with
    # padding. For this kind of model the padding has to go on the left.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, device_map="auto")
    model.eval()
    return tokenizer, model


def model_commit(cfg):
    try:
        return model_info(cfg["model"]["name"], revision=cfg["model"]["revision"]).sha
    except Exception:
        return None  # e.g. no internet


@torch.no_grad()  # we only use the model, we do not train it here
def generate(tokenizer, model, prompts, cfg):
    answers = []
    batch_size = cfg["batch_size"]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        # The chat template already added the special tokens, so we do not add them again.
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = inputs.to(model.device)
        output = model.generate(
            **inputs,
            do_sample=cfg["decoding"]["do_sample"],  # False = always take the most likely word
            max_new_tokens=cfg["decoding"]["max_new_tokens"],
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        # The output starts with the prompt itself; keep only the new tokens.
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        answers += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        print(f"  {start + len(batch)}/{len(prompts)}", flush=True)
    return answers


def make_prompt(tokenizer, text, cfg, demos):
    messages = build_messages(text, cfg["prompt"]["variant"], demos)
    # Extra settings for the chat template, e.g. enable_thinking=False for Qwen3.
    template_settings = cfg.get("chat_template_kwargs") or {}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_settings
    )


def evaluate_languages(tokenizer, model, records, languages, cfg, demos, out_dir):
    # also used by peft/evaluate_peft.py
    all_metrics = {}
    runtime = {}
    for lang in languages:
        print(f"[{cfg['experiment']}] {lang}: {len(records)} examples")

        # demos in the same language as the test line
        lang_demos = []
        for demo in demos:
            lang_demos.append((demo[f"text_{lang}"], demo["labels"]))

        prompts = []
        for record in records:
            prompts.append(make_prompt(tokenizer, record[f"text_{lang}"], cfg, lang_demos))
        (out_dir / f"prompt_example_{lang}.txt").write_text(prompts[0], encoding="utf-8")

        start = time.time()
        answers = generate(tokenizer, model, prompts, cfg)
        runtime[lang] = round(time.time() - start, 1)

        gold, predictions, status = [], [], []
        with open(out_dir / f"predictions_{lang}.jsonl", "w", encoding="utf-8") as f:
            for record, answer in zip(records, answers):
                labels, answer_status = parse_output(answer)
                gold.append(record["labels"])
                predictions.append(labels)
                status.append(answer_status)
                row = {
                    "align_id": record["align_id"],
                    "gold": record["labels"],
                    "raw_output": answer,
                    "pred": labels,
                    "status": answer_status,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        m = compute_metrics(gold, predictions, status)
        all_metrics[lang] = m
        print(
            f"  micro-F1 {m['micro_f1']:.3f} | Jaccard {m['jaccard_samples']:.3f} "
            f"| macro-F1 {m['macro_f1']:.3f} | malformed {m['malformed_rate']:.1%} "
            f"(empty list {m['empty_list_rate']:.1%}, invalid {m['invalid_rate']:.1%})"
        )

    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    return runtime


def environment_info():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
    return {
        "device": device,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, help="only use the first N lines (quick test)")
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], help="other prompt wording (dev only)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # Prompt sensitivity: run the same experiment with another prompt, saved in its own folder.
    if args.variant is not None and args.variant != cfg["prompt"]["variant"]:
        if args.split == "test":
            raise ValueError("prompt variants are only compared on dev")
        cfg["prompt"]["variant"] = args.variant
        cfg["experiment"] = cfg["experiment"] + "_" + args.variant

    transformers.set_seed(cfg["seed"])

    records = load_split(args.split)
    if args.limit:
        records = records[: args.limit]

    if cfg["prompt"]["n_shots"] > 0:
        demos = load_demos(cfg["prompt"]["demos_file"])
    else:
        demos = []

    run_name = args.split
    if args.limit:
        run_name = f"{args.split}_limit{args.limit}"
    out_dir = Path("results/prompting") / cfg["experiment"] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(cfg)
    runtime = evaluate_languages(tokenizer, model, records, cfg["languages"], cfg, demos, out_dir)

    info = {
        "config": cfg,
        "config_file": args.config,
        "split": args.split,
        "limit": args.limit,
        "n_examples": len(records),
        "demo_ids": [demo["align_id"] for demo in demos],
        "model_commit": model_commit(cfg),
        "runtime_seconds": runtime,
    }
    info.update(environment_info())
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
