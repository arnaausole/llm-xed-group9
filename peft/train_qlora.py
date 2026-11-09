"""QLoRA fine-tuning of Qwen3-8B (EN, ES or both, depending on the config).

The base model is loaded in 4 bit and frozen, only the LoRA layers are trained.
Input = the zero-shot prompt, output = the gold labels as a JSON list, and the loss is
only on the output. We keep the epoch with the lowest dev loss.
The adapter and train_info.json go to outputs/peft/<experiment>/ (too big for git).
"""

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

# Let Python find our own folders. This comes after the imports above because our
# folder is also called "peft" and would hide the real peft library.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting.prompts import answer_turn
from prompting.run_prompting import environment_info, load_split, make_prompt, model_commit


def build_dataset(records, languages, tokenizer, cfg):
    # bilingual: first all English examples, then all Spanish
    rows = []
    for lang in languages:
        for record in records:
            prompt = make_prompt(tokenizer, record[f"text_{lang}"], cfg, [])
            answer = answer_turn(record["labels"])
            # The trainer adds the end token (<|im_end|> for Qwen3) after the answer.
            rows.append({"prompt": prompt, "completion": answer})
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    transformers.set_seed(cfg["seed"])
    out_dir = Path("outputs/peft") / cfg["experiment"]

    # 1. Data
    name = cfg["model"]["name"]
    revision = cfg["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    languages = cfg["train_languages"]
    train_data = build_dataset(load_split("train"), languages, tokenizer, cfg)
    dev_data = build_dataset(load_split("dev"), languages, tokenizer, cfg)
    print(f"[{cfg['experiment']}] train {len(train_data)} | dev {len(dev_data)} ({languages})")

    # 2. The base model, compressed to 4 bits (the "Q" in QLoRA)
    q = cfg["quantization"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],  # nf4
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=getattr(torch, cfg["model"]["dtype"]),
        quantization_config=quantization,
        device_map="auto",
    )

    # 3. The small LoRA layers we train
    lora = cfg["lora"]
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        target_modules=lora["target_modules"],
    )

    # 4. Training settings
    t = cfg["training"]
    training_config = SFTConfig(
        output_dir=str(out_dir),
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        max_length=t["max_length"],
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=cfg["model"]["dtype"] == "bfloat16",
        completion_only_loss=True,  # only learn from the answer, not from the prompt
        eval_strategy="epoch",  # compute the dev loss after every epoch
        save_strategy="epoch",  # save a checkpoint after every epoch
        load_best_model_at_end=True,  # at the end, go back to the best checkpoint
        metric_for_best_model="eval_loss",  # best = lowest dev loss
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_data,
        eval_dataset=dev_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

    # 5. Train
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    runtime = round(time.time() - start, 1)

    # 6. Save the best adapter and the training info
    trainer.save_model(str(out_dir / "adapter"))
    info = {
        "config": cfg,
        "config_file": args.config,
        "n_train": len(train_data),
        "n_dev": len(dev_data),
        "model_commit": model_commit(cfg),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_loss": trainer.state.best_metric,
        "train_runtime_seconds": runtime,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "log_history": trainer.state.log_history,  # all losses during training
    }
    info.update(environment_info())
    info["versions"]["peft"] = peft.__version__
    info["versions"]["trl"] = trl.__version__
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"Best checkpoint: {info['best_checkpoint']} (dev loss {info['best_dev_loss']})")
    print(f"Adapter saved to {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
