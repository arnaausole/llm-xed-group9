# Cross-Lingual Multi-Label Emotion Classification with LLMs (XED), Group 9

Course project for *Introduction to Large Language Models* (University of Groningen).

We compare prompting (zero-shot and five-shot) with parameter-efficient fine-tuning
(QLoRA) for multi-label emotion classification of movie subtitles from
[XED](https://github.com/Helsinki-NLP/XED), in English and Spanish (our assigned
language, band 5).

Labels: `anger, anticipation, disgust, fear, joy, sadness, surprise, trust`.

## Repository

```
data/        download XED, link English-Spanish pairs, remove duplicates, split
splits/      the fixed train/dev/test alignment IDs and the five-shot IDs
eval/        output parser and metrics (used by every experiment)
prompting/   prompts, five-shot example selection, zero-/five-shot runs
peft/        QLoRA training and evaluation
configs/     one YAML file per experiment
jobs/        SLURM scripts for Hábrók
results/     predictions, metrics and run info of every experiment
```

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone git@github.com:arnaausole/llm-xed-group9.git
cd llm-xed-group9
uv sync
```

Llama 3.1 is gated: accept the licence on its Hugging Face page and log in with a read
token (`uv run hf auth login`). Never commit the token. On Hábrók, set
`HF_HOME=/scratch/$USER/hf_cache` before logging in, since the jobs use that folder.

## Data

The dataset is not in this repository.

```bash
bash data/download_xed.sh                 # XED at a fixed commit
uv run python data/prepare_data.py        # data/processed/ + splits/
uv run python prompting/select_demos.py   # splits/fewshot_ids.txt
```

Both scripts use seed 42 and must give exactly the committed files in `splits/`
(the experiment scripts check this).

Of the 11,261 Spanish lines we could link to `pairs-es.txt`, 6,294 are aligned to a
Danish subtitle file (`da/...`), not an English one. This is also the case in the latest
XED version. English could only be reached through a second Danish-English alignment,
which is noisier, so we drop these lines. That leaves 4,967 English-Spanish pairs:
3,443 train / 1,016 dev / 508 test (70/20/10, multi-label stratified, duplicates kept in
the same partition).

## Experiments

| Condition | Model | Training / examples | Evaluated on |
|---|---|---|---|
| Zero-shot | Qwen3-8B + Llama-3.1-8B-Instruct | none | EN + ES |
| Five-shot | Qwen3-8B + Llama-3.1-8B-Instruct | 5 fixed training pairs | EN + ES |
| English QLoRA | Qwen3-8B | EN train | EN + ES |
| Spanish QLoRA | Qwen3-8B | ES train | ES |
| Bilingual QLoRA | Qwen3-8B | EN + ES train | EN + ES |

All settings (model version, prompt, decoding, LoRA, quantization, training, seed) are in
the config files. Every run also saves its settings, the model commit, library versions,
GPU and runtime in `run_info.json` (and `train_info.json` for training).

Main choices:
- Greedy decoding, max 64 new tokens. Qwen3 thinking mode is off.
- QLoRA: 4-bit NF4 base, LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP
  projections, lr 2e-4, 3 epochs, effective batch 16, max length 512. We keep the epoch
  with the lowest dev loss.
- For evaluation the adapter is put on the normal bf16 model (the same one used for
  prompting), so the adapter is the only difference.
- Metrics: micro-F1 and sample Jaccard (primary), macro-F1, micro precision/recall and
  Hamming loss (secondary), per-label scores, exact match and malformed-output rate.
  scikit-learn, fixed label order, `zero_division=0`.

## Running on Hábrók

Run from the repository root. Jobs write a `slurm-*.out` log in the same folder.

Development set (all decisions are made here):

```bash
# prompting
for c in qwen3-8b_zero-shot qwen3-8b_five-shot llama3.1-8b_zero-shot llama3.1-8b_five-shot; do
  sbatch jobs/prompting.slurm configs/prompting/$c.yaml dev
done

# prompt sensitivity: same runs with prompt v2 and v3
for c in qwen3-8b_zero-shot qwen3-8b_five-shot llama3.1-8b_zero-shot llama3.1-8b_five-shot; do
  for v in v2 v3; do
    sbatch jobs/prompting.slurm configs/prompting/$c.yaml dev --variant $v
  done
done

# QLoRA: train, then evaluate on dev
sbatch jobs/peft_train.slurm configs/peft/qwen3-8b_qlora_en.yaml
sbatch jobs/peft_train.slurm configs/peft/qwen3-8b_qlora_es.yaml
sbatch --time=10:00:00 jobs/peft_train.slurm configs/peft/qwen3-8b_qlora_bilingual.yaml
```

Test set (final evaluation, run once):

```bash
for c in qwen3-8b_zero-shot qwen3-8b_five-shot llama3.1-8b_zero-shot llama3.1-8b_five-shot; do
  sbatch jobs/prompting.slurm configs/prompting/$c.yaml test
done
for c in en es bilingual; do
  sbatch jobs/peft_eval.slurm configs/peft/qwen3-8b_qlora_$c.yaml test
done
```

Results are saved in `results/prompting/<experiment>/<split>/` and
`results/peft/<experiment>/<split>/`. Adapters and checkpoints go to `outputs/` and are
not committed.

## Reference

Öhman, E., Pàmies, M., Kajava, K. and Tiedemann, J. (2020). XED: A Multilingual Dataset
for Sentiment Analysis and Emotion Detection. *COLING 2020*.
