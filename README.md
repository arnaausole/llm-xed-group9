# Cross-Lingual Multi-Label Emotion Classification with LLMs (XED) — Group 9

Course project for *Introduction to Large Language Models* (University of Groningen).

We compare **prompt engineering** (zero-shot and five-shot) with **parameter-efficient
fine-tuning** (QLoRA) for multi-label emotion classification of movie subtitles from the
[XED dataset](https://github.com/Helsinki-NLP/XED), on **English** and **Spanish**
(assigned language, resource band 5, ~11,303 projected lines; 4,967 matched English–Spanish pairs after removing Danish-source pairs, see Data).

Label space (Plutchik): `anger, anticipation, disgust, fear, joy, sadness, surprise, trust`.

## Repository structure

```
data/        loading XED, pairing EN–ES by alignment ID, deduplication, stratified split
splits/      fixed train/dev/test alignment IDs (committed for reproducibility)
eval/        output parser and multi-label metrics (shared by all experiments)
prompting/   zero-shot and five-shot prompting
peft/        QLoRA training (English-only, Spanish-only, bilingual)
configs/     one YAML config per experiment
jobs/        SLURM job scripts for Hábrók
results/     predictions and metrics per experiment
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone <repo-url>
cd llm-xed-group9
uv sync
```

On Hábrók, keep caches out of the home directory:

```bash
export HF_HOME=/scratch/$USER/hf_cache
export UV_CACHE_DIR=/scratch/$USER/uv_cache
```

## Data

The dataset is **not** included in this repository.

```bash
bash data/download_xed.sh                                     # XED at a pinned commit
uv run python data/prepare_data.py --xed data/raw/XED --seed 42
```

Of the 11,261 Spanish projection rows linked to `pairs-es.txt`, 6,294 come from a
**Danish** source subtitle file (`da/...`), so their "English" side is Danish. This is
the same in the latest upstream XED version and in every `pairs-*.txt` file. English
lines could only be reached through a second, Danish–English alignment
(`pairs-en.txt`), which is noisier, so these pairs are dropped (see
`data/prepare_data.py`). This leaves **4,967 directly aligned English–Spanish pairs**:
3,443 train / 1,016 dev / 508 test.
The committed `splits/*_ids.txt` must be reproduced exactly; the scripts check this.

## Reproducing results

### Prompting

Models: primary `Qwen/Qwen3-8B` (thinking mode disabled), second
`meta-llama/Llama-3.1-8B-Instruct` (gated: accept the licence on Hugging Face and log in
with a read token on the machine that runs it; never commit the token).

```bash
# five fixed demonstrations from the training set (already committed in splits/)
uv run python prompting/select_demos.py --seed 42

# development set (all prompt decisions are made here)
sbatch jobs/prompting.slurm configs/prompting/qwen3-8b_zero-shot.yaml dev
sbatch jobs/prompting.slurm configs/prompting/qwen3-8b_five-shot.yaml dev
sbatch jobs/prompting.slurm configs/prompting/llama3.1-8b_zero-shot.yaml dev
sbatch jobs/prompting.slurm configs/prompting/llama3.1-8b_five-shot.yaml dev

# final evaluation, once the prompt is frozen: replace dev with test
```

Outputs go to `results/prompting/<experiment>/<split>/` (predictions, rendered prompt,
metrics, run info).

_To be completed: PEFT training and evaluation._

## Experimental matrix

| Condition | Model(s) | Training / demonstrations | Evaluated on |
|---|---|---|---|
| Zero-shot | Primary + second | None | EN + ES |
| Five-shot | Primary + second | 5 fixed aligned train examples | EN + ES |
| English-only PEFT | Primary | EN train | EN + ES |
| Spanish-only PEFT | Primary | ES train | ES |
| Bilingual PEFT | Primary | EN + ES train | EN + ES |

## Citation

Öhman, E., Pàmies, M., Kajava, K. and Tiedemann, J. (2020). XED: A Multilingual Dataset for
Sentiment Analysis and Emotion Detection. *COLING 2020*.