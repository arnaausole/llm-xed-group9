"""Make all tables and figures for the report from the saved results.

It only reads files in results/ (no GPU needed). It writes to results/analysis/:
  tables.md     all tables in Markdown, ready to copy into the report
  fig_*.png     the figures

Confidence intervals: we use the bootstrap. We pick 508 test lines at random (with
replacement), compute the metric, and repeat this 1000 times. The middle 95% of these
1000 values is the 95% confidence interval. To compare two systems we use the same
random lines for both ("paired bootstrap"): if the interval of the difference does not
contain 0, the difference is probably real and not luck.

How to run it:
  uv run python analysis/analyze.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, do not open a window
import matplotlib.pyplot as plt
import numpy as np

# Let Python find our own folders.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import LABELS, to_matrix

RESULTS = Path("results")
OUT = RESULTS / "analysis"
N_BOOTSTRAP = 1000
SEED = 42

# (folder in results/, name in tables and figures)
SYSTEMS = [
    ("prompting/qwen3-8b_zero-shot", "Qwen zero-shot"),
    ("prompting/qwen3-8b_five-shot", "Qwen five-shot"),
    ("peft/qwen3-8b_qlora_en", "Qwen QLoRA EN"),
    ("peft/qwen3-8b_qlora_es", "Qwen QLoRA ES"),
    ("peft/qwen3-8b_qlora_bilingual", "Qwen QLoRA EN+ES"),
    ("prompting/llama3.1-8b_zero-shot", "Llama zero-shot"),
    ("prompting/llama3.1-8b_five-shot", "Llama five-shot"),
]
LANGUAGES = ["en", "es"]

# Paired comparisons on test: (language, system A, system B). We report A - B.
COMPARISONS = [
    ("en", "Qwen five-shot", "Qwen zero-shot"),
    ("en", "Qwen QLoRA EN", "Qwen five-shot"),
    ("en", "Qwen QLoRA EN+ES", "Qwen QLoRA EN"),
    ("en", "Llama five-shot", "Qwen QLoRA EN"),
    ("es", "Qwen five-shot", "Qwen zero-shot"),
    ("es", "Qwen QLoRA EN", "Qwen five-shot"),
    ("es", "Qwen QLoRA ES", "Qwen QLoRA EN"),
    ("es", "Qwen QLoRA EN+ES", "Qwen QLoRA ES"),
    ("es", "Llama five-shot", "Qwen QLoRA ES"),
]

# Colors (checked to be readable for colorblind people)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LANGUAGE_COLOR = {"en": BLUE, "es": ORANGE}
LANGUAGE_NAME = {"en": "English", "es": "Spanish"}


# reading results


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_predictions(folder, split, lang):
    """The predictions of one system, or None if this system was not run on this language."""
    path = RESULTS / folder / split / f"predictions_{lang}.jsonl"
    if not path.exists():
        return None
    return read_jsonl(path)


def load_metrics(folder, split):
    path = RESULTS / folder / split / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# metrics per line, so the bootstrap is fast


def line_counts(rows):
    """For every line: true positives, false positives, false negatives and Jaccard."""
    gold = to_matrix([row["gold"] for row in rows])
    pred = to_matrix([row["pred"] for row in rows])
    tp = (gold & pred).sum(axis=1)
    fp = ((1 - gold) & pred).sum(axis=1)
    fn = (gold & (1 - pred)).sum(axis=1)
    union = (gold | pred).sum(axis=1)
    jaccard = np.where(union > 0, tp / np.maximum(union, 1), 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "jaccard": jaccard}


def micro_f1(counts, lines):
    tp = counts["tp"][lines].sum()
    fp = counts["fp"][lines].sum()
    fn = counts["fn"][lines].sum()
    return 2 * tp / (2 * tp + fp + fn)


def mean_jaccard(counts, lines):
    return counts["jaccard"][lines].mean()


def bootstrap_samples(n_lines):
    """1000 lists of random line numbers (the same for every system)."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n_lines, size=(N_BOOTSTRAP, n_lines))


def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return low, high


# small helpers for the tables


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def f3(x):
    return f"{x:.3f}"


def percent(x):
    return f"{100 * x:.1f}%"


# the tables


def results_table(split):
    header = ["System", "Lang", "micro-F1", "Jaccard", "macro-F1", "micro-P", "micro-R",
              "Hamming", "Exact match", "Malformed"]  # fmt: skip
    rows = []
    for folder, name in SYSTEMS:
        metrics = load_metrics(folder, split)
        for lang in LANGUAGES:
            if lang not in metrics:
                continue
            m = metrics[lang]
            rows.append([name, lang, f3(m["micro_f1"]), f3(m["jaccard_samples"]), f3(m["macro_f1"]),
                         f3(m["micro_precision"]), f3(m["micro_recall"]), f3(m["hamming_loss"]),
                         f3(m["exact_match"]), percent(m["malformed_rate"])])  # fmt: skip
    return markdown_table(header, rows)


def confidence_tables(counts, samples):
    """Table with 95% intervals for each system, and table with paired differences."""
    rows = []
    ci = {}  # used again for the figure
    for _, name in SYSTEMS:
        for lang in LANGUAGES:
            if (name, lang) not in counts:
                continue
            c = counts[(name, lang)]
            all_lines = np.arange(len(c["tp"]))
            f1_values = [micro_f1(c, lines) for lines in samples]
            jac_values = [mean_jaccard(c, lines) for lines in samples]
            f1_low, f1_high = interval(f1_values)
            jac_low, jac_high = interval(jac_values)
            ci[(name, lang)] = (micro_f1(c, all_lines), f1_low, f1_high)
            rows.append([name, lang,
                         f"{micro_f1(c, all_lines):.3f} [{f1_low:.3f}, {f1_high:.3f}]",
                         f"{mean_jaccard(c, all_lines):.3f} [{jac_low:.3f}, {jac_high:.3f}]"])  # fmt: skip
    systems_table = markdown_table(["System", "Lang", "micro-F1 [95% CI]", "Jaccard [95% CI]"], rows)

    rows = []
    for lang, a, b in COMPARISONS:
        ca, cb = counts[(a, lang)], counts[(b, lang)]
        all_lines = np.arange(len(ca["tp"]))
        for metric_name, metric in [("micro-F1", micro_f1), ("Jaccard", mean_jaccard)]:
            difference = metric(ca, all_lines) - metric(cb, all_lines)
            differences = [metric(ca, lines) - metric(cb, lines) for lines in samples]
            low, high = interval(differences)
            real = "yes" if low > 0 or high < 0 else "no"
            rows.append([lang, f"{a} − {b}", metric_name, f"{difference:+.3f}",
                         f"[{low:+.3f}, {high:+.3f}]", real])  # fmt: skip
    header = ["Lang", "Comparison", "Metric", "Difference", "95% CI", "CI excludes 0"]
    comparisons_table = markdown_table(header, rows)
    return systems_table, comparisons_table, ci


def language_gap_table(counts, samples):
    """English minus Spanish for the same system. The lines are aligned, so this is paired too."""
    rows = []
    for _, name in SYSTEMS:
        if (name, "en") not in counts or (name, "es") not in counts:
            continue
        c_en, c_es = counts[(name, "en")], counts[(name, "es")]
        all_lines = np.arange(len(c_en["tp"]))
        difference = micro_f1(c_en, all_lines) - micro_f1(c_es, all_lines)
        differences = [micro_f1(c_en, lines) - micro_f1(c_es, lines) for lines in samples]
        low, high = interval(differences)
        rows.append([name, f"{difference:+.3f}", f"[{low:+.3f}, {high:+.3f}]"])
    return markdown_table(["System", "micro-F1 EN − ES", "95% CI"], rows)


def label_count_table(split):
    """How many labels each system predicts per line, compared with the gold labels."""
    rows = []
    for folder, name in SYSTEMS:
        for lang in LANGUAGES:
            preds = load_predictions(folder, split, lang)
            if preds is None:
                continue
            n_pred = np.mean([len(row["pred"]) for row in preds])
            n_gold = np.mean([len(row["gold"]) for row in preds])
            m = load_metrics(folder, split)[lang]
            rows.append([name, lang, f"{n_pred:.2f}", f"{n_gold:.2f}",
                         f3(m["micro_precision"]), f3(m["micro_recall"])])  # fmt: skip
    header = ["System", "Lang", "Labels/line (pred)", "Labels/line (gold)", "micro-P", "micro-R"]
    return markdown_table(header, rows)


def per_label_table(split, lang):
    header = ["System"] + LABELS
    rows = []
    support = None
    for folder, name in SYSTEMS:
        metrics = load_metrics(folder, split)
        if lang not in metrics:
            continue
        per_label = metrics[lang]["per_label"]
        rows.append([name] + [f"{per_label[label]['f1']:.2f}" for label in LABELS])
        support = [str(per_label[label]["support"]) for label in LABELS]
    rows.append(["*support (gold lines)*"] + support)
    return markdown_table(header, rows)


def prompt_sensitivity_table():
    rows = []
    for folder, name in SYSTEMS:
        if not folder.startswith("prompting/"):
            continue
        for lang in LANGUAGES:
            scores = []
            for suffix in ["", "_v2", "_v3"]:
                scores.append(load_metrics(folder + suffix, "dev")[lang]["micro_f1"])
            rows.append([name, lang] + [f3(s) for s in scores] + [f3(max(scores) - min(scores))])
    return markdown_table(["System", "Lang", "v1 (main)", "v2", "v3", "Range"], rows)


def malformed_table(split):
    rows = []
    for folder, name in SYSTEMS:
        metrics = load_metrics(folder, split)
        for lang in LANGUAGES:
            if lang not in metrics:
                continue
            m = metrics[lang]
            rows.append([name, lang, percent(m["malformed_rate"]), percent(m["empty_list_rate"]),
                         percent(m["invalid_rate"])])  # fmt: skip
    return markdown_table(["System", "Lang", "Malformed", "Empty list []", "Invalid"], rows)


def cost_table():
    rows = []
    for folder, name in SYSTEMS:
        run_info = json.loads((RESULTS / folder / "test" / "run_info.json").read_text())
        seconds = sum(run_info["runtime_seconds"].values())
        per_line = 1000 * seconds / (run_info["n_examples"] * len(run_info["runtime_seconds"]))
        train_minutes, memory, trainable = "–", "–", "–"
        train_info_path = RESULTS / folder / "train_info.json"
        if train_info_path.exists():
            train_info = json.loads(train_info_path.read_text())
            train_minutes = f"{train_info['train_runtime_seconds'] / 60:.0f}"
            memory = f"{train_info['peak_gpu_memory_gb']:.1f}"
            trainable = f"{train_info['trainable_parameters'] / 1e6:.1f}M ({100 * train_info['trainable_parameters'] / train_info['total_parameters']:.2f}%)"
        rows.append([name, f"{per_line:.0f}", train_minutes, memory, trainable])
    header = [
        "System",
        "Inference ms/line (test)",
        "Training min",
        "Training peak GPU GB",
        "Trainable params",
    ]
    return markdown_table(header, rows) + f"\n\nGPU: {run_info['device']}."


def training_table():
    rows = []
    for folder, name in SYSTEMS:
        path = RESULTS / folder / "train_info.json"
        if not path.exists():
            continue
        info = json.loads(path.read_text())
        dev_losses = [entry["eval_loss"] for entry in info["log_history"] if "eval_loss" in entry]
        best_epoch = dev_losses.index(min(dev_losses)) + 1
        rows.append([name, info["n_train"]] + [f"{loss:.3f}" for loss in dev_losses] + [best_epoch])
    return markdown_table(
        ["System", "Train examples", "Dev loss ep. 1", "ep. 2", "ep. 3", "Best epoch"], rows
    )


# figures


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_axisbelow(True)


def figure_main_results(ci):
    """micro-F1 on test with 95% intervals, English and Spanish next to each other."""
    names = [name for _, name in SYSTEMS]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bar_height = 0.38
    for i, lang in enumerate(LANGUAGES):
        for row, name in enumerate(names):
            if (name, lang) not in ci:
                continue
            value, low, high = ci[(name, lang)]
            y = row + (i - 0.5) * (bar_height + 0.02)
            ax.barh(y, value, height=bar_height, color=LANGUAGE_COLOR[lang],
                    label=LANGUAGE_NAME[lang] if row == 0 else None)  # fmt: skip
            ax.plot([low, high], [y, y], color=INK, linewidth=1)
            ax.text(high + 0.008, y, f"{value:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("micro-F1 on test (line = 95% bootstrap CI)")
    ax.grid(axis="x", color=GRID)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_results.png", dpi=200)
    plt.close(fig)


def figure_precision_recall(split):
    """Precision against recall: shows which systems predict many or few labels."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True)
    for ax, lang in zip(axes, LANGUAGES):
        for folder, name in SYSTEMS:
            metrics = load_metrics(folder, split)
            if lang not in metrics:
                continue
            if "QLoRA" in name:
                color = AQUA
            elif "Llama" in name:
                color = ORANGE
            else:
                color = BLUE
            p, r = metrics[lang]["micro_precision"], metrics[lang]["micro_recall"]
            ax.scatter(r, p, s=45, color=color, edgecolors="white", linewidths=1.5, zorder=3)
            ax.annotate(name, (r, p), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(LANGUAGE_NAME[lang], color=INK)
        ax.set_xlabel("micro recall")
        ax.grid(color=GRID)
        clean_axes(ax)
    axes[0].set_ylabel("micro precision")
    fig.tight_layout()
    fig.savefig(OUT / "fig_precision_recall.png", dpi=200)
    plt.close(fig)


def figure_per_label(split):
    """Heatmap of F1 per emotion: rows are systems, columns are emotions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, lang in zip(axes, LANGUAGES):
        names, table = [], []
        for folder, name in SYSTEMS:
            metrics = load_metrics(folder, split)
            if lang not in metrics:
                continue
            names.append(name)
            table.append([metrics[lang]["per_label"][label]["f1"] for label in LABELS])
        table = np.array(table)
        ax.imshow(table, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                text_color = "white" if table[row, col] > 0.5 else INK
                ax.text(
                    col, row, f"{table[row, col]:.2f}", ha="center", va="center", fontsize=7, color=text_color
                )
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_title(f"F1 per emotion, {LANGUAGE_NAME[lang]} (test)", color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_label_f1.png", dpi=200)
    plt.close(fig)


def figure_training_curves():
    """Training loss during training and dev loss after each epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    colors = [BLUE, ORANGE, AQUA]
    qlora_systems = [(folder, name) for folder, name in SYSTEMS if "QLoRA" in name]
    for (folder, name), color in zip(qlora_systems, colors):
        info = json.loads((RESULTS / folder / "train_info.json").read_text())
        train = [(e["epoch"], e["loss"]) for e in info["log_history"] if "loss" in e]
        dev = [(e["epoch"], e["eval_loss"]) for e in info["log_history"] if "eval_loss" in e]
        axes[0].plot([x for x, _ in train], [y for _, y in train], color=color, linewidth=1.5, label=name)
        axes[1].plot(
            [x for x, _ in dev], [y for _, y in dev], color=color, linewidth=2, marker="o", label=name
        )
    axes[0].set_title("Training loss", color=INK)
    axes[0].set_ylim(0, 0.8)
    axes[1].set_title("Dev loss (after each epoch)", color=INK)
    axes[1].set_xticks([1, 2, 3])
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(color=GRID)
        clean_axes(ax)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curves.png", dpi=200)
    plt.close(fig)


def figure_prompt_sensitivity():
    """micro-F1 on dev for prompts v1, v2 and v3; the grey line shows the range."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    variants = [("", "v1 (main)", "o", BLUE), ("_v2", "v2", "s", ORANGE), ("_v3", "v3", "^", AQUA)]
    labels = []
    row = 0
    for folder, name in SYSTEMS:
        if not folder.startswith("prompting/"):
            continue
        for lang in LANGUAGES:
            scores = [load_metrics(folder + suffix, "dev")[lang]["micro_f1"] for suffix, _, _, _ in variants]
            ax.plot([min(scores), max(scores)], [row, row], color=GRID, linewidth=4, zorder=1)
            for score, (_, variant_name, marker, color) in zip(scores, variants):
                ax.scatter(score, row, marker=marker, s=50, color=color, edgecolors="white",
                           linewidths=1.5, zorder=3, label=variant_name if row == 0 else None)  # fmt: skip
            labels.append(f"{name} ({lang})")
            row += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("micro-F1 on dev")
    ax.grid(axis="x", color=GRID)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prompt_sensitivity.png", dpi=200)
    plt.close(fig)


# main


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Counts per line for every system and language on test
    counts = {}
    for folder, name in SYSTEMS:
        for lang in LANGUAGES:
            preds = load_predictions(folder, "test", lang)
            if preds is not None:
                counts[(name, lang)] = line_counts(preds)
    n_test = len(next(iter(counts.values()))["tp"])
    samples = bootstrap_samples(n_test)

    systems_ci, comparisons, ci = confidence_tables(counts, samples)

    sections = [
        ("1. Test results", results_table("test")),
        ("2. Dev results", results_table("dev")),
        (f"3. Test results with 95% bootstrap confidence intervals ({N_BOOTSTRAP} samples, seed {SEED})",
         systems_ci),
        ("4. Paired comparisons on test (A − B)", comparisons),
        ("5. Language gap on test (same lines in English and Spanish)", language_gap_table(counts, samples)),
        ("6. Number of predicted labels, precision and recall (test)", label_count_table("test")),
        ("7a. F1 per emotion, English (test)", per_label_table("test", "en")),
        ("7b. F1 per emotion, Spanish (test)", per_label_table("test", "es")),
        ("8. Prompt sensitivity, micro-F1 on dev", prompt_sensitivity_table()),
        ("9a. Malformed outputs (test)", malformed_table("test")),
        ("9b. Malformed outputs (dev)", malformed_table("dev")),
        ("10. QLoRA training (dev loss per epoch)", training_table()),
        ("11. Computational cost", cost_table()),
    ]  # fmt: skip
    text = "# Results tables\n\nMade by analysis/analyze.py from the files in results/.\n"
    for title, table in sections:
        text += f"\n## {title}\n\n{table}\n"
    (OUT / "tables.md").write_text(text, encoding="utf-8")

    figure_main_results(ci)
    figure_precision_recall("test")
    figure_per_label("test")
    figure_training_curves()
    figure_prompt_sensitivity()
    print(f"Saved tables.md and figures to {OUT}")


if __name__ == "__main__":
    main()
