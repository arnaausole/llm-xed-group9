"""Shared output parser and multi-label metrics (project brief §3 and §7).

Every experiment (prompting and PEFT) must use this module so that all numbers are
computed from the same binarised eight-label predictions with the same policy.

Parser policy
-------------
- Find the first JSON list in the model output.
- Normalise case and surrounding whitespace of each element.
- Keep only the eight permitted English labels; NO synonym mapping ("happy" != "joy").
- Remove duplicate labels.
- Malformed output (no parsable JSON list) or a list with no permitted label
  -> empty prediction, flagged as malformed and reported separately.

Metric policy
-------------
- Fixed label order: LABELS below.
- zero_division=0 for every sklearn metric.
"""

from __future__ import annotations

import json
import re

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

LABELS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]
ZERO_DIVISION = 0

_LIST_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def parse_output(text: str) -> tuple[list[str], bool]:
    """Return (labels, malformed)."""
    for match in _LIST_RE.finditer(text or ""):
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(items, list):
            continue
        labels = []
        for item in items:
            if isinstance(item, str):
                lab = item.strip().lower()
                if lab in LABELS and lab not in labels:
                    labels.append(lab)
        labels.sort(key=LABELS.index)
        return labels, len(labels) == 0
    return [], True


def binarize(label_lists: list[list[str]]) -> np.ndarray:
    y = np.zeros((len(label_lists), len(LABELS)), dtype=int)
    for i, labs in enumerate(label_lists):
        for lab in labs:
            y[i, LABELS.index(lab)] = 1
    return y


def compute_metrics(
    gold: list[list[str]],
    pred: list[list[str]],
    malformed: list[bool] | None = None,
) -> dict:
    y_true, y_pred = binarize(gold), binarize(pred)
    zd = ZERO_DIVISION
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=zd
    )
    out = {
        # primary
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=zd),
        "jaccard_samples": jaccard_score(
            y_true, y_pred, average="samples", zero_division=zd
        ),
        # secondary
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=zd),
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=zd
        ),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=zd),
        "hamming_loss": hamming_loss(y_true, y_pred),
        # diagnostic
        "exact_match": accuracy_score(y_true, y_pred),
        "per_label": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(LABELS)
        },
        "n": len(gold),
        "label_order": LABELS,
        "zero_division": zd,
        "sklearn_version": sklearn.__version__,
    }
    if malformed is not None:
        out["malformed_rate"] = float(np.mean(malformed)) if malformed else 0.0
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}


if __name__ == "__main__":
    # Quick self-check: uv run python eval/metrics.py
    outputs = [
        '["joy", "Trust"]',
        'Answer: ["anger"]',
        '["happy"]',  # synonym -> not mapped -> malformed
        "I think it's fear",  # no JSON -> malformed
        '["fear", "fear", "surprise"]',
    ]
    gold = [["joy", "trust"], ["anger"], ["joy"], ["fear"], ["fear"]]
    parsed = [parse_output(o) for o in outputs]
    pred, mal = [p for p, _ in parsed], [m for _, m in parsed]
    print(pred, mal)
    m = compute_metrics(gold, pred, mal)
    print({k: v for k, v in m.items() if k != "per_label"})
