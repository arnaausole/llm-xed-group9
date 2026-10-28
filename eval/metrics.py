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
- For diagnosis, every output gets a status: "ok", "empty_list" (a valid but empty
  JSON list, i.e. the model answered "no emotion") or "invalid" (no JSON list, or only
  non-permitted labels). Malformed = "empty_list" or "invalid".

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


def parse_output(text: str) -> tuple[list[str], str]:
    """Return (labels, status) with status "ok", "empty_list" or "invalid"."""
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
        if labels:
            return labels, "ok"
        return [], "empty_list" if not items else "invalid"
    return [], "invalid"


def binarize(label_lists: list[list[str]]) -> np.ndarray:
    y = np.zeros((len(label_lists), len(LABELS)), dtype=int)
    for i, labs in enumerate(label_lists):
        for lab in labs:
            y[i, LABELS.index(lab)] = 1
    return y


def compute_metrics(
    gold: list[list[str]],
    pred: list[list[str]],
    status: list[str] | None = None,
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
    if status is not None:
        n = max(len(status), 1)
        out["malformed_rate"] = sum(s != "ok" for s in status) / n
        out["empty_list_rate"] = status.count("empty_list") / n
        out["invalid_rate"] = status.count("invalid") / n
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}


if __name__ == "__main__":
    # Quick self-check: uv run python eval/metrics.py
    outputs = [
        '["joy", "Trust"]',
        'Answer: ["anger"]',
        '["happy"]',  # synonym -> not mapped -> invalid
        "I think it's fear",  # no JSON -> invalid
        '["fear", "fear", "surprise"]',
        "[]",  # valid JSON, no emotion -> empty_list
    ]
    gold = [["joy", "trust"], ["anger"], ["joy"], ["fear"], ["fear"], ["sadness"]]
    parsed = [parse_output(o) for o in outputs]
    pred, status = [p for p, _ in parsed], [s for _, s in parsed]
    print(pred, status)
    m = compute_metrics(gold, pred, status)
    print({k: v for k, v in m.items() if k != "per_label"})
