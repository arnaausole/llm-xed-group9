"""Read the answer of the model and compute the evaluation metrics.

All experiments (prompting and QLoRA) use this file, so they are scored the same way.
"""

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

# The eight emotions, always in this order.
LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def parse_output(text):
    """Turn the text of the model into a list of labels.

    Returns (labels, status). The status is:
      "ok"          we found at least one valid label
      "empty_list"  the model answered with an empty list: []
      "invalid"     there is no JSON list, or the list has no valid label
    For "empty_list" and "invalid" the labels are [] (we call these answers malformed).
    """
    if text is None:
        text = ""

    # Look for something like [ ... ] in the text and use the first one that is valid JSON.
    for match in re.finditer(r"\[[^\[\]]*\]", text):
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

        labels = []
        for item in items:
            if not isinstance(item, str):
                continue
            label = item.strip().lower()  # "  Joy " -> "joy"
            # Only the eight labels count; we do not translate synonyms ("happy" is not "joy").
            if label in LABELS and label not in labels:
                labels.append(label)
        labels.sort(key=LABELS.index)

        if len(labels) > 0:
            return labels, "ok"
        if len(items) == 0:
            return [], "empty_list"
        return [], "invalid"

    # No JSON list at all.
    return [], "invalid"


def to_matrix(label_lists):
    """Turn lists of labels into a 0/1 table: one row per line, one column per emotion."""
    matrix = np.zeros((len(label_lists), len(LABELS)), dtype=int)
    for row, labels in enumerate(label_lists):
        for label in labels:
            matrix[row, LABELS.index(label)] = 1
    return matrix


def compute_metrics(gold, pred, status=None):
    """Compare the predicted labels with the gold labels.

    If a metric divides by zero (e.g. an emotion is never predicted) it counts as 0.
    """
    y_true = to_matrix(gold)
    y_pred = to_matrix(pred)

    metrics = {}
    # primary metrics
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["jaccard_samples"] = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    # secondary metrics
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["micro_precision"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["micro_recall"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))  # lower is better
    # diagnostic metrics
    metrics["exact_match"] = float(accuracy_score(y_true, y_pred))  # all 8 labels right

    # precision, recall, F1 and number of gold examples for each emotion
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["per_label"] = {}
    for i, label in enumerate(LABELS):
        metrics["per_label"][label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics["n"] = len(gold)
    metrics["label_order"] = LABELS
    metrics["zero_division"] = 0
    metrics["sklearn_version"] = sklearn.__version__

    # how often the model gave no usable answer
    if status is not None:
        n = len(status)
        metrics["malformed_rate"] = (n - status.count("ok")) / n
        metrics["empty_list_rate"] = status.count("empty_list") / n
        metrics["invalid_rate"] = status.count("invalid") / n
    return metrics
