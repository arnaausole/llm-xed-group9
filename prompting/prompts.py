"""Prompt templates shared by prompting and PEFT (project brief §3).

The instruction is always in English, for English and Spanish subtitles alike. It goes in
the system message; every subtitle (demonstration or query) is one user turn and every
answer one assistant turn holding a JSON list. Five-shot prompting therefore uses exactly
the same turns as zero-shot, with five demonstration turn pairs in front of the query.

Variants (for the prompt-sensitivity analysis on the development set only):
- v1: main prompt, labels with short definitions, alphabetical order.
- v2: same definitions, reworded and more compact instruction.
- v3: v1 with the labels listed in a different (fixed) order.
"""

from __future__ import annotations

import json

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

DEFINITIONS = {
    "anger": "annoyance, rage, hostility",
    "anticipation": "expectation, interest, looking forward to or planning something",
    "disgust": "revulsion, contempt, strong disapproval",
    "fear": "worry, anxiety, terror",
    "joy": "happiness, pleasure, amusement",
    "sadness": "grief, disappointment, regret",
    "surprise": "astonishment, disbelief, reaction to something unexpected",
    "trust": "acceptance, admiration, reassurance, confidence in someone",
}

V3_ORDER = [
    "trust",
    "fear",
    "joy",
    "disgust",
    "surprise",
    "anger",
    "sadness",
    "anticipation",
]


def _label_block(order: list[str]) -> str:
    return "\n".join(f"- {lab}: {DEFINITIONS[lab]}" for lab in order)


_V1 = """You are an annotator of emotions in movie subtitles.
Read one subtitle line and decide which emotions it expresses. Use only these eight labels:
{labels}

A line can express one emotion or several emotions at the same time; return every label that applies, and at least one.
Answer with a JSON list that contains only the label names above, written in English exactly as shown, and nothing else."""

_V2 = """Task: multi-label emotion classification of a single movie subtitle line.
Allowed labels:
{labels}

Return all labels that apply (one or more) as a JSON list of English label names, e.g. ["<label>", "<label>"]. Output only the JSON list."""

SYSTEM_PROMPTS = {
    "v1": _V1.format(labels=_label_block(LABELS)),
    "v2": _V2.format(labels=_label_block(LABELS)),
    "v3": _V1.format(labels=_label_block(V3_ORDER)),
}


def user_turn(text: str) -> str:
    return f"Subtitle: {text}"


def answer_turn(labels: list[str]) -> str:
    return json.dumps(sorted(labels, key=LABELS.index))


def build_messages(
    text: str,
    variant: str = "v1",
    demos: list[tuple[str, list[str]]] | None = None,
) -> list[dict]:
    """Chat messages for one query; ``demos`` is a list of (text, labels) pairs."""
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[variant]}]
    for demo_text, demo_labels in demos or []:
        messages.append({"role": "user", "content": user_turn(demo_text)})
        messages.append({"role": "assistant", "content": answer_turn(demo_labels)})
    messages.append({"role": "user", "content": user_turn(text)})
    return messages
