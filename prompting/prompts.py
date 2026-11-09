"""Prompts used for prompting and for QLoRA.

The instruction is in English for both languages and goes in the system message.
v1 is the main prompt. v2 (other wording) and v3 (other label order) are only run on dev.
"""

import json

from eval.metrics import LABELS

# short description of each emotion
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

V3_ORDER = ["trust", "fear", "joy", "disgust", "surprise", "anger", "sadness", "anticipation"]

V1 = """You are an annotator of emotions in movie subtitles.
Read one subtitle line and decide which emotions it expresses. Use only these eight labels:
{labels}

A line can express one emotion or several emotions at the same time; return every label that applies.
Every subtitle line expresses at least one of these emotions, even if it is subtle. Never return an empty list; if unsure, choose the most likely emotion.
Answer with a JSON list that contains only the label names above, written in English exactly as shown, and nothing else."""

V2 = """Task: multi-label emotion classification of a single movie subtitle line.
Allowed labels:
{labels}

Every line expresses at least one of these emotions, even if subtle; never return an empty list.
Return all labels that apply (one or more) as a JSON list of English label names, e.g. ["<label>", "<label>"]. Output only the JSON list."""


def label_list(order):
    lines = []
    for label in order:
        lines.append(f"- {label}: {DEFINITIONS[label]}")
    return "\n".join(lines)


SYSTEM_PROMPTS = {
    "v1": V1.format(labels=label_list(LABELS)),
    "v2": V2.format(labels=label_list(LABELS)),
    "v3": V1.format(labels=label_list(V3_ORDER)),
}


def answer_turn(labels):
    # e.g. '["fear", "sadness"]'
    return json.dumps(sorted(labels, key=LABELS.index))


def build_messages(text, variant="v1", demos=()):
    # demos = list of (text, labels), empty for zero-shot
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[variant]}]
    for demo_text, demo_labels in demos:
        messages.append({"role": "user", "content": f"Subtitle: {demo_text}"})
        messages.append({"role": "assistant", "content": answer_turn(demo_labels)})
    messages.append({"role": "user", "content": f"Subtitle: {text}"})
    return messages
