"""The prompts we send to the models (used by prompting and by QLoRA).

The instruction is always in English, also for Spanish subtitles. It goes in the
"system" message. Then every subtitle is a "user" message and every answer an
"assistant" message. For five-shot we put five solved examples before the real question.

We have three versions of the instruction:
  v1  the main prompt (used for all final results)
  v2  the same idea with other words
  v3  v1 with the emotions listed in another order
v2 and v3 are only used on the dev set, to see how much the wording matters.
"""

import json

from eval.metrics import LABELS

# A short explanation of each emotion, shown to the model.
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

# The order of the emotions in prompt v3.
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
    """Write the emotions as a list, e.g. "- anger: annoyance, rage, hostility"."""
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
    """The correct answer as text, e.g. '["fear", "sadness"]' (labels in the fixed order)."""
    return json.dumps(sorted(labels, key=LABELS.index))


def build_messages(text, variant="v1", demos=()):
    """Make the chat messages for one subtitle.

    demos is a list of (subtitle, labels) examples; empty for zero-shot.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[variant]}]
    for demo_text, demo_labels in demos:
        messages.append({"role": "user", "content": f"Subtitle: {demo_text}"})
        messages.append({"role": "assistant", "content": answer_turn(demo_labels)})
    messages.append({"role": "user", "content": f"Subtitle: {text}"})
    return messages
