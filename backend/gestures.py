"""
Canonical server-side gesture labels for data collection and training.

Keep labels lowercase, trimmed, and unique. The frontend can display them in
title case, but the model should always learn one clean spelling per sign.
"""

GESTURES = [
    "hello",
    "yes",
    "no",
    "please",
    "thank you",
    "sorry",
    "help",
    "stop",
    "wait",
    "come",
    "go",
    "good",
    "ok",
    "friend",
    "you",
    "love",
    "call me",
    "rock on",
    "give",
    "home",
    "eat",
    "drink",
    "sleep",
    "bathroom",
    "money",
    "work",
    "school",
    "together",
    "bye",
]

LABEL_ALIASES = {
    "": None,
    "money ": "money",
    "i love you": "love",
    "goodbye": "bye",
    "thanks": "thank you",
    "come here": "come",
    "okay": "ok",
}


def canonical_label(label):
    """Return the single model label used for a raw collected label."""
    clean = str(label or "").strip().lower()
    return LABEL_ALIASES.get(clean, clean)


def validate_gestures():
    seen = set()
    duplicates = []
    for gesture in GESTURES:
        if gesture != gesture.strip() or gesture != gesture.lower():
            raise ValueError(f"Gesture label must be lowercase and trimmed: {gesture!r}")
        if gesture in seen:
            duplicates.append(gesture)
        seen.add(gesture)
    if duplicates:
        raise ValueError(f"Duplicate gesture labels: {duplicates}")
