"""The reading measures the model predicts, in the order of its output layer.

Eye-tracking (ET) measures:

* ``ffd``: first-fixation duration, the first fixation on the word during first pass.
* ``gd``: gaze duration, all first-pass fixations before the eyes leave the word.
* ``gpt``: go-past time, from entering the word until a word to its right is fixated,
  regressions to earlier text included.
* ``trt``: total reading time, every fixation on the word.
* ``fix_prob``: probability that the word is fixated during first pass (not skipped).
* ``reg_prob``: probability that the first-pass reading of the word ends with a
  regression to earlier text.

Self-paced reading (SPR) measure:

* ``rt``: time the word stays on screen before the reader asks for the next one.

Durations are population averages over readers: geometric means in milliseconds, and for
the eye-tracking durations only over the readers who fixated the word.
"""

from __future__ import annotations

DURATIONS = ("ffd", "gd", "gpt", "trt")
"""Eye-tracking durations, in milliseconds."""

PROBABILITIES = ("fix_prob", "reg_prob")
"""Eye-tracking probabilities, between 0 and 1."""

SPR = ("rt",)
"""Self-paced reading times, in milliseconds."""

TARGETS = DURATIONS + PROBABILITIES + SPR
"""Every measure, in the order of the model's output layer."""

LOG_TARGETS = DURATIONS + SPR
"""Measures predicted as z-scored log milliseconds."""

PARADIGMS = ("et", "spr")
"""Measurement paradigms: eye tracking and self-paced reading."""

READER_GROUPS = ("L1", "L2")
"""Native (L1) and non-native (L2) readers."""

PARADIGM_MEASURES = {"et": DURATIONS + PROBABILITIES, "spr": SPR}
"""Measures reported for each paradigm."""

DESCRIPTIONS = {
    "ffd": "first-fixation duration (ms)",
    "gd": "gaze duration (ms)",
    "gpt": "go-past time (ms)",
    "trt": "total reading time (ms)",
    "fix_prob": "first-pass fixation probability",
    "reg_prob": "first-pass regression probability",
    "rt": "self-paced reading time (ms)"
}
"""Short description of every measure."""
