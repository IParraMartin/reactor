"""Text handling: joining words, conditioning prompts, windows and subword alignment.

The model reads a text as a single string preceded by a short plain-text prompt that
names the language, the measurement paradigm, the reading task and the reader group.
Predictions are made per word, so every word must be mapped back to the subword tokens
the backbone produced for it; long texts are cut into overlapping windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_SPACE_LANGUAGES = frozenset({"zh", "ja", "yue", "th"})
"""Languages written without spaces between words; their words are concatenated."""

LANGUAGE_NAMES = {
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "he": "Hebrew",
    "hi": "Hindi",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "no": "Norwegian",
    "pt": "Portuguese",
    "ru": "Russian",
    "sr": "Serbian",
    "sv": "Swedish",
    "tr": "Turkish",
    "yue": "Cantonese",
    "zh": "Chinese"
}
"""English names of ISO 639 codes, as written in the conditioning prompt."""

PARADIGM_NAMES = {"et": "eye-tracking", "spr": "self-paced reading"}
"""Prompt names of the two measurement paradigms."""

READER_GROUP_NAMES = {"L1": "native speakers", "L2": "non-native speakers"}
"""Prompt names of the two reader groups."""


def join_words(words: list[str], language: str) -> tuple[str, list[tuple[int, int]]]:
    """Joins words into the string the model reads.

    Args:
        words: Words as displayed to readers, punctuation attached.
        language: ISO 639 code; languages in ``NO_SPACE_LANGUAGES`` are concatenated,
            all others are joined with single spaces.

    Returns:
        The joined text and the ``(start, end)`` character span of every word in it.
    """
    separator = "" if language in NO_SPACE_LANGUAGES else " "
    spans = []
    position = 0
    for i, word in enumerate(words):
        if i > 0:
            position += len(separator)
        spans.append((position, position + len(word)))
        position += len(word)
    return separator.join(words), spans


def conditioning_prompt(language: str, paradigm: str, condition: str, reader_group: str) -> str:
    """Builds the plain-text prefix that tells the model whose reading it predicts.

    The prompt is part of the model input exactly as during training, e.g.
    ``"Language: English. Measure: eye-tracking. Task: reading for comprehension.
    Readers: native speakers.\\n\\n"``.

    Args:
        language: ISO 639 code of the text. Codes without an entry in
            ``LANGUAGE_NAMES`` are written as given.
        paradigm: ``"et"`` (eye tracking) or ``"spr"`` (self-paced reading).
        condition: Human-readable task description, e.g. ``"reading for comprehension"``.
        reader_group: ``"L1"`` (native) or ``"L2"`` (non-native readers).

    Returns:
        The prefix, ending in a blank line after which the text follows.
    """
    return (
        f"Language: {LANGUAGE_NAMES.get(language, language)}. "
        f"Measure: {PARADIGM_NAMES[paradigm]}. "
        f"Task: {condition}. "
        f"Readers: {READER_GROUP_NAMES[reader_group]}.\n\n"
    )


def align_words_to_tokens(
    offsets: np.ndarray,
    spans: list[tuple[int, int]],
    text_start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Finds the first and last subword token of every word.

    The last token of a word is the first token whose end reaches the end of the word;
    for a causal model it is the earliest position that has seen the whole word. A token
    that straddles two words (possible in languages without spaces) counts as the last
    token of the first word and the first token of the second.

    Args:
        offsets: ``(n_tokens, 2)`` character offsets into the full input string, as
            returned by a fast tokenizer; special tokens have ``(0, 0)``.
        spans: Word spans relative to the start of the text.
        text_start: Position of the text inside the full input string, i.e. the
            length of any prompt before it.

    Returns:
        ``first`` and ``last`` token indices per word, -1 for words without tokens
        (e.g. cut off by truncation).
    """
    starts = offsets[:, 0]
    ends = offsets[:, 1]
    real = ends > starts
    token_ids = np.flatnonzero(real & (ends > text_start))
    token_ends = ends[token_ids]
    token_starts = starts[token_ids]
    first = np.full(len(spans), -1, dtype="int64")
    last = np.full(len(spans), -1, dtype="int64")
    for i, (start, end) in enumerate(spans):
        start += text_start
        end += text_start
        j = np.searchsorted(token_ends, start, side="right")
        if j >= len(token_ids) or token_starts[j] >= end:
            continue
        k = np.searchsorted(token_ends, end, side="left")
        if k >= len(token_ids):
            continue
        first[i] = token_ids[j]
        last[i] = token_ids[k]
    return first, last


@dataclass(frozen=True)
class Window:
    """A slice of a long text: context on both sides plus the words it predicts.

    Attributes:
        context_start: First word fed to the model.
        target_start: First word whose predictions are kept.
        target_end: One past the last word whose predictions are kept.
        context_end: One past the last word fed to the model.
    """

    context_start: int
    target_start: int
    target_end: int
    context_end: int


def make_windows(n_words: int, target_words: int, context_words: int) -> list[Window]:
    """Splits a text into overlapping windows whose target regions tile it exactly.

    A text of at most ``target_words + 2 * context_words`` words is a single window.
    Longer texts get consecutive target regions of ``target_words`` words, each with up
    to ``context_words`` words of context on either side.

    Args:
        n_words: Length of the text in words.
        target_words: Words predicted per window.
        context_words: Extra words of context on each side.

    Returns:
        Windows covering every word exactly once in their target regions.
    """
    if n_words <= target_words + 2 * context_words:
        return [Window(0, 0, n_words, n_words)]
    windows = []
    for target_start in range(0, n_words, target_words):
        target_end = min(target_start + target_words, n_words)
        windows.append(Window(
            max(0, target_start - context_words),
            target_start,
            target_end,
            min(n_words, target_end + context_words)
        ))
    return windows
