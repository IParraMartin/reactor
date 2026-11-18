"""Model inputs: texts with their features, tokenized windows, and padded batches."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from reactor.text import align_words_to_tokens, conditioning_prompt, join_words, make_windows

log = logging.getLogger(__name__)

GENERIC_DATASET = "<generic>"
"""Name of the dataset-embedding row that stands for an unspecified reader population."""


@dataclass
class TextInput:
    """One text to predict, with the setup it is read under.

    Attributes:
        dataset: Training corpus whose readers and setup to predict, or
            ``GENERIC_DATASET``.
        language: ISO 639 code.
        paradigm: ``"et"`` or ``"spr"``.
        condition: Human-readable task description.
        reader_group: ``"L1"`` or ``"L2"``.
        words: Words as displayed.
        features: ``(n_words, n_features)`` word predictors.
    """

    dataset: str
    language: str
    paradigm: str
    condition: str
    reader_group: str
    words: list[str]
    features: np.ndarray


@dataclass
class WindowInput:
    """Tokenized model input for one window of a text.

    Attributes:
        text: Index of the source text.
        word_slice: Words of the text fed to the model, as ``(start, end)``.
        target_slice: Words whose predictions are kept, as ``(start, end)``.
        input_ids: Token ids of the conditioning prompt plus the window text.
        pool_index: Token position pooled for each word in ``word_slice``; -1 if the
            word lost its tokens to truncation.
        dataset_index: Row of the dataset embedding.
        length: Number of tokens.
    """

    text: int
    word_slice: tuple[int, int]
    target_slice: tuple[int, int]
    input_ids: np.ndarray
    pool_index: np.ndarray
    dataset_index: int
    length: int = field(init=False)

    def __post_init__(self) -> None:
        self.length = len(self.input_ids)


def tokenize_text(
    text: TextInput,
    text_index: int,
    tokenizer,
    dataset_index: int,
    max_tokens: int,
    target_words: int,
    context_words: int,
    pool: str
) -> list[WindowInput]:
    """Splits a text into windows and tokenizes each one after the conditioning prompt.

    Args:
        text: The text.
        text_index: Position of ``text`` in the list of texts being predicted.
        tokenizer: Fast tokenizer of the backbone.
        dataset_index: Row of the dataset embedding for this text.
        max_tokens: Inputs are truncated to this many tokens.
        target_words: Words predicted per window.
        context_words: Context words on each side of a window.
        pool: ``"last"`` or ``"first"`` subword token per word.

    Returns:
        One input per window.
    """
    prompt = conditioning_prompt(text.language, text.paradigm, text.condition, text.reader_group)
    windows = []
    for window in make_windows(len(text.words), target_words, context_words):
        joined, spans = join_words(text.words[window.context_start:window.context_end], text.language)
        encoding = tokenizer(
            prompt + joined,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_tokens
        )
        offsets = np.asarray(encoding["offset_mapping"], dtype="int64").reshape(-1, 2)
        first, last = align_words_to_tokens(offsets, spans, len(prompt))
        windows.append(WindowInput(
            text=text_index,
            word_slice=(window.context_start, window.context_end),
            target_slice=(window.target_start, window.target_end),
            input_ids=np.asarray(encoding["input_ids"], dtype="int64"),
            pool_index=last if pool == "last" else first,
            dataset_index=dataset_index
        ))
    return windows


def build_windows(
    texts: list[TextInput],
    tokenizer,
    dataset_vocab: dict[str, int],
    max_tokens: int,
    target_words: int,
    context_words: int,
    pool: str
) -> list[WindowInput]:
    """Tokenizes every text; datasets missing from the vocabulary map to the generic row.

    Args:
        texts: Texts to predict.
        tokenizer: Fast tokenizer of the backbone.
        dataset_vocab: Dataset name to embedding row.
        max_tokens: Inputs are truncated to this many tokens.
        target_words: Words predicted per window.
        context_words: Context words on each side of a window.
        pool: ``"last"`` or ``"first"`` subword token per word.

    Returns:
        The windows of all texts, in order.
    """
    windows = []
    for i, text in enumerate(texts):
        windows.extend(tokenize_text(
            text,
            i,
            tokenizer,
            dataset_vocab.get(text.dataset, 0),
            max_tokens,
            target_words,
            context_words,
            pool
        ))
    truncated = sum(int((w.pool_index < 0).any()) for w in windows)
    if truncated:
        log.warning("%d of %d windows lost words to truncation", truncated, len(windows))
    return windows


def collate(windows: list[WindowInput], texts: list[TextInput], pad_token_id: int) -> dict:
    """Pads windows into model-ready tensors.

    Args:
        windows: Windows to batch.
        texts: Texts the windows point into.
        pad_token_id: Token id used for padding.

    Returns:
        A dict of tensors (``input_ids``, ``attention_mask``, ``pool_index``,
        ``word_mask``, ``positions``, ``features``, ``dataset_index``) plus ``meta``,
        one ``(text, text positions, batch positions)`` triple per window that locates
        its predicted words in the text and in the batch.
    """
    batch_size = len(windows)
    max_tokens = max(w.length for w in windows)
    max_words = max(w.word_slice[1] - w.word_slice[0] for w in windows)
    n_features = texts[0].features.shape[1]
    input_ids = torch.full((batch_size, max_tokens), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    pool_index = torch.zeros((batch_size, max_words), dtype=torch.long)
    word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
    features = torch.zeros((batch_size, max_words, n_features))
    dataset_index = torch.zeros(batch_size, dtype=torch.long)
    meta = []
    for b, window in enumerate(windows):
        text = texts[window.text]
        start, end = window.word_slice
        n_words = end - start
        input_ids[b, :window.length] = torch.from_numpy(window.input_ids)
        attention_mask[b, :window.length] = 1
        has_token = window.pool_index >= 0
        pool_index[b, :n_words] = torch.from_numpy(np.where(has_token, window.pool_index, 0))
        word_mask[b, :n_words] = torch.from_numpy(has_token)
        features[b, :n_words] = torch.from_numpy(text.features[start:end])
        dataset_index[b] = window.dataset_index
        kept = np.zeros(n_words, dtype=bool)
        kept[window.target_slice[0] - start:window.target_slice[1] - start] = True
        kept &= has_token
        meta.append((window.text, np.flatnonzero(kept) + start, np.flatnonzero(kept)))
    positions = torch.arange(max_words).expand(batch_size, max_words).clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pool_index": pool_index,
        "word_mask": word_mask,
        "positions": positions,
        "features": features,
        "dataset_index": dataset_index,
        "meta": meta
    }
