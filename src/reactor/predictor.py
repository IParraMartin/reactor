"""Low-level prediction: word tables of every measure for lists of texts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reactor.batching import length_sorted_batches
from reactor.features import feature_matrix, lm_word_features, word_start_token_mask
from reactor.inputs import GENERIC_DATASET, TextInput, build_windows, collate
from reactor.measures import PARADIGM_MEASURES, PROBABILITIES, TARGETS
from reactor.model import MODEL_INPUTS, ReadingTimeModel
from reactor.text import NO_SPACE_LANGUAGES

MAX_TOKENS = 4096
"""Default padded-token budget per backbone call."""

MAX_BATCH_SIZE = 32
"""Default maximum number of sequences per backbone call."""


def default_device() -> torch.device:
    """The fastest available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lookup_stats(stats: dict, dataset: str, condition: str, reader_group: str, language: str, paradigm: str) -> dict:
    """Returns the most specific normalization statistics available for a reading setup.

    Statistics of the exact training domain (dataset, condition, reader group) take
    precedence over those pooled by language, paradigm and reader group, which take
    precedence over those pooled by paradigm.

    Args:
        stats: The ``stats`` of a model configuration.
        dataset: Training corpus, or ``GENERIC_DATASET``.
        condition: Task description.
        reader_group: ``"L1"`` or ``"L2"``.
        language: ISO 639 code.
        paradigm: ``"et"`` or ``"spr"``.

    Returns:
        ``(mean, std)`` of the log duration per measure.
    """
    candidates = (
        stats["domain"].get(f"{dataset}|{condition}|{reader_group}"),
        stats["language"].get(f"{language}|{paradigm}|{reader_group}"),
        stats["paradigm"].get(paradigm)
    )
    merged: dict = {}
    for entry in reversed([c for c in candidates if c]):
        merged.update(entry)
    return merged


def split_words(text: str | Sequence[str], language: str) -> list[str]:
    """Splits a string on whitespace; a sequence is taken as the words.

    Args:
        text: A string or a sequence of words.
        language: ISO 639 code of the text.

    Returns:
        The words.

    Raises:
        ValueError: If the text has no words, or is a string in a language written
            without spaces (such text must be segmented into words first).
    """
    if isinstance(text, str):
        if language in NO_SPACE_LANGUAGES:
            raise ValueError(f"pass a list of words for {language!r}; it is written without spaces")
        words = text.split()
    else:
        words = [str(word) for word in text]
    if not words:
        raise ValueError("a text has no words")
    return words


class ReadingTimePredictor:
    """Predicts every measure for every word of raw texts with a loaded model.

    Args:
        model: Loaded model.
        device: Device the model lives on.
    """

    def __init__(self, model: ReadingTimeModel, device: torch.device) -> None:
        self.model = model
        self.config = model.config
        self.device = device
        self.word_start_mask = word_start_token_mask(model.tokenizer, model.lm_head.weight.shape[0])

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None, dtype: str | None = None) -> ReadingTimePredictor:
        """Loads a saved model directory.

        Args:
            directory: Directory with ``config.json`` and the head weights.
            device: ``"cuda"``, ``"mps"`` or ``"cpu"``; None picks the fastest available.
            dtype: Backbone weight type; None uses the model's own (``bfloat16``).

        Returns:
            The predictor.
        """
        device = torch.device(device) if device else default_device()
        return cls(ReadingTimeModel.load(directory, device, dtype), device)

    @property
    def autocast(self) -> torch.autocast:
        """bf16 autocast on CUDA; a no-op elsewhere, where the backbone keeps its own dtype."""
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    @torch.no_grad()
    def predict_batch(
        self,
        texts: Sequence[str | Sequence[str]],
        language: str,
        paradigm: str = "et",
        condition: str = "reading for comprehension",
        reader_group: str = "L1",
        dataset: str | None = None,
        max_tokens: int = MAX_TOKENS,
        max_batch_size: int = MAX_BATCH_SIZE
    ) -> list[pd.DataFrame]:
        """Predicts every measure for every word of several texts read in the same setup.

        Each text is read on its own, from its first word. Texts of similar length share
        padded backbone calls, for the surprisal pass and for the prediction pass; the
        result matches predicting the texts one at a time up to floating-point
        precision. Ensemble members are averaged: durations in log space (a geometric
        mean in milliseconds) and probabilities as probabilities.

        Args:
            texts: Strings, split on whitespace, or sequences of words.
            language: ISO 639 code shared by the texts.
            paradigm: ``"et"`` or ``"spr"``.
            condition: Task description, as named in training.
            reader_group: ``"L1"`` or ``"L2"``.
            dataset: Training corpus to calibrate to; None, or a name the model does not
                know, uses the generic reader.
            max_tokens: Maximum ``batch size * longest sequence`` in tokens per backbone
                call; lower it if the device runs out of memory.
            max_batch_size: Maximum sequences per backbone call.

        Returns:
            One table per text, in input order, with a row per word: ``word``, the
            backbone's ``surprisal`` in bits, and the paradigm's measures (durations in
            milliseconds, probabilities between 0 and 1). A word that lost its tokens to
            truncation has NaN predictions.
        """
        word_lists = [split_words(text, language) for text in texts]
        if not word_lists:
            return []
        lm = self.lm_features(word_lists, language, max_tokens, max_batch_size)
        config = self.config
        name = dataset if dataset in config.dataset_vocab else GENERIC_DATASET
        inputs = []
        for words, (surprisal, entropy) in zip(word_lists, lm):
            n = len(words)
            features = feature_matrix(
                words,
                language,
                np.full(n, np.nan),
                np.full(n, -1),
                np.full(n, -1),
                surprisal,
                entropy,
                np.arange(n) if config.document_position else None
            )
            inputs.append(TextInput(name, language, paradigm, condition, reader_group, words, features))
        windows = build_windows(
            inputs,
            self.model.tokenizer,
            config.dataset_vocab,
            config.windows["max_tokens"],
            config.windows["target_words"],
            config.windows["context_words"],
            config.pool
        )
        members = len(self.model.heads)
        outputs = [np.full((members, len(words), len(TARGETS)), np.nan) for words in word_lists]
        for batch_indices in length_sorted_batches([w.length for w in windows], max_tokens, max_batch_size):
            batch = collate([windows[i] for i in batch_indices], inputs, self.model.tokenizer.pad_token_id)
            with self.autocast:
                predicted = self.model(**{k: batch[k].to(self.device) for k in MODEL_INPUTS}).float().cpu().numpy()
            for b, (text, text_positions, batch_positions) in enumerate(batch["meta"]):
                outputs[text][:, text_positions] = predicted[:, b, batch_positions]
        stats = lookup_stats(config.stats, name, condition, reader_group, language, paradigm)
        keep = PARADIGM_MEASURES[paradigm]
        return [self.to_table(words, surprisal, values, stats, keep) for words, (surprisal, _), values in zip(word_lists, lm, outputs)]

    def lm_features(
        self,
        texts: list[list[str]],
        language: str,
        max_tokens: int,
        max_batch_size: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Surprisal and entropy, in bits, of every word of every text from the frozen backbone."""
        with self.autocast:
            return lm_word_features(
                self.model.body,
                self.model.lm_head,
                self.model.tokenizer,
                texts,
                language,
                self.word_start_mask,
                max_tokens=max_tokens,
                max_batch_size=max_batch_size
            )

    @staticmethod
    def to_table(words: list[str], surprisal: np.ndarray, values: np.ndarray, stats: dict, keep: tuple[str, ...]) -> pd.DataFrame:
        """Averages the ensemble outputs of one text into a word table.

        Args:
            words: Words of the text.
            surprisal: Their surprisal in bits.
            values: Outputs of every member, ``(members, words, targets)``: z-scored
                log durations and probability logits.
            stats: Normalization statistics of the reading setup.
            keep: Measures to report.

        Returns:
            The word table described in ``predict_batch``.
        """
        table = pd.DataFrame({"word": words, "surprisal": surprisal})
        for j, target in enumerate(TARGETS):
            if target not in keep:
                continue
            if target in PROBABILITIES:
                table[target] = (1.0 / (1.0 + np.exp(-values[:, :, j]))).mean(axis=0)
            elif target in stats:
                mean, std = stats[target]
                table[target] = np.exp(values[:, :, j].mean(axis=0) * std + mean)
        return table
