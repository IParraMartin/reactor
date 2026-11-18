"""``NeuralReact``: human reading times for any text from the trained neural model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reactor.hub import resolve_model
from reactor.inputs import GENERIC_DATASET
from reactor.measures import PARADIGM_MEASURES, READER_GROUPS
from reactor.predictor import MAX_BATCH_SIZE, MAX_TOKENS, ReadingTimePredictor

SUMMED_MEASURES = {"et": ("trt", "gd", "ffd", "gpt"), "spr": ("rt",)}
"""Measures that a prediction's total can sum, per paradigm."""

DEFAULT_MEASURE = {"et": "trt", "spr": "rt"}
"""Measure summed into the total when none is chosen."""

SETTINGS = ("language", "paradigm", "measure", "condition", "reader_group", "dataset")
"""Names of the reading-setup settings, which ``predict`` accepts as overrides."""


@dataclass(frozen=True)
class Prediction:
    """Predicted reading times of one text.

    Printing a prediction shows the text's total reading time in milliseconds: the sum
    of the chosen measure over its words. Every measure the model predicts is kept per
    word in ``words``.

    Attributes:
        text: The input as given (a string, or a list of words).
        measure: The measure summed into ``total``: ``"trt"`` (total reading time),
            ``"gd"`` (gaze duration), ``"ffd"`` (first-fixation duration), ``"gpt"``
            (go-past time) or ``"rt"`` (self-paced reading time).
        words: One row per word: the word, the backbone's surprisal in bits, every
            predicted duration in milliseconds and, for eye tracking, ``fix_prob`` and
            ``reg_prob``, the probabilities that a reader fixates the word in first pass
            and that the first-pass reading ends in a regression.
    """

    text: str | list[str]
    measure: str
    words: pd.DataFrame

    @property
    def per_word(self) -> list[float]:
        """The chosen measure for every word, in milliseconds."""
        return self.words[self.measure].tolist()

    @property
    def total(self) -> float:
        """The chosen measure summed over all words, in milliseconds."""
        return float(self.words[self.measure].sum())

    def __float__(self) -> float:
        return self.total

    def __str__(self) -> str:
        return f"{self.total:.2f}ms"

    def __repr__(self) -> str:
        return f"Prediction(total={self.total:.2f}ms, measure={self.measure!r}, words={len(self.words)})"


class NeuralReact:
    """Predicts human reading times for text with the trained neural model.

    The model reads text with a frozen Qwen3.5-4B-Base language model and maps it to
    word-level reading measures with heads trained on 31 eye-tracking and self-paced
    reading corpora in 22 languages; the default model averages three heads trained
    with different seeds over one copy of the backbone. Predictions are population
    averages: geometric
    means over readers, and for eye-tracking durations over the readers who fixate the
    word, so the total counts every word as fixated.

    On first use the head weights (about 80 MB) are downloaded from the Hugging Face Hub
    and the backbone (about 9 GB) from its own repository; both are cached, so later
    runs load offline. A CUDA GPU is fastest; Apple silicon (MPS) and CPU also work.

    Args:
        model: Local model directory or Hub repository id; defaults to
            ``$REACTOR_MODEL`` or ``reactor.hub.DEFAULT_REPO``.
        language: ISO 639 code of the texts, e.g. ``"en"``, ``"de"``, ``"zh"``.
        paradigm: ``"et"`` for eye tracking or ``"spr"`` for self-paced reading.
        measure: Measure summed into a prediction's total; defaults to ``"trt"`` for eye
            tracking and ``"rt"`` for self-paced reading.
        condition: Reading task, as named in training, e.g. ``"reading for
            comprehension"`` or ``"information seeking"`` (see ``model_info()``).
        reader_group: ``"L1"`` for native readers, ``"L2"`` for second-language readers.
        dataset: Training corpus whose readers, setup and speed to predict (see
            ``datasets``); None predicts a generic reader of the language.
        device: ``"cuda"``, ``"mps"`` or ``"cpu"``; None picks the fastest available.
        dtype: Backbone weight type (``"bfloat16"``, ``"float16"`` or ``"float32"``);
            None keeps the model's ``bfloat16``.
        revision: Hub revision of the model; None uses the release this package was
            tested with.
        cache_dir: Download cache; None uses the Hugging Face default (``HF_HOME``).
        batch_size: Most sequences per backbone call in ``predict_batch``.
        max_tokens: Most padded tokens per backbone call; lower it if the device runs
            out of memory.

    Raises:
        ValueError: On an invalid paradigm, measure or reader group.

    Example:
        >>> import reactor as rt
        >>> react = rt.NeuralReact()
        >>> result = react.predict("the house is red")
        >>> print(result)
        1034.54ms
        >>> result.words  # every measure for every word
        >>> results = react.predict_batch(["the house is red", "the old man the boat"])
    """

    def __init__(
        self,
        model: str | Path | None = None,
        language: str = "en",
        paradigm: str = "et",
        measure: str | None = None,
        condition: str = "reading for comprehension",
        reader_group: str = "L1",
        dataset: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = MAX_BATCH_SIZE,
        max_tokens: int = MAX_TOKENS
    ) -> None:
        self.settings = {
            "language": language,
            "paradigm": paradigm,
            "measure": measure,
            "condition": condition,
            "reader_group": reader_group,
            "dataset": dataset
        }
        # validated now; the default measure is filled in per call, as it depends on the paradigm
        self.resolve(dict(self.settings))
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.predictor = ReadingTimePredictor.load(resolve_model(model, revision, cache_dir), device, dtype)

    @property
    def datasets(self) -> list[str]:
        """Training corpora the model can be calibrated to with ``dataset``."""
        return sorted(name for name in self.predictor.config.dataset_vocab if name != GENERIC_DATASET)

    @property
    def device(self) -> str:
        """Device the model runs on."""
        return str(self.predictor.device)

    def resolve(self, settings: dict) -> dict:
        """Checks a reading setup and fills in the default measure.

        Args:
            settings: Values of every name in ``SETTINGS``; changed in place.

        Returns:
            ``settings``.

        Raises:
            ValueError: On an unknown paradigm or reader group, or a measure the
                paradigm does not have.
        """
        paradigm = settings["paradigm"]
        if paradigm not in PARADIGM_MEASURES:
            raise ValueError(f"paradigm must be one of {sorted(PARADIGM_MEASURES)}, not {paradigm!r}")
        settings["measure"] = settings["measure"] or DEFAULT_MEASURE[paradigm]
        if settings["measure"] not in SUMMED_MEASURES[paradigm]:
            raise ValueError(f"{paradigm} measures are {SUMMED_MEASURES[paradigm]}, not {settings['measure']!r}")
        if settings["reader_group"] not in READER_GROUPS:
            raise ValueError(f"reader_group must be one of {READER_GROUPS}, not {settings['reader_group']!r}")
        return settings

    def predict(self, text: str | Sequence[str], **overrides: str | None) -> Prediction:
        """Predicts the reading times of one text.

        Args:
            text: The text, split into words on whitespace, or a list of words (needed
                for languages written without spaces, such as Chinese or Japanese).
            **overrides: Any of ``language``, ``paradigm``, ``measure``, ``condition``,
                ``reader_group`` and ``dataset``, for this call only.

        Returns:
            The prediction; ``print`` shows its total in milliseconds and ``words``
            holds every measure per word.

        Raises:
            TypeError: On an unknown override.
            ValueError: On an invalid setting, an unknown dataset or an empty text.
        """
        return self.predict_batch([text], **overrides)[0]

    def predict_batch(self, texts: Sequence[str | Sequence[str]], **overrides: str | None) -> list[Prediction]:
        """Predicts the reading times of many texts, batched on the device.

        Each text is read on its own, from its first word, exactly as in ``predict``;
        texts of similar length share backbone calls of up to ``max_tokens`` padded
        tokens, which is much faster than predicting them one by one.

        Args:
            texts: Texts, each a string or a list of words (see ``predict``).
            **overrides: Settings for this call, as in ``predict``; they apply to every
                text.

        Returns:
            One prediction per text, in input order.

        Raises:
            TypeError: If ``texts`` is a single string, or on an unknown override.
            ValueError: On an invalid setting, an unknown dataset or an empty text.
        """
        if isinstance(texts, str):
            raise TypeError("predict_batch takes a list of texts; use predict for a single string")
        unknown = set(overrides) - set(SETTINGS)
        if unknown:
            raise TypeError(f"unknown settings: {sorted(unknown)}; choose from {SETTINGS}")
        settings = self.resolve({**self.settings, **overrides})
        if settings["dataset"] is not None and settings["dataset"] not in self.datasets:
            raise ValueError(f"unknown dataset {settings['dataset']!r}; choose one of {self.datasets}")
        inputs = [text if isinstance(text, str) else list(text) for text in texts]
        tables = self.predictor.predict_batch(
            inputs,
            language=settings["language"],
            paradigm=settings["paradigm"],
            condition=settings["condition"],
            reader_group=settings["reader_group"],
            dataset=settings["dataset"],
            max_tokens=self.max_tokens,
            max_batch_size=self.batch_size
        )
        return [Prediction(text=text, measure=settings["measure"], words=table) for text, table in zip(inputs, tables)]
