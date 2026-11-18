"""What the default model is: training data, languages, training recipe and evaluation results.

The information ships with the package (``reactor/resources/model_info.json``), so it is
available without downloading the model. The full per-cell results are in the
``results/`` directory of the repository and of the Hub model repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

import pandas as pd


@dataclass(frozen=True)
class ModelInfo:
    """Description of a released model.

    Attributes:
        name: Hub repository id.
        revision: Release tag.
        licence: Licence of the model weights.
        backbone: Hub id and revision of the frozen language model.
        training: How the model was trained: data mix, splits, targets, architecture,
            optimisation and compute.
        datasets: One entry per training corpus: name, paradigm, languages, conditions,
            reader groups, texts, words, mean readers per word, measures, licence and
            citation.
        languages: ISO 639 code to English name of every training language.
        conditions: Reading tasks named in training, usable as ``condition``.
        results: Evaluation summaries: held-out test scores, per-measure noise-ceiling
            ratios, benchmark comparisons and cross-corpus generalisation.
    """

    name: str
    revision: str
    licence: str
    backbone: dict
    training: dict
    datasets: list
    languages: dict
    conditions: list
    results: dict

    def datasets_table(self) -> pd.DataFrame:
        """The training corpora as a table, one row per corpus."""
        return pd.DataFrame(self.datasets)

    def results_table(self, key: str) -> pd.DataFrame:
        """One of the result tables in ``results`` (e.g. ``"per_dataset"``) as a DataFrame.

        Raises:
            KeyError: If there is no such table; ``results.keys()`` lists them.
        """
        return pd.DataFrame(self.results[key])


@cache
def model_info() -> ModelInfo:
    """Returns the description of the default model, read from the installed package."""
    text = resources.files("reactor").joinpath("resources/model_info.json").read_text(encoding="utf-8")
    raw = json.loads(text)
    return ModelInfo(**{name: raw[name] for name in ModelInfo.__dataclass_fields__})
