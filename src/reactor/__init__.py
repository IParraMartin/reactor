"""reactor: human reading times for any text.

``NeuralReact`` predicts word-level eye-tracking and self-paced reading measures with a
neural model trained on 31 reading corpora in 22 languages.

Example:
    >>> import reactor as rt
    >>> react = rt.NeuralReact()
    >>> print(react.predict("the house is red"))
    1034.54ms

``model_info()`` describes the model: its training data, languages, recipe and results.
"""

from __future__ import annotations

from reactor.info import ModelInfo, model_info
from reactor.neural import NeuralReact, Prediction

__version__ = "0.1.0"

__all__ = ["ModelInfo", "NeuralReact", "Prediction", "__version__", "model_info"]
