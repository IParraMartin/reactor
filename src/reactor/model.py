"""The reading-time model: a frozen language-model backbone shared by an ensemble of heads.

A saved model is a directory with ``config.json`` and the weights of every ensemble
member's head in ``heads/`` (e.g. ``heads/seed0.safetensors``). The backbone is not part
of it: it is loaded from the Hugging Face Hub at the revision pinned in the
configuration, once, and every head reads the same hidden states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from reactor.backbone import Backbone, load_backbone
from reactor.heads import ReadingTimeHead
from reactor.measures import TARGETS

FORMAT = "reactor-neural"
"""Value of ``format`` in the ``config.json`` of a saved model."""

FORMAT_VERSION = 1
"""Version of the saved-model format this release reads."""

MODEL_INPUTS = (
    "input_ids",
    "attention_mask",
    "pool_index",
    "word_mask",
    "positions",
    "features",
    "dataset_index"
)
"""Keys of a collated batch that the model takes as inputs."""


@dataclass(frozen=True)
class ModelConfig:
    """The contents of a saved model's ``config.json``.

    Attributes:
        backbone: ``name``, ``revision``, ``dtype`` and ``attn_implementation`` of the
            frozen language model.
        head: Head hyperparameters (see ``ReadingTimeHead``), shared by every member.
        pool: Subword token that represents a word: ``"last"`` or ``"first"``.
        num_features: Width of the word-feature matrix.
        document_position: Whether the optional ``document_position`` feature is used.
        windows: ``max_tokens``, ``target_words`` and ``context_words`` of the input
            windows.
        dataset_vocab: Training corpus name to dataset-embedding row; row 0 is the
            generic reader.
        stats: Normalization statistics: mean and standard deviation of log durations
            per ``domain`` (dataset, condition, reader group), per ``language``
            (language, paradigm, reader group) and per ``paradigm``.
        members: One entry per ensemble member: ``name``, ``file`` (its head weights,
            relative to the model directory) and its provenance (training seed,
            selected step and validation score).
        targets: Measure of every output unit, in order.
    """

    backbone: dict
    head: dict
    pool: str
    num_features: int
    document_position: bool
    windows: dict
    dataset_vocab: dict
    stats: dict
    members: list
    targets: list

    @classmethod
    def from_file(cls, path: str | Path) -> ModelConfig:
        """Reads and checks a ``config.json``.

        Args:
            path: The file.

        Returns:
            The configuration.

        Raises:
            ValueError: If the file is not a reactor model of a supported version, has
                no ensemble members, or predicts other measures than this release.
        """
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get("format") != FORMAT:
            raise ValueError(f"{path} is not a reactor model (format {raw.get('format')!r})")
        if raw.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(f"{path} needs a newer reactor (format version {raw['format_version']}); upgrade the package")
        if tuple(raw["targets"]) != TARGETS:
            raise ValueError(f"{path} predicts {raw['targets']}, not {list(TARGETS)}")
        if not raw["members"]:
            raise ValueError(f"{path} lists no ensemble members")
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})


class ReadingTimeModel(nn.Module):
    """A frozen backbone and the heads of every ensemble member.

    Args:
        backbone: Loaded backbone.
        config: Model configuration.
    """

    def __init__(self, backbone: Backbone, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = backbone.tokenizer
        self.body = backbone.decoder
        self.lm_head = backbone.lm_head
        self.heads = nn.ModuleList([
            ReadingTimeHead(
                backbone.hidden_size,
                backbone.num_layers + 1,
                config.num_features,
                max(config.dataset_vocab.values()) + 1,
                len(TARGETS),
                config.head
            )
            for _ in config.members
        ])

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: str | torch.device = "cpu",
        dtype: str | None = None
    ) -> ReadingTimeModel:
        """Loads a saved model and its backbone.

        Args:
            directory: Directory with ``config.json`` and the head weights.
            device: Device to place the model on.
            dtype: Backbone weight type; None uses the one in the configuration.

        Returns:
            The model in eval mode.
        """
        directory = Path(directory)
        config = ModelConfig.from_file(directory / "config.json")
        spec = config.backbone
        backbone = load_backbone(spec["name"], spec["revision"], dtype or spec["dtype"], spec["attn_implementation"])
        model = cls(backbone, config)
        for head, member in zip(model.heads, config.members):
            head.load_state_dict(load_file(directory / member["file"]))
        return model.to(device).eval()

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Runs the backbone and returns the hidden states of every layer."""
        outputs = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
        return outputs.hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_index: torch.Tensor,
        word_mask: torch.Tensor,
        positions: torch.Tensor,
        features: torch.Tensor,
        dataset_index: torch.Tensor
    ) -> torch.Tensor:
        """Runs the backbone once and every head on its hidden states.

        Args:
            input_ids: Token ids, ``(batch, tokens)``.
            attention_mask: 1 for real tokens.
            pool_index: Pooled token position per word, ``(batch, words)``.
            word_mask: True for real words.
            positions: Word position within the window.
            features: Word features, ``(batch, words, n_features)``.
            dataset_index: Dataset embedding row per example.

        Returns:
            Predictions of shape ``(members, batch, words, n_targets)``: z-scored log
            durations and probability logits.
        """
        hidden_states = self.hidden_states(input_ids, attention_mask)
        return torch.stack([
            head(hidden_states, pool_index, word_mask, positions, features, dataset_index)
            for head in self.heads
        ])
