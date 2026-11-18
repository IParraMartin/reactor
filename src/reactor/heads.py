"""The reading-time head: layer mixing, feature fusion, dataset conditioning and word context.

The head turns the hidden states of a frozen backbone into per-word predictions:

1. ``ScalarMix`` takes, for every word, the hidden state of its pooled subword token from
   every layer and averages the layers with learned weights (two mixes, so early-pass and
   late-pass measures can draw on different depths).
2. The mixed states are projected and added to an embedding of the word features
   (length, frequency, surprisal, position).
3. A dataset embedding scales and shifts the result (FiLM), so one model represents the
   readers and setups of every training corpus; row 0 is a generic reader.
4. ``ContextMixer`` mixes in the neighbouring words with a local convolution and a small
   bidirectional Transformer, which a causal backbone cannot provide on its own.
5. A linear layer outputs every measure, plus a linear skip from the features of each
   word and its neighbours.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ScalarMix(nn.Module):
    """Learned softmax-weighted averages of layer-normalized hidden states.

    Args:
        num_states: Number of hidden-state tensors (layers plus the embedding output).
        num_mixes: Number of independent mixes.
    """

    def __init__(self, num_states: int, num_mixes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_mixes, num_states))
        self.scale = nn.Parameter(torch.ones(num_mixes))

    def forward(self, hidden_states: tuple[torch.Tensor, ...], index: torch.Tensor) -> torch.Tensor:
        """Gathers word positions from every layer and mixes them.

        Args:
            hidden_states: ``num_states`` tensors of shape ``(batch, tokens, hidden)``.
            index: Token position of every word, shape ``(batch, words)``.

        Returns:
            Tensor of shape ``(batch, num_mixes, words, hidden)`` in float32.
        """
        probs = torch.softmax(self.logits, dim=-1)
        gather_index = index.unsqueeze(-1).expand(-1, -1, hidden_states[0].shape[-1])
        mixed = None
        for layer, states in enumerate(hidden_states):
            words = torch.gather(states, 1, gather_index).float()
            words = F.layer_norm(words, words.shape[-1:])
            term = probs[:, layer].view(1, -1, 1, 1) * words.unsqueeze(1)
            mixed = term if mixed is None else mixed + term
        return self.scale.view(1, -1, 1, 1) * mixed


class ContextMixer(nn.Module):
    """Bidirectional mixing over the word sequence.

    A depthwise convolution adds the immediate neighbours (spillover from the previous
    words, preview of the next one) and a small Transformer adds longer-range context.
    Padded word slots are zeroed before the convolution, after its layer norm, so they
    act like the convolution's own zero padding and a text's output does not depend on
    the other texts in its batch.

    Args:
        d_model: Width of the word representations.
        num_layers: Transformer layers; 0 keeps only the local convolution.
        num_heads: Attention heads.
        kernel_size: Width of the local convolution, in words.
        max_words: Number of learned word-position embeddings; later positions share
            the last one.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        kernel_size: int,
        max_words: int,
        dropout: float
    ) -> None:
        super().__init__()
        self.position = nn.Embedding(max_words, d_model)
        self.local = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.local_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False) if num_layers else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mixes context into word representations.

        Args:
            x: Word representations, shape ``(batch, words, d_model)``.
            positions: Word position within the window, shape ``(batch, words)``.
            mask: True for real words, shape ``(batch, words)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        x = x + self.position(positions.clamp(max=self.position.num_embeddings - 1))
        x = x * mask.unsqueeze(-1)
        local = self.local((self.local_norm(x) * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
        x = x + F.gelu(local)
        if self.transformer is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        return self.norm(x)


class ReadingTimeHead(nn.Module):
    """Maps backbone hidden states and word features to reading-measure predictions.

    Args:
        hidden_size: Backbone width.
        num_states: Number of backbone hidden-state tensors.
        num_features: Number of word features.
        num_datasets: Rows of the dataset embedding, row 0 being the generic dataset.
        num_targets: Number of predicted measures.
        cfg: Head configuration with ``d_model``, ``num_mixes``, ``dropout``,
            ``mixer_layers``, ``mixer_heads``, ``kernel_size``, ``max_words`` and
            optionally ``feature_skip_kernel`` (0 or absent disables the feature skip).
    """

    def __init__(
        self,
        hidden_size: int,
        num_states: int,
        num_features: int,
        num_datasets: int,
        num_targets: int,
        cfg: dict
    ) -> None:
        super().__init__()
        d_model = cfg["d_model"]
        num_mixes = cfg["num_mixes"]
        self.mix = ScalarMix(num_states, num_mixes)
        self.project = nn.Linear(num_mixes * hidden_size, d_model)
        self.features = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.fuse_norm = nn.LayerNorm(d_model)
        self.dataset_film = nn.Embedding(num_datasets, 2 * d_model)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.mixer = ContextMixer(
            d_model,
            cfg["mixer_layers"],
            cfg["mixer_heads"],
            cfg["kernel_size"],
            cfg["max_words"],
            cfg["dropout"]
        )
        self.output = nn.Linear(d_model, num_targets)
        kernel = cfg.get("feature_skip_kernel", 0)
        self.feature_skip = nn.Conv1d(num_features, num_targets, kernel, padding=kernel // 2) if kernel else None

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        pool_index: torch.Tensor,
        word_mask: torch.Tensor,
        positions: torch.Tensor,
        features: torch.Tensor,
        dataset_index: torch.Tensor
    ) -> torch.Tensor:
        """Predicts every measure for every word.

        Args:
            hidden_states: Backbone hidden states of every layer.
            pool_index: Token position pooled for each word, ``(batch, words)``.
            word_mask: True for real words, ``(batch, words)``.
            positions: Word position within the window, ``(batch, words)``.
            features: Word features, ``(batch, words, num_features)``.
            dataset_index: Dataset embedding row per example, ``(batch,)``.

        Returns:
            Predictions of shape ``(batch, words, num_targets)``: z-scored log
            durations and probability logits.
        """
        mixed = self.mix(hidden_states, pool_index)
        batch, num_mixes, words, hidden = mixed.shape
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, words, num_mixes * hidden)
        x = self.fuse_norm(self.project(self.dropout(mixed)) + self.features(features.float()))
        scale, shift = self.dataset_film(dataset_index).unsqueeze(1).chunk(2, dim=-1)
        x = x * (1 + scale) + shift
        x = self.mixer(self.dropout(x), positions, word_mask)
        output = self.output(x)
        if self.feature_skip is not None:
            skip = self.feature_skip((features.float() * word_mask.unsqueeze(-1)).transpose(1, 2))
            output = output + skip.transpose(1, 2)
        return output
