"""Loading of the frozen pretrained language model that the heads read from."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
"""Supported backbone weight types, by name."""


@dataclass
class Backbone:
    """A pretrained causal language model split into the parts the reading-time model uses.

    Attributes:
        decoder: Transformer body returning the hidden states of every layer.
        lm_head: Vocabulary projection, used for surprisal and entropy.
        tokenizer: Matching fast tokenizer.
        hidden_size: Width of the hidden states.
        num_layers: Number of transformer layers (hidden states are one more).
    """

    decoder: nn.Module
    lm_head: nn.Module
    tokenizer: object
    hidden_size: int
    num_layers: int


def load_backbone(name: str, revision: str, dtype: str = "bfloat16", attn_implementation: str = "sdpa") -> Backbone:
    """Loads a causal language model from the Hugging Face Hub or a local directory.

    The model is loaded through ``AutoModelForCausalLM``, so multimodal checkpoints such
    as Qwen3.5 come back text-only, and the transformer body is taken from
    ``get_decoder()``. Every parameter is frozen.

    Args:
        name: Hub repository id or local path, e.g. ``"Qwen/Qwen3.5-4B-Base"``.
        revision: Git revision (commit hash, branch or tag) to load.
        dtype: Weight type: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
        attn_implementation: Attention implementation passed to transformers.

    Returns:
        The loaded backbone in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=DTYPES[dtype],
        attn_implementation=attn_implementation
    )
    model.requires_grad_(False)
    model.eval()
    decoder = model.get_decoder()
    decoder.config.use_cache = False
    config = decoder.config
    if getattr(config, "layer_types", None) and "linear_attention" in config.layer_types:
        report_linear_attention_kernels()
    return Backbone(decoder, model.get_output_embeddings(), tokenizer, config.hidden_size, config.num_hidden_layers)


def report_linear_attention_kernels() -> dict[str, bool]:
    """Logs whether the fast kernels for Gated DeltaNet layers are importable.

    Without ``flash-linear-attention`` and ``causal-conv1d``, transformers falls back to
    a pure PyTorch implementation that gives the same results but is several times
    slower. The kernels need a CUDA GPU.

    Returns:
        Availability of each kernel package.
    """
    available = {
        "fla": importlib.util.find_spec("fla") is not None,
        "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None
    }
    missing = [package for package, found in available.items() if not found]
    if missing:
        log.info("%s not installed; linear-attention layers use the slower PyTorch path", " and ".join(missing))
    return available
