"""Shared fixtures: a tiny tokenizer, tiny random backbones and reactor models built on them.

Nothing is downloaded; every model is randomly initialized, so the tests check the
mechanics (alignment, batching, ensembling, the API), not prediction quality.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from reactor.backbone import load_backbone
from reactor.features import FEATURE_NAMES
from reactor.heads import ReadingTimeHead
from reactor.measures import TARGETS
from reactor.model import FORMAT, FORMAT_VERSION

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Reading times depend on word length, frequency and predictability.",
    "Readers skip short words and regress after surprising ones.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "我们 今天 读 了 一 篇 很 长 的 文章 。",
    "Language: English. Measure: eye-tracking. Task: reading for comprehension. Readers: native speakers."
]
STATS = {
    "domain": {"toy|reading for comprehension|L1": {t: [5.6, 0.5] for t in ("ffd", "gd", "gpt", "trt")}},
    "language": {},
    "paradigm": {"et": {t: [5.4, 0.4] for t in ("ffd", "gd", "gpt", "trt")}, "spr": {"rt": [5.9, 0.3]}}
}
HEAD = {
    "d_model": 32,
    "num_mixes": 2,
    "dropout": 0.1,
    "dataset_dropout": 0.1,
    "mixer_layers": 1,
    "mixer_heads": 4,
    "kernel_size": 5,
    "max_words": 512,
    "feature_skip_kernel": 5
}


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory) -> Path:
    """A byte-level BPE tokenizer trained on a few sentences."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(CORPUS * 20, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    path = tmp_path_factory.mktemp("tokenizer")
    fast.save_pretrained(path)
    return path


def save_tiny_backbone(config, tokenizer_dir: Path, path: Path) -> Path:
    """Saves a randomly initialized causal language model with the tiny tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def dense_backbone_dir(tokenizer_dir, tmp_path_factory) -> Path:
    """A two-layer dense Qwen3 decoder."""
    from transformers import AutoTokenizer, Qwen3Config

    config = Qwen3Config(
        vocab_size=len(AutoTokenizer.from_pretrained(tokenizer_dir)),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=True
    )
    return save_tiny_backbone(config, tokenizer_dir, tmp_path_factory.mktemp("dense"))


@pytest.fixture(scope="session")
def hybrid_backbone_dir(tokenizer_dir, tmp_path_factory) -> Path:
    """A four-layer Qwen3.5 decoder mixing linear (Gated DeltaNet) and full attention."""
    # transformers binds the fla kernels at import time, and their Triton code needs a GPU
    if importlib.util.find_spec("fla") is not None and not torch.cuda.is_available():
        pytest.skip("flash-linear-attention is installed but no GPU is available")
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=len(AutoTokenizer.from_pretrained(tokenizer_dir)),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=2048,
        tie_word_embeddings=True
    )
    return save_tiny_backbone(config, tokenizer_dir, tmp_path_factory.mktemp("hybrid"))


def make_model_dir(backbone_dir: Path, path: Path, seeds: tuple[int, ...] = (0, 1)) -> Path:
    """Writes a reactor model directory with one randomly initialized head per seed.

    Args:
        backbone_dir: Local backbone, recorded as the model's backbone.
        path: Directory to write.
        seeds: Seed of every ensemble member's random head.

    Returns:
        ``path``.
    """
    backbone = load_backbone(str(backbone_dir), "main", "float32")
    (path / "heads").mkdir(parents=True, exist_ok=True)
    members = []
    for seed in seeds:
        torch.manual_seed(seed)
        head = ReadingTimeHead(backbone.hidden_size, backbone.num_layers + 1, len(FEATURE_NAMES), 2, len(TARGETS), HEAD)
        with torch.no_grad():
            # fresh output layers start at zero; perturb them so predictions vary by word
            for p in head.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        file = f"heads/seed{seed}.safetensors"
        save_file({k: v.contiguous() for k, v in head.state_dict().items()}, path / file)
        members.append({"name": f"seed{seed}", "file": file, "seed": seed})
    config = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "backbone": {"name": str(backbone_dir), "revision": "main", "dtype": "float32", "attn_implementation": "sdpa"},
        "head": HEAD,
        "pool": "last",
        "num_features": len(FEATURE_NAMES),
        "document_position": False,
        "windows": {"max_tokens": 1024, "target_words": 24, "context_words": 8},
        "dataset_vocab": {"<generic>": 0, "toy": 1},
        "stats": STATS,
        "members": members,
        "targets": list(TARGETS)
    }
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def model_dir(dense_backbone_dir, tmp_path_factory) -> Path:
    """A two-member ensemble on the dense backbone."""
    return make_model_dir(dense_backbone_dir, tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="session")
def hybrid_model_dir(hybrid_backbone_dir, tmp_path_factory) -> Path:
    """A two-member ensemble on the hybrid linear-attention backbone."""
    return make_model_dir(hybrid_backbone_dir, tmp_path_factory.mktemp("hybrid_model"))
