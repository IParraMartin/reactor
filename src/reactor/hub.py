"""Locating trained models: local directories or repositories on the Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO = "IParraMartin/reactor-qwen3.5-4b"
"""Hub repository of the default model."""

DEFAULT_REVISION = "v0.1.0"
"""Release tag of the default model that this version of the package was tested with."""

MODEL_FILES = ("config.json", "heads/*.safetensors")
"""Files of a model repository needed for prediction; documentation is not downloaded."""


def resolve_model(
    model: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None
) -> Path:
    """Returns a local directory holding a model, downloading it from the Hub if needed.

    Resolution order: ``model`` if given, else the ``REACTOR_MODEL`` environment
    variable, else ``DEFAULT_REPO``. An existing local directory is used as is; anything
    else is taken as a Hub repository id and downloaded once into the Hugging Face cache
    (``HF_HOME``, or ``cache_dir``), after which it loads offline. The backbone is
    downloaded separately, from its own repository, when the model is loaded.

    Args:
        model: Local directory or Hub repository id.
        revision: Hub revision (tag, branch or commit); defaults to ``DEFAULT_REVISION``
            for the default repository and to ``"main"`` for any other.
        cache_dir: Download cache; None uses the Hugging Face default.

    Returns:
        Directory with ``config.json`` and the head weights.

    Raises:
        FileNotFoundError: If a local path is given but does not contain ``config.json``.
    """
    model = model or os.environ.get("REACTOR_MODEL") or DEFAULT_REPO
    path = Path(model).expanduser()
    if path.exists():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{path} has no config.json; it is not a reactor model directory")
        return path
    from huggingface_hub import snapshot_download

    if revision is None:
        revision = DEFAULT_REVISION if str(model) == DEFAULT_REPO else "main"
    return Path(snapshot_download(
        repo_id=str(model),
        revision=revision,
        allow_patterns=list(MODEL_FILES),
        cache_dir=cache_dir
    ))
