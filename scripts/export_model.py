"""Converts training runs into a reactor model directory for the Hugging Face Hub.

Each input is a training run directory of the training code: ``best/model.json`` (model
configuration and normalization statistics), ``best/weights.pt`` (head weights),
``config.json`` and ``run.json``. The runs must keep the backbone frozen and differ only
in their training seed; each becomes one ensemble member. The output directory gets
``config.json`` and every member's head as ``heads/<name>.safetensors``, a format that
cannot execute code when loaded.

Example: python scripts/export_model.py out/model --backbone-revision 1001bb4d --run seed0=runs/a --run seed1=runs/b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from reactor.measures import TARGETS
from reactor.model import FORMAT, FORMAT_VERSION

# metadata every member must share with the first
SHARED = ("model", "dataset_vocab", "num_features", "stats", "data")

def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    """Reads a training run's saved model, its training configuration and its run summary.

    Args:
        run_dir: Training run directory with ``best/``, ``config.json`` and ``run.json``.

    Returns:
        The saved model metadata, the training configuration and the run summary.

    Raises:
        ValueError: If the run tuned its backbone, so its weights are not head-only.
    """
    with open(run_dir / "best" / "model.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(run_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(run_dir / "run.json", encoding="utf-8") as f:
        summary = json.load(f)
    if meta["model"]["finetune"] != "frozen":
        raise ValueError(f"{run_dir} fine-tuned its backbone; only frozen-backbone runs can be exported")
    return meta, config, summary


def head_state(run_dir: Path) -> dict[str, torch.Tensor]:
    """The head weights of a run, without the ``head.`` prefix, as contiguous tensors."""
    state = torch.load(run_dir / "best" / "weights.pt", map_location="cpu", weights_only=True)
    others = [name for name in state if not name.startswith("head.")]
    if others:
        raise ValueError(f"{run_dir} has weights outside the head, e.g. {others[0]}")
    return {name.removeprefix("head."): value.contiguous() for name, value in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="model directory to write")
    parser.add_argument("--run", action="append", required=True, help="name=run_dir, once per ensemble member")
    parser.add_argument("--backbone-revision", required=True, help="commit hash of the backbone the runs used")
    args = parser.parse_args()
    (args.output / "heads").mkdir(parents=True, exist_ok=True)
    members, meta = [], None
    for item in args.run:
        name, run_dir = item.split("=", 1)
        run_dir = Path(run_dir)
        run_meta, train_config, summary = load_run(run_dir)
        meta = meta or run_meta
        for key in SHARED:
            if run_meta[key] != meta[key]:
                raise ValueError(f"{run_dir} differs from the first member in {key!r}")
        file = f"heads/{name}.safetensors"
        save_file(head_state(run_dir), args.output / file, metadata={"format": FORMAT, "member": name})
        members.append({
            "name": name,
            "file": file,
            "seed": train_config["seed"],
            "best_step": summary["best_step"],
            "steps": summary["steps"],
            "val_macro_pearson": summary["best_val_macro_pearson"]
        })
    model, data = meta["model"], meta["data"]
    config = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "backbone": {
            "name": model["name"],
            "revision": args.backbone_revision,
            "dtype": model["dtype"],
            "attn_implementation": model["attn_implementation"]
        },
        "head": model["head"],
        "pool": model["pool"],
        "num_features": meta["num_features"],
        "document_position": bool(data.get("document_position", False)),
        "windows": {key: data[key] for key in ("max_tokens", "target_words", "context_words")},
        "dataset_vocab": meta["dataset_vocab"],
        "stats": meta["stats"],
        "members": members,
        "targets": list(TARGETS)
    }
    with open(args.output / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(args.output)


if __name__ == "__main__":
    main()
