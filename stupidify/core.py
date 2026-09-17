from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .config import DamageConfig, DamageStats
from .mutations import mutate_tensor, should_mutate

Progress = Callable[[str], None]


def _say(progress: Progress | None, text: str) -> None:
    if progress:
        progress(text)


def resolve_model(model: str, cache_dir: str | None = None) -> Path:
    local = Path(model).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError("model must be a Hugging Face repo id or a local model directory")
        return local.resolve()
    downloaded = snapshot_download(repo_id=model, cache_dir=cache_dir)
    return Path(downloaded)


def _prepare_output(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Refusing to overwrite the source model. Choose a separate output folder.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == ".git":
            continue
        target = output / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        elif child.suffix != ".safetensors":
            shutil.copy2(child, target)


def stupidify_live_model(
    model: torch.nn.Module,
    config: DamageConfig | None = None,
    *,
    progress: Progress | None = None,
) -> DamageStats:
    """Damage an already-loaded PyTorch model in place.

    This is useful for experiments and the web demo because no second checkpoint copy is needed.
    The caller owns the model instance; reloading it is how you reset it.
    """
    cfg = config or DamageConfig()
    cfg.validate()
    stats = DamageStats()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            tensor = parameter.data
            stats.tensors_seen += 1
            stats.parameters_seen += tensor.numel()
            if not tensor.is_floating_point():
                stats.skipped_non_float += 1
            elif should_mutate(name, tensor, cfg):
                changed = mutate_tensor(name, tensor, cfg)
                stats.tensors_changed += 1
                stats.parameters_changed += min(changed, tensor.numel())
            else:
                stats.skipped_scope += 1

    _say(progress, f"Changed {stats.tensors_changed}/{stats.tensors_seen} live tensors.")
    return stats


def stupidify_model(
    model: str,
    output_dir: str,
    config: DamageConfig | None = None,
    *,
    cache_dir: str | None = None,
    progress: Progress | None = print,
    dry_run: bool = False,
) -> DamageStats:
    cfg = config or DamageConfig()
    cfg.validate()
    source = resolve_model(model, cache_dir=cache_dir)
    output = Path(output_dir).expanduser().resolve()

    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(
            "No .safetensors weight files were found. Stupidify intentionally refuses to load "
            "pickle-based .bin checkpoints. Convert the model to safetensors first."
        )

    stats = DamageStats()
    if not dry_run:
        _prepare_output(source, output)

    _say(progress, f"Source: {source}")
    _say(progress, f"Found {len(weight_files)} safetensors shard(s)")

    for file_index, weight_file in enumerate(weight_files, 1):
        _say(progress, f"[{file_index}/{len(weight_files)}] {weight_file.name}")
        tensors = load_file(str(weight_file), device="cpu")
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        metadata["stupidified"] = "true"
        changed_in_file = False
        output_tensors = {}

        for name, source_tensor in tensors.items():
            tensor = source_tensor.clone()
            stats.tensors_seen += 1
            stats.parameters_seen += tensor.numel()

            if not tensor.is_floating_point():
                stats.skipped_non_float += 1
            elif should_mutate(name, tensor, cfg):
                if dry_run:
                    changed = int(tensor.numel() * cfg.severity) if cfg.method not in {"bitcrush", "noise"} else tensor.numel()
                else:
                    changed = mutate_tensor(name, tensor, cfg)
                stats.tensors_changed += 1
                stats.parameters_changed += min(changed, tensor.numel())
                changed_in_file = True
            else:
                stats.skipped_scope += 1

            if not dry_run:
                output_tensors[name] = tensor

        if changed_in_file:
            stats.files_changed += 1
        if not dry_run:
            save_file(output_tensors, str(output / weight_file.name), metadata=metadata)

    if not dry_run:
        manifest = {
            "format": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": model,
            "resolved_source": str(source),
            "config": cfg.to_dict(),
            "stats": stats.to_dict(),
            "warning": "This checkpoint was intentionally degraded by Stupidify.",
        }
        (output / "stupidify_recipe.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _say(progress, f"Changed {stats.tensors_changed}/{stats.tensors_seen} tensors across {stats.files_changed} shard(s).")
    return stats
