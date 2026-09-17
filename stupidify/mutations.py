from __future__ import annotations

import math

import torch

from .config import DamageConfig


_EMBED_HINTS = ("embed", "embedding", "wte", "wpe", "tok_embeddings")
_NORM_HINTS = ("norm", "ln_", "layernorm")
_BIAS_HINTS = ("bias",)


def should_mutate(name: str, tensor: torch.Tensor, cfg: DamageConfig) -> bool:
    lowered = name.lower()
    if not tensor.is_floating_point():
        return False
    if tensor.ndim < cfg.min_ndim or tensor.numel() < cfg.min_numel:
        return False
    if cfg.include and not any(part.lower() in lowered for part in cfg.include):
        return False
    if cfg.exclude and any(part.lower() in lowered for part in cfg.exclude):
        return False
    if not cfg.include_embeddings and any(hint in lowered for hint in _EMBED_HINTS):
        return False
    if not cfg.include_norms and any(hint in lowered for hint in _NORM_HINTS):
        return False
    if not cfg.include_biases and any(hint in lowered for hint in _BIAS_HINTS):
        return False
    return True


def _generator(seed: int, name: str, device: torch.device) -> torch.Generator:
    value = seed & 0x7FFF_FFFF
    for byte in name.encode("utf-8"):
        value = ((value * 131) + byte) & 0x7FFF_FFFF
    gen = torch.Generator(device=device)
    gen.manual_seed(value)
    return gen


def _iter_chunks(flat: torch.Tensor, chunk_size: int):
    for start in range(0, flat.numel(), chunk_size):
        end = min(flat.numel(), start + chunk_size)
        yield flat[start:end]


def random_zero(tensor: torch.Tensor, severity: float, gen: torch.Generator, chunk_size: int) -> int:
    changed = 0
    flat = tensor.view(-1)
    for chunk in _iter_chunks(flat, chunk_size):
        mask = torch.rand(chunk.shape, generator=gen, device=chunk.device) < severity
        changed += int(mask.sum().item())
        chunk[mask] = 0
    return changed


def gaussian_noise(tensor: torch.Tensor, severity: float, gen: torch.Generator, chunk_size: int) -> int:
    if severity <= 0:
        return 0
    source = tensor.float()
    std = float(source.std(unbiased=False).item())
    if not math.isfinite(std) or std == 0:
        std = max(float(source.abs().mean().item()), 1e-6)
    scale = std * severity
    flat = tensor.view(-1)
    for chunk in _iter_chunks(flat, chunk_size):
        noise = torch.randn(chunk.shape, generator=gen, device=chunk.device, dtype=torch.float32) * scale
        chunk.add_(noise.to(dtype=chunk.dtype))
    return tensor.numel()


def sign_flip(tensor: torch.Tensor, severity: float, gen: torch.Generator, chunk_size: int) -> int:
    changed = 0
    flat = tensor.view(-1)
    for chunk in _iter_chunks(flat, chunk_size):
        mask = torch.rand(chunk.shape, generator=gen, device=chunk.device) < severity
        changed += int(mask.sum().item())
        chunk[mask] = -chunk[mask]
    return changed


def bitcrush(tensor: torch.Tensor, bits: int) -> int:
    max_abs = float(tensor.float().abs().max().item())
    if not math.isfinite(max_abs) or max_abs == 0:
        return 0
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    work = tensor.float()
    work = torch.clamp(torch.round(work / scale), -qmax, qmax) * scale
    tensor.copy_(work.to(dtype=tensor.dtype))
    return tensor.numel()


def shuffle_values(tensor: torch.Tensor, severity: float, gen: torch.Generator) -> int:
    if severity <= 0:
        return 0
    flat = tensor.view(-1)
    count = int(flat.numel() * severity)
    if count < 2:
        return 0
    count = min(count, 2_000_000)
    indexes = torch.randint(0, flat.numel(), (count,), generator=gen, device=flat.device)
    values = flat[indexes].clone()
    order = torch.randperm(count, generator=gen, device=flat.device)
    flat[indexes] = values[order]
    return count


def mutate_tensor(name: str, tensor: torch.Tensor, cfg: DamageConfig) -> int:
    """Mutate a tensor in place and return an approximate number of changed parameters."""
    gen = _generator(cfg.seed, name, tensor.device)
    method = cfg.method

    if method == "zero":
        return random_zero(tensor, cfg.severity, gen, cfg.chunk_size)
    if method == "noise":
        return gaussian_noise(tensor, cfg.severity, gen, cfg.chunk_size)
    if method == "signflip":
        return sign_flip(tensor, cfg.severity, gen, cfg.chunk_size)
    if method == "bitcrush":
        return bitcrush(tensor, cfg.bits)
    if method == "shuffle":
        return shuffle_values(tensor, cfg.severity, gen)
    if method == "goblin":
        changed = bitcrush(tensor, cfg.bits)
        changed += random_zero(tensor, cfg.severity * 0.45, gen, cfg.chunk_size)
        changed += gaussian_noise(tensor, cfg.severity * 0.30, gen, cfg.chunk_size)
        return changed

    raise AssertionError(f"unhandled method: {method}")
