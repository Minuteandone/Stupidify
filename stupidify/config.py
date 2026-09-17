from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DamageConfig:
    method: str = "goblin"
    severity: float = 0.10
    seed: int = 42
    bits: int = 4
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    include_embeddings: bool = False
    include_norms: bool = False
    include_biases: bool = False
    min_ndim: int = 2
    min_numel: int = 128
    chunk_size: int = 1_000_000

    def validate(self) -> None:
        methods = {"zero", "noise", "signflip", "bitcrush", "shuffle", "goblin"}
        if self.method not in methods:
            raise ValueError(f"Unknown method {self.method!r}. Choose from {sorted(methods)}")
        if not 0 <= self.severity <= 1:
            raise ValueError("severity must be between 0 and 1")
        if not 2 <= self.bits <= 16:
            raise ValueError("bits must be between 2 and 16")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DamageStats:
    tensors_seen: int = 0
    tensors_changed: int = 0
    parameters_seen: int = 0
    parameters_changed: int = 0
    files_changed: int = 0
    skipped_non_float: int = 0
    skipped_scope: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
