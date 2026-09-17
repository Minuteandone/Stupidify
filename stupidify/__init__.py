"""Stupidify: intentionally degrade model checkpoints without training."""

from .config import DamageConfig, DamageStats
from .core import stupidify_live_model, stupidify_model

__all__ = ["DamageConfig", "DamageStats", "stupidify_model", "stupidify_live_model"]
__version__ = "0.1.0"
