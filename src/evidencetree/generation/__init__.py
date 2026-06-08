"""Configurable generation backends (mock / HF transformers / external API)."""

from .base import Generator, GenerationConfig, build_generator

__all__ = ["Generator", "GenerationConfig", "build_generator"]
