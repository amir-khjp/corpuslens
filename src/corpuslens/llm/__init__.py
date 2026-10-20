"""LLM backend factory. All backends are optional; the default is offline."""

from __future__ import annotations

import os

from corpuslens.llm.base import LLMBackend
from corpuslens.llm.noop import NoOpBackend

__all__ = ["LLMBackend", "NoOpBackend", "get_backend"]


def get_backend(name: str | None = None) -> LLMBackend:
    """Return an LLM backend by name ("anthropic", "openai", "none").

    With no `name`, auto-detects from environment variables
    (ANTHROPIC_API_KEY, then OPENAI_API_KEY), falling back to NoOpBackend.
    """
    if name is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "none"

    if name == "none":
        return NoOpBackend()
    if name == "anthropic":
        from corpuslens.llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend()
    if name == "openai":
        from corpuslens.llm.openai_backend import OpenAIBackend

        return OpenAIBackend()

    raise ValueError(f"Unknown LLM backend: {name!r}")
