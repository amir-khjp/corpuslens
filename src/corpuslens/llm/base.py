"""Pluggable LLM backend interface.

CorpusLens never requires an LLM: every backend is optional, and
`NoOpBackend` is the default so the tool works fully offline. Backends are
small adapters — see `anthropic_backend.py` for the reference implementation
if you want to add another provider.
"""

from __future__ import annotations


class LLMBackend:
    """Subclass this to plug a new LLM provider into CorpusLens."""

    def summarize(self, text: str) -> str:
        """Return a short plain-English summary of `text`."""
        raise NotImplementedError

    def label_cluster(self, keywords: list[str], samples: list[str]) -> str:
        """Return a short human-readable label for a topic cluster."""
        raise NotImplementedError
