"""OpenAI backend. Requires `pip install corpuslens[openai]` and an
OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import os

from corpuslens.llm.base import LLMBackend

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAIBackend requires the 'openai' package: pip install corpuslens[openai]"
            ) from exc

        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def _complete(self, prompt: str, max_tokens: int = 200) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this dataset description in one or two plain-English sentences:\n\n{text}"
        return self._complete(prompt, max_tokens=120)

    def label_cluster(self, keywords: list[str], samples: list[str]) -> str:
        sample_block = "\n".join(f"- {s[:150]}" for s in samples)
        prompt = (
            "Give a short (3-6 word) topic label for a cluster of text documents.\n"
            f"Top keywords: {', '.join(keywords)}\n"
            f"Example documents:\n{sample_block}\n\n"
            "Respond with only the label, no punctuation or explanation."
        )
        return self._complete(prompt, max_tokens=20)
