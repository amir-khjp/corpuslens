from corpuslens.llm import get_backend
from corpuslens.llm.noop import NoOpBackend


def test_default_backend_is_noop_without_api_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = get_backend()
    assert isinstance(backend, NoOpBackend)


def test_noop_label_cluster_joins_keywords():
    backend = NoOpBackend()
    assert backend.label_cluster(["alpha", "beta", "gamma", "delta"], []) == "alpha / beta / gamma"
