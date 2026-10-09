"""APIGenerator endpoint handling (no network: clients are never called)."""

import pytest

pytest.importorskip("openai")

from evidencetree.generation.api_backend import APIGenerator  # noqa: E402
from evidencetree.generation.base import GenerationConfig  # noqa: E402


def _gen(base_url, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "x")
    return APIGenerator(GenerationConfig.from_dict({
        "backend": "api", "provider": "openai", "model": "m",
        "base_url": base_url, "api_key_env": "TEST_KEY",
    }))


def test_single_endpoint(monkeypatch):
    gen = _gen("http://127.0.0.1:8000/v1", monkeypatch)
    assert gen._next_client() is gen._next_client() is gen._client


def test_endpoint_list_rotates(monkeypatch):
    gen = _gen(["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"], monkeypatch)
    first, second, third = gen._next_client(), gen._next_client(), gen._next_client()
    assert first is not second and first is third
    assert [str(c.base_url).rstrip("/") for c in gen._clients] == [
        "http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1",
    ]


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        APIGenerator(GenerationConfig.from_dict({
            "backend": "api", "provider": "openai", "model": "m", "api_key_env": "NO_SUCH_KEY",
        }))


def test_anthropic_provider_accepts_base_url(monkeypatch):
    """Anthropic-protocol endpoints of other vendors (e.g. Kimi) via base_url."""
    import sys
    import types

    created = []

    class FakeAnthropic:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setenv("KIMI_TEST_KEY", "x")
    APIGenerator(GenerationConfig.from_dict({
        "backend": "api", "provider": "anthropic", "model": "m",
        "base_url": "https://example.invalid/anthropic", "api_key_env": "KIMI_TEST_KEY",
    }))
    APIGenerator(GenerationConfig.from_dict({
        "backend": "api", "provider": "anthropic", "model": "m", "api_key_env": "KIMI_TEST_KEY",
    }))
    assert created[0]["base_url"] == "https://example.invalid/anthropic"
    assert "base_url" not in created[1]  # official endpoint by default
