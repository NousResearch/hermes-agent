"""Tests for the Cohere embed + rerank built-in tools.

Tests exercise the public handler functions (``cohere_embed`` and
``cohere_rerank``) with the ``cohere.ClientV2`` mocked at the build-client
seam (``agent.cohere_adapter.build_cohere_client``) so no network calls
happen and tests pass in the hermetic environment.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.cohere_embed import (
    cohere_embed,
    check_cohere_requirements as check_embed_requirements,
    _handle_cohere_embed,
    COHERE_EMBED_SCHEMA,
)
from tools.cohere_rerank import (
    cohere_rerank,
    check_cohere_requirements as check_rerank_requirements,
    _handle_cohere_rerank,
    COHERE_RERANK_SCHEMA,
)


# ── check_fn gating ─────────────────────────────────────────────────────


class TestCheckRequirements:
    def test_unset_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        assert check_embed_requirements() is False
        assert check_rerank_requirements() is False

    def test_cohere_api_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        assert check_embed_requirements() is True
        assert check_rerank_requirements() is True

    def test_co_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.setenv("CO_API_KEY", "test-key")
        assert check_embed_requirements() is True
        assert check_rerank_requirements() is True


# ── Schema sanity ───────────────────────────────────────────────────────


class TestSchemas:
    def test_embed_schema_required_fields(self):
        params = COHERE_EMBED_SCHEMA["parameters"]
        assert set(params["required"]) == {"texts", "input_type"}
        assert "model" in params["properties"]

    def test_rerank_schema_required_fields(self):
        params = COHERE_RERANK_SCHEMA["parameters"]
        assert set(params["required"]) == {"query", "documents"}
        assert "top_n" in params["properties"]
        assert "model" in params["properties"]


# ── cohere_embed handler ────────────────────────────────────────────────


def _make_embed_response(vectors):
    """Build a SimpleNamespace mimicking a Cohere v2 embed response."""
    return SimpleNamespace(
        embeddings=SimpleNamespace(float_=vectors),
    )


class TestCohereEmbed:
    def test_missing_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        out = cohere_embed(texts=["hello"], input_type="search_document")
        data = json.loads(out)
        assert "error" in data
        assert "COHERE_API_KEY" in data["error"]

    def test_empty_texts_returns_error(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        out = cohere_embed(texts=[], input_type="search_document")
        data = json.loads(out)
        assert "error" in data

    def test_invalid_input_type_returns_error(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        out = cohere_embed(texts=["x"], input_type="not_real")
        data = json.loads(out)
        assert "error" in data

    def test_happy_path(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        fake_client = MagicMock()
        fake_client.embed.return_value = _make_embed_response(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            out = cohere_embed(
                texts=["a", "b"], input_type="search_document", model="embed-v4.0"
            )
        data = json.loads(out)
        assert data["success"] is True
        assert data["count"] == 2
        assert data["dim"] == 3
        assert data["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert data["model"] == "embed-v4.0"
        # Verify the SDK was called with the right keyword shape.
        fake_client.embed.assert_called_once()
        kwargs = fake_client.embed.call_args.kwargs
        assert kwargs["texts"] == ["a", "b"]
        assert kwargs["model"] == "embed-v4.0"
        assert kwargs["input_type"] == "search_document"
        assert kwargs["embedding_types"] == ["float"]

    def test_handler_wraps_cohere_embed(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        fake_client = MagicMock()
        fake_client.embed.return_value = _make_embed_response([[0.0, 0.0]])
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            out = _handle_cohere_embed(
                {"texts": ["x"], "input_type": "clustering"}
            )
        data = json.loads(out)
        assert data["success"] is True
        assert data["dim"] == 2

    def test_sdk_exception_surfaced_as_error(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        fake_client = MagicMock()
        fake_client.embed.side_effect = RuntimeError("boom")
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            out = cohere_embed(texts=["x"], input_type="search_document")
        data = json.loads(out)
        assert "error" in data
        assert "boom" in data["error"]


# ── cohere_rerank handler ───────────────────────────────────────────────


def _make_rerank_response(items):
    """Build a SimpleNamespace mimicking a Cohere v2 rerank response."""
    return SimpleNamespace(
        results=[
            SimpleNamespace(index=i, relevance_score=s) for (i, s) in items
        ]
    )


class TestCohereRerank:
    def test_missing_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        out = cohere_rerank(query="q", documents=["a", "b"])
        data = json.loads(out)
        assert "error" in data

    def test_empty_query_returns_error(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        out = cohere_rerank(query="", documents=["a"])
        data = json.loads(out)
        assert "error" in data

    def test_empty_documents_returns_error(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        out = cohere_rerank(query="q", documents=[])
        data = json.loads(out)
        assert "error" in data

    def test_happy_path(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        docs = ["the cat sat", "the dog ran", "a fish swam"]
        fake_client = MagicMock()
        fake_client.rerank.return_value = _make_rerank_response(
            [(1, 0.95), (0, 0.42), (2, 0.10)]
        )
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            out = cohere_rerank(query="dog?", documents=docs, top_n=3)
        data = json.loads(out)
        assert data["success"] is True
        assert data["model"] == "rerank-v3.5"
        assert len(data["results"]) == 3
        assert data["results"][0]["index"] == 1
        assert data["results"][0]["document"] == "the dog ran"
        assert data["results"][0]["relevance_score"] == pytest.approx(0.95)
        # Verify call shape
        kwargs = fake_client.rerank.call_args.kwargs
        assert kwargs["query"] == "dog?"
        assert kwargs["documents"] == docs
        assert kwargs["top_n"] == 3

    def test_top_n_clamped_to_doc_count(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        fake_client = MagicMock()
        fake_client.rerank.return_value = _make_rerank_response([(0, 0.9)])
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            cohere_rerank(query="q", documents=["only one"], top_n=100)
        # SDK should have received top_n=1 (clamped to len(documents))
        kwargs = fake_client.rerank.call_args.kwargs
        assert kwargs["top_n"] == 1

    def test_invalid_index_dropped(self, monkeypatch):
        """If Cohere returns an out-of-range index, drop that result."""
        monkeypatch.setenv("COHERE_API_KEY", "k")
        fake_client = MagicMock()
        fake_client.rerank.return_value = _make_rerank_response(
            [(0, 0.9), (99, 0.8)]
        )
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            out = cohere_rerank(query="q", documents=["a", "b"])
        data = json.loads(out)
        assert len(data["results"]) == 1
        assert data["results"][0]["index"] == 0

    def test_handler_wraps_cohere_rerank(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        fake_client = MagicMock()
        fake_client.rerank.return_value = _make_rerank_response([(0, 0.5)])
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            out = _handle_cohere_rerank(
                {"query": "q", "documents": ["one"], "top_n": 1}
            )
        data = json.loads(out)
        assert data["success"] is True

    def test_sdk_exception_surfaced_as_error(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "k")
        fake_client = MagicMock()
        fake_client.rerank.side_effect = RuntimeError("nope")
        with patch(
            "agent.cohere_adapter.build_cohere_client",
            return_value=fake_client,
        ):
            out = cohere_rerank(query="q", documents=["a"])
        data = json.loads(out)
        assert "error" in data
        assert "nope" in data["error"]
