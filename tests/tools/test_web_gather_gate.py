"""Tests for the anti-fabrication gather gate on web_search / web_extract.

Covers the community-reported gap where a model could fabricate placeholder data
when a gather step returned nothing: an empty-but-successful search, a failed
search, or an extract whose entries all carry errors. The gate must stamp an
explicit ``gather_aborted`` marker so the model reports the gap instead.
"""
from __future__ import annotations

import json

from tools import web_tools as wt
from tools import web_tools_gather_gate as gate
from tools.registry import registry


# ─── gate_search_result ─────────────────────────────────────────────────

def test_search_with_results_is_untouched():
    resp = {"success": True, "data": {"web": [{"url": "https://a", "title": "t"}]}}
    out = gate.gate_search_result(resp, "tavily")
    assert out is resp  # same object, no marker on real data
    assert "gather_aborted" not in out


def test_search_empty_success_gets_gate_marker():
    resp = {"success": True, "data": {"web": []}}
    out = gate.gate_search_result(resp, "tavily")
    assert out["gather_aborted"] is True
    assert "no real data" in out["gather_instruction"]
    # original success field is preserved so no contradictory flag is emitted
    assert out["success"] is True


def test_search_failure_gets_gate_marker_with_reason():
    resp = {"success": False, "error": "429 rate limited"}
    out = gate.gate_search_result(resp, "tavily")
    assert out["gather_aborted"] is True
    assert out["gather_reason"] == "429 rate limited"


def test_search_failure_without_error_uses_provider_name():
    out = gate.gate_search_result({"success": False}, "exa")
    assert out["gather_aborted"] is True
    assert "exa" in out["gather_reason"]


def test_gate_instruction_never_varies():
    # determinism: the exact model-facing text is stable across all gate results
    a = gate.gate_search_result({"success": False, "error": "x"}, "p")["gather_instruction"]
    b = gate.gate_search_result({"success": True, "data": {"web": []}}, "p")["gather_instruction"]
    assert a == b


# ─── gate_extract_results ───────────────────────────────────────────────

def test_extract_with_usable_content_returns_none():
    results = [{"url": "https://a", "content": "real page text", "error": None}]
    assert gate.gate_extract_results(results, "firecrawl") is None


def test_extract_all_errors_gets_gate_marker():
    results = [{"url": "https://a", "content": "", "error": "timeout"},
               {"url": "https://b", "content": "", "error": "blocked"}]
    out = gate.gate_extract_results(results, "firecrawl")
    assert out is not None
    assert out["gather_aborted"] is True
    assert out["results"] == results  # nothing lost, diagnostics preserved
    assert "timeout" in out["gather_reason"]


def test_extract_partial_usable_is_not_gated():
    results = [{"url": "https://a", "content": "real", "error": None},
               {"url": "https://b", "content": "", "error": "timeout"}]
    assert gate.gate_extract_results(results, "firecrawl") is None


# ─── E2E via the real web_search_tool handler ───────────────────────────

class _FakeProvider:
    name = "fake"
    display_name = "Fake"

    def supports_search(self) -> bool:
        return True

    def search(self, query, limit):
        # returns empty success — the fabrication trap
        return {"success": True, "data": {"web": [], "search": query}}


def test_gate_extract_failure_stamps_serialized_payload():
    out = gate.gate_extract_failure('{"success": false, "error": "search-only backend"}')
    parsed = json.loads(out)
    assert parsed["success"] is False
    assert parsed["error"] == "search-only backend"
    assert parsed["gather_aborted"] is True
    assert "do not invent" in parsed["gather_instruction"].lower()


def test_web_search_never_configured_emits_gate(tmp_path, monkeypatch):
    """No provider at all (get_active_search_provider returns None) must still be gated."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wt, "_get_search_backend", lambda: "tavily")
    import agent.web_search_registry as reg
    monkeypatch.setattr(reg, "get_provider", lambda backend: None)
    monkeypatch.setattr(reg, "get_active_search_provider", lambda: None)

    result = wt.web_search_tool("anything", limit=5)
    parsed = json.loads(result)
    assert parsed["success"] is False
    assert parsed["gather_aborted"] is True
    assert "no real data" in parsed["gather_instruction"]


def test_web_search_strict_selection_emits_gate(tmp_path, monkeypatch):
    """A stored-but-unregistered selection (strict-selection error) must be gated."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wt, "_get_search_backend", lambda: "tavily")
    import agent.web_search_registry as reg
    monkeypatch.setattr(reg, "get_provider", lambda backend: None)
    # selection_exists("web") True → the strict-selection path fires.
    monkeypatch.setattr(wt, "selection_exists", lambda name: name == "web")

    result = wt.web_search_tool("anything", limit=5)
    parsed = json.loads(result)
    assert parsed["success"] is False
    assert parsed["gather_aborted"] is True
    assert "no real data" in parsed["gather_instruction"]


def test_web_search_tool_empty_emits_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _fake_backend():
        return "fake"

    # Force the fake provider through resolution.
    monkeypatch.setattr(wt, "_get_search_backend", _fake_backend)
    from agent import web_search_registry as reg

    def _fake_get_provider(backend):
        return _FakeProvider() if backend == "fake" else None

    monkeypatch.setattr(reg, "get_provider", _fake_get_provider)
    monkeypatch.setattr(reg, "get_active_search_provider", lambda: _FakeProvider())

    result = wt.web_search_tool("anything", limit=5)
    parsed = json.loads(result)
    assert parsed["gather_aborted"] is True
    assert "no real data" in parsed["gather_instruction"]