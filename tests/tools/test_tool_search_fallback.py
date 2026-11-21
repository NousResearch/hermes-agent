"""Regression tests for the silent tool_search fallback.

When the snowballstemmer import fails, tools/tool_search_catalog.py used to raise at
import time, which made model_tools.py swallow the whole tool_search assembly with a
quiet ``logger.warning`` and ship every raw schema (~3x token cliff per request).
These tests pin both halves of the fix:

* the catalog degrades gracefully (unstemmed, consistent matching) and warns ONCE;
* a failed assembly is LOUD, once per process, and still returns raw schemas (never-break).
"""

from __future__ import annotations

import importlib
import io
import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _BlockStemmer:
    """sys.meta_path finder that makes importing snowballstemmer raise ImportError."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "snowballstemmer" or fullname.startswith("snowballstemmer."):
            raise ImportError("snowballstemmer blocked by test")


_DEGRADED_WARNING = ""


@pytest.fixture
def degraded_catalog():
    """Reload the catalog with snowballstemmer unavailable; restore everything after."""
    import tools.tool_search_catalog as catalog

    global _DEGRADED_WARNING
    saved_stemmer = sys.modules.pop("snowballstemmer", None)
    blocker = _BlockStemmer()
    saved_stderr = sys.stderr
    capture = io.StringIO()
    sys.meta_path.insert(0, blocker)
    try:
        sys.stderr = capture  # the degraded warning fires at module load
        mod = importlib.reload(catalog)
        _DEGRADED_WARNING = "".join(line for line in capture.getvalue().splitlines(True)
                                    if "tool search quality degraded" in line)
        yield mod
    finally:
        sys.stderr = saved_stderr
        sys.meta_path.remove(blocker)
        if saved_stemmer is not None:
            sys.modules["snowballstemmer"] = saved_stemmer
        else:
            sys.modules.pop("snowballstemmer", None)
        # Re-import with the real dependency so later tests see the normal path.
        importlib.reload(catalog)


def test_degraded_catalog_imports_and_matches_unstemmed_once(degraded_catalog, capsys):
    """Import succeeds with no stemmer; tokens are consistent/lowercased; warn exactly once."""
    catalog = degraded_catalog
    assert catalog.snowballstemmer is None

    tokens = catalog._tokenize("create_issue create issues")
    assert tokens and tokens == catalog._tokenize("create_issue create issues")
    assert all(t == t.lower() for t in tokens)
    assert "issues" in tokens  # unstemmed: the real stemmer would yield "issu"

    assert catalog._stem("Issues") == "issues"  # lowercased, never raised

    assert _DEGRADED_WARNING.count("tool search quality degraded") == 1
    assert "tool assembly is unaffected" in _DEGRADED_WARNING
    assert capsys.readouterr().err == ""  # tokenizing never re-warns


def test_degraded_catalog_still_assembles_deferred_shape(degraded_catalog):
    """A missing stemmer must not collapse the deferred shape back to raw schemas."""
    import model_tools

    names = [t["function"]["name"] for t in model_tools.get_tool_definitions(quiet_mode=True)]
    assert "tool_search" in names
    assert len(names) < 40
    assert not any(n.startswith("omh_") or n.startswith("mcp__") for n in names)


def test_model_tools_loud_fallback_once_and_never_break(monkeypatch, capsys):
    """A failed assembly prints a loud stderr warning once, sets the flag, keeps raw tools."""
    import model_tools

    def _boom(*args, **kwargs):
        raise RuntimeError("catalog exploded")

    # Baseline: with assembly working the shape is deferred and carries the bridge.
    deferred = model_tools.get_tool_definitions(quiet_mode=False)
    deferred_names = [t["function"]["name"] for t in deferred]
    assert "tool_search" in deferred_names
    capsys.readouterr()  # discard any activation chatter before measuring the warning

    monkeypatch.setattr("tools.tool_search.assemble_tool_defs", _boom)
    monkeypatch.setattr(model_tools, "TOOL_SEARCH_ASSEMBLY_FAILED", False)
    monkeypatch.setattr(model_tools, "_TOOL_SEARCH_ASSEMBLY_WARNED", False)

    raw = model_tools.get_tool_definitions(quiet_mode=False)
    err1 = capsys.readouterr().err
    assert "assembly failed" in err1
    assert "catalog exploded" in err1  # the exception is named
    assert "extra tokens/request" in err1  # the cost is named
    assert "hermes doctor" in err1
    assert model_tools.TOOL_SEARCH_ASSEMBLY_FAILED is True
    raw_names = [t["function"]["name"] for t in raw]
    assert len(raw) > len(deferred)  # never-break: more raw schemas than the deferred shape
    assert set(raw_names) - set(deferred_names)  # deferrable tools are back as raw schemas
    assert "tool_search" not in raw_names  # assembly never ran, so no bridge

    # A second call retries assembly but must NOT print again (once per process).
    model_tools.get_tool_definitions(quiet_mode=False)
    assert capsys.readouterr().err == ""


def test_real_stemmer_path_still_active():
    """Guard did not disable real stemming when snowballstemmer is present."""
    import tools.tool_search_catalog as catalog

    assert catalog.snowballstemmer is not None
    assert catalog._stem("issues") != "issues"
    assert catalog._stem("running") != "running"
