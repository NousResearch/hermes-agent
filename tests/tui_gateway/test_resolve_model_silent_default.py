"""Tests: the TUI's no-model fallback follows the one silent-default constant.

``_resolve_model`` with no env seed and no configured model lands on the
cached catalog's labeled default, and on ``PREFERRED_SILENT_DEFAULT_MODEL``
when that lookup fails. A hardcoded copy of the id there drifts every time
the default rotates.
"""

from __future__ import annotations

import pytest

import hermes_cli.models as models_mod
from hermes_cli.models_catalog_static import PREFERRED_SILENT_DEFAULT_MODEL
from tui_gateway import server


@pytest.fixture
def no_model_configured(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(server, "_load_cfg", dict)


def test_unset_model_uses_catalog_labeled_default(monkeypatch, no_model_configured):
    monkeypatch.setattr(
        models_mod, "get_preferred_silent_default_model", lambda provider="openrouter": "example/labeled"
    )

    assert server._resolve_model() == "example/labeled"


def test_unset_model_falls_back_to_the_constant_when_lookup_fails(monkeypatch, no_model_configured):
    def _boom(provider="openrouter"):
        raise RuntimeError("catalog lookup unavailable")

    monkeypatch.setattr(models_mod, "get_preferred_silent_default_model", _boom)

    assert server._resolve_model() == PREFERRED_SILENT_DEFAULT_MODEL
