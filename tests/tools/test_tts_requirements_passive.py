"""``check_tts_requirements`` is the ``text_to_speech`` tool's ``check_fn``: a PASSIVE probe.

Regression: the edge / elevenlabs / mistral entries called the SDK importers, and those install
their pm extra on import — so a tool listing (every turn, and on boot) could start a full
dependency rebuild under the install lock. A provider whose SDK installs at first use still has
to read as READY here, or the tool disappears from the schema for exactly the users the install
exists for; the install itself belongs to synthesis.
"""

import pytest

from tools import tts_tool


def _lazy_installs(monkeypatch, allowed: bool) -> None:
    """Stand in for ``security.allow_lazy_installs``.

    The submodule must be fetched through ``import_module``: ``pm.ensure`` on the pm PACKAGE is the
    ``ensure()`` function, so ``pm.install.lazy_installs_allowed`` is not the seam the production
    import reads.
    """
    import importlib

    monkeypatch.setattr(importlib.import_module("pm.install"), "lazy_installs_allowed", lambda: allowed)


def _no_extra(monkeypatch) -> None:
    """pm's availability answer: nothing installed. A real function in every revision, so the
    check reaches its own logic instead of being stubbed out."""
    import importlib

    monkeypatch.setattr(importlib.import_module("pm.extras"), "available", lambda extra: False)


@pytest.fixture
def installs(monkeypatch):
    """Records any install a requirement check attempts, and fails loudly on one."""
    import pm

    calls: list[str] = []

    def fake_ensure(extra):
        calls.append(extra)
        raise AssertionError(f"a TTS requirement check installed {extra!r}")

    monkeypatch.setattr(pm, "ensure_import", fake_ensure)
    return calls


@pytest.fixture
def edge_without_sdk(monkeypatch):
    """The default provider, with edge-tts absent and no NeuTTS fallback."""
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *a, **k: None)
    _no_extra(monkeypatch)
    monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)


def test_installable_sdk_reads_ready_without_installing(monkeypatch, installs, edge_without_sdk):
    _lazy_installs(monkeypatch, True)
    assert tts_tool.check_tts_requirements() is True
    assert installs == []


def test_missing_sdk_reports_unavailable_when_install_cannot_help(monkeypatch, installs, edge_without_sdk):
    """Lazy installs off: nothing to be ready about, and still no install."""
    _lazy_installs(monkeypatch, False)
    assert tts_tool.check_tts_requirements() is False
    assert installs == []


def test_second_provider_needs_its_own_credential_before_first_use_counts(monkeypatch, installs):
    """An install cannot conjure a key: elevenlabs without ELEVENLABS_API_KEY stays unavailable."""
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *a, **k: None)
    _no_extra(monkeypatch)
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **k: "")
    _lazy_installs(monkeypatch, True)
    assert tts_tool.check_tts_requirements() is False
    assert installs == []
