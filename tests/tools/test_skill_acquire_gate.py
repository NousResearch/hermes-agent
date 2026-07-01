"""Tests for routing a community/untrusted skill_acquire through the P5.3 gate.

Covers the acquire-half wiring (Phase 6 employee model):
  * trusted source + clean scan          -> auto-install (unchanged)
  * untrusted, gate ENABLED              -> staged via the tool-approval gate
                                            (pending record + approval card)
  * untrusted, gate DISABLED             -> requires_approval CLI fallback
  * approved replay (replay token set)   -> install regardless of trust, once
  * already installed                    -> no-op

The Skills-Hub network/IO helpers are stubbed so the test is hermetic; only the
branching policy in ``skill_acquire`` is exercised.
"""

import json
import os
import shutil
import tempfile
import types

import pytest

from tools import skill_discover_tool as sd
from tools import tool_gate
from tools import write_approval as wa


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_acq_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _set_gate(**gate):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("approvals", {})["tool_gate"] = gate
    cfg.save_config(c)


class _Bundle:
    def __init__(self, name):
        self.name = name


class _Meta:
    def __init__(self, trust):
        self.trust_level = trust


class _Scan:
    def __init__(self, verdict="ok"):
        self.verdict = verdict


@pytest.fixture
def stub_hub(monkeypatch, tmp_path):
    """Stub the lazily-imported hub helpers; return a record of side effects."""
    state = {
        "installed": [],          # (qpath, name, category) per install_from_quarantine
        "already_installed": set(),
        "trust": "community",
        "scan_verdict": "ok",
        "scan_allows": True,
        "scan_reason": "",
        "bundle_name": "demo-skill",
    }

    import tools.skills_hub as hub
    import tools.skills_guard as guard
    import hermes_cli.skills_hub as cli_hub

    monkeypatch.setattr(hub, "GitHubAuth", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(hub, "create_source_router", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(hub, "quarantine_bundle", lambda bundle: str(tmp_path / "q"), raising=False)

    def _install(qpath, name, category, bundle, scan):
        state["installed"].append((qpath, name, category))
    monkeypatch.setattr(hub, "install_from_quarantine", _install, raising=False)

    class _Lock:
        def get_installed(self, name):
            return name in state["already_installed"]
    monkeypatch.setattr(hub, "HubLockFile", lambda *a, **k: _Lock(), raising=False)

    monkeypatch.setattr(guard, "scan_skill",
                        lambda path, source="community": _Scan(state["scan_verdict"]),
                        raising=False)
    monkeypatch.setattr(guard, "should_allow_install",
                        lambda scan, force=False: (state["scan_allows"], state["scan_reason"]),
                        raising=False)

    monkeypatch.setattr(cli_hub, "_resolve_short_name",
                        lambda ident, sources, console: f"owner/{ident}", raising=False)
    monkeypatch.setattr(cli_hub, "_resolve_source_meta_and_bundle",
                        lambda ident, sources: (_Meta(state["trust"]), _Bundle(state["bundle_name"]), ident),
                        raising=False)
    return state


def _call(identifier="demo-skill"):
    return json.loads(sd.skill_acquire(identifier))


# ---------------------------------------------------------------------------


def test_trusted_clean_scan_auto_installs(hermes_home, stub_hub):
    stub_hub["trust"] = "trusted"
    out = _call()
    assert out["installed"] is True
    assert out["approved_via"] is None
    assert len(stub_hub["installed"]) == 1


def test_already_installed_is_noop(hermes_home, stub_hub):
    stub_hub["already_installed"].add("demo-skill")
    out = _call()
    assert out["installed"] is True and out["already_installed"] is True
    assert stub_hub["installed"] == []


def test_untrusted_gate_enabled_stages(hermes_home, stub_hub):
    stub_hub["trust"] = "community"
    _set_gate(enabled=True, deferred={"board_assignee": "alon"})
    out = _call()
    assert out["status"] == "staged"
    assert out["installed"] is False
    assert stub_hub["installed"] == []          # nothing installed at stage time
    pid = out["pending_id"]
    assert pid
    rec = wa.get_pending(tool_gate.SUBSYSTEM, pid)
    assert rec is not None
    assert rec["tool_name"] == "skill_acquire"
    assert rec["payload"]["args"]["identifier"] == "owner/demo-skill"
    assert out["card_id"]                        # approval card opened


def test_untrusted_gate_disabled_falls_back_to_cli(hermes_home, stub_hub):
    stub_hub["trust"] = "community"
    # No gate config at all -> gate disabled.
    out = _call()
    assert out["status"] == "requires_approval"
    assert stub_hub["installed"] == []
    assert "hermes skills install" in out["note"]


def test_flagged_trusted_scan_stages_when_gate_on(hermes_home, stub_hub):
    # Trusted source but the scanner refuses -> still needs approval.
    stub_hub["trust"] = "trusted"
    stub_hub["scan_allows"] = False
    stub_hub["scan_reason"] = "suspicious shell call"
    _set_gate(enabled=True, deferred={"board_assignee": "alon"})
    out = _call()
    assert out["status"] == "staged"
    assert stub_hub["installed"] == []


def test_replay_token_installs_regardless_of_trust(hermes_home, stub_hub):
    stub_hub["trust"] = "community"
    _set_gate(enabled=True, deferred={"board_assignee": "alon"})
    # Simulate the worker replaying an approved acquire: set the one-shot token.
    tok = tool_gate.set_replay_token(
        {"pending_id": "abc123", "tool_name": "skill_acquire", "token": "t"})
    try:
        out = _call()
    finally:
        tool_gate.reset_replay_token(tok)
    assert out["installed"] is True
    assert out["approved_via"] == "abc123"
    assert len(stub_hub["installed"]) == 1


def test_replay_token_is_one_shot(hermes_home, stub_hub):
    """A token for skill_acquire is consumed on first use; a second acquire in
    the same context re-stages instead of silently installing again."""
    stub_hub["trust"] = "community"
    _set_gate(enabled=True, deferred={"board_assignee": "alon"})
    tok = tool_gate.set_replay_token(
        {"pending_id": "abc123", "tool_name": "skill_acquire", "token": "t"})
    try:
        first = _call()
        second = _call()
    finally:
        tool_gate.reset_replay_token(tok)
    assert first["installed"] is True
    assert second["status"] == "staged"          # token already consumed
    assert len(stub_hub["installed"]) == 1
