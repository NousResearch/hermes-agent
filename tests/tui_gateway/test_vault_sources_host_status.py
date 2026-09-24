"""Invariants for ``vault.sources`` — the Desktop's login-source discovery door.

Contracts:
1. Every row names the host the manager lives on and carries a distinct ``status``, so the
   Desktop can say installed/disconnected/auth-required/available and WHICH machine is
   affected, instead of collapsing everything into "Not detected".
2. Re-calling the method re-probes: that is the reconnect path, no restart required.
3. One failing backend degrades to a ``disconnected`` row instead of failing the whole RPC
   (the shape ``vault.list`` already has).
4. It stays metadata-only: no secret value, token or resolvable material reaches the Desktop,
   and the desktop-facing surface never resolves a reference — resolution stays server-side,
   under the existing approval and unlock flow.

Regression for the "1Password installed on the host but the panel says Not detected, with no
explanation and nowhere to fix it" class.
"""

from __future__ import annotations

import json

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _rows(rid=90):
    envelope = srv._methods["vault.sources"](rid, {})
    assert "error" not in envelope, envelope
    return {row["name"]: row for row in envelope["result"]["sources"]}


def _probe(status, *, installed=True, host="greenmini-jonathon", reason=""):
    from agent.vault_backends.base import SourceProbe, SourceStatus

    return SourceProbe(name="onepassword", installed=installed, status=SourceStatus(status),
                       host=host, reason=reason)


# ── host identity + distinct states reach the Desktop ─────────────────────────────────────────


def test_sources_row_names_the_owning_host(home, monkeypatch):
    monkeypatch.setattr("agent.vault_backends.base.probe",
                        lambda name: _probe("available") if name == "onepassword"
                        else _probe("not_installed", installed=False))

    row = _rows()["onepassword"]
    assert row["host"] == "greenmini-jonathon"
    assert row["status"] == "available"


@pytest.mark.parametrize("status,installed", [
    ("not_installed", False),
    ("disconnected", True),
    ("auth_required", True),
    ("available", True),
])
def test_every_status_survives_the_rpc_distinctly(home, monkeypatch, status, installed):
    """The four states must not collapse: disconnected and auth_required are both ``installed``,
    so a client reading only ``installed`` cannot tell them apart — the row must carry ``status``."""
    monkeypatch.setattr("agent.vault_backends.base.probe",
                        lambda name: _probe(status, installed=installed, reason="because reasons"))

    row = _rows()["onepassword"]
    assert row["status"] == status
    assert row["installed"] is installed


def test_non_available_rows_explain_what_is_wrong_and_where(home, monkeypatch):
    monkeypatch.setattr(
        "agent.vault_backends.base.probe",
        lambda name: _probe("auth_required",
                            reason="1Password is locked on greenmini-jonathon — authenticate there to use it"))

    row = _rows()["onepassword"]
    assert "greenmini-jonathon" in row["reason"], "an error must name the affected host"
    assert row["status"] == "auth_required"


def test_local_source_is_always_available(home):
    row = _rows()["local"]
    assert row["status"] == "available"
    assert row["installed"] is True and row["host"]


# ── reconnect + failure isolation ─────────────────────────────────────────────────────────────


def test_recalling_sources_reprobes(home, monkeypatch):
    """Reconnect = re-query. A host that came back must be reported as available on the next call
    without a gateway restart, so the Desktop's refetch is a working reconnect affordance."""
    calls = {"onepassword": 0}

    def probe(name):
        if name != "onepassword":
            return _probe("not_installed", installed=False)
        calls[name] += 1
        return _probe("disconnected" if calls[name] == 1 else "available")

    monkeypatch.setattr("agent.vault_backends.base.probe", probe)

    assert _rows(1)["onepassword"]["status"] == "disconnected"
    assert _rows(2)["onepassword"]["status"] == "available"
    assert calls["onepassword"] == 2, "each call must re-probe, not serve a cached verdict"


def test_one_raising_backend_does_not_fail_the_whole_rpc(home, monkeypatch):
    """``vault.list`` already wraps its backends; ``vault.sources`` must too, or a single broken
    manager blanks the entire Passwords & Logins panel."""
    def probe(name):
        if name == "onepassword":
            raise RuntimeError("op exploded")
        return _probe("available")

    monkeypatch.setattr("agent.vault_backends.base.probe", probe)

    rows = _rows()
    assert rows["onepassword"]["status"] == "disconnected"
    assert rows["bitwarden"]["status"] == "available"
    assert rows["local"]["status"] == "available"


def test_enable_toggle_still_agrees_with_detection(home, monkeypatch):
    """The zero-config contract survives the new shape: a detected manager the user has not
    opted out of is enabled (#109546)."""
    monkeypatch.setattr("agent.vault_backends.base.probe",
                        lambda name: _probe("available") if name == "bitwarden"
                        else _probe("not_installed", installed=False))

    rows = _rows()
    assert rows["bitwarden"]["enabled"] is True
    assert rows["onepassword"]["enabled"] is False


# ── the desktop deals in references, never values ─────────────────────────────────────────────


def test_sources_response_carries_no_secret_material(home, monkeypatch):
    """Discovery is metadata. A token planted in the environment and any value-shaped key must
    appear nowhere in the serialized response the Desktop receives. (``display_name`` legitimately
    contains the product name "1Password", so this asserts on VALUES and value-carrying keys, not
    on the substring "password" anywhere in the frame.)"""
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_nonsecret_fixture_token_value")
    monkeypatch.setattr("agent.vault_backends.base.probe", lambda name: _probe("available"))

    envelope = srv._methods["vault.sources"](1, {})
    dumped = json.dumps(envelope)
    assert "ops_nonsecret_fixture_token_value" not in dumped
    assert "op://" not in dumped, "a sources response never carries a secret reference"
    value_keys = {"password", "secret", "token", "value", "session_token", "master_password"}
    for row in envelope["result"]["sources"]:
        assert not (value_keys & set(row)), f"value-carrying key in a sources row: {sorted(row)}"


def test_unapproved_resolution_is_refused_and_leaks_nothing(home, monkeypatch):
    """The authorization contract the Desktop depends on: a reference is resolved ONLY on the
    authorized host under the existing unlock/approval flow. A locked manager refuses, the
    refusal names no value, and the manager CLI is never invoked."""
    from agent.vault_backends.base import UnlockRequired
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    backend = OnePasswordLoginBackend({"enabled": True})
    monkeypatch.setattr(OnePasswordLoginBackend, "is_unlocked", lambda self: False)

    def must_not_run(*_a, **_kw):
        raise AssertionError("an unapproved resolution must never invoke the manager CLI")

    monkeypatch.setattr("agent.secret_sources.base.run_cli", must_not_run)
    monkeypatch.setattr("agent.vault_backends.unlock.get_session_token", lambda name: None)

    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("op:some-item")
    assert "op://" not in str(caught.value)


def test_no_rpc_method_hands_the_desktop_a_resolved_value(home):
    """Structural contract: the vault RPC surface exposes listing, status, toggle, unlock, lock,
    add and remove — and nothing that returns a resolved secret. Filling happens server-side."""
    vault_methods = {name for name in srv._methods if name.startswith("vault.")}
    assert vault_methods == {"vault.list", "vault.sources", "vault.source.set", "vault.unlock",
                             "vault.lock", "vault.add", "vault.remove"}
