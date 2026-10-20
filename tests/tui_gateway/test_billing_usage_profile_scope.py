"""Usage and billing RPCs run under the requested profile's scope once the process multiplexes (#141859).

``session.usage`` fetches Nous credits and the billing/subscription RPCs call the Portal. They ran with
no profile secret scope, so the Portal override read raised ``UnscopedSecretError`` (logged and dropped:
the fetch went to the default Portal) and a secondary profile's request resolved the launch home.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tui_gateway import launch_profile_policy as lpp
from tui_gateway import server

LAUNCH_PORTAL = "https://launch-portal.example.invalid"
B_PORTAL = "https://b-portal.example.invalid"


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home (root) + secondary ``profiles/b``, each with its own Portal override, multiplexing on."""
    root = tmp_path / "hermes_home"
    b = root / "profiles" / "b"
    b.mkdir(parents=True)
    (root / ".env").write_text(f"HERMES_PORTAL_BASE_URL={LAUNCH_PORTAL}\n", encoding="utf-8")
    (b / ".env").write_text(f"HERMES_PORTAL_BASE_URL={B_PORTAL}\n", encoding="utf-8")
    for home in (root, b):
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", LAUNCH_PORTAL)  # the launch process loaded its own .env
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(lpp, "_snapshot", None)
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    lpp.activate_multi_profile_hosting()
    return root, b


def _recorder(calls: list, result):
    """Stand-in for a Portal fetch: records the Portal override and home it would have used."""
    def fake(*_args, **_kwargs):
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home
        try:
            calls.append((get_secret("HERMES_PORTAL_BASE_URL"), Path(get_hermes_home()).resolve()))
        except Exception as exc:  # UnscopedSecretError on the unscoped path
            calls.append(type(exc).__name__)
        return result
    return fake


def _session(profile_home):
    return {"agent": None, "session_key": "sess", "profile_home": profile_home,
            "history": [], "history_lock": threading.Lock()}


def test_session_usage_fetches_credits_in_the_sessions_profile_scope(two_homes, monkeypatch):
    root, b = two_homes
    calls: list = []
    monkeypatch.setattr("agent.account_usage.nous_credits_lines", _recorder(calls, ["credits"]))
    monkeypatch.setattr(server, "_sessions", {"sa": _session(None), "sb": _session(str(b))})

    for sid in ("sa", "sb"):
        resp = server._methods["session.usage"]("rid", {"session_id": sid})
        assert resp["result"]["credits_lines"] == ["credits"], resp

    assert calls == [(LAUNCH_PORTAL, root.resolve()), (B_PORTAL, b.resolve())]


_BILLING_RPCS = [
    ("billing.state", "agent.billing_view.build_billing_state", {}),
    ("usage.bars", "agent.billing_usage.build_usage_model", {}),
    ("subscription.state", "agent.subscription_view.build_subscription_state", {}),
    ("subscription.preview", "hermes_cli.nous_billing.post_subscription_preview", {"subscription_type_id": "t"}),
    ("billing.charge_status", "hermes_cli.nous_billing.get_charge_status", {"charge_id": "c"}),
    ("billing.step_up", "hermes_cli.auth.step_up_nous_billing_scope", {}),
]


@pytest.mark.parametrize(("name", "target", "params"), _BILLING_RPCS, ids=[r[0] for r in _BILLING_RPCS])
def test_billing_rpcs_call_the_portal_in_the_requested_profile_scope(two_homes, monkeypatch, name, target, params):
    root, b = two_homes
    calls: list = []
    monkeypatch.setattr(target, _recorder(calls, {}))
    monkeypatch.setattr("hermes_cli.anon_auth.has_free_tier_account", lambda: False)

    server._methods[name]("rid", {**params, "profile": "b"})
    server._methods[name]("rid", dict(params))

    assert calls == [(B_PORTAL, b.resolve()), (LAUNCH_PORTAL, root.resolve())]
