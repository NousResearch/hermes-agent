"""Multi-profile hosting in the TUI gateway is fail-closed and every profile-scoped RPC runs under
the FULL runtime scope of the requested profile (home + secrets + terminal), the launch profile
included once the process multiplexes.

Regression for the silent cross-profile secret leak class: ``hermes serve`` hosted many profile
homes but never called ``set_multiplex_active(True)``, so every unscoped ``get_secret`` read for a
secondary silently returned the LAUNCH profile's ``os.environ`` value; ``@_profile_scoped`` bound
only HERMES_HOME. And the launch-profile asymmetry: a default-member hosted-room turn in a
``multiplex_profiles: true`` gateway died at agent build with ``UnscopedSecretError``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tui_gateway.server as server
from tui_gateway import launch_profile_policy as lpp

A_VAL = "a-only-secret-0001"
B_VAL = "b-only-secret-0002"
ENV_VAL = "systemd-injected-0003"


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home (root) + secondary ``profiles/b``; B's config references both tokens."""
    root = tmp_path / "hermes_home"
    b = root / "profiles" / "b"
    b.mkdir(parents=True)
    (root / ".env").write_text(f"A_ONLY_TOKEN={A_VAL}\n", encoding="utf-8")
    (b / ".env").write_text(f"B_ONLY_TOKEN={B_VAL}\n", encoding="utf-8")
    for home in (root, b):
        (home / "config.yaml").write_text(
            "probe:\n  a_ref: ${A_ONLY_TOKEN}\n  b_ref: ${B_ONLY_TOKEN}\n  env_ref: ${INJECTED_TOKEN}\n",
            encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("A_ONLY_TOKEN", A_VAL)  # the launch process loaded its own .env
    monkeypatch.setenv("INJECTED_TOKEN", ENV_VAL)  # systemd / op run credential injection
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(lpp, "_snapshot", None)
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    return root, b


def _probe(profile: str | None) -> dict:
    params = {"key": "full"}
    if profile:
        params["profile"] = profile
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    resp = server._methods["config.get"]("rid", params)
    assert "error" not in resp, resp
    return resp["result"]["config"]["probe"]


def test_config_get_for_secondary_resolves_only_its_own_secrets_and_flips_fail_closed(two_homes):
    from agent.secret_scope import UnscopedSecretError, get_secret, is_multiplex_active

    root, _b = two_homes
    assert not is_multiplex_active()  # single-profile so far

    probe_b = _probe("b")
    assert probe_b["b_ref"] == B_VAL
    assert probe_b["a_ref"] == "${A_ONLY_TOKEN}"  # never the launch profile's value
    assert probe_b["env_ref"] == "${INJECTED_TOKEN}"  # never the launch process env
    # Hosting a second home flipped the process: an unscoped read now raises instead of borrowing.
    assert is_multiplex_active()
    with pytest.raises(UnscopedSecretError):
        get_secret("A_ONLY_TOKEN")
    assert os.environ["A_ONLY_TOKEN"] == A_VAL  # never mutated

    # The launch profile is a profile too: its RPC keeps its own .env AND its injected env.
    probe_a = _probe(None)
    assert probe_a["a_ref"] == A_VAL
    assert probe_a["env_ref"] == ENV_VAL
    assert probe_a["b_ref"] == "${B_ONLY_TOKEN}"


def test_single_profile_serve_keeps_environ_fallthrough(two_homes):
    """Control: with no secondary ever requested the launch profile stays unscoped, so credentials
    injected only via the process env (systemd, ``op run``) keep resolving."""
    from agent.secret_scope import get_secret, is_multiplex_active

    probe = _probe(None)
    assert probe["env_ref"] == ENV_VAL
    assert not is_multiplex_active()
    assert get_secret("INJECTED_TOKEN") == ENV_VAL


def test_rpc_scope_reaches_llm_oneshot_and_model_options(two_homes, monkeypatch):
    """The scope must wrap the body of every credential-reading RPC, not only config.get."""
    from agent.secret_scope import get_secret

    root, b = two_homes
    seen = {}

    def fake_oneshot(**kwargs):
        seen["oneshot"] = (Path(os.environ.get("HERMES_HOME", "")), get_secret("B_ONLY_TOKEN"), get_secret("A_ONLY_TOKEN"))
        from hermes_constants import get_hermes_home
        seen["oneshot_home"] = Path(get_hermes_home())
        return "t"

    monkeypatch.setattr("agent.oneshot.run_oneshot", fake_oneshot)
    monkeypatch.setattr(server, "_model_picker_context", lambda agent: object())

    def build_payload(ctx, **kwargs):
        from hermes_constants import get_hermes_home
        seen["options"] = (Path(get_hermes_home()), get_secret("B_ONLY_TOKEN"), get_secret("A_ONLY_TOKEN"))
        return {"providers": []}

    monkeypatch.setattr("hermes_cli.inventory.build_model_options_payload", build_payload)

    r = server._methods["llm.oneshot"]("r1", {"profile": "b", "instructions": "x", "input": "y"})
    assert r["result"]["text"] == "t"
    assert seen["oneshot_home"] == b and seen["oneshot"][1:] == (B_VAL, None)
    r = server._methods["model.options"]("r2", {"profile": "b"})
    assert r["result"] == {"providers": []}
    assert seen["options"] == (b, B_VAL, None)


def test_launch_profile_agent_build_is_scoped_once_multiplexing(two_homes, monkeypatch):
    """The C6 asymmetry: a default-profile session (``profile_home`` None) in a multiplexing process
    must bind the launch profile's own scope for its agent build instead of running unscoped."""
    from agent.secret_scope import current_secret_scope, set_multiplex_active
    from hermes_constants import get_hermes_home

    root, _b = two_homes
    set_multiplex_active(True)  # the messaging gateway's flip (GatewayRunner.__init__)
    scopes = server._bind_build_profile_scopes(None)
    try:
        scope = current_secret_scope()
        assert scope is not None and scope["A_ONLY_TOKEN"] == A_VAL and scope["INJECTED_TOKEN"] == ENV_VAL
        assert "B_ONLY_TOKEN" not in scope
        assert Path(get_hermes_home()) == root
    finally:
        server._release_build_profile_scopes(scopes)
    assert current_secret_scope() is None
