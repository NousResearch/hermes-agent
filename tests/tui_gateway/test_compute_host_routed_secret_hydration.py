"""A routed compute-host session hydrates the profile's EXTERNAL secret sources before building its
scope (#119521): the isolated turn process never ran the launch dotenv path for that profile, so
without hydration a vault-only provider key is absent from the scope and the agent build fails closed."""

from __future__ import annotations

import io
import types
from pathlib import Path

import pytest

from tui_gateway import server
from tui_gateway.compute_host import ComputeHost


@pytest.mark.platforms("linux")
def test_routed_session_build_sees_the_profiles_external_secret_source(monkeypatch, tmp_path):
    """Two routed homes A→B→A, each with a ``secrets.command`` helper that is the ONLY holder of its
    provider key. Inside each build the scope resolves that profile's key and not the other's."""
    from agent.secret_scope import get_secret, set_multiplex_active
    from hermes_cli import env_loader

    launch = tmp_path / ".hermes"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("VAULT_ONLY_KEY", "from-launch-env")
    env_loader.reset_secret_source_cache()
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    set_multiplex_active(True)

    homes = {}
    for name in ("a", "b"):
        home = launch / "profiles" / name
        home.mkdir(parents=True)
        (home / ".env").write_text(f"{name.upper()}_MARKER=1\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            "secrets:\n  command:\n    enabled: true\n"
            f"    command: \"printf 'VAULT_ONLY_KEY=vault-{name}\\\\n'\"\n",
            encoding="utf-8")
        homes[name] = home

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    seen = []

    def fake_make_agent(sid, key, **kwargs):
        seen.append(get_secret("VAULT_ONLY_KEY"))
        return types.SimpleNamespace(session_id=key)

    def fake_init_session(sid, key, agent, history, **kwargs):
        monkeypatch.setitem(server._sessions, sid, {"agent": agent, "session_key": key})

    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda agent, db: False)
    monkeypatch.setattr(server, "_init_session", fake_init_session)
    try:
        for name in ("a", "b", "a"):
            host._build_server_session(
                server, {"sid": f"s-{name}", "session_key": f"k-{name}", "history": [],
                         "profile_home": str(homes[name])}, f"s-{name}")
    finally:
        host.close()
        env_loader.reset_secret_source_cache()
        for name in ("a", "b"):
            server._sessions.pop(f"s-{name}", None)
    # The wrong side is ABSENT: never the launch env, never the other profile's vault.
    assert seen == ["vault-a", "vault-b", "vault-a"]
    assert Path(homes["a"]).exists()
