"""Profile session builds hydrate plugin secret sources before provider resolution."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

from agent.secret_scope import current_secret_scope, get_secret
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tui_gateway import server


SECRET_NAME = "TUI_GATEWAY_SOURCE_ONLY_API_KEY"
SECRET_VALUE = "source-only-profile-key"


class _SessionDB:
    def __init__(self, db_path=None, **_kwargs):
        self.db_path = db_path
        self.closed = False

    def close(self):
        self.closed = True


def test_profile_session_build_hydrates_registered_source_before_scope(
    tmp_path, monkeypatch
):
    """A cold profile's source-only key is in scope when its agent is built."""
    from agent.secret_sources import registry
    from hermes_cli import env_loader, plugins
    import agent.credits_tracker as credits_tracker
    import tools.approval as approval
    import tui_gateway.entry as entry

    launch_home = tmp_path / "launch-home"
    profile_home = tmp_path / "profiles" / "cold"
    plugin_dir = profile_home / "plugins" / "source-only"
    launch_home.mkdir()
    plugin_dir.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: [source-only]\n"
        "secrets:\n"
        "  source_only:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: source-only\nversion: 0.1.0\n", encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from agent.secret_sources.base import FetchResult, SecretSource\n\n"
        "class SourceOnly(SecretSource):\n"
        "    name = 'source_only'\n"
        "    label = 'Source only'\n"
        "    shape = 'mapped'\n\n"
        "    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:\n"
        f"        return FetchResult(secrets={{{SECRET_NAME!r}: {SECRET_VALUE!r}}})\n\n"
        "def register(ctx):\n"
        "    ctx.register_secret_source(SourceOnly())\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.delenv(SECRET_NAME, raising=False)
    plugins._reset_plugin_managers_for_tests()
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()

    profile_token = set_hermes_home_override(profile_home)
    try:
        plugins.discover_plugins()
    finally:
        reset_hermes_home_override(profile_token)

    # Plugin discovery re-pulls enabled sources for ordinary startup. Clear
    # both outputs so the TUI build starts from a registered-but-cold profile.
    env_loader.reset_secret_source_cache()
    monkeypatch.delenv(SECRET_NAME, raising=False)
    assert not (profile_home / ".env").exists()

    monkeypatch.setattr("hermes_state.SessionDB", _SessionDB)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(approval, "register_gateway_notify", lambda *_args: None)
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: None)
    monkeypatch.setattr(
        credits_tracker, "seed_credits_at_session_start", lambda _agent: None
    )
    for name, value in (
        ("_set_session_context", lambda _key: []),
        ("_clear_session_context", lambda _tokens: None),
        ("_wire_callbacks", lambda _sid: None),
        ("_config_model_target", lambda: None),
        ("_load_memory_notifications", lambda: False),
        ("_start_notification_poller", lambda _sid, _session: None),
        ("_notify_session_boundary", lambda *_args, **_kwargs: None),
        ("_session_info", lambda *_args, **_kwargs: {}),
        ("_probe_config_health", lambda _cfg: None),
        ("_load_cfg", lambda: {}),
        ("_emit", lambda *_args, **_kwargs: None),
        ("_schedule_mcp_late_refresh", lambda *_args, **_kwargs: None),
        ("_session_source", lambda _session: "desktop"),
        ("_child_run_active", lambda _key: False),
    ):
        monkeypatch.setattr(server, name, value)

    seen = {}

    def build_agent(_sid, _key, session_db=None, **_kwargs):
        seen["scope"] = dict(current_secret_scope() or {})
        seen["resolved"] = get_secret(SECRET_NAME)
        return SimpleNamespace(_session_db=session_db, _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", build_agent)
    sid = "cold-secret-source-profile"
    session = {
        "session_key": "cold-secret-source-key",
        "agent_ready": threading.Event(),
        "profile_home": str(profile_home),
    }
    with server._sessions_lock:
        server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert session["agent_ready"].wait(timeout=10), "agent build did not finish"

        assert session.get("agent_error") is None
        assert seen["scope"][SECRET_NAME] == SECRET_VALUE
        assert seen["resolved"] == SECRET_VALUE
        assert SECRET_NAME not in os.environ
        assert not (profile_home / ".env").exists()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        agent = session.get("agent")
        if agent is not None and agent._session_db is not None:
            agent._session_db.close()
        plugins._reset_plugin_managers_for_tests()
        registry._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()
