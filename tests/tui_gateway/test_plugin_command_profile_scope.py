"""TUI-route plugin slash commands must resolve and run under the chat's profile (#133244).

``command.dispatch``'s plugin stage and ``slash.exec``'s plugin branch both resolved the handler
with a bare (launch-profile) registry lookup and ran it with no secret scope: on a multiplexed
host a plugin enabled only in the launch profile fired in another profile's chat, and every
credential read failed closed with ``UnscopedSecretError`` — even in the launch profile's own
Desktop chat. Mirrors the messaging-gateway dispatch fix (gateway/run_inbound.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _plugin_pkg(home, slug: str, command: str) -> None:
    pkg = home / "plugins" / slug
    pkg.mkdir(parents=True)
    (pkg / "plugin.yaml").write_text(
        f"name: {slug}\nversion: 1.0.0\ndescription: probe\nauthor: probe\n",
        encoding="utf-8",
    )
    (pkg / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def _handler(args):\n"
        "        import hermes_constants\n"
        "        from agent.secret_scope import get_secret\n"
        "        try:\n"
        "            key = get_secret('PROBE_KEY')\n"
        "        except Exception as exc:\n"
        "            key = 'ERR:' + type(exc).__name__\n"
        "        return 'probe@' + hermes_constants.get_hermes_home().name + ':' + str(key)\n"
        "    ctx.register_command(%r, handler=_handler, description='probe')\n"
        % command,
        encoding="utf-8",
    )


@pytest.fixture
def plugin_homes(tmp_path, monkeypatch):
    """Launch home with a launch-only plugin; secondary profile with its own plugin + secret."""
    launch = tmp_path / ".hermes"
    launch.mkdir()
    (launch / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - probe-launch\n", encoding="utf-8"
    )
    (launch / ".env").write_text("PROBE_KEY=launch-key\n", encoding="utf-8")
    _plugin_pkg(launch, "probe-launch", "launchonly")

    secondary = launch / "profiles" / "infra"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - probe-infra\n", encoding="utf-8"
    )
    (secondary / ".env").write_text("PROBE_KEY=infra-key\n", encoding="utf-8")
    _plugin_pkg(secondary, "probe-infra", "infraonly")

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("PROBE_KEY", "launch-key")

    from hermes_cli import plugins as plugins_mod

    plugins_mod._reset_plugin_managers_for_tests()
    yield {"launch": launch, "secondary": secondary}
    plugins_mod._reset_plugin_managers_for_tests()


def _infra_session(home) -> dict:
    return {"session_key": "s-infra", "cwd": "", "profile_home": str(home)}


def test_launch_only_plugin_command_not_fired_in_secondary_chat(plugin_homes):
    """command.dispatch's plugin stage must consult the SESSION's registry: a command registered
    only in the launch profile returns None here (falls through) instead of running a foreign
    plugin with no credential scope."""
    import tui_gateway.server as server

    res = server._dispatch_plugin(
        1, {}, _infra_session(plugin_homes["secondary"]), "launchonly", "hi"
    )
    assert res is None


def test_secondary_plugin_command_runs_under_its_own_scope(plugin_homes):
    """The secondary's own plugin resolves in its registry AND runs with its secrets: the handler
    sees the secondary's home and PROBE_KEY, not the launch profile's."""
    import tui_gateway.server as server

    res = server._dispatch_plugin(
        1, {}, _infra_session(plugin_homes["secondary"]), "infraonly", "hi"
    )
    assert res is not None
    output = res["result"]["output"]
    assert output.startswith("probe@infra:"), output
    assert output.endswith(":infra-key"), output


def test_slash_exec_launch_only_falls_through_in_secondary_chat(plugin_homes):
    """slash.exec's plugin branch takes the same scoped lookup: /launchonly in a secondary chat
    must NOT run the plugin — it falls through to the slash worker like any unknown command."""
    import tui_gateway.server as server

    session = _infra_session(plugin_homes["secondary"])
    worker = MagicMock()
    worker.run.return_value = "worker-fallback"
    worker.pop_seed.return_value = (
        ""  # not a /prompt//blueprint compose; plain output path
    )
    session["slash_worker"] = worker
    server._sessions["sid-infra"] = session
    try:
        res = server._methods["slash.exec"](
            1, {"session_id": "sid-infra", "command": "/launchonly hi"}
        )
    finally:
        server._sessions.pop("sid-infra", None)
    assert res["result"]["output"] == "worker-fallback"
    worker.run.assert_called_once_with("/launchonly hi")


def test_launch_profile_chat_reads_its_own_secret_once_multiplexed(plugin_homes):
    """The repro from #133244: with multiplexing active even the launch profile's own chat has no
    ambient scope, so the bare dispatch failed closed on get_secret. The scoped dispatch must read
    the launch profile's .env instead of raising UnscopedSecretError."""
    from agent.secret_scope import set_multiplex_active

    import tui_gateway.server as server

    session = {"session_key": "s-launch", "cwd": "", "profile_home": None}
    server._sessions["sid-launch"] = session
    set_multiplex_active(True)
    try:
        res = server._methods["slash.exec"](
            1, {"session_id": "sid-launch", "command": "/launchonly hi"}
        )
    finally:
        set_multiplex_active(False)
        server._sessions.pop("sid-launch", None)
    output = res["result"]["output"]
    assert output.startswith("probe@"), output
    assert output.endswith(":launch-key"), output
