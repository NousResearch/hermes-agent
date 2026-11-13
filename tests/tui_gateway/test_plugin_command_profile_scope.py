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
        f"    ctx.register_command({command!r}, handler=_handler, description='probe')\n",
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
    from hermes_agent.tui_gateway import server

    res = server._dispatch_plugin(
        1, {}, _infra_session(plugin_homes["secondary"]), "launchonly", "hi"
    )
    assert res is None


def test_secondary_plugin_command_runs_under_its_own_scope(plugin_homes):
    """The secondary's own plugin resolves in its registry AND runs with its secrets: the handler
    sees the secondary's home and PROBE_KEY, not the launch profile's."""
    from hermes_agent.tui_gateway import server

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
    from hermes_agent.tui_gateway import server

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

    from hermes_agent.tui_gateway import server

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


def test_builtin_slash_exec_runs_when_the_profile_secret_bind_fails(plugin_homes, monkeypatch):
    """Only a plugin command needs the profile's secrets: a secret source that fails to load must
    not turn every built-in reaching slash.exec into ``Plugin command error`` (review of #153538)."""
    from hermes_cli import env_loader

    from hermes_agent.tui_gateway import server

    def _boom(home):
        raise RuntimeError("secret source unavailable")

    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", _boom)
    worker = MagicMock()
    worker.run.return_value = "help-output"
    worker.pop_seed.return_value = ""
    session = {**_infra_session(plugin_homes["secondary"]), "slash_worker": worker}
    server._sessions["sid-infra"] = session
    try:
        builtin = server._methods["slash.exec"](1, {"session_id": "sid-infra", "command": "/help"})
        plugin = server._methods["slash.exec"](1, {"session_id": "sid-infra", "command": "/infraonly hi"})
    finally:
        server._sessions.pop("sid-infra", None)
    assert builtin["result"]["output"] == "help-output"
    assert plugin["result"]["output"] == "Plugin command error: secret source unavailable"


@pytest.mark.parametrize("name, expected", [("infraonly", ":infra-key"), ("launchonly", None)])
def test_sessionless_dispatch_runs_the_named_profiles_plugin(plugin_homes, monkeypatch, name, expected):
    """``command.dispatch`` with ``profile`` and no session resolves AND runs in that profile: its
    own plugin with its own secret, and never the launch profile's plugin with the launch secret."""
    from agent.secret_scope import set_multiplex_active

    from hermes_agent.tui_gateway import server

    monkeypatch.setattr(server, "_profile_home",
                        lambda p: plugin_homes["secondary"] if p == "infra" else None)
    set_multiplex_active(True)
    try:
        res = server._methods["command.dispatch"](1, {"name": name, "arg": "hi", "profile": "infra"})
    finally:
        set_multiplex_active(False)
    if expected is None:
        assert res["error"]["code"] == 4018, res
    else:
        assert res["result"]["output"].endswith(expected), res


def test_plugin_that_reads_a_secret_while_registering_loads_in_its_profile(plugin_homes):
    """The first plugin-command lookup discovers the profile's plugins, so ``register(ctx)`` runs
    then: it must run under the profile's secret scope, or a plugin that reads a secret while
    registering fails to load (UnscopedSecretError) and its command is never found."""
    from agent.secret_scope import set_multiplex_active

    from hermes_agent.tui_gateway import server

    secondary = plugin_homes["secondary"]
    pkg = secondary / "plugins" / "probe-reg"
    pkg.mkdir(parents=True)
    (pkg / "plugin.yaml").write_text("name: probe-reg\nversion: 1.0.0\ndescription: p\nauthor: p\n", encoding="utf-8")
    (pkg / "__init__.py").write_text(
        "def register(ctx):\n"
        "    from agent.secret_scope import get_secret\n"
        "    key = get_secret('PROBE_KEY')\n"
        "    ctx.register_command('regkey', handler=lambda args: 'reg:' + str(key), description='p')\n",
        encoding="utf-8")
    (secondary / "config.yaml").write_text("plugins:\n  enabled:\n    - probe-infra\n    - probe-reg\n", encoding="utf-8")
    set_multiplex_active(True)
    try:
        res = server._dispatch_plugin(1, {}, _infra_session(secondary), "regkey", "")
    finally:
        set_multiplex_active(False)
    assert res is not None and res["result"]["output"] == "reg:infra-key", res


def test_absent_command_probe_reads_the_registry_under_the_discovery_lock(plugin_homes, monkeypatch):
    """A reload holds the discovery lock while it clears and refills the registry; the no-discovery
    probe must read it under that lock, never between the loaded check and the read."""
    import threading

    from hermes_cli import plugins as plugins_mod

    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    assert "launchonly" in manager._plugin_commands

    class _Probe(dict):
        def __contains__(self, key):
            def other_thread():  # an RLock: only another thread (a reload) sees it held
                got = manager._discovery_lock.acquire(blocking=False)
                if got:
                    manager._discovery_lock.release()
                reads.append(got)
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join()
            return dict.__contains__(self, key)

    reads: list = []
    monkeypatch.setattr(manager, "_plugin_commands", _Probe(manager._plugin_commands))
    assert plugins_mod.plugin_command_absent_nowait("launchonly") is False
    assert reads == [False]
