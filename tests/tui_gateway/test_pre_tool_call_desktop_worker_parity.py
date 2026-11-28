"""Desktop and gateway-worker turns give a fail-closed guard the same verdict (regression for #160512).

The report: a ``fail_closed`` guard plugin that could not start (its external binary was missing)
let ``terminal`` run on Desktop sessions. Both homes here (the launch profile and a secondary,
multiplexed profile) carry their own copy of the guard, loaded through real plugin discovery. A
real ``AIAgent`` turn (scripted model client, real executor, real local terminal) runs once under
the Desktop session binding (``tui_gateway`` session context, ``source=desktop``) and once under the
gateway worker binding (``gateway.run`` profile scope, ``source=local``); both must block ``terminal``
with the identical tool result.
"""

from __future__ import annotations

import json
import shlex
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.secret_scope import set_multiplex_active
from hermes_cli import plugins as plugins_mod
from tui_gateway import launch_profile_policy as lpp
from tui_gateway import server

pytestmark = pytest.mark.platforms("posix")

_GUARDS = {
    "blocks": ("def _pre(tool_name='', **kw):\n"
               "    from hermes_constants import get_hermes_home\n"
               "    if tool_name == 'terminal':\n"
               "        return {'action': 'block', 'message': 'BLOCKED-BY-GUARD in ' + get_hermes_home().name}\n"
               "def register(ctx):\n"
               "    ctx.register_hook('pre_tool_call', _pre)\n"),
    "binary_missing": ("import shutil\n"
                       "def register(ctx):\n"
                       "    if shutil.which('guard-binary-160512-not-installed') is None:\n"
                       "        raise RuntimeError('guard-binary-160512-not-installed not found on PATH')\n"),
}


@pytest.fixture
def homes(tmp_path, monkeypatch, request):
    launch = tmp_path / "hermes_home"
    routed = launch / "profiles" / "beta"
    for home in (launch, routed):
        pdir = home / "plugins" / "guard"
        pdir.mkdir(parents=True)
        (pdir / "plugin.yaml").write_text(
            "name: guard\nversion: '0.0.1'\ndescription: g\nprovides_hooks: [pre_tool_call]\nfail_closed: true\n",
            encoding="utf-8")
        (pdir / "__init__.py").write_text(_GUARDS[request.param], encoding="utf-8")
        (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["guard"]}}), encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(lpp, "_snapshot", None)
    monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "off")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    plugins_mod._reset_plugin_managers_for_tests()
    yield SimpleNamespace(launch=launch, routed=routed, tmp=tmp_path)
    set_multiplex_active(False)
    plugins_mod._reset_plugin_managers_for_tests()


@contextmanager
def _desktop_binding(home, launch_home, key):
    from gateway.session_context import clear_session_vars

    scopes = server._profile_runtime_scope_tokens(None if home == launch_home else home)
    server._sessions[key] = {"session_key": key, "source": "desktop", "cwd": str(home)}
    tokens = server._set_session_context(key)
    try:
        yield
    finally:
        clear_session_vars(tokens)
        server._release_profile_runtime_scope_tokens(scopes)


@contextmanager
def _worker_binding(home, launch_home, key):
    from gateway.run import _profile_runtime_scope
    from gateway.session_context import clear_session_vars, set_session_vars

    scope = lpp.launch_profile_runtime_scope(launch_home) if home == launch_home else _profile_runtime_scope(home, {})
    with scope:
        tokens = set_session_vars(platform="local", source="local", session_key=key, session_id=key)
        try:
            yield
        finally:
            clear_session_vars(tokens)


def _turn(tmp, label):
    from run_agent import AIAgent
    from tools.terminal_tool import TERMINAL_SCHEMA

    marker = tmp / f"{label}-ran"
    call = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(
        name="terminal", arguments=json.dumps({"command": f"touch {shlex.quote(str(marker))}"})))
    replies = [SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="" if calls else "done", tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop")], model="test/model", usage=None)
        for calls in ([call], None)]
    with (
        patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": TERMINAL_SCHEMA}]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(model="test/model", api_key="test-key", base_url="http://127.0.0.1:1/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=3,
                        platform="desktop" if label.startswith("desktop") else "local")
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = replies
    agent._cached_system_prompt = "Use the available tools."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    result = agent.run_conversation("Run the command.")
    from tools.terminal_tool_lifecycle import cleanup_vm
    cleanup_vm("default")
    return marker.exists(), [m["content"] for m in result["messages"] if m["role"] == "tool"]


@pytest.mark.parametrize("homes", sorted(_GUARDS), indirect=True)
@pytest.mark.parametrize("profile", ["launch", "secondary"])
def test_desktop_and_worker_block_terminal_identically(homes, profile, monkeypatch):
    home = homes.launch if profile == "launch" else homes.routed
    if profile == "secondary":
        set_multiplex_active(True)
    monkeypatch.setattr(server, "_sessions", {})
    with _desktop_binding(home, homes.launch, "desktop-key"):
        desktop_ran, desktop_results = _turn(homes.tmp, f"desktop-{profile}")
    with _worker_binding(home, homes.launch, "worker-key"):
        worker_ran, worker_results = _turn(homes.tmp, f"worker-{profile}")

    assert not desktop_ran, f"Desktop ({profile} profile) ran terminal past the fail-closed guard"
    assert not worker_ran, f"gateway worker ({profile} profile) ran terminal past the fail-closed guard"
    assert desktop_results == worker_results
    assert len(desktop_results) == 1 and "BLOCKED" in desktop_results[0]
    if "BLOCKED-BY-GUARD" in desktop_results[0]:  # the guard of the profile the session is bound to
        assert desktop_results[0].endswith(f'in {home.name}"}}'), desktop_results[0]
