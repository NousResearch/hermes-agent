"""A child spawned FOR a served profile carries that profile's env, never the launch profile's.

Under ``gateway.multiplex_profiles`` one process serves several profiles and ``os.environ`` holds the
LAUNCH profile's ``.env``. Every ``hermes -p X`` / helper child (slash worker, relay delivery, A2A
forward, ``key_cmd`` helper, browser driver) must start from X's env: X's home pinned, X's own
secrets, and none of the launch profile's residue. The same build reaches every site through
``served_profile_child_env``; the slash worker is spawned through its production class here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_PROBE = ("import json,os;print(json.dumps({k:os.environ.get(k) for k in "
          "('HERMES_HOME','A_MARKER','B_MARKER','TERMINAL_ENV','HERMES_MODEL','FIRECRAWL_API_KEY')}))")


@pytest.fixture
def mux_homes(tmp_path, monkeypatch):
    """Launch home A (its .env mirrored into os.environ, as the multiplexer loads it) and served home B."""
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    (a / ".env").write_text("A_MARKER=a\nHERMES_MODEL=a-model\nTERMINAL_ENV=docker\nFIRECRAWL_API_KEY=a-fc\n", encoding="utf-8")
    (b / ".env").write_text("B_MARKER=b\nFIRECRAWL_API_KEY=b-fc\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(a))
    for key, val in (("A_MARKER", "a"), ("HERMES_MODEL", "a-model"), ("TERMINAL_ENV", "docker"),
                     ("FIRECRAWL_API_KEY", "a-fc")):
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("B_MARKER", raising=False)
    set_multiplex_active(True)
    try:
        yield a, b
    finally:
        set_multiplex_active(False)


def _child_view(env: dict) -> dict:
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _assert_is_b_env(seen: dict, b: Path, *, with_secrets: bool):
    assert seen["HERMES_HOME"] == str(b)
    assert seen["A_MARKER"] is None and seen["TERMINAL_ENV"] is None and seen["HERMES_MODEL"] is None
    assert seen["B_MARKER"] == ("b" if with_secrets else None)


def test_slash_worker_child_runs_in_the_served_profiles_env(mux_homes, monkeypatch):
    """The real ``_SlashWorker`` spawn, observed from INSIDE the child: B's home and secrets, no A residue."""
    import tui_gateway.server as server

    a, b = mux_homes
    captured = {}

    class _Popen:
        def __init__(self, argv, **kw):
            captured["env"] = kw["env"]
            self.stdout = self.stderr = iter(())
            self.stdin = None

        def poll(self):
            return 0

    with monkeypatch.context() as m:  # restored before the probe child spawns through the real Popen
        m.setattr(server.subprocess, "Popen", _Popen)
        token = set_hermes_home_override(str(b))
        try:
            server._SlashWorker("sess", "", profile_home=str(b))
        finally:
            reset_hermes_home_override(token)
        served_env = captured["env"]
        # Outside multiplex the launch profile's own worker keeps its env untouched.
        set_multiplex_active(False)
        server._SlashWorker("sess", "", profile_home=None)
        assert captured["env"]["A_MARKER"] == "a" and captured["env"]["HERMES_HOME"] == str(a)
    _assert_is_b_env(_child_view(served_env), b, with_secrets=True)


def test_helper_children_resolve_secrets_through_the_served_profile(mux_homes):
    """``key_cmd`` helpers and the browser driver spawned during B's turn see B's key, never A's."""
    from agent.command_token_source import _mint
    from gateway.run import _profile_runtime_scope
    from tools.browser_tool import _build_browser_env

    a, b = mux_homes
    helper = (f"{sys.executable} -c \"import os;print(os.environ.get('B_MARKER','-')+'|'"
              f"+os.environ.get('A_MARKER','-')+'|'+os.environ.get('HERMES_HOME',''))\"")
    with _profile_runtime_scope(b, hydrate_secrets=False):
        token, _ttl = _mint(helper, "b-provider")
        browser_env = _build_browser_env()
    assert token == f"b|-|{b}"
    seen = _child_view(browser_env)
    _assert_is_b_env(seen, b, with_secrets=False)  # provider tier stays scrubbed for the browser
    assert seen["FIRECRAWL_API_KEY"] == "b-fc"  # the passthrough key is B's, not the launch profile's
