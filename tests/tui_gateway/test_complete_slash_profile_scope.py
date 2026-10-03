"""``complete.slash`` runs its argument completers in the calling session's profile scope.

``/tools`` and ``/personality`` offer the MCP servers and personalities in config.yaml, and
``/handoff`` offers the platforms whose credentials are in the profile's ``.env``. In a ``serve``
process that hosts several profile homes, a session on a secondary profile must get that
profile's rows, never the launch profile's, and completion must not write to ``os.environ``.
"""

from __future__ import annotations

import os
import sys

import pytest

import hermes_cli.commands_completion as commands_completion
import tui_gateway.server as server
from tui_gateway import launch_profile_policy as lpp


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home (root) with a launch session, and ``profiles/b`` with a session on it."""
    root = tmp_path / "hermes_home"
    b = root / "profiles" / "b"
    b.mkdir(parents=True)
    for home, name in ((root, "launch"), (b, "other")):
        (home / "config.yaml").write_text(
            f"mcp_servers:\n  {name}srv:\n    command: 'true'\n"
            f"personalities:\n  {name}persona: {name} persona\n", encoding="utf-8")
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=launch-token\n", encoding="utf-8")
    (b / ".env").write_text("DISCORD_BOT_TOKEN=b-token\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "launch-token")  # the launch process loaded its own .env
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(lpp, "_snapshot", None)
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    lpp.activate_multi_profile_hosting()
    # A serve process does not import ``cli`` at startup, so a completion can be its first import.
    monkeypatch.delitem(sys.modules, "cli", raising=False)
    monkeypatch.setattr(commands_completion, "_personalities_memo", None)
    monkeypatch.setattr(server, "_sessions", {
        "sid-launch": {"session_key": "key-launch"},
        "sid-b": {"session_key": "key-b", "profile_home": str(b)}})
    return root, b


def _completions(scope: dict, text: str) -> list[str]:
    resp = server._methods["complete.slash"]("r", {"text": text, **scope})
    assert "error" not in resp, resp
    return [item["text"] for item in resp["result"]["items"]]


@pytest.mark.parametrize(("text", "keep", "launch_rows", "secondary_rows"), [
    ("/tools enable ", lambda row: row.endswith(":"), ["launchsrv:"], ["othersrv:"]),
    ("/handoff ", lambda _row: True, ["telegram"], ["discord"]),
    ("/personality ", lambda row: row.endswith("persona"), ["launchpersona"], ["otherpersona"]),
])
def test_argument_completions_follow_the_session_profile(two_homes, text, keep, launch_rows, secondary_rows):
    # A new-chat draft has no session and names its profile instead.
    scopes = ({"session_id": "sid-launch"}, {"session_id": "sid-b"}, {"profile": "b"}, {"session_id": "sid-launch"})
    environ = dict(os.environ)
    offered = [[row for row in _completions(scope, text) if keep(row)] for scope in scopes]
    assert offered == [launch_rows, secondary_rows, secondary_rows, launch_rows]
    assert dict(os.environ) == environ
