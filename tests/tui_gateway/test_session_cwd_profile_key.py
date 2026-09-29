"""A secondary profile's session.create cwd record is readable inside that profile's turn scope.

``session.create`` is a plain ``@method`` (no profile scope bound), while the turn that reads the
record runs under the session's profile home. Since the terminal keys its cwd/override records by
the routed home (#123989), the writer must bind the same home or the record lands under the raw
key and ``get_session_cwd`` misses until the first ``cd``.
"""

from __future__ import annotations

import tools.terminal_tool as terminal_tool
import tui_gateway.server as server


def test_secondary_profile_session_cwd_is_found_inside_its_scope(monkeypatch, tmp_path):
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(server, "_effective_terminal_backend", lambda: "local")
    home = tmp_path / "profiles" / "research"
    home.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = {"session_key": "sess-b", "cwd": str(workspace), "explicit_cwd": True,
               "source": "desktop", "profile_home": str(home)}
    secret_scope.set_multiplex_active(True)
    try:
        server._register_session_cwd(session)  # session.create path: no scope bound by the caller
        token = set_hermes_home_override(str(home))
        try:
            assert terminal_tool.get_session_cwd("sess-b") == str(workspace)
            assert terminal_tool.resolve_task_overrides("sess-b")["cwd"] == str(workspace)
        finally:
            reset_hermes_home_override(token)
        assert terminal_tool.get_session_cwd("sess-b") is None  # not leaked onto the launch profile's key
    finally:
        secret_scope.set_multiplex_active(False)
