"""tools.list / tools.show read back the SESSION's effective toolsets (#117977).

``profiles.configure`` pins ``platform_toolsets.cli`` in the profile's own config.yaml — the key the
agent build reads through ``_load_enabled_toolsets`` under the session's profile scope. The read-back
RPCs consulted only a BUILT agent; a session whose agent had not been built yet (``session.create``
with no prompt, exactly the editor's flow) read back as "everything enabled", and a session-less call
resolved against the launch home. Invariant: for a session_id, both RPCs answer with the toolsets the
session's own profile would build with; the launch profile's session still sees its own pin (A→B→A).
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

import tui_gateway.server as server
from tui_gateway.methods_profiles import _save_toolset_pin

LAUNCH_PIN = ["web", "browser", "terminal"]
WORKER_PIN = ["file", "clarify"]


def _pin(home, names):
    """Write the pin the way ``profiles.configure`` does (its writer, into that home's config.yaml)."""
    home.mkdir(parents=True, exist_ok=True)
    _save_toolset_pin({}, names, save_config=lambda cfg: (home / "config.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8"))


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home pinned to LAUNCH_PIN, secondary ``profiles/work`` pinned to WORKER_PIN; one
    multiplexing backend serves an agent-less session per home."""
    root = tmp_path / "hermes_home"
    worker = root / "profiles" / "work"
    _pin(root, LAUNCH_PIN)
    _pin(worker, WORKER_PIN)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    monkeypatch.setattr(server, "_hermes_home", root)
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    sessions = {
        "launch": {"agent": None, "profile_home": None, "cwd": str(tmp_path), "source": "tui"},
        "work": {"agent": None, "profile_home": str(worker), "cwd": str(tmp_path), "source": "tui"},
    }
    monkeypatch.setattr(server, "_sessions", sessions)
    return sessions


def _enabled(sid: str) -> set[str]:
    resp = server._methods["tools.list"]("rid", {"session_id": sid})
    assert "error" not in resp, resp
    return {row["name"] for row in resp["result"]["toolsets"] if row["enabled"]}


def _sections(sid: str) -> set[str]:
    resp = server._methods["tools.show"]("rid", {"session_id": sid})
    assert "error" not in resp, resp
    return {section["name"] for section in resp["result"]["sections"]}


def test_tools_list_reads_back_the_sessions_own_pin_a_b_a(two_homes):
    from hermes_constants import get_hermes_home_override

    work_first = _enabled("work")
    launch = _enabled("launch")
    work_again = _enabled("work")
    assert set(WORKER_PIN) <= work_first == work_again, sorted(work_first)
    assert work_first.isdisjoint({"web", "browser", "terminal"}), sorted(work_first)
    assert set(LAUNCH_PIN) <= launch and "file" not in launch, sorted(launch)
    assert get_hermes_home_override() is None  # scope released after each answer


def test_tools_show_sections_follow_the_sessions_pin(two_homes):
    """Sections are per resolved TOOL, so credential-gated toolsets (web, browser) may be absent
    from both; ``terminal`` vs ``file`` is the pair that tells the two pins apart."""
    work, launch = _sections("work"), _sections("launch")
    assert "file" in work and "terminal" not in work, sorted(work)
    assert "terminal" in launch and "file" not in launch, sorted(launch)
