"""Persistence-boundary write guard: cron/jobs.json, LaunchAgents, shell rc files.

A cron job or rc line executes as the user on later ticks/terminals, so a generic
file write there is a durable code/state vector. The guard must ALWAYS ask and fail
closed with no human channel — matching the protected-instruction gate — and must
NOT over-match ordinary project files (a project named ``LaunchAgents``, a ``.zshrc``
not at the home root, or a ``jobs.json`` not under a ``cron/`` dir).
"""

import json

import pytest


@pytest.fixture(autouse=True)
def _no_human(monkeypatch):
    """No approval callback and no platform env → the always-ask gate fails closed."""
    from tools.terminal_tool import set_approval_callback
    for name in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    set_approval_callback(None)
    yield
    set_approval_callback(None)


# ── unit: matcher ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("/opt/hermes/cron/jobs.json", "cron jobs file"),
    ("/some/where/cron/jobs.json", "cron jobs file"),
    ("/Users/x/Library/LaunchAgents/ai.hermes.gateway.plist", "macOS LaunchAgents entry"),
])
def test_persistence_path_reason_matches(path, expected):
    from tools.file_tools_write_guards import _persistence_path_reason
    assert _persistence_path_reason(path) == expected


def test_home_root_rc_matches(tmp_path, monkeypatch):
    from tools.file_tools_write_guards import _persistence_path_reason
    monkeypatch.setattr("os.path.expanduser", lambda _: str(tmp_path))
    assert _persistence_path_reason(str(tmp_path / ".zshrc")) == "shell rc file (.zshrc)"
    assert _persistence_path_reason(str(tmp_path / ".bashrc")) == "shell rc file (.bashrc)"


@pytest.mark.parametrize("path", [
    "/some/where/plain.txt",
    "/some/where/cron/executions.db",            # sibling, not jobs.json
    "/some/where/notcron/jobs.json",             # parent is not exactly ``cron``
    "/opt/proj/LaunchAgents/x.plist",            # not macOS ``~/Library/LaunchAgents``
])
def test_persistence_path_reason_non_matches(path):
    from tools.file_tools_write_guards import _persistence_path_reason
    assert _persistence_path_reason(path) is None


def test_rc_not_at_home_root_does_not_match(tmp_path, monkeypatch):
    from tools.file_tools_write_guards import _persistence_path_reason
    monkeypatch.setattr("os.path.expanduser", lambda _: str(tmp_path))
    assert _persistence_path_reason(str(tmp_path / "sub" / ".zshrc")) is None


# ── gate: does not over-match ─────────────────────────────────────────────

def test_plain_paths_not_gated():
    from tools.file_tools_write_guards import _check_persistence_path_write
    assert _check_persistence_path_write(["/some/where/plain.txt"]) is None
    assert _check_persistence_path_write(["/some/where/notcron/jobs.json"]) is None


# ── end-to-end: fail-closed gate via write_file_tool ──────────────────────

@pytest.mark.parametrize("rel,content", [
    ("cron/jobs.json", '{"jobs": []}'),
    ("Library/LaunchAgents/com.example.hi.plist", "<plist/>"),
    (".zshrc", "alias hi='echo hi'"),
], ids=["jobs.json", "LaunchAgents", "zshrc-at-home"])
def test_persistence_write_is_blocked_without_human(tmp_path, monkeypatch, rel, content):
    """No interactive/gateway channel → the write is refused and nothing is written."""
    from tools.file_tools import write_file_tool
    home = str(tmp_path)
    monkeypatch.setattr("os.path.expanduser", lambda _: home)

    if rel == ".zshrc":
        target = tmp_path / ".zshrc"          # at the (monkeypatched) home root
    else:
        target_dir = tmp_path / ("cron" if rel.startswith("cron") else "Library/LaunchAgents")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (rel.split("/")[-1])

    res = json.loads(write_file_tool(str(target), content))

    assert "BLOCKED" in res["error"]
    assert "persistence boundary" in res["error"]
    assert not target.exists()
