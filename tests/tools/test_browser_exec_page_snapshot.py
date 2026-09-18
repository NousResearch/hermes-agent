"""Invariants for the snapshot helper Hermes installs into every browser_exec workspace.

The helper is what makes one browser round trip enough to see a page: the exec namespace gets
``snapshot()`` / ``snapshot_table()`` / ``find_entry()`` / ``point()`` / ``page_changed()`` without
touching the browser-use CLI. Two things must stay true:

* the helper installs, survives an agent-authored ``agent_helpers.py``, and is importable;
* a planted symlink can never redirect the write (workspaces are agent-writable).
"""

import json
import sys
import types
from pathlib import Path

from tools import browser_exec_page_snapshot as snap_mod

HELPERS = ("snapshot", "snapshot_table", "find_entry", "point", "page_changed")


def _fake_js(_expression: str) -> str:
    """Stand-in for browser_harness.helpers.js(): one call, one JSON blob."""
    return json.dumps({
        "ok": True, "title": "Fixture", "url": "https://example.test/", "count": 1, "total": 1,
        "mut": 7, "seq": 1, "sy": 0, "sx": 0, "vw": 800, "vh": 600, "page_ms": 1.0,
        "elements": [{"i": 1, "role": "button", "name": "Save", "cx": 10, "cy": 20,
                      "w": 40, "h": 20, "in_view": True, "covered_by": None}],
    })


def _load_agent_helpers(monkeypatch, workspace: Path) -> dict:
    """Execute the installed agent_helpers.py the way browser-harness does."""
    harness = types.ModuleType("browser_harness")
    helpers = types.ModuleType("browser_harness.helpers")
    helpers.js = _fake_js
    monkeypatch.setitem(sys.modules, "browser_harness", harness)
    monkeypatch.setitem(sys.modules, "browser_harness.helpers", helpers)
    monkeypatch.syspath_prepend(str(workspace))

    agent = workspace / "agent_helpers.py"
    namespace = {"__file__": str(agent)}
    exec(compile(agent.read_text(encoding="utf-8"), str(agent), "exec"), namespace)
    return namespace


def test_installs_importable_helper_without_clobbering_agent_helpers(tmp_path, monkeypatch):
    """Install is additive, idempotent, and the imported helper actually works."""
    assert snap_mod.ensure_workspace_helpers(None) is None    # no workspace dir to install into
    assert snap_mod.ensure_workspace_helpers("") is None

    workspace = tmp_path / "workspace"
    agent = workspace / "agent_helpers.py"
    agent.parent.mkdir(parents=True)
    agent.write_text("def my_helper():\n    return 'mine'\n", encoding="utf-8")

    assert snap_mod.ensure_workspace_helpers(str(workspace))
    assert (workspace / snap_mod.HELPER_MODULE_NAME).is_file()
    first = agent.read_text(encoding="utf-8")
    assert "def my_helper" in first                      # the agent's own helper survives
    assert first.count(snap_mod._MARKER_BEGIN) == 1

    snap_mod.ensure_workspace_helpers(str(workspace))    # a second call must not duplicate
    assert agent.read_text(encoding="utf-8") == first

    namespace = _load_agent_helpers(monkeypatch, workspace)
    for name in HELPERS:
        assert callable(namespace[name]), name
    assert namespace["my_helper"]() == "mine"

    snap = namespace["snapshot"]()
    assert snap["count"] == 1 and snap["elements"][0]["name"] == "Save"
    assert namespace["point"](snap["elements"][0]) == (10, 20)
    assert namespace["find_entry"](snap, "sav")[0]["name"] == "Save"
    assert namespace["page_changed"](snap)["changed"] is False


def test_planted_symlink_is_not_followed(tmp_path):
    """A symlink in the agent-writable workspace must not become an arbitrary-file write."""
    victim = tmp_path / "victim.py"
    victim.write_text("untouched\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in (snap_mod.AGENT_HELPERS_NAME, snap_mod.HELPER_MODULE_NAME):
        (workspace / name).symlink_to(victim)

    assert snap_mod.ensure_workspace_helpers(str(workspace))

    assert victim.read_text(encoding="utf-8") == "untouched\n"
    for name in (snap_mod.AGENT_HELPERS_NAME, snap_mod.HELPER_MODULE_NAME):
        assert not (workspace / name).is_symlink()
        assert (workspace / name).read_text(encoding="utf-8")


def test_non_utf8_agent_helpers_is_not_clobbered(tmp_path):
    """A non-UTF-8 agent_helpers.py must stay put; the generated helper still installs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = workspace / snap_mod.AGENT_HELPERS_NAME
    original = b"\xff\xfe not utf-8 \x80"
    agent.write_bytes(original)

    assert snap_mod.ensure_workspace_helpers(str(workspace))
    assert agent.read_bytes() == original
    assert (workspace / snap_mod.HELPER_MODULE_NAME).is_file()


def test_helper_source_is_valid_python():
    """The generated workspace module must compile; leftover quote sentinels are a bug."""
    src = snap_mod.helper_source()
    compile(src, snap_mod.HELPER_MODULE_NAME, "exec")
    assert "def snapshot(" in src
    assert "@OPEN@" not in src
    assert "@CLOSE@" not in src
    assert "__MAX__" in src


def test_half_deleted_marker_is_repaired(tmp_path):
    """A begin-marker without its end is replaced, not duplicated; agent code above it stays."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = workspace / snap_mod.AGENT_HELPERS_NAME
    agent.write_text(
        "def my_helper():\n    return 'mine'\n\n"
        + snap_mod._MARKER_BEGIN
        + "\nGARBAGE left after a partial delete\n",
        encoding="utf-8",
    )

    assert snap_mod.ensure_workspace_helpers(str(workspace))
    text = agent.read_text(encoding="utf-8")
    assert "def my_helper" in text
    assert "GARBAGE" not in text
    assert text.count(snap_mod._MARKER_BEGIN) == 1
    assert snap_mod._MARKER_END in text
