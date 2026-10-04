"""``kanban_create`` assignee guard + read-only ``kanban_discover``.

Covers, per the work order:
  - ``kanban_create`` rejects a nonexistent assignee BEFORE any task / edge /
    event / workspace side effect, reusing the authoritative profile
    enumeration (``hermes_cli.profiles`` — the same kernel the CLI and the
    dispatcher's spawn gate use), with ``default`` included correctly and
    profile-root / env semantics respected
  - ``kanban_discover`` lists every profile, with optional ``profile.yaml``
    descriptor metadata; profiles WITHOUT a descriptor are included, malformed
    or unreadable descriptors are reported explicitly, and no config,
    credential, ``SOUL`` or prompt content is ever returned
  - discovery is read-only: no board is opened, nothing is written and no
    descriptor is generated; output strings are length-bounded
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- Fixtures

def _install_profile(home: Path, name: str) -> Path:
    """A live named profile: ``config.yaml`` is the identity marker
    (``named_profile_has_identity``) that makes the directory a profile."""
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    return profile_dir


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    """Orchestrator-context isolated home with no dispatcher worker env."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_SESSION_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_DELEGATED_CHILD_CONTEXT",
                "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _db():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    return kb, kbc


def _dispatch(tool, args):
    """Real registry dispatch, tolerant of str or dict handler results."""
    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    from tools.registry import registry
    out = registry.dispatch(tool, args)
    return out if isinstance(out, dict) else json.loads(out)


def _discover_raw() -> dict:
    """``kanban_discover`` straight from the handler, bypassing the registry's
    own result normalization/redaction — so a leak here is this tool's leak."""
    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    return json.loads(kt._handle_discover({}))


def _board_fingerprint(db_path: Path):
    """Every count a refused create must leave untouched, plus the file itself."""
    import sqlite3

    stat = db_path.stat()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("tasks", "task_events", "task_links", "task_runs", "task_comments")
        }
    finally:
        conn.close()
    return (stat.st_mtime_ns, stat.st_ino, counts)


def _profile_dir_snapshot(home: Path):
    """``{path: (mtime_ns, size)}`` for every file under ``home/profiles``."""
    root = home / "profiles"
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# --------------------------------------------------------------------------- kanban_create

def test_create_rejects_an_unknown_assignee_before_any_side_effect(board_env):
    kb, kbc = _db()
    db_path = board_env / "kanban.db"
    before = _board_fingerprint(db_path)

    out = _dispatch("kanban_create", {"title": "typo'd profile", "assignee": "nope"})
    assert out.get("ok") is not True
    # Wording contract: profiles are "not found" (they are not installable
    # packages) and the refusal always names the roster + the discover tool.
    assert "profile 'nope' was not found" in out["error"], out
    assert "not installed" not in out["error"], out
    # The roster is named back, and the fix is pointed at.
    assert "default" in out["error"]
    assert "kanban_discover" in out["error"]
    assert out["error"].endswith("Nothing changed.")
    assert _board_fingerprint(db_path) == before, "a refused create wrote to the board"


def test_create_still_requires_an_assignee(board_env):
    out = _dispatch("kanban_create", {"title": "no owner"})
    assert out.get("ok") is not True
    assert "assignee is required" in out["error"]
    # Not an "unknown parameter" failure — the key set itself is legal.
    assert "unknown parameter" not in out["error"]


def test_create_accepts_the_default_profile_without_a_profiles_dir(board_env):
    """``default`` is the implicit profile rooted at HERMES_HOME itself: it must
    resolve even when ``profiles/`` has never been created."""
    assert not (board_env / "profiles").is_dir()
    out = _dispatch("kanban_create", {"title": "root card", "assignee": "default"})
    assert out["ok"] is True, out


def test_create_accepts_an_installed_named_profile(board_env):
    _install_profile(board_env, "worker2")
    out = _dispatch("kanban_create", {"title": "named card", "assignee": "worker2"})
    assert out["ok"] is True, out
    kb, kbc = _db()
    with kbc.connect() as conn:
        assert kb.get_task(conn, out["task_id"]).assignee == "worker2"


def test_create_rejects_a_tombstoned_or_markerless_directory(board_env):
    """A directory that is not a live profile must not pass: no identity marker
    (cron/logging side effects can leave marker-less shells behind)."""
    ghost = board_env / "profiles" / "ghost"
    ghost.mkdir(parents=True)
    out = _dispatch("kanban_create", {"title": "ghost card", "assignee": "ghost"})
    assert out.get("ok") is not True
    assert "profile 'ghost' was not found" in out["error"]
    assert "not installed" not in out["error"]


def test_every_roster_name_discover_returns_is_accepted_by_create(board_env):
    """Discovery and the create guard share ONE enumeration, so they can never
    disagree about what a valid assignee is."""
    _install_profile(board_env, "alpha")
    _install_profile(board_env, "beta")

    roster = _dispatch("kanban_discover", {})
    assert roster["ok"] is True
    names = [p["name"] for p in roster["profiles"]]
    assert "default" in names and "alpha" in names and "beta" in names

    for name in names:
        out = _dispatch("kanban_create", {"title": f"for {name}", "assignee": name})
        assert out["ok"] is True, (name, out)


# --------------------------------------------------------------------------- kanban_discover

def test_discover_lists_default_and_named_profiles(board_env):
    _install_profile(board_env, "worker2")
    out = _dispatch("kanban_discover", {})
    assert out["ok"] is True
    by_name = {p["name"]: p for p in out["profiles"]}
    assert set(by_name) == {"default", "worker2"}
    assert by_name["default"]["is_default"] is True
    assert by_name["worker2"]["is_default"] is False
    assert out["count"] == len(out["profiles"])


def test_discover_includes_profiles_without_a_descriptor_and_does_not_generate_one(board_env):
    """A profile with no ``profile.yaml`` is listed with an explicit status, and
    discovery must never run the describer (which would WRITE one)."""
    profile_dir = _install_profile(board_env, "plain")
    before = _profile_dir_snapshot(board_env)

    out = _dispatch("kanban_discover", {})
    entry = {p["name"]: p for p in out["profiles"]}["plain"]
    assert entry["descriptor"] == {"status": "missing"}
    assert "description" not in entry["descriptor"]

    assert not (profile_dir / "profile.yaml").exists(), "discovery generated a descriptor"
    assert _profile_dir_snapshot(board_env) == before, "discovery wrote to a profile dir"


def test_discover_returns_descriptor_metadata(board_env):
    profile_dir = _install_profile(board_env, "described")
    (profile_dir / "profile.yaml").write_text(
        "description: Reads and edits Python repositories.\ndisplay_name: Reader\n",
        encoding="utf-8")

    out = _dispatch("kanban_discover", {})
    descriptor = {p["name"]: p for p in out["profiles"]}["described"]["descriptor"]
    assert descriptor["status"] == "ok"
    assert descriptor["description"] == "Reads and edits Python repositories."
    assert descriptor["display_name"] == "Reader"
    assert descriptor["role"] is None


def test_discover_reports_a_malformed_descriptor_explicitly(board_env):
    profile_dir = _install_profile(board_env, "broken")
    (profile_dir / "profile.yaml").write_text(
        "description: [this never closes\n  bad: : :\n", encoding="utf-8")

    out = _dispatch("kanban_discover", {})
    entry = {p["name"]: p for p in out["profiles"]}["broken"]
    assert entry["descriptor"]["status"] == "invalid", entry
    # Only the exception class: a YAML scanner error may quote the document.
    assert entry["descriptor"]["detail"] == "ParserError"
    assert "never closes" not in json.dumps(out), "descriptor contents leaked"


def test_discover_reports_a_non_mapping_descriptor_explicitly(board_env):
    profile_dir = _install_profile(board_env, "listy")
    (profile_dir / "profile.yaml").write_text("- one\n- two\n", encoding="utf-8")

    out = _dispatch("kanban_discover", {})
    entry = {p["name"]: p for p in out["profiles"]}["listy"]
    assert entry["descriptor"] == {"status": "invalid", "detail": "not-a-mapping"}


def test_discover_degrades_unhashable_role_to_none_without_aborting_the_roster(board_env):
    """Valid YAML mapping, malformed VALUE: ``role: [setup]`` is unhashable in
    the canonical loader's ``PROFILE_ROLES`` membership test, which used to
    raise ``TypeError`` straight out of ``_handle_discover``. The approved fix
    in ``read_profile_meta`` tests the type before the membership check, so the
    bad VALUE degrades to ``None`` exactly like an unknown scalar: the record is
    a valid descriptor, the rest of the document still comes through, and the
    roster survives — profiles listed after it included. The former
    ``invalid``/``TypeError`` expectation is obsolete with that root fix; the
    discovery contract it guarded (never abort, never leak exception detail)
    still holds here."""
    _install_profile(board_env, "broken_meta")
    _install_profile(board_env, "zz_after")  # enumerated after the broken one
    (board_env / "profiles" / "broken_meta" / "profile.yaml").write_text(
        "description: 'DescriptorBodyMarker'\n"
        "display_name: 'MarkerName'\n"
        "role: [setup]\n",
        encoding="utf-8")

    for payload in (_discover_raw(), _dispatch("kanban_discover", {})):
        assert payload["ok"] is True, payload
        by_name = {p["name"]: p for p in payload["profiles"]}
        descriptor = by_name["broken_meta"]["descriptor"]
        # The guard isolates the bad VALUE: the record is valid, and the rest of
        # the document still comes through instead of degrading to defaults.
        assert descriptor == {"status": "ok", "description": "DescriptorBodyMarker",
                              "display_name": "MarkerName", "role": None}, descriptor
        # Not aborted: every profile the roster enumerated still appears.
        assert set(by_name) == {"default", "broken_meta", "zz_after"}, sorted(by_name)
        # No exception detail or message is surfaced for the degraded value.
        dumped = json.dumps(payload)
        assert "TypeError" not in dumped
        assert "unhashable" not in dumped


def test_discover_reports_an_unreadable_descriptor_without_raising(board_env):
    profile_dir = _install_profile(board_env, "locked")
    descriptor = profile_dir / "profile.yaml"
    descriptor.write_text("description: secret-ish\n", encoding="utf-8")
    os.chmod(descriptor, 0o000)
    try:
        if os.access(descriptor, os.R_OK):
            pytest.skip("descriptor still readable (running as root?)")
        out = _dispatch("kanban_discover", {})
    finally:
        os.chmod(descriptor, 0o644)
    assert out["ok"] is True, out
    entry = {p["name"]: p for p in out["profiles"]}["locked"]
    assert entry["descriptor"]["status"] == "unreadable"


def test_discover_returns_no_config_credentials_soul_or_prompt_content(board_env):
    """Only ``profile.yaml``'s descriptor fields are read — never ``config.yaml``,
    ``.env``, ``SOUL.md`` or anything under ``skills/`` — and even the descriptor
    text crosses the shared redaction boundary."""
    profile_dir = _install_profile(board_env, "loaded")
    (profile_dir / "config.yaml").write_text(
        "model: super-secret-model\napi_key: CONFIG_SECRET_VALUE\n", encoding="utf-8")
    (profile_dir / "SOUL.md").write_text("SOUL_MARKER_PROMPT_CONTENT\n", encoding="utf-8")
    (profile_dir / ".env").write_text("HERMES_API_TOKEN=ENV_SECRET_VALUE\n", encoding="utf-8")
    (profile_dir / "profile.yaml").write_text(
        "description: 'handles ghp_" + "A" * 40 + "'\n", encoding="utf-8")

    raw = _discover_raw()
    dispatched = _dispatch("kanban_discover", {})
    dumped = json.dumps(raw) + json.dumps(dispatched)
    for marker in ("CONFIG_SECRET_VALUE", "super-secret-model", "SOUL_MARKER_PROMPT_CONTENT",
                   "ENV_SECRET_VALUE"):
        assert marker not in dumped, marker
    # The credential inside the descriptor itself is masked too.
    assert "ghp_" + "A" * 40 not in dumped

    entry = {p["name"]: p for p in dispatched["profiles"]}["loaded"]
    assert entry["descriptor"]["status"] == "ok"


def test_discover_bounds_long_descriptors(board_env):
    profile_dir = _install_profile(board_env, "verbose")
    (profile_dir / "profile.yaml").write_text(
        "description: '" + ("word " * 400) + "'\ndisplay_name: '" + ("D" * 300) + "'\n",
        encoding="utf-8")

    out = _dispatch("kanban_discover", {})
    descriptor = {p["name"]: p for p in out["profiles"]}["verbose"]["descriptor"]
    assert len(descriptor["description"]) <= 400, len(descriptor["description"])
    assert len(descriptor["display_name"]) <= 64, len(descriptor["display_name"])
    assert descriptor["description"].endswith("…")


def test_discover_is_read_only_never_opens_or_initializes_a_board(board_env):
    kb, kbc = _db()
    _install_profile(board_env, "worker2")
    db_path = board_env / "kanban.db"
    before = _board_fingerprint(db_path)
    profiles_before = _profile_dir_snapshot(board_env)

    out = _dispatch("kanban_discover", {})
    assert out["ok"] is True

    assert _board_fingerprint(db_path) == before, "discovery touched the board"
    assert _profile_dir_snapshot(board_env) == profiles_before, "discovery wrote a file"


def test_discover_is_scoped_to_the_hermes_home_root(board_env, tmp_path):
    """Profiles outside this home's root are not this home's assignees."""
    _install_profile(board_env, "mine")
    elsewhere = tmp_path / "elsewhere"
    stray = elsewhere / "profiles" / "stray"
    stray.mkdir(parents=True)
    (stray / "config.yaml").write_text("{}\n", encoding="utf-8")

    names = [p["name"] for p in _dispatch("kanban_discover", {})["profiles"]]
    assert "mine" in names
    assert "stray" not in names


def test_discover_schema_is_registered_with_no_required_arguments(board_env):
    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    from tools.registry import registry
    from toolsets import TOOLSETS

    schema = registry.get_schema("kanban_discover")
    assert schema is not None, "kanban_discover is not registered"
    assert schema["parameters"]["required"] == []
    assert "board" in schema["parameters"]["properties"]
    assert "kanban_discover" in TOOLSETS["kanban"]["tools"]
    # Read-only roster work is not a board-routing move: workers may call it.
    assert "kanban_discover" not in kt._ORCHESTRATOR_TOOLS
