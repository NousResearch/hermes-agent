"""Tests for the capabilities tool surface (tools/capabilities_tool.py).

Verifies:
  - The tool is gated to orchestrator profiles (not exposed when
    ``HERMES_KANBAN_TASK`` is set, even if the toolset is enabled).
  - The handler enumerates every profile, returns the expected shape,
    and applies the ``skills.disabled`` filter (with the platform
    overlay) the rest of Hermes honors.
  - Symlink-reached SKILL.md entries are dropped — a worker can't
    enumerate a sibling's skills by planting a symlink in its own
    ``skills/`` tree.
  - Worker-writable description text is sanitized (length-capped,
    control characters stripped) before it lands in the orchestrator's
    LLM prompt context.
  - An INFO audit-log line is emitted on every call.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _write_skill(skills_dir: Path, name: str, description: str = "", category: str = "") -> Path:
    """Create a SKILL.md under skills_dir. Returns the SKILL.md path.

    Empty descriptions are written as ``"."`` rather than blank — an
    actual empty value would parse as YAML null and trip
    ``_find_skills_in_profile``'s missing-len guard. The cross-profile
    behavior under test does not depend on description content.
    """
    if category:
        skill_dir = skills_dir / category / name
    else:
        skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    safe_description = description if description else "."
    md.write_text(
        f"---\nname: {name}\ndescription: {safe_description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return md


def _build_profile_tree(home: Path, profile_skills: dict[str, list[tuple[str, str, str]]]) -> None:
    """Materialize a multi-profile filesystem layout under ``home``.

    ``profile_skills`` maps profile name → list of
    ``(skill_name, description, category)`` triples. ``"default"`` is
    written at ``home/skills``; named profiles at
    ``home/profiles/<name>/skills``.
    """
    for profile_name, skills in profile_skills.items():
        if profile_name == "default":
            base = home
        else:
            base = home / "profiles" / profile_name
        base.mkdir(parents=True, exist_ok=True)
        skills_dir = base / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill_name, description, category in skills:
            _write_skill(skills_dir, skill_name, description, category)


def _isolated_home(monkeypatch, tmp_path: Path) -> Path:
    """Set HERMES_HOME to an isolated temp dir outside ``~/.hermes`` so
    the profiles module treats it as a Docker-style root."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return home


def _call_capabilities(args: dict | None = None) -> list[dict]:
    """Invoke the capabilities_list handler and return parsed JSON.

    Forces a re-import so module-level caches (registry / config) pick
    up the test's HERMES_HOME monkeypatch.
    """
    import importlib

    import tools.capabilities_tool as mod
    importlib.reload(mod)
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()

    result = mod._handle_capabilities_list(args or {})
    return json.loads(result)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_capabilities_tool_hidden_for_dispatched_workers(monkeypatch, tmp_path):
    """A worker spawned via the kanban dispatcher (HERMES_KANBAN_TASK
    set) never sees ``capabilities_list`` in its schema — even if its
    profile config lists the ``capabilities`` toolset.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    (home / "config.yaml").write_text("toolsets:\n  - capabilities\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")

    import tools.capabilities_tool  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "capabilities_list" not in names, (
        "capabilities_list must be gated off for dispatched workers"
    )


def test_capabilities_tool_visible_for_orchestrator(monkeypatch, tmp_path):
    """Orchestrator profile (no HERMES_KANBAN_TASK, capabilities toolset
    enabled) sees the tool."""
    home = _isolated_home(monkeypatch, tmp_path)
    (home / "config.yaml").write_text("toolsets:\n  - capabilities\n", encoding="utf-8")

    import tools.capabilities_tool  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "capabilities_list" in names


def test_capabilities_tool_hidden_when_toolset_not_enabled(monkeypatch, tmp_path):
    """An orchestrator profile that hasn't opted into the capabilities
    toolset doesn't see the tool — same as the kanban gating story.
    """
    _isolated_home(monkeypatch, tmp_path)

    import tools.capabilities_tool  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "capabilities_list" not in names


# ---------------------------------------------------------------------------
# Handler happy paths
# ---------------------------------------------------------------------------

def test_happy_path_three_profiles_six_skills(monkeypatch, tmp_path):
    """3 profiles × 2 skills = 6 entries with correct profile attribution."""
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "default": [
            ("kanban-orchestrator", "Routes work via kanban", "devops"),
            ("status-board", "Reads kanban state", "devops"),
        ],
        "creative": [
            ("live-fully", "Live Fully brand voice", "brand"),
            ("image-gen-prompt-engineer", "Prompt-engineer image-gen", "creative"),
        ],
        "researcher": [
            ("deep-web-research", "Multi-source web research", "research"),
            ("citation-formatter", "Cite sources cleanly", "research"),
        ],
    })

    out = _call_capabilities()
    assert len(out) == 6
    by_profile = {}
    for entry in out:
        by_profile.setdefault(entry["profile"], set()).add(entry["name"])
    assert by_profile == {
        "default": {"kanban-orchestrator", "status-board"},
        "creative": {"live-fully", "image-gen-prompt-engineer"},
        "researcher": {"deep-web-research", "citation-formatter"},
    }
    # Shape sanity-check.
    sample = out[0]
    assert set(sample.keys()) == {"profile", "name", "description", "category"}


def test_profile_filter_scopes_result_to_one_profile(monkeypatch, tmp_path):
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "default": [("a", "", "")],
        "creative": [("b", "", "")],
        "researcher": [("c", "", "")],
    })

    out = _call_capabilities({"profile": "creative"})
    assert len(out) == 1
    assert out[0]["profile"] == "creative"
    assert out[0]["name"] == "b"


def test_unknown_profile_returns_empty(monkeypatch, tmp_path):
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {"default": [("a", "", "")]})

    out = _call_capabilities({"profile": "does-not-exist"})
    assert out == []


# ---------------------------------------------------------------------------
# Disabled-skills filter
# ---------------------------------------------------------------------------

def test_global_disabled_skills_excluded(monkeypatch, tmp_path):
    """``skills.disabled`` in profile config.yaml hides those skills."""
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "creative": [("keep", "", ""), ("hide", "", "")],
    })
    (home / "profiles" / "creative" / "config.yaml").write_text(
        "skills:\n  disabled:\n    - hide\n",
        encoding="utf-8",
    )

    out = _call_capabilities()
    names = {(e["profile"], e["name"]) for e in out}
    assert ("creative", "keep") in names
    assert ("creative", "hide") not in names


def test_platform_overlay_overrides_global(monkeypatch, tmp_path):
    """When ``platform_disabled.<platform>`` is present, it REPLACES the
    global disabled list (mirrors ``get_disabled_skills`` semantics)."""
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "creative": [("only-cli", "", ""), ("global-disabled", "", "")],
    })
    # Global disables "global-disabled"; cli overlay disables "only-cli"
    # instead. With platform="cli" the cli list wins → only "only-cli"
    # is hidden.
    (home / "profiles" / "creative" / "config.yaml").write_text(
        "skills:\n"
        "  disabled:\n"
        "    - global-disabled\n"
        "  platform_disabled:\n"
        "    cli:\n"
        "      - only-cli\n",
        encoding="utf-8",
    )

    out = _call_capabilities({"platform": "cli"})
    creative = [e["name"] for e in out if e["profile"] == "creative"]
    assert creative == ["global-disabled"], (
        f"platform=cli should swap the global list for the cli overlay; "
        f"got {creative}"
    )


# ---------------------------------------------------------------------------
# Symlink isolation
# ---------------------------------------------------------------------------

def test_symlinked_skill_dirs_are_dropped(monkeypatch, tmp_path):
    """A worker that plants a symlink inside its own ``skills/`` tree
    pointing at a sibling's skills directory must not enumerate the
    sibling's skills under its own profile.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "default": [],  # exists so default profile is materialized but empty
        "creative": [("real-creative-skill", "", "")],
        "victim": [("sensitive", "", "")],
    })

    # Worker plants symlink at profiles/creative/skills/peek -> profiles/victim/skills
    src = home / "profiles" / "victim" / "skills"
    dst = home / "profiles" / "creative" / "skills" / "peek"
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("filesystem does not support symlinks")

    out = _call_capabilities({"profile": "creative"})
    names = {e["name"] for e in out}
    assert "real-creative-skill" in names
    assert "sensitive" not in names, (
        "symlinked SKILL.md leaked across profile boundary"
    )


def test_symlinked_skills_dir_itself_is_dropped(monkeypatch, tmp_path):
    """If the profile's ``skills/`` dir is itself a symlink pointing at
    a sibling profile's ``skills/`` tree, every entry returned for the
    spoofing profile would otherwise be victim-sourced. The hardened
    filter must drop the whole tree.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "default": [],
        "victim": [("sensitive", "vd", "")],
    })

    spoof_dir = home / "profiles" / "spoof"
    spoof_dir.mkdir(parents=True, exist_ok=True)
    victim_skills = home / "profiles" / "victim" / "skills"
    try:
        os.symlink(victim_skills, spoof_dir / "skills")
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("filesystem does not support symlinks")

    out = _call_capabilities({"profile": "spoof"})
    assert out == [], (
        "skills/ as a symlink to a sibling profile must not leak entries"
    )


def test_symlinked_skill_md_file_is_dropped(monkeypatch, tmp_path):
    """File-level symlink: a worker creates a real ``skills/spoof/``
    directory, then symlinks ``SKILL.md`` inside it at a victim
    profile's SKILL.md. Every intermediate directory is real, so the
    dir-chain check passes — but the leaf file is a symlink.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "default": [],
        "creative": [("real-skill", "rd", "")],
        "victim": [("sensitive", "vd", "")],
    })

    spoof_dir = home / "profiles" / "creative" / "skills" / "spoof"
    spoof_dir.mkdir(parents=True, exist_ok=True)
    victim_md = home / "profiles" / "victim" / "skills" / "sensitive" / "SKILL.md"
    try:
        os.symlink(victim_md, spoof_dir / "SKILL.md")
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("filesystem does not support symlinks")

    out = _call_capabilities({"profile": "creative"})
    names = {e["name"] for e in out}
    assert "real-skill" in names
    assert "sensitive" not in names, (
        "file-level SKILL.md symlink leaked across profile boundary"
    )


# ---------------------------------------------------------------------------
# Description sanitization
# ---------------------------------------------------------------------------

def test_description_length_capped(monkeypatch, tmp_path):
    home = _isolated_home(monkeypatch, tmp_path)
    long_desc = "X" * 2000
    _build_profile_tree(home, {"creative": [("long", long_desc, "")]})

    out = _call_capabilities({"profile": "creative"})
    assert len(out) == 1
    # ``_find_skills_in_profile`` first truncates to 1024 (skill_tool
    # cap); our handler further trims to 500 with an ellipsis sentinel.
    assert len(out[0]["description"]) <= 500
    assert out[0]["description"].endswith("...")


def test_description_control_chars_stripped(monkeypatch, tmp_path):
    home = _isolated_home(monkeypatch, tmp_path)
    # NUL, BEL, ESC, DEL. YAML can't carry literal NUL/BEL in plain
    # scalars without quoting, but the sanitizer runs on the loaded
    # string regardless of how it got into the frontmatter — set it
    # directly via the SKILL.md write path.
    skills_dir = home / "profiles" / "creative" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = skills_dir / "spicy"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: spicy\n"
        "description: \"clean\\x07text\\x1bhere\\x7fend\"\n"
        "---\n\n# spicy\n",
        encoding="utf-8",
    )

    out = _call_capabilities({"profile": "creative"})
    assert len(out) == 1
    desc = out[0]["description"]
    for forbidden in ("\x07", "\x1b", "\x7f"):
        assert forbidden not in desc, (
            f"control char {forbidden!r} survived sanitization: {desc!r}"
        )
    assert "cleantexthereend" in desc


def test_description_unicode_format_chars_stripped(monkeypatch, tmp_path):
    """Bidi overrides, zero-width chars, and BOMs must be stripped —
    they bypass naive C0+DEL filters but carry prompt-injection
    payloads into the orchestrator's LLM context.
    """
    skills_dir = home = _isolated_home(monkeypatch, tmp_path)
    profile_skills = home / "profiles" / "creative" / "skills" / "tricky"
    profile_skills.mkdir(parents=True, exist_ok=True)
    # U+202E RIGHT-TO-LEFT OVERRIDE, U+200B ZERO WIDTH SPACE,
    # U+FEFF BYTE ORDER MARK, U+2066 LEFT-TO-RIGHT ISOLATE
    (profile_skills / "SKILL.md").write_text(
        "---\n"
        "name: tricky\n"
        "description: \"head‮middle​tail﻿!⁦end\"\n"
        "---\n\n# tricky\n",
        encoding="utf-8",
    )

    out = _call_capabilities({"profile": "creative"})
    assert len(out) == 1
    desc = out[0]["description"]
    for forbidden in ("‮", "​", "﻿", "⁦"):
        assert forbidden not in desc, (
            f"format char {forbidden!r} survived sanitization: {desc!r}"
        )
    # Verify the visible content is preserved.
    assert "head" in desc and "middle" in desc and "tail" in desc and "end" in desc


# ---------------------------------------------------------------------------
# Defensive fallbacks
# ---------------------------------------------------------------------------

def test_malformed_config_yaml_logs_warning_and_falls_back(monkeypatch, tmp_path, caplog):
    """A malformed config.yaml must not crash the whole call; the
    handler logs a warning, treats the profile as if no skills were
    disabled, and continues.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {"creative": [("a", "ad", "")]})
    # Truncated YAML — unbalanced quote → parse error.
    (home / "profiles" / "creative" / "config.yaml").write_text(
        "skills:\n  disabled:\n    - 'unterminated\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING, logger="tools.capabilities_tool")
    out = _call_capabilities({"profile": "creative"})

    names = {e["name"] for e in out}
    assert "a" in names, "skills should still appear with config-load fallback"
    assert any(
        "could not load config" in rec.getMessage() and "creative" in rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    ), f"expected config-load warning; got {[r.getMessage() for r in caplog.records]}"


def test_skills_null_in_config_does_not_crash(monkeypatch, tmp_path, caplog):
    """``skills: null`` (or any non-dict value at the ``skills`` key)
    triggers an AttributeError inside ``get_disabled_skills``. A single
    malformed profile must not block discovery for the rest of the host.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {
        "creative": [("a", "ad", "")],
        "researcher": [("b", "bd", "")],
    })
    (home / "profiles" / "creative" / "config.yaml").write_text(
        "skills:\n",  # YAML loads this as skills: None
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING, logger="tools.capabilities_tool")
    out = _call_capabilities()
    by_profile = {e["profile"]: e["name"] for e in out}
    assert by_profile.get("researcher") == "b", (
        "researcher discovery must not be blocked by creative's malformed config"
    )
    # creative's skills should still appear since the fallback treats
    # the disabled set as empty rather than failing closed.
    assert by_profile.get("creative") == "a"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_handler_accepts_dispatcher_kwargs(monkeypatch, tmp_path):
    """The registry dispatches handlers as ``handler(args, **kwargs)`` and
    the gateway injects context kwargs (``task_id``, etc.) on every
    invocation. The handler must accept those kwargs gracefully — a
    missing ``**kw`` catch-all surfaces in production as
    ``TypeError: unexpected keyword argument 'task_id'`` and the model
    falls back to manual filesystem walking.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {"creative": [("foo", "fd", "")]})

    import importlib
    import tools.capabilities_tool as mod
    importlib.reload(mod)

    # Direct positional+kwargs call matching the dispatcher's exact shape.
    result = mod._handle_capabilities_list({"profile": "creative"}, task_id="t_smoke_test")
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "foo"


def test_info_log_emitted_on_call(monkeypatch, tmp_path, caplog):
    """Every invocation emits an INFO-level audit line naming the
    calling profile, so operators can grep gateway.log for cross-
    profile discovery activity.
    """
    home = _isolated_home(monkeypatch, tmp_path)
    _build_profile_tree(home, {"creative": [("foo", "", "")]})
    monkeypatch.setenv("HERMES_PROFILE", "artemis")

    caplog.set_level(logging.INFO, logger="tools.capabilities_tool")
    _call_capabilities({"profile": "creative", "platform": "cli"})

    matching = [
        rec for rec in caplog.records
        if rec.name == "tools.capabilities_tool"
        and rec.levelno == logging.INFO
        and "capabilities_list called" in rec.getMessage()
    ]
    assert matching, (
        f"expected an INFO audit line; got {[r.getMessage() for r in caplog.records]}"
    )
    msg = matching[0].getMessage()
    assert "artemis" in msg
    assert "creative" in msg
    assert "cli" in msg
