"""Regression tests for profile-scoped TTS default output dir (#98749).

``DEFAULT_OUTPUT_DIR`` was resolved once at import time, so long-lived
multi-profile runtimes (dashboard console, TUI/Desktop backend, cron, kanban
workers) kept writing synthesized audio into the launch profile's
``cache/audio`` even while the request was scoped to a different profile via
``HERMES_HOME`` or ``set_hermes_home_override()``. The call-time accessor
``_default_output_dir()`` re-resolves from the live profile-scoped home;
these pins keep the synthesis paths from re-freezing the launch profile.
"""

import importlib
from pathlib import Path


def _reload_tts_tool(import_home: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(import_home))
    import tools.tts_tool as tts_tool

    return importlib.reload(tts_tool)


def test_default_output_dir_follows_contextvar_profile_override(tmp_path, monkeypatch):
    """The web server scopes profiles via set_hermes_home_override() rather
    than mutating the process env; the accessor must follow that override."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "ramona"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)

    tts_tool = _reload_tts_tool(default_home, monkeypatch)

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(str(profile_home))
    try:
        assert tts_tool._default_output_dir() == str(
            profile_home / "cache" / "audio"
        )
    finally:
        reset_hermes_home_override(token)

    # Outside the override scope the launch home applies again.
    assert tts_tool._default_output_dir() == str(default_home / "cache" / "audio")


def test_explicit_default_output_dir_monkeypatch_still_wins(tmp_path, monkeypatch):
    """Existing tests and external patchers can still override
    tools.tts_tool.DEFAULT_OUTPUT_DIR directly."""
    default_home = tmp_path / "default-home"
    default_home.mkdir(parents=True)

    tts_tool = _reload_tts_tool(default_home, monkeypatch)

    monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", "/custom/audio")

    assert tts_tool._default_output_dir() == "/custom/audio"
