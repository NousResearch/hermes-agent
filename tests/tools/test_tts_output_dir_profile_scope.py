"""Regression tests for profile-scoped TTS default output dir (#98749).

Long-lived multi-profile runtimes (dashboard console, TUI/Desktop backend,
cron, kanban workers) must write synthesized audio into the active profile's
``cache/audio`` when a request is scoped via ``set_hermes_home_override()``,
not into the launch profile's.
"""

from tools import tts_tool


def test_default_output_dir_follows_contextvar_profile_override(tmp_path, monkeypatch):
    """The web server scopes profiles via set_hermes_home_override() rather
    than mutating the process env; the accessor must follow that override."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "ramona"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(default_home))

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
