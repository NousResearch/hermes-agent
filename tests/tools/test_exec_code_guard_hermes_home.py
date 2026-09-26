"""Per-call Hermes-home regressions for PR #65592 / issue #113421."""

import os
from types import SimpleNamespace

import pytest

import hermes_constants as homes
import tools.exec_code_policy as policy


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    # Exercise real path/context resolvers; never replace the resolver under test.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    token = homes.set_hermes_home_override(None)
    try:
        yield tmp_path
    finally:
        homes.reset_hermes_home_override(token)


@pytest.fixture
def windows_home(monkeypatch, isolated_home):
    # Patch only this module's platform input, not process-global sys.platform
    # or os.name (which would change pathlib/pytest behaviour on POSIX runners).
    monkeypatch.setattr(homes, "sys", SimpleNamespace(platform="win32"))
    local = isolated_home / "AppData" / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    return local / "hermes"


def test_candidates_include_windows_default_and_legacy(windows_home):
    candidates = policy._hermes_home_candidates()
    assert str(windows_home) in candidates
    assert os.path.expanduser("~/.hermes") in candidates
    assert isinstance(candidates, tuple)
    assert all(isinstance(value, str) for value in candidates)
    assert len(candidates) == len(set(candidates))


@pytest.mark.parametrize("local_appdata", [None, "", "   "])
def test_candidates_windows_fallback(monkeypatch, windows_home, local_appdata):
    if local_appdata is None:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
    else:
        monkeypatch.setenv("LOCALAPPDATA", local_appdata)
    assert str(windows_home) in policy._hermes_home_candidates()


def test_candidates_recompute_localappdata(monkeypatch, isolated_home, windows_home):
    assert str(windows_home) in policy._hermes_home_candidates()
    other = isolated_home / "other-local"
    monkeypatch.setenv("LOCALAPPDATA", str(other))
    candidates = policy._hermes_home_candidates()
    assert str(other / "hermes") in candidates
    assert str(windows_home) not in candidates


def test_candidates_preserve_override_and_native(monkeypatch, isolated_home, windows_home):
    custom = isolated_home / "custom-home"
    monkeypatch.setenv("HERMES_HOME", str(custom))
    candidates = policy._hermes_home_candidates()
    assert str(custom) in candidates
    assert str(windows_home) in candidates
    assert os.path.expanduser("~/.hermes") in candidates


def test_candidates_recompute_environment(monkeypatch, isolated_home, windows_home):
    first, second = isolated_home / "first", isolated_home / "second"
    monkeypatch.setenv("HERMES_HOME", str(first))
    assert str(first) in policy._hermes_home_candidates()
    monkeypatch.setenv("HERMES_HOME", str(second))
    candidates = policy._hermes_home_candidates()
    assert str(second) in candidates
    assert str(first) not in candidates


def test_candidates_follow_context_without_changing_environment(
    monkeypatch, isolated_home, windows_home,
):
    process_home = isolated_home / "process-home"
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    first, second = isolated_home / "profile-a", isolated_home / "profile-b"
    for active, inactive in ((first, second), (second, first), (first, second)):
        token = homes.set_hermes_home_override(active)
        try:
            candidates = policy._hermes_home_candidates()
            assert str(active) in candidates
            assert str(inactive) not in candidates
            assert str(process_home) in candidates
            assert str(windows_home) in candidates
            assert os.environ["HERMES_HOME"] == str(process_home)
        finally:
            homes.reset_hermes_home_override(token)
    assert str(first) not in policy._hermes_home_candidates()
    assert str(second) not in policy._hermes_home_candidates()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_candidates_preserve_posix_default(monkeypatch, isolated_home, platform):
    monkeypatch.setattr(homes, "sys", SimpleNamespace(platform=platform))
    candidates = policy._hermes_home_candidates()
    assert str(isolated_home / ".hermes") in candidates
    assert len(candidates) == len(set(candidates))


def _configure_mode(monkeypatch, mode):
    import tools.approval as approval
    import tools.approval_context as context

    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", mode == "yolo")
    monkeypatch.setattr(context, "_get_approval_mode", lambda: "off" if mode == "off" else "manual")
    return approval


@pytest.mark.parametrize("mode", ["normal", "yolo", "off"])
@pytest.mark.parametrize("writer", ["open", "pathlib", "library"])
@pytest.mark.parametrize("separators", ["forward", "backward"])
def test_guard_windows_default_config_hard_blocked(
    monkeypatch, windows_home, mode, writer, separators,
):
    approval = _configure_mode(monkeypatch, mode)
    path = str(windows_home / "config.yaml").replace("\\", "/")
    if separators == "backward":
        path = path.replace("/", "\\")
    if writer == "open":
        code = f"open({path!r}, 'w').write('test')"
    elif writer == "pathlib":
        code = f"from pathlib import Path\nPath({path!r}).write_text('test')"
    else:
        code = f"frame.to_csv({path!r})"
    result = approval.check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


def test_guard_tracks_active_config_per_call(monkeypatch, isolated_home, windows_home):
    from hermes_cli.config import get_config_path

    approval = _configure_mode(monkeypatch, "yolo")
    for name in ("profile-a", "profile-b", "profile-a"):
        active = isolated_home / name
        token = homes.set_hermes_home_override(active)
        try:
            config = get_config_path()
            assert config.parent == active
            result = approval.check_execute_code_guard(
                f"open({str(config)!r}, 'w').write('test')", env_type="local",
            )
            assert result["approved"] is False
            assert result["outcome"] == "hard_blocked"
        finally:
            homes.reset_hermes_home_override(token)


@pytest.mark.parametrize("relative", ["skills/data.json", "logs/run.log", "cache/data.json"])
def test_guard_keeps_windows_workspace_writable(monkeypatch, windows_home, relative):
    approval = _configure_mode(monkeypatch, "yolo")
    path = str(windows_home / relative)
    assert policy._write_target_is_sensitive(path) is False
    result = approval.check_execute_code_guard(
        f"open({path!r}, 'w').write('test')", env_type="local",
    )
    assert result["approved"] is True


def test_guard_allows_read_only_config(monkeypatch, windows_home):
    approval = _configure_mode(monkeypatch, "yolo")
    result = approval.check_execute_code_guard(
        f"open({str(windows_home / 'config.yaml')!r}, 'r').read()", env_type="local",
    )
    assert result["approved"] is True


def test_guard_does_not_protect_similarly_prefixed_directory(windows_home):
    sibling = windows_home.with_name("hermes-unrelated") / "config.yaml"
    assert policy._write_target_is_sensitive(str(sibling)) is False
