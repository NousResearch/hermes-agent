"""Real temporary config files; approval-state operations only, no commands."""
from contextlib import contextmanager

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import approval


@contextmanager
def selected_home(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def write_allowlist(home, entries):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "fixture"}, "command_allowlist": entries,
    }), encoding="utf-8")


def disk_allowlist(home):
    return set(yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["command_allowlist"])


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, named, other = (tmp_path / name for name in ("launch", "named", "other"))
    write_allowlist(launch, ["launch-only"])
    write_allowlist(named, ["revoked-op", "kept-op"])
    write_allowlist(other, ["other-only"])
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})
    monkeypatch.setattr(approval, "_permanent_baseline_by_home", {})
    monkeypatch.setattr(approval, "_session_approved", {})
    with selected_home(None):
        approval.load_permanent_allowlist()
        yield launch, named, other


@pytest.mark.parametrize("initial_load", ["lazy", "explicit"])
def test_named_home_save_preserves_disk_edits_without_resurrecting_revocation(homes, initial_load):
    launch, named, other = homes
    launch_bytes = (launch / "config.yaml").read_bytes()
    other_bytes = (other / "config.yaml").read_bytes()
    with selected_home(named):
        if initial_load == "explicit":
            approval.load_permanent_allowlist()
        assert approval.is_approved("named-session", "revoked-op")
        assert not approval.is_approved("named-session", "launch-only")
        write_allowlist(named, ["kept-op", "operator-added"])
        # Revocation is intentionally synchronized on reload/save, not watched.
        assert approval.is_approved("named-session", "revoked-op")
        approval._persist_choice("named-session", "always", [("new-grant", "Fixture", False)])
        assert disk_allowlist(named) == {"kept-op", "operator-added", "new-grant"}
        assert not approval.is_approved("another-session", "revoked-op")
        assert approval.is_approved("another-session", "new-grant")
    assert (launch / "config.yaml").read_bytes() == launch_bytes
    assert (other / "config.yaml").read_bytes() == other_bytes
    assert approval.is_approved("launch-session", "launch-only")
    assert not approval.is_approved("launch-session", "new-grant")
    with selected_home(other):
        assert approval.is_approved("other-session", "other-only")
        assert not approval.is_approved("other-session", "new-grant")


@pytest.mark.parametrize("scoped", [False, True])
def test_empty_reload_revokes_then_explicit_new_grant_persists_only_in_selected_home(homes, scoped):
    launch, named, other = homes
    home = named if scoped else launch
    with selected_home(home if scoped else None):
        previous = approval.load_permanent_allowlist()
        approval.approve_session("ongoing-session", "session-only")
        write_allowlist(home, [])
        assert approval.load_permanent_allowlist() == set()
        assert all(not approval.is_approved("fresh-session", key) for key in previous)
        assert approval.is_approved("ongoing-session", "session-only")
        approval._persist_choice("fresh-session", "always", [("new-grant", "Fixture", False)])
        assert disk_allowlist(home) == {"new-grant"}
        assert approval.is_approved("later-session", "new-grant")
    assert disk_allowlist(other) == {"other-only"}


def test_reload_of_one_home_does_not_replace_another_cached_home(homes):
    _, named, other = homes
    with selected_home(other):
        approval.load_permanent_allowlist()
    with selected_home(named):
        approval.load_permanent_allowlist()
        write_allowlist(named, [])
        approval.load_permanent_allowlist()
    with selected_home(other):
        assert approval.is_approved("other-session", "other-only")
        approval._persist_choice("other-session", "always", [("other-new", "Fixture", False)])
        assert disk_allowlist(other) == {"other-only", "other-new"}
    assert disk_allowlist(named) == set()
