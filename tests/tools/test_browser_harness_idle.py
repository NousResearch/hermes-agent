"""Idle harness daemons: a daemon no browser_exec has started for the idle window is stopped; a recently used one is kept."""

import pytest

from tools import browser_use_cli as cli


@pytest.fixture
def driven(monkeypatch):
    monkeypatch.setattr(cli, "_driven_daemons", {})
    monkeypatch.setattr(cli, "_driven_daemon_last_used", {})
    reloaded: list = []
    monkeypatch.setattr(cli, "_reload_daemons", lambda daemons: reloaded.append(list(daemons)))
    return reloaded


def test_idle_daemon_is_stopped_and_recent_one_kept(driven):
    old, new = ("bu", "old"), ("bu", "new")
    cli._driven_daemons.update({old: "home", new: "home"})
    cli._driven_daemon_last_used.update({old: 1_000.0, new: 1_900.0})

    stopped = cli.stop_idle_harness_daemons(idle_s=500, now=2_000.0)

    assert stopped == [old]
    assert driven == [[old]]
    assert set(cli._driven_daemons) == {new}
    assert old not in cli._driven_daemon_last_used


def test_nothing_idle_means_no_reload(driven):
    key = ("bu", "x")
    cli._driven_daemons[key] = "home"
    cli._driven_daemon_last_used[key] = 1_000.0
    assert cli.stop_idle_harness_daemons(idle_s=500, now=1_100.0) == []
    assert driven == [[]]
    assert key in cli._driven_daemons


def test_stop_by_home_also_clears_idle_clock(driven, monkeypatch):
    key = ("bu", "h")
    cli._driven_daemons[key] = "home-a"
    cli._driven_daemon_last_used[key] = 1.0
    cli.stop_harness_daemons("home-a")
    assert key not in cli._driven_daemons
    assert key not in cli._driven_daemon_last_used
    assert driven == [[key]]
