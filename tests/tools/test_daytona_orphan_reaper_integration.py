"""Integration tests for Daytona orphan-reaper wiring in terminal_tool."""

from unittest.mock import patch

import tools.terminal_tool as terminal_tool


def _reset_reaper_gate():
    terminal_tool._daytona_orphan_reaper_ran = False


def test_maybe_reap_runs_once_per_process(monkeypatch):
    _reset_reaper_gate()
    call_count = {"reap": 0}

    def _fake_reap(**kwargs):
        call_count["reap"] += 1
        return 0

    with patch("tools.environments.daytona.reap_orphan_sandboxes", _fake_reap):
        config = {"daytona_orphan_reaper": True}
        terminal_tool._maybe_reap_daytona_orphans(config, "current")
        terminal_tool._maybe_reap_daytona_orphans(config, "current")
        terminal_tool._maybe_reap_daytona_orphans(config, "current")

    assert call_count["reap"] == 1


def test_maybe_reap_respects_disable_flag(monkeypatch):
    _reset_reaper_gate()
    call_count = {"reap": 0}

    def _fake_reap(**kwargs):
        call_count["reap"] += 1
        return 0

    with patch("tools.environments.daytona.reap_orphan_sandboxes", _fake_reap):
        terminal_tool._maybe_reap_daytona_orphans(
            {"daytona_orphan_reaper": False},
            "current",
        )

    assert call_count["reap"] == 0
    assert terminal_tool._daytona_orphan_reaper_ran is False


def test_maybe_reap_passes_current_task_profile_and_age(monkeypatch):
    _reset_reaper_gate()
    captured_args = {}

    def _fake_reap(**kwargs):
        captured_args.update(kwargs)
        return 0

    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "300")
    with patch("tools.environments.daytona.reap_orphan_sandboxes", _fake_reap), \
         patch("tools.environments.daytona._get_active_profile_name", return_value="research"):
        terminal_tool._maybe_reap_daytona_orphans(
            {"daytona_orphan_reaper": True},
            "current",
        )

    assert captured_args["max_age_seconds"] == 600
    assert captured_args["current_task_id"] == "current"
    assert captured_args["profile_filter"] == "research"


def test_maybe_reap_swallows_exceptions(monkeypatch):
    _reset_reaper_gate()

    def _exploding_reap(**kwargs):
        raise RuntimeError("sdk down")

    with patch("tools.environments.daytona.reap_orphan_sandboxes", _exploding_reap):
        terminal_tool._maybe_reap_daytona_orphans(
            {"daytona_orphan_reaper": True},
            "current",
        )


def test_create_environment_runs_daytona_reaper_before_constructor(monkeypatch):
    _reset_reaper_gate()
    events = []

    class FakeDaytonaEnvironment:
        def __init__(self, **kwargs):
            events.append(("create", kwargs["task_id"]))

    def _fake_reap(config, task_id):
        events.append(("reap", task_id, config["daytona_orphan_reaper"]))

    monkeypatch.setattr(terminal_tool, "_maybe_reap_daytona_orphans", _fake_reap)
    monkeypatch.setattr(
        "tools.environments.daytona.DaytonaEnvironment",
        FakeDaytonaEnvironment,
    )

    terminal_tool._create_environment(
        env_type="daytona",
        image="test-image",
        cwd="/home/daytona",
        timeout=60,
        container_config={"daytona_orphan_reaper": True},
        task_id="current",
    )

    assert events == [
        ("reap", "current", True),
        ("create", "current"),
    ]
