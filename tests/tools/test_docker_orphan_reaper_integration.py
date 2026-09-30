"""Integration tests for the docker orphan-reaper wiring in terminal_tool.

The reaper itself is unit-tested in tests/tools/test_docker_environment.py
under the "Orphan reaper" section. These tests cover the terminal_tool-side
gates: once-per-process behavior, the disable flag, and the
``lifetime_seconds`` doubling that determines the reaper's age threshold.

Issue #20561 — without these gates, parallel subagents would each fire the
reaper on container creation, and the ``terminal.docker_orphan_reaper: false``
opt-out would silently do nothing.
"""

import os
from unittest.mock import patch

import tools.terminal_tool as terminal_tool


def _reset_reaper_gate():
    """Clear the once-per-process flag between tests."""
    terminal_tool._docker_orphan_reaper_ran = False


def test_maybe_reap_runs_once_per_process(monkeypatch):
    """The reaper sweep must run at most once per Python interpreter.
    Parallel subagents that each call _create_environment(env_type='docker')
    would otherwise fire N concurrent docker ps + inspect storms against the
    daemon and waste 5–10s of startup."""
    _reset_reaper_gate()
    call_count = {"reap": 0}

    def _fake_reap(**kwargs):
        call_count["reap"] += 1
        return 0

    with patch("tools.environments.docker.reap_orphan_containers", _fake_reap):
        config = {"docker_orphan_reaper": True}
        terminal_tool._maybe_reap_docker_orphans(config)
        terminal_tool._maybe_reap_docker_orphans(config)
        terminal_tool._maybe_reap_docker_orphans(config)

    assert call_count["reap"] == 1, (
        f"reaper must run exactly once per process; got {call_count['reap']} calls"
    )


def test_maybe_reap_passes_current_profile_as_filter(monkeypatch):
    """The reaper must be scoped to the current Hermes profile — a research
    profile must NEVER reap default's containers. Verifies the
    profile-filter wiring."""
    _reset_reaper_gate()
    captured_args = {}

    def _fake_reap(**kwargs):
        captured_args.update(kwargs)
        return 0

    with patch("tools.environments.docker.reap_orphan_containers", _fake_reap), \
         patch("tools.environments.docker._get_active_profile_name", return_value="research-bot"):
        terminal_tool._maybe_reap_docker_orphans({"docker_orphan_reaper": True})

    assert captured_args.get("profile_filter") == "research-bot", (
        f"expected profile_filter='research-bot', got {captured_args.get('profile_filter')!r}"
    )


def test_maybe_reap_swallows_exceptions(monkeypatch):
    """A reaper crash (docker daemon down, parse error in helper) must NOT
    block env creation. The reaper is best-effort plumbing, not a critical
    path; failures get logged at debug level and execution continues."""
    _reset_reaper_gate()

    def _exploding_reap(**kwargs):
        raise RuntimeError("docker daemon ate the cat")

    with patch("tools.environments.docker.reap_orphan_containers", _exploding_reap):
        # Must not raise
        terminal_tool._maybe_reap_docker_orphans({"docker_orphan_reaper": True})
