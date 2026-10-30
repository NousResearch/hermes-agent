"""FIX-2068-e: the UNSUPERVISED (CLI/interactive) raw-kick defer gate.

Regression target: the 2026-10-25 FIX-2068 storm, where a CLI session issued 6 bare
``launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`` terminal calls, each SIGTERM
killing every in-flight kanban worker. The supervised branch of ``gateway_lifecycle_block``
hard-blocks in-gateway lifecycle commands, but by design let the CLI scope through with ZERO
in-flight awareness. These tests pin the new dual mode: the CLI scope now defers a destructive
gateway-lifecycle command while kanban workers are in flight (log + FIX-946 channel + force
hint), still allows it at 0 workers (the legitimate manual-restart channel), honours the
explicit force bypass, and never touches non-lifecycle commands or unrelated launchd labels.
"""

import json
import logging

import pytest

import tools.terminal_tool_guards as g
from tools import process_registry

STORM_CMD = "launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway"


@pytest.fixture(autouse=True)
def _isolate_cooldown_file(tmp_path, monkeypatch):
    """Point the defer cooldown at a fresh temp file so tests never read/write the real
    ~/.hermes/data/l3-cooldown.json (and so the first-defer vs repeat-defer dedup is clean)."""
    monkeypatch.setattr(g, "_KICK_COOLDOWN_FILE", tmp_path / "l3-cooldown.json")


def _unsupervised(monkeypatch):
    """Put the guard in the UNSUPERVISED (CLI) scope: the supervised probe returns False, so
    ``gateway_lifecycle_block`` routes to the new defer gate."""
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)


def _block_for(command, env_type="local", cwd="/tmp", session_key="test"):
    return g.gateway_lifecycle_block(
        command=command, env=None, env_type=env_type, cwd=cwd, workdir=None, session_key=session_key,
    )


# --- core defer / allow decisions (injection seams: fully controlled, no live ps) -------------


def test_defers_storm_form_when_workers_in_flight(monkeypatch):
    _unsupervised(monkeypatch)
    monkeypatch.setattr(g, "_live_gateway_pid", lambda: 12345)
    monkeypatch.setattr(g, "kanban_workers_in_flight", lambda pid: 3)
    monkeypatch.delenv(g._KICK_FORCE_ENV, raising=False)

    blocked = _block_for(STORM_CMD)

    assert blocked is not None
    result = json.loads(blocked)
    assert result["exit_code"] == 1
    assert result["status"] == "deferred"
    # The defer message must name the in-flight count, the FIX-946 out-of-process channel, and
    # the explicit force bypass (acceptance: log/message carries count + recommended channel).
    assert "3" in result["error"]
    assert "gateway-restart-pending" in result["error"]  # FIX-946 signal file
    assert g._KICK_FORCE_ENV in result["error"]


def test_allows_storm_form_at_zero_workers(monkeypatch):
    """The legitimate manual-restart channel: 0 workers in flight -> allow (log only, no block)."""
    _unsupervised(monkeypatch)
    monkeypatch.setattr(g, "_live_gateway_pid", lambda: 12345)
    monkeypatch.setattr(g, "kanban_workers_in_flight", lambda pid: 0)
    monkeypatch.delenv(g._KICK_FORCE_ENV, raising=False)

    assert _block_for(STORM_CMD) is None


def test_allows_storm_form_with_no_gateway_running(monkeypatch):
    """No gateway pid found -> nothing in flight to kill -> allow (a kick just (re)starts it)."""
    _unsupervised(monkeypatch)
    monkeypatch.setattr(g, "_live_gateway_pid", lambda: None)
    monkeypatch.setattr(g, "kanban_workers_in_flight", lambda pid: 0)
    monkeypatch.delenv(g._KICK_FORCE_ENV, raising=False)

    assert _block_for(STORM_CMD) is None


def test_force_bypass_allows_even_with_workers_in_flight(monkeypatch):
    _unsupervised(monkeypatch)
    monkeypatch.setattr(g, "_live_gateway_pid", lambda: 12345)
    monkeypatch.setattr(g, "kanban_workers_in_flight", lambda pid: 4)
    monkeypatch.setenv(g._KICK_FORCE_ENV, "1")

    assert _block_for(STORM_CMD) is None


# --- scope / label safety: the gate must never touch what it doesn't own ----------------------


def test_non_lifecycle_command_passes_even_with_workers_in_flight(monkeypatch):
    _unsupervised(monkeypatch)
    monkeypatch.setattr(g, "_live_gateway_pid", lambda: 12345)
    monkeypatch.setattr(g, "kanban_workers_in_flight", lambda pid: 5)
    monkeypatch.delenv(g._KICK_FORCE_ENV, raising=False)

    assert _block_for("ls -la") is None
    assert _block_for("git status") is None


def test_unrelated_launchd_label_passes_even_with_workers_in_flight(monkeypatch):
    """Label anchoring scopes this to the hermes-gateway; other services stay runnable."""
    _unsupervised(monkeypatch)
    monkeypatch.setattr(g, "_live_gateway_pid", lambda: 12345)
    monkeypatch.setattr(g, "kanban_workers_in_flight", lambda pid: 5)
    monkeypatch.delenv(g._KICK_FORCE_ENV, raising=False)

    assert _block_for("launchctl bootout gui/501/com.example.unrelated") is None
    assert _block_for("launchctl remove ai.hermes.update-checker") is None


def test_supervised_scope_unchanged_still_hard_blocks(monkeypatch):
    """Regression guard: the SUPERVISED branch must keep hard-blocking (not route to the new
    defer gate). Mock the probe True and confirm the existing block still fires."""
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    # The supervised branch blocks hermes-gateway lifecycle commands unconditionally.
    blocked = _block_for(STORM_CMD)
    assert blocked is not None
    result = json.loads(blocked)
    assert result["exit_code"] == 1
    # Supervised rejections are "error" status, never the new "deferred" status.
    assert result["status"] != "deferred"


# --- cooldown log-dedup (2h pattern) ----------------------------------------------------------


def test_first_defer_writes_cooldown_repeat_defer_dedups(monkeypatch, caplog):
    _unsupervised(monkeypatch)
    monkeypatch.setattr(g, "_live_gateway_pid", lambda: 12345)
    monkeypatch.setattr(g, "kanban_workers_in_flight", lambda pid: 2)
    monkeypatch.delenv(g._KICK_FORCE_ENV, raising=False)

    with caplog.at_level(logging.INFO, logger="tools.terminal_tool"):
        b1 = _block_for(STORM_CMD)
        first_msgs = [r.message for r in caplog.records]
        b2 = _block_for(STORM_CMD)
        second_msgs = [r.message for r in caplog.records[len(first_msgs):]]

    assert b1 is not None and b2 is not None
    # First defer: full line naming count + FIX-946 channel + force bypass, and it marks cooldown.
    assert any("DEFERRED" in m and "2 kanban worker" in m and "gateway-restart-pending" in m
               for m in first_msgs)
    assert g._in_kick_cooldown() is True
    # Repeat defer inside the 2h window: still blocks, but the log line is the dedup variant.
    assert any("still DEFERRED" in m and "cooldown active" in m for m in second_msgs)


# --- live helpers (ps-based): must be total (never raise) and sane ----------------------------


def test_kanban_workers_in_flight_none_pid_is_zero(monkeypatch):
    assert g.kanban_workers_in_flight(None) == 0


def test_kanban_workers_in_flight_dead_pid_is_zero(monkeypatch):
    # A pid that has no kanban children (999999 is almost certainly dead) -> 0, no crash.
    assert g.kanban_workers_in_flight(999999) == 0


def test_live_gateway_pid_returns_int_or_none():
    pid = g._live_gateway_pid()
    assert pid is None or isinstance(pid, int)
