from __future__ import annotations

from types import SimpleNamespace

import tools.kanban_tools as kanban_tools


class _Connection:
    def close(self) -> None:
        pass


def _reset_heartbeat_window() -> None:
    kanban_tools._auto_heartbeat_last_attempt = 0.0


def test_auto_heartbeat_fences_worker_after_confirmed_claim_loss(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "Mac:123")
    kb = SimpleNamespace(
        heartbeat_claim=lambda *_args, **_kwargs: False,
        heartbeat_worker=lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(kanban_tools, "_connect", lambda: (kb, _Connection()))
    _reset_heartbeat_window()
    lost: list[str] = []

    attempted = kanban_tools.heartbeat_current_worker_from_env(
        on_lease_lost=lost.append
    )

    assert attempted is True
    assert lost == ["t_deadbeef"]


def test_auto_heartbeat_fences_worker_after_confirmed_run_loss(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "Mac:123")
    kb = SimpleNamespace(
        heartbeat_claim=lambda *_args, **_kwargs: True,
        heartbeat_worker=lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(kanban_tools, "_connect", lambda: (kb, _Connection()))
    _reset_heartbeat_window()
    lost: list[str] = []

    kanban_tools.heartbeat_current_worker_from_env(on_lease_lost=lost.append)

    assert lost == ["t_deadbeef"]


def test_auto_heartbeat_does_not_fence_on_unavailable_board(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    monkeypatch.setattr(
        kanban_tools,
        "_connect",
        lambda: (_ for _ in ()).throw(OSError("database busy")),
    )
    _reset_heartbeat_window()
    lost: list[str] = []

    attempted = kanban_tools.heartbeat_current_worker_from_env(
        on_lease_lost=lost.append
    )

    assert attempted is False
    assert lost == []
