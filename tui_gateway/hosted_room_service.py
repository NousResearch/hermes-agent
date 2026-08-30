"""Production coordinator for same-gateway hosted Discussion rooms."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import (
    HostedRoomPolicyCheckpoint,
    PolicySnapshot,
)
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC


_HOSTED_ROOM_IDLE_FALLBACK_SECONDS = 5.0
_HOSTED_ROOM_ACTIVE_POLL_SECONDS = 0.25
_HOSTED_ROOM_TERMINAL_GRACE_SECONDS = 30.0


def _hosted_room_turn_timeout_seconds() -> float:
    try:
        agent_timeout = float(os.getenv("HERMES_AGENT_TIMEOUT", "1800"))
    except (TypeError, ValueError):
        agent_timeout = 1800.0
    if agent_timeout <= 0:
        agent_timeout = 1800.0
    return agent_timeout + _HOSTED_ROOM_TERMINAL_GRACE_SECONDS


class HostedRoomService:
    """Own the hosted Discussion policy and its transport-free worker."""

    def __init__(
        self, server: ModuleType, *, db_path: Path | str | None = None
    ) -> None:
        self.server = server
        self.db_path = Path(db_path or hosted_rooms.default_db_path())
        hosted_rooms.prune_disbanded_rooms(self.db_path)
        self._policy_lock = threading.RLock()
        self._pending_actions: dict[tuple[str, str], dict[str, Any]] = {}
        self.policy_checkpoint = HostedRoomPolicyCheckpoint(self.db_path)
        self.rpc = HostedRoomServerRPC(server)
        self.runtime = HostedRoomRuntime(
            db_path=self.db_path,
            rooms=self.bindings,
            rpc=self.rpc,
            turn_lock=self._turn_lock,
            prepare_room=self.prepare_room,
            publish_terminal=self.publish_terminal,
            pending_action=self._set_pending_action,
            poll_interval_seconds=_HOSTED_ROOM_IDLE_FALLBACK_SECONDS,
            active_poll_interval_seconds=_HOSTED_ROOM_ACTIVE_POLL_SECONDS,
            turn_timeout_seconds=_hosted_room_turn_timeout_seconds(),
        )

    @property
    def root(self) -> Path:
        return self.db_path.parent

    def local_profiles(self) -> tuple[str, ...]:
        profiles = {"default"}
        profiles_dir = self.root / "profiles"
        if profiles_dir.is_dir():
            profiles.update(
                path.name for path in profiles_dir.iterdir() if path.is_dir()
            )
        return tuple(sorted(profiles))

    def bindings(self) -> tuple[HostedRoomBinding, ...]:
        local_gateway_id = hosted_rooms.local_authority_gateway_id()
        return tuple(
            HostedRoomBinding(
                room_id=str(room["room_id"]),
                gateway_id=str(room["authority_gateway_id"]),
                authority_epoch=int(room["authority_epoch"]),
            )
            for room in hosted_rooms.list_rooms(self.db_path)
            if str(room["authority_gateway_id"]) == local_gateway_id
        )

    def _owned_room(self, room_id: str) -> dict[str, Any]:
        room = hosted_rooms.room_state(self.db_path, room_id=room_id)
        if str(room["authority_gateway_id"]) != (
            hosted_rooms.local_authority_gateway_id()
        ):
            raise hosted_rooms.AuthorityConflictError(
                "This Group Chat is managed by another gateway."
            )
        return room

    @contextlib.contextmanager
    def _turn_lock(self, profile: str) -> Iterator[None]:
        from tools.bot_relay import acquire_turn_lock

        with acquire_turn_lock(self.root, profile):
            yield

    def start(self) -> None:
        self.runtime.start()

    def stop(self, *, timeout: float = 5.0) -> bool:
        return self.runtime.stop(timeout=timeout)

    def wakeup(self) -> None:
        self.runtime.wakeup()

    def _set_pending_action(
        self,
        room_id: str,
        member_id: str,
        action: Mapping[str, Any] | None,
    ) -> None:
        key = (room_id, member_id)
        with self._policy_lock:
            if action is None:
                self._pending_actions.pop(key, None)
            else:
                self._pending_actions[key] = {**action, "member_id": member_id}

    def _events(self, room_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        cursor = 0
        while True:
            page = hosted_rooms.read_events(
                self.db_path,
                room_id=room_id,
                since_seq=cursor,
                limit=hosted_rooms.MAX_LOG_LIMIT,
            )
            rows = page.get("events")
            if isinstance(rows, list):
                events.extend(row for row in rows if isinstance(row, dict))
            next_cursor = int(page.get("cursor") or cursor)
            if not page.get("has_more"):
                return events
            if next_cursor <= cursor:
                raise RuntimeError("hosted room replay cursor did not advance")
            cursor = next_cursor

    def _append_plan(self, room_id: str, plan: discussion.PublicationPlan) -> None:
        for event in plan.events:
            hosted_rooms.append_event(
                self.db_path,
                **event.append_kwargs(room_id),
            )

    def _policy_snapshot(self, room: Mapping[str, Any]) -> PolicySnapshot:
        return self.policy_checkpoint.snapshot(
            room_id=str(room["room_id"]),
            latest_seq=int(room["latest_seq"]),
        )

    def _publish_terminal_tasks(
        self,
        room: Mapping[str, Any],
    ) -> bool:
        changed = False
        local_profiles = self.local_profiles()
        for status in ("deferred", "settled", "failed", "cancelled"):
            for task in driver.list_tasks(
                self.db_path,
                room_id=str(room["room_id"]),
                status=status,
            ):
                identity = task["identity"]
                if self.policy_checkpoint.publication_exists(
                    room_id=str(room["room_id"]),
                    task_id=identity.task_id,
                    status=status,
                    execution_generation=int(task["execution_generation"]),
                ):
                    continue
                task_events = self.policy_checkpoint.events_for_task(
                    room_id=str(room["room_id"]),
                    source_event_seq=int(task["payload"]["source_event_seq"]),
                )
                plan = discussion.reconstruct_task_plan(
                    room,
                    task_events,
                    task,
                    local_profiles=local_profiles,
                )
                publication = discussion.plan_publication(
                    room,
                    task_events,
                    plan,
                    status=status,
                    result=task.get("result"),
                    execution_generation=(
                        int(task["execution_generation"])
                        if status == "deferred"
                        else None
                    ),
                    local_profiles=local_profiles,
                )
                self._append_plan(str(room["room_id"]), publication)
                changed = True
        return changed

    def _append_room_status(
        self,
        room: Mapping[str, Any],
        decision: discussion.DiscussionDecision,
    ) -> None:
        if decision.discussion_event_id is None:
            return
        hosted_rooms.append_event(
            self.db_path,
            room_id=str(room["room_id"]),
            event_id=f"dactivity:{decision.discussion_event_id}:{decision.reason}",
            kind="room.activity",
            actor={"kind": "gateway", "id": str(room["authority_gateway_id"])},
            payload={
                "status": decision.status,
                "reason_code": decision.reason,
                "thread_id": decision.thread_id,
                "discussion_event_id": decision.discussion_event_id,
            },
            authority_gateway_id=str(room["authority_gateway_id"]),
            authority_epoch=int(room["authority_epoch"]),
        )

    def prepare_room(self, binding: HostedRoomBinding) -> None:
        with self._policy_lock:
            room = hosted_rooms.room_state(self.db_path, room_id=binding.room_id)
            snapshot = self._policy_snapshot(room)
            events = list(snapshot.events)
            if self._publish_terminal_tasks(room):
                room = hosted_rooms.room_state(
                    self.db_path,
                    room_id=binding.room_id,
                )
                snapshot = self._policy_snapshot(room)
                events = list(snapshot.events)
            self.policy_checkpoint.compact_completed(room_id=binding.room_id)
            driver.prune_published_terminal_tasks(
                self.db_path,
                room_id=binding.room_id,
                clock=self.runtime.clock,
            )
            if any(
                driver.list_tasks(
                    self.db_path,
                    room_id=binding.room_id,
                    status=status,
                )
                for status in ("queued", "running", "stopping")
            ):
                return
            decision = discussion.plan_next_task(
                room,
                events,
                local_profiles=self.local_profiles(),
                initial_watermarks=snapshot.watermarks,
            )
            if decision.status == "task" and decision.task is not None:
                driver.admit_task(
                    self.db_path,
                    decision.task.identity,
                    payload=decision.task.payload,
                    clock=time.time,
                )
                # A stop can race the policy read from another process. Re-read
                # after admission and cancel before the runtime can execute a
                # task whose source event is now behind the room stop fence.
                fresh_room = hosted_rooms.room_state(
                    self.db_path,
                    room_id=binding.room_id,
                )
                stopped_through_seq = self._policy_snapshot(
                    fresh_room
                ).stopped_through_seq
                if (
                    decision.source_event_seq is not None
                    and decision.source_event_seq < stopped_through_seq
                ):
                    self.runtime.cancel(
                        decision.task.identity,
                        cancel_id=f"stop-fence:{stopped_through_seq}",
                    )
            elif decision.status in {"settled", "bounded"}:
                self._append_room_status(room, decision)

    def publish_terminal(
        self,
        binding: HostedRoomBinding,
        _task: Mapping[str, Any],
    ) -> None:
        self.prepare_room(binding)
        self.runtime.wakeup()

    def create_room(self, *, room_id: str, name: str, members: Any) -> dict[str, Any]:
        normalized = discussion.validate_roster(
            members,
            local_profiles=self.local_profiles(),
        )
        room = hosted_rooms.create_room(
            self.db_path,
            room_id=room_id,
            name=name,
            members=[
                {
                    "member_id": member.member_id,
                    "profile": member.profile,
                    "handle": member.handle,
                    **(
                        {"display_name": member.display_name}
                        if member.display_name
                        else {}
                    ),
                }
                for member in normalized
            ],
            authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        )
        self.runtime.wakeup()
        return room

    def send(
        self,
        *,
        room_id: str,
        event_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        normalized = discussion.validate_user_payload(payload)
        room = self._owned_room(room_id)
        event = hosted_rooms.append_event(
            self.db_path,
            room_id=room_id,
            event_id=event_id,
            kind="message.user",
            actor={"kind": "user", "id": "desktop"},
            payload=normalized,
            authority_gateway_id=str(room["authority_gateway_id"]),
            authority_epoch=int(room["authority_epoch"]),
        )
        binding = next(
            (
                candidate
                for candidate in self.bindings()
                if candidate.room_id == room_id
            ),
            None,
        )
        if binding is None:
            raise hosted_rooms.RoomNotFoundError("hosted room not found")
        self.prepare_room(binding)
        self.runtime.wakeup()
        return event

    def stop_room(
        self,
        room_id: str,
        *,
        cancel_id: str,
        require_acknowledged: bool = False,
    ) -> int:
        room = self._owned_room(room_id)
        hosted_rooms.request_room_stop(
            self.db_path,
            room_id=room_id,
            cancel_id=cancel_id,
            expected_gateway_id=str(room["authority_gateway_id"]),
            expected_epoch=int(room["authority_epoch"]),
        )
        cancelled = 0
        pending = 0
        with self._policy_lock:
            tasks = {}
            for status in (
                "queued",
                "running",
                "indeterminate",
                "deferred",
                "stopping",
            ):
                for task in driver.list_tasks(
                    self.db_path,
                    room_id=room_id,
                    status=status,
                ):
                    identity = task["identity"]
                    tasks[(identity.room_id, identity.task_id)] = task
            for task in tasks.values():
                task_cancel_id = (
                    str(task.get("cancel_id") or "")
                    if task.get("status") == "stopping"
                    else ""
                )
                result = self.runtime.cancel(
                    task["identity"],
                    cancel_id=task_cancel_id or cancel_id,
                )
                cancelled += 1
                if result["status"] == "stopping":
                    pending += 1
        if require_acknowledged and pending:
            raise RuntimeError(
                "room work is still stopping; retry deletion after Stop completes"
            )
        self.runtime.wakeup()
        return cancelled

    def retry_room_task(self, room_id: str, *, task_id: str) -> dict[str, Any]:
        """Retry one uncertain or deferred task only after explicit user action."""

        task = next(
            (
                candidate
                for status in ("indeterminate", "deferred")
                for candidate in driver.list_tasks(
                    self.db_path, room_id=room_id, status=status
                )
                if candidate["identity"].task_id == task_id
            ),
            None,
        )
        if task is None:
            raise driver.InvalidTaskTransitionError(
                "no retryable room task matches task_id"
            )
        return self.runtime.retry_indeterminate(task["identity"])

    def approve_room_task(
        self,
        room_id: str,
        *,
        member_id: str,
        task_id: str,
        execution_generation: int,
        choice: str,
        request_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Resolve one exact local approval and wake room observation."""

        key = (room_id, member_id)
        with self._policy_lock:
            action = self._pending_actions.get(key)
        requested_approval_id = str(request_id or "")
        pending_approval_id = str((action or {}).get("request_id") or "")
        if (
            action is None
            or action.get("task_id") != task_id
            or int(action.get("execution_generation") or 0) != execution_generation
            or not requested_approval_id
            or requested_approval_id != pending_approval_id
        ):
            raise RuntimeError("room approval is no longer pending")
        if choice not in {"once", "deny"}:
            raise RuntimeError("room approval choice must be once or deny")
        session_id = str(action.get("session_id") or "")
        if not session_id:
            raise RuntimeError("local room approval identity is unavailable")
        result = self.rpc.approve(
            session_id=session_id,
            request_id=requested_approval_id,
            choice=choice,
        )
        if result is None:
            raise RuntimeError("room approval target is unavailable")
        with self._policy_lock:
            current = self._pending_actions.get(key)
            if (
                current is not None
                and str(current.get("request_id") or "") == requested_approval_id
                and current.get("task_id") == task_id
                and int(current.get("execution_generation") or 0)
                == execution_generation
            ):
                self._pending_actions.pop(key, None)
        self.runtime.wakeup()
        return result

    def status(self, room_id: str | None = None) -> dict[str, Any]:
        runtime = self.runtime.status()
        if room_id is None:
            return runtime
        tasks = driver.list_tasks(self.db_path, room_id=room_id)
        counts = Counter(str(task["status"]) for task in tasks)
        pending_actions = [
            {
                "kind": "retry",
                "task_id": task["identity"].task_id,
            }
            for task in tasks
            if task["status"] in {"indeterminate", "deferred"}
        ]
        with self._policy_lock:
            pending_actions.extend(
                dict(action)
                for (
                    action_room_id,
                    _member_id,
                ), action in self._pending_actions.items()
                if action_room_id == room_id
            )
        return {
            "running": runtime["running"],
            "working": bool(
                counts.get("running") or counts.get("queued") or counts.get("stopping")
            ),
            "blocked": room_id in runtime["blocked_rooms"]
            or bool(counts.get("indeterminate") or counts.get("stopping")),
            "counts": dict(counts),
            "pending_actions": pending_actions,
        }
