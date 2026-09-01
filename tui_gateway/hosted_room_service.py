"""Production coordinator for same-gateway hosted Discussion rooms."""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
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
        self.rpc = HostedRoomServerRPC(
            server,
            profile_available=self._profile_available,
        )
        self.runtime = HostedRoomRuntime(
            db_path=self.db_path,
            rooms=self.bindings,
            rpc=self.rpc,
            turn_lock=self._turn_lock,
            profile_available=self._profile_available,
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

    def _profile_available(self, profile: str) -> bool:
        return profile in self.local_profiles()

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
        if action is None:
            driver.clear_member_approval_requests(
                self.db_path,
                room_id=room_id,
                member_id=member_id,
            )
            with self._policy_lock:
                self._pending_actions.pop(key, None)
            return
        durable_action = {**action, "member_id": member_id}
        identity = driver.TaskIdentity(
            room_id=room_id,
            task_id=str(action.get("task_id") or ""),
            thread_id=str(action.get("thread_id") or ""),
            turn_id=str(action.get("turn_id") or ""),
        )
        request = driver.publish_approval_request(
            self.db_path,
            identity,
            execution_generation=int(action.get("execution_generation") or 0),
            member_id=member_id,
            request_id=str(action.get("request_id") or ""),
            session_id=str(action.get("session_id") or ""),
            action=durable_action,
            clock=time.time,
        )
        with self._policy_lock:
            self._pending_actions[key] = durable_action
        choice = request.get("choice")
        if choice not in {"once", "deny"}:
            return
        result = self.rpc.approve(
            session_id=str(request["session_id"]),
            request_id=str(request["request_id"]),
            choice=str(choice),
        )
        if not isinstance(result, Mapping) or not bool(result.get("resolved")):
            return
        if not driver.mark_approval_consumed(
            self.db_path,
            identity,
            execution_generation=int(request["execution_generation"]),
            member_id=member_id,
            request_id=str(request["request_id"]),
            choice=str(choice),
            clock=time.time,
        ):
            raise RuntimeError("room approval decision changed before acknowledgement")
        with self._policy_lock:
            self._pending_actions.pop(key, None)
        self.runtime.wakeup()

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

    def _append_plan(
        self,
        room_id: str,
        plan: discussion.PublicationPlan,
        *,
        expected_latest_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        return hosted_rooms.append_events(
            self.db_path,
            events=[event.append_kwargs(room_id) for event in plan.events],
            allow_terminal_recovery=True,
            expected_latest_seq=expected_latest_seq,
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
        expected_latest_seq = int(room["latest_seq"])
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
                appended = self._append_plan(
                    str(room["room_id"]),
                    publication,
                    expected_latest_seq=expected_latest_seq,
                )
                expected_latest_seq = max(
                    expected_latest_seq,
                    *(int(event["seq"]) for event in appended),
                )
                changed = True
        return changed

    def _append_room_status(
        self,
        room: Mapping[str, Any],
        decision: discussion.DiscussionDecision,
    ) -> None:
        if decision.discussion_event_id is None:
            return
        hosted_rooms.append_events(
            self.db_path,
            events=[
                {
                    "room_id": str(room["room_id"]),
                    "event_id": (
                        f"dactivity:{decision.discussion_event_id}:{decision.reason}"
                    ),
                    "kind": "room.activity",
                    "actor": {
                        "kind": "gateway",
                        "id": str(room["authority_gateway_id"]),
                    },
                    "payload": {
                        "status": decision.status,
                        "reason_code": decision.reason,
                        "thread_id": decision.thread_id,
                        "discussion_event_id": decision.discussion_event_id,
                    },
                    "authority_gateway_id": str(room["authority_gateway_id"]),
                    "authority_epoch": int(room["authority_epoch"]),
                }
            ],
            expected_latest_seq=int(room["latest_seq"]),
        )

    def _finish_room_demotion(self, intent: Mapping[str, Any]) -> dict[str, Any]:
        from gateway.hosted_room_replicas import demote_room

        room_id = str(intent["room_id"])
        gateway_id = str(intent["gateway_id"])
        authority_epoch = int(intent["authority_epoch"])
        stopped_locally = True
        try:
            self.stop_room(
                room_id,
                cancel_id=str(intent["cancel_id"]),
                require_acknowledged=True,
            )
        except hosted_rooms.AuthorityConflictError:
            # Another process may have completed the same durable intent after
            # this runtime read it. The replica primitive verifies that the
            # current lineage exactly matches the intended target.
            stopped_locally = False
        if stopped_locally:
            room = hosted_rooms.room_state(
                self.db_path,
                room_id=room_id,
            )
            if (
                str(room["authority_gateway_id"]) == gateway_id
                and int(room["authority_epoch"]) == authority_epoch
            ):
                try:
                    published = self._publish_terminal_tasks(room)
                except hosted_rooms.AuthorityConflictError:
                    # A competing process crossed the authority CAS after our
                    # old-lineage snapshot. The replica primitive below still
                    # verifies that it reached this exact intended target.
                    pass
                else:
                    if published:
                        refreshed = hosted_rooms.room_state(
                            self.db_path,
                            room_id=room_id,
                        )
                        self._policy_snapshot(refreshed)
        return demote_room(
            self.db_path,
            room_id=room_id,
            observed_gateway_id=str(intent["observed_gateway_id"]),
            observed_epoch=int(intent["observed_epoch"]),
        )

    def prepare_room(self, binding: HostedRoomBinding) -> None:
        with self._policy_lock:
            pending_demotion = driver.pending_room_demotion(
                self.db_path,
                room_id=binding.room_id,
            )
            if pending_demotion is not None:
                self._finish_room_demotion(pending_demotion)
                return
            room = hosted_rooms.room_state(self.db_path, room_id=binding.room_id)
            driver.reconcile_stop_fenced_inactive_tasks(
                self.db_path,
                room_id=binding.room_id,
                clock=self.runtime.clock,
            )
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
            driver.prune_closed_published_deferred_tasks(
                self.db_path,
                room_id=binding.room_id,
            )
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
            local_profiles = self.local_profiles()
            decision = discussion.plan_next_task(
                room,
                events,
                local_profiles=local_profiles,
                initial_watermarks=snapshot.watermarks,
            )
            while decision.status == "task" and decision.task is not None:
                local_profiles = self.local_profiles()
                unavailable = discussion.plan_unavailable_member_deferral(
                    room,
                    events,
                    decision.task,
                    local_profiles=local_profiles,
                )
                if unavailable is None:
                    break
                self._append_plan(
                    binding.room_id,
                    unavailable,
                    expected_latest_seq=int(room["latest_seq"]),
                )
                room = hosted_rooms.room_state(
                    self.db_path,
                    room_id=binding.room_id,
                )
                snapshot = self._policy_snapshot(room)
                events = list(snapshot.events)
                local_profiles = self.local_profiles()
                decision = discussion.plan_next_task(
                    room,
                    events,
                    local_profiles=local_profiles,
                    initial_watermarks=snapshot.watermarks,
                )
            if decision.status == "task" and decision.task is not None:
                current_profiles = self.local_profiles()
                if decision.task.member.profile not in current_profiles:
                    # A frozen roster member can disappear after policy replay
                    # but before durable admission. Publish the same retryable
                    # terminal result used by runtime recovery without ever
                    # resolving the missing profile through the launch DB.
                    publication = discussion.plan_publication(
                        room,
                        events,
                        decision.task,
                        status="deferred",
                        result={"reason": "member_unavailable", "retryable": True},
                        execution_generation=1,
                        local_profiles=current_profiles,
                    )
                    self._append_plan(
                        binding.room_id,
                        publication,
                        expected_latest_seq=int(room["latest_seq"]),
                    )
                    return
                try:
                    driver.admit_task(
                        self.db_path,
                        decision.task.identity,
                        payload=decision.task.payload,
                        clock=time.time,
                    )
                except driver.TaskAdmissionBlockedError:
                    # The user event remains durable, but this runtime lost the
                    # atomic Stop or authority-demotion admission race.
                    return
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
            require_open_admissions=True,
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
        try:
            self.prepare_room(binding)
        except hosted_rooms.RoomConflictError:
            # The user event is already durable. A concurrent append only
            # invalidates this policy snapshot, so reschedule from fresh state
            # instead of reporting that the accepted send failed.
            pass
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
        stop_event = hosted_rooms.request_room_stop(
            self.db_path,
            room_id=room_id,
            cancel_id=cancel_id,
            expected_gateway_id=str(room["authority_gateway_id"]),
            expected_epoch=int(room["authority_epoch"]),
        )
        stop_seq = int(stop_event["seq"])
        cancelled = 0
        pending = 0
        with self._policy_lock:
            driver.prune_closed_published_deferred_tasks(
                self.db_path,
                room_id=room_id,
            )
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
                    if int(task["payload"]["source_event_seq"]) > stop_seq:
                        continue
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

    def demote_room(
        self,
        room_id: Any,
        *,
        observed_gateway_id: Any,
        observed_epoch: Any,
    ) -> dict[str, Any]:
        """Stop accepted local work before committing a newer authority."""

        from gateway.hosted_room_replicas import (
            demote_room,
            validate_demotion_observation,
        )

        with self._policy_lock:
            room_id, observed_gateway_id, observed_epoch = (
                validate_demotion_observation(
                    room_id=room_id,
                    observed_gateway_id=observed_gateway_id,
                    observed_epoch=observed_epoch,
                )
            )
            room = hosted_rooms.room_state(self.db_path, room_id=room_id)
            current_gateway = str(room["authority_gateway_id"])
            current_epoch = int(room["authority_epoch"])
            local_gateway = hosted_rooms.local_authority_gateway_id()

            # Preserve the primitive's idempotent/error semantics without
            # stopping work for a rejected or already-applied observation.
            if current_gateway != local_gateway or observed_epoch <= current_epoch:
                return demote_room(
                    self.db_path,
                    room_id=room_id,
                    observed_gateway_id=observed_gateway_id,
                    observed_epoch=observed_epoch,
                )

            pending_demotion = driver.pending_room_demotion(
                self.db_path,
                room_id=room_id,
            )
            if pending_demotion is not None:
                if (
                    pending_demotion["observed_gateway_id"] != observed_gateway_id
                    or int(pending_demotion["observed_epoch"]) != observed_epoch
                ):
                    raise driver.TaskConflictError(
                        "room already has a different pending demotion intent"
                    )
                return self._finish_room_demotion(pending_demotion)

            intent = driver.begin_room_demotion(
                self.db_path,
                room_id=room_id,
                expected_gateway_id=current_gateway,
                expected_epoch=current_epoch,
                observed_gateway_id=observed_gateway_id,
                observed_epoch=observed_epoch,
                cancel_id=f"authority-demote:{observed_epoch}:{uuid.uuid4().hex}",
                clock=time.time,
            )
            return self._finish_room_demotion(intent)

    def retry_room_task(self, room_id: str, *, task_id: str) -> dict[str, Any]:
        """Retry one uncertain or deferred task only after explicit user action."""

        with self._policy_lock:
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
            source_event_seq = None
            if task["status"] == "deferred":
                room = hosted_rooms.room_state(self.db_path, room_id=room_id)
                self.policy_checkpoint.sync(
                    room_id=room_id,
                    latest_seq=int(room["latest_seq"]),
                )
                source_event_seq = int(task["payload"]["source_event_seq"])
                if not self.policy_checkpoint.events_for_task(
                    room_id=room_id,
                    source_event_seq=source_event_seq,
                ):
                    raise driver.InvalidTaskTransitionError(
                        "cannot retry deferred task because its source discussion "
                        "is no longer active"
                    )
            return self.runtime.retry_indeterminate(
                task["identity"],
                require_active_source_event_seq=source_event_seq,
            )

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

        requested_approval_id = str(request_id or "")
        pending = next(
            (
                request
                for request in driver.list_pending_approval_requests(
                    self.db_path,
                    room_id=room_id,
                )
                if request["identity"].task_id == task_id
                and request["execution_generation"] == execution_generation
                and request["member_id"] == member_id
                and request["request_id"] == requested_approval_id
            ),
            None,
        )
        if pending is None:
            raise RuntimeError("room approval is no longer pending")
        result = driver.decide_approval_request(
            self.db_path,
            pending["identity"],
            execution_generation=execution_generation,
            member_id=member_id,
            request_id=requested_approval_id,
            choice=choice,
            clock=time.time,
        )
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
        pending_actions.extend(
            {
                **request["action"],
                **(
                    {"decision": request["choice"]}
                    if request.get("choice") is not None
                    else {}
                ),
            }
            for request in driver.list_pending_approval_requests(
                self.db_path,
                room_id=room_id,
            )
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
