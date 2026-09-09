"""Peer-backed session transport for one hosted-room member task: implements
:class:`InternalSessionRPC` without canonical Bot Chat. The remote client must resolve a hidden
``Group: <room_id>`` session with ``source=bot_room`` and verify the scoped grant at admission."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_peer import HostedMemberDispatch, PROTOCOL_VERSION, attachment_manifest_digest
from tui_gateway.hosted_room_driver import (
    ROOM_SESSION_SOURCE, HostedRoomBinding, InternalSessionRPC, room_session_title)


class HostedRoomPeerClient(Protocol):
    """Authenticated client for a target gateway's narrow room-member API."""

    def bind_room_scope(self, **scope: Any) -> None: ...
    def prepare(self, *, room_id: str, profile: str, source: str, grant: str, create: bool,
                expected_session_id: str | None = None) -> Mapping[str, Any] | None: ...
    def dispatch(self, *, dispatch: Mapping[str, Any], grant: str) -> Mapping[str, Any]: ...
    def history(self, *, room_id: str, profile: str, session_id: str, grant: str
                ) -> Sequence[Mapping[str, Any]]: ...
    def status(self, *, room_id: str, profile: str, session_id: str, grant: str
               ) -> Mapping[str, Any]: ...
    def stop(self, *, dispatch: Mapping[str, Any], grant: str) -> Mapping[str, Any] | None: ...
    def stop_receipt(self, *, task_id: str, execution_generation: int, grant: str
                     ) -> Mapping[str, Any] | None: ...


    def discard_attachments(
        self,
        *,
        task_id: str,
        execution_generation: int,
        grant: str,
    ) -> Mapping[str, Any]: ...


    def stage_attachments(
        self,
        *,
        dispatch: Mapping[str, Any],
        attachments: Sequence[Mapping[str, Any]],
        grant: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RoomLinkCandidate:
    """One address/provider for the same authenticated target gateway."""

    name: str
    mode: str
    target_install_id: str
    client: HostedRoomPeerClient


class FailoverHostedRoomPeerClient:
    """Try alternate links without changing target or logical task identity."""

    def __init__(
        self, candidates: Sequence[RoomLinkCandidate], *, reprobe_interval_seconds: float = 60,
        clock: Callable[[], float] = time.monotonic) -> None:
        if not candidates:
            raise ValueError("at least one RoomLink candidate is required")
        if len({candidate.target_install_id for candidate in candidates}) != 1:
            raise ValueError("RoomLink candidates must target one installation")
        if reprobe_interval_seconds <= 0:
            raise ValueError("reprobe_interval_seconds must be positive")
        self.candidates, self._active, self.clock = tuple(candidates), 0, clock
        self.reprobe_interval_seconds = float(reprobe_interval_seconds)
        self._last_primary_probe = 0.0

    @property
    def active_link(self) -> RoomLinkCandidate:
        return self.candidates[self._active]

    def _call(self, method: str, **kwargs):
        """Try the active link (re-probing the primary after a cooldown), then the rest. Ambiguous
        or non-retryable failures propagate: failing over after an ambiguous dispatch could run
        the same task twice."""
        now = self.clock()
        order = [self._active]
        if self._active != 0 and now - self._last_primary_probe >= self.reprobe_interval_seconds:
            self._last_primary_probe = now
            order = [0, self._active]
        order.extend(index for index in range(len(self.candidates)) if index not in order)
        last_error = None
        for index in order:
            try:
                result = getattr(self.candidates[index].client, method)(**kwargs)
            except Exception as exc:
                if (method != "stage_attachments" and getattr(exc, "ambiguous", False)) or not getattr(exc, "retryable", False):
                    raise
                last_error = exc
                continue
            self._active = index
            return result
        raise last_error if last_error is not None else RuntimeError("no RoomLink candidate was attempted")

    def _delegate(method: str):
        def call(self, **kwargs):
            return self._call(method, **kwargs)
        call.__name__ = method
        return call

    prepare, dispatch, history, status, stop = map(_delegate, ("prepare", "dispatch", "history", "status", "stop"))
    del _delegate

    def bind_room_scope(self, **kwargs):
        for candidate in self.candidates:
            if callable(bind := getattr(candidate.client, "bind_room_scope", None)):
                bind(**kwargs)


    def discard_attachments(self, **kwargs):
        return self._call("discard_attachments", **kwargs)


    def stage_attachments(self, **kwargs):
        return self._call("stage_attachments", **kwargs)


@dataclass(frozen=True)
class PeerMemberRoute:
    """Secret-free target coordinates plus a separately stored room grant."""

    home_install_id: str
    member_id: str
    target_install_id: str
    target_profile: str
    capability_digest: str
    cancellation_scope_id: str
    trace_id: str
    grant: str
    execution_policy_digest: str = ""
    attachments: bool = False


def build_member_dispatch(
    *, binding: HostedRoomBinding, route: PeerMemberRoute, room_id: str, task_id: str,
    target_profile: str, execution_generation: int, source_event_seq: int, prompt: str,
    trace_id: str, attachment_digest: str | None = None) -> HostedMemberDispatch:
    """Build the fully fenced member dispatch shared by submit and recovery."""
    return HostedMemberDispatch.from_mapping({
        "protocol_version": PROTOCOL_VERSION, "room_id": room_id,
        "home_install_id": route.home_install_id, "authority_gateway_id": binding.gateway_id,
        "authority_epoch": binding.authority_epoch, "member_id": route.member_id,
        "target_install_id": route.target_install_id, "target_profile": target_profile,
        "task_id": task_id, "execution_generation": execution_generation,
        "source_event_seq": source_event_seq, "cancellation_scope_id": route.cancellation_scope_id,
        "prompt": prompt, "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "capability_digest": route.capability_digest,
        "execution_policy_digest": route.execution_policy_digest, "trace_id": trace_id,
        **({"attachment_manifest_digest": attachment_digest} if attachment_digest is not None else {})})


class PeerHostedRoomTransport(InternalSessionRPC):
    """Translate runtime session operations into recipient-validated peer RPC."""

    def __init__(
        self,
        *,
        binding: HostedRoomBinding,
        route: PeerMemberRoute,
        client: HostedRoomPeerClient,
        source_event_seq: int = 1,
        task_id: str | None = None,
        execution_generation: int | None = None,
    ) -> None:
        self.binding = binding
        self.route = route
        self.client = client
        if isinstance(source_event_seq, bool) or source_event_seq < 1:
            raise ValueError("peer room source_event_seq must be positive")
        self.source_event_seq = int(source_event_seq)
        self.task_id = task_id
        self.execution_generation = execution_generation
        self._session_id: str | None = None
        self._dispatch: HostedMemberDispatch | None = None
        if callable(bind_scope := getattr(self.client, "bind_room_scope", None)):
            bind_scope(
                room_id=binding.room_id,
                home_install_id=route.home_install_id,
                authority_gateway_id=binding.gateway_id,
                authority_epoch=binding.authority_epoch,
                member_id=route.member_id,
                target_install_id=route.target_install_id,
                target_profile=route.target_profile,
            )
        self._attachment_attempt: tuple[str, int] | None = None
        self._pending_attachments: list[dict[str, Any]] = []

    def _validate_coordinates(self, *, profile: str, source: str, title: str | None = None) -> None:
        if source != ROOM_SESSION_SOURCE:
            raise ValueError("peer room transport requires source=bot_room")
        if profile != self.route.target_profile:
            raise ValueError("peer room transport profile does not match its grant")
        if title is not None and title != room_session_title(self.binding.room_id):
            raise ValueError("peer room transport title does not match room identity")

    def _scoped(self, **extra: Any) -> dict[str, Any]:
        """Room id + grant keyword arguments shared by every scoped client call."""
        return {"room_id": self.binding.room_id, "grant": self.route.grant, **extra}

    def _prepare(self, *, profile: str, source: str, create: bool, title: str | None = None, **extra):
        """Validate coordinates, then the scoped ``prepare`` call."""
        self._validate_coordinates(profile=profile, source=source, title=title)
        return self.client.prepare(**self._scoped(profile=profile, source=source, create=create, **extra))

    def resolve_exact(self, *, profile: str, title: str, source: str) -> Mapping[str, Any] | None:
        return self._prepare(profile=profile, source=source, create=False, title=title)

    def create(self, *, profile: str, title: str, source: str) -> Mapping[str, Any]:
        session = self._prepare(profile=profile, source=source, create=True, title=title)
        if session is None:
            raise RuntimeError("peer did not create the room session")
        self._session_id = str(session.get("session_id") or session.get("id") or "")
        return session

    def resume(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        session = self._prepare(profile=profile, source=source, create=False, expected_session_id=session_id)
        if session is None:
            raise RuntimeError("peer room session is unavailable")
        self._session_id = session_id
        return session

    def submit(
        self, *, profile: str, session_id: str, prompt: str, source: str, task: TaskIdentity,
        execution_generation: int, on_terminal: Callable[[Mapping[str, Any]], None],
    ) -> Mapping[str, Any]:
        self._validate_coordinates(profile=profile, source=source)
        if self._session_id not in {None, session_id}:
            raise ValueError("peer room session changed during admission")
        pending = list(self._pending_attachments)
        manifest = [{key: item[key] for key in ("attachment_id", "kind", "name", "size", "mime", "sha256")}
                    for item in pending]
        dispatch = build_member_dispatch(
            binding=self.binding, route=self.route, room_id=task.room_id, task_id=task.task_id,
            target_profile=profile, execution_generation=execution_generation,
            source_event_seq=self.source_event_seq, prompt=prompt,
            trace_id=self.route.trace_id or f"trace-{uuid.uuid4().hex}",
            attachment_digest=attachment_manifest_digest(manifest) if manifest else None)
        self._dispatch = dispatch
        self._session_id = session_id
        if pending:
            try:
                self.client.stage_attachments(dispatch=dispatch.as_mapping(), attachments=pending, grant=self.route.grant)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 413 and not getattr(exc, "retryable", False):
                    # A definitive byte rejection precedes model admission: settle once.
                    self._discard_terminal_attachments()
                    receipt = {
                        "status": "failed",
                        "settlement_id": f"attachment-rejected:{dispatch.task_id}:{dispatch.execution_generation}",
                        "error": "A Group Chat file exceeded the peer gateway's upload limit."}
                    on_terminal(receipt)
                    return receipt
                # A lost idempotent upload response cannot mean a model run was admitted.
                self._discard_terminal_attachments()
                try:
                    exc.not_admitted = True
                    exc.ambiguous = False
                except Exception:
                    pass
                raise
        try:
            result = self.client.dispatch(dispatch=dispatch.as_mapping(), grant=self.route.grant)
        except Exception as exc:
            if getattr(exc, "not_admitted", False):
                self._discard_terminal_attachments()
            raise
        if result.get("status") in {"settled", "failed", "cancelled"}:
            on_terminal(result)
            self._discard_terminal_attachments()
        return result

    def history(self, *, profile: str, session_id: str, source: str) -> Sequence[Mapping[str, Any]]:
        self._validate_coordinates(profile=profile, source=source)
        history = self.client.history(**self._scoped(profile=profile, session_id=session_id))
        if any(str(item.get("task_id") or "") == str(self.task_id or "")
               and item.get("status") in {"cancelled", "failed", "settled"}
               for item in history if isinstance(item, Mapping)):
            self._discard_terminal_attachments()
        return history

    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        self._validate_coordinates(profile=profile, source=source)
        return self.client.status(**self._scoped(profile=profile, session_id=session_id))

    def interrupt(
        self, *, profile: str, session_id: str, source: str, expected_task_id: str
    ) -> Mapping[str, Any] | None:
        self._validate_coordinates(profile=profile, source=source)
        dispatch = self._dispatch
        if dispatch is not None:
            if dispatch.task_id != expected_task_id:
                return None
            result = self.client.stop(dispatch=dispatch.as_mapping(), grant=self.route.grant)
            if isinstance(result, Mapping) and result.get("status") in {"cancelled", "failed", "settled"}:
                self._discard_terminal_attachments()
            return result
        if (self.task_id != expected_task_id or not self.execution_generation
                or not hasattr(self.client, "stop_receipt")):
            return None
        return self.client.stop_receipt(
            task_id=expected_task_id, execution_generation=self.execution_generation,
            grant=self.route.grant)


    def rollback_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        """Drop local bytes; target-side partial batches expire without admission."""

        self.commit_attachment_staging(
            profile=profile,
            session_id=session_id,
            source=source,
            execution_generation=execution_generation,
        )


    def _discard_terminal_attachments(self) -> None:
        dispatch = self._dispatch
        if dispatch is None or dispatch.attachment_manifest_digest is None:
            return
        discard = getattr(self.client, "discard_attachments", None)
        if not callable(discard):
            return
        try:
            discard(
                task_id=dispatch.task_id,
                execution_generation=dispatch.execution_generation,
                grant=self.route.grant,
            )
        except Exception:
            # Terminal observation retries this cleanup; target TTL and quotas
            # remain the crash backstop.
            return


    def commit_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        """Forget local bytes once target run admission becomes authoritative."""

        self._validate_coordinates(profile=profile, source=source)
        if self._attachment_attempt == (str(self.task_id or ""), int(execution_generation)):
            self._attachment_attempt = None
            self._pending_attachments = []


    def stage_attachment(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        attachment: Mapping[str, Any],
        data: bytes,
        execution_generation: int,
    ) -> Mapping[str, Any]:
        """Buffer verified home-owned bytes for one pre-admission peer push."""

        self._validate_coordinates(profile=profile, source=source)
        attempt = (str(self.task_id or ""), int(execution_generation))
        if self._session_id not in {None, session_id} or self._attachment_attempt != attempt:
            raise ValueError("peer attachment staging is outside its fenced attempt")
        payload = bytes(data)
        if int(attachment.get("size") or -1) != len(payload):
            raise ValueError("peer attachment bytes no longer match their manifest")
        manifest = {
            "attachment_id": str(attachment.get("attachment_id") or ""),
            "kind": str(attachment.get("kind") or ""),
            "name": str(attachment.get("name") or ""),
            "size": len(payload),
            "mime": str(attachment.get("mime") or ""),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "data": payload,
        }
        self._pending_attachments.append(manifest)
        return {"attached": True}


    def begin_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        """Start one peer-upload batch without admitting the target run."""

        self._validate_coordinates(profile=profile, source=source)
        if self._session_id not in {None, session_id}:
            raise ValueError("peer room session changed during attachment staging")
        if not self.task_id or execution_generation < 1:
            raise ValueError("peer attachment attempt identity is unavailable")
        attempt = (self.task_id, int(execution_generation))
        if self._attachment_attempt not in {None, attempt}:
            raise ValueError("peer attachment attempt changed during staging")
        self._attachment_attempt = attempt
        self._pending_attachments = []
