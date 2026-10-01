"""Run Matrix requests on their owning loop and bound unavailable-loop waits."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine

from gateway.session_context import get_session_env, get_session_routing_identity, get_session_transport
from gateway.session_identity import RoutingIdentity
from hermes_constants import get_hermes_home
from tools.interrupt import acting_for_tid, is_thread_interrupted

_monotonic = time.monotonic

_SESSION_KEYS = (
    "HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_USER_ID",
    "HERMES_SESSION_KEY", "HERMES_SESSION_PROFILE",
)


@dataclass(frozen=True)
class MatrixOwner:
    """The Matrix adapter, client and session that a tool call was issued for."""

    adapter: Any
    client: Any
    crypto: Any
    bot: str | None
    profile: str | None
    home: Path
    session: tuple[str, ...]
    identity: RoutingIdentity | None

    @classmethod
    def capture(cls) -> MatrixOwner:
        adapter, _ = get_session_transport()
        client = getattr(adapter, "_client", None)
        return cls(
            adapter, client, getattr(client, "crypto", None), getattr(adapter, "_user_id", None),
            getattr(adapter, "_owner_profile", None), get_hermes_home(), _current_session(),
            get_session_routing_identity(),
        )

    def matches(self) -> bool:
        adapter, loop = get_session_transport()
        client = getattr(adapter, "_client", None)
        return (adapter is self.adapter and loop is asyncio.get_running_loop()
                and client is self.client and getattr(client, "crypto", None) is self.crypto
                and getattr(adapter, "_user_id", None) == self.bot
                and getattr(adapter, "_owner_profile", None) == self.profile
                and get_session_routing_identity() is self.identity
                and (self.identity is None or self.identity.adapter() is adapter)
                and get_hermes_home() == self.home and _current_session() == self.session)


def _current_session() -> tuple[str, ...]:
    return tuple(get_session_env(key) or "" for key in _SESSION_KEYS)


async def run_matrix_mutation(
    owner_loop: asyncio.AbstractEventLoop,
    operation_factory: Callable[[Callable[[], bool], Callable[[], None]], Coroutine[Any, Any, dict[str, Any]]],
    *, operation_label: str, next_step: str,
) -> str:
    worker_id = threading.get_ident()
    parent_id = acting_for_tid.get()
    cancelled = threading.Event()
    write_started = threading.Event()

    def interrupted() -> bool:
        return cancelled.is_set() or is_thread_interrupted(worker_id) or is_thread_interrupted(parent_id)

    if interrupted():
        return json.dumps({"error": f"{operation_label} interrupted"})

    if not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})

    owner = MatrixOwner.capture()
    change = operation_factory(interrupted, write_started.set)
    owner_task: asyncio.Task[dict[str, Any]] | None = None

    async def run_change() -> dict[str, Any]:
        nonlocal owner_task
        owner_task = asyncio.current_task()
        if cancelled.is_set():
            change.close()
            return {"error": f"{operation_label} interrupted"}
        secrets = None
        if owner.identity is not None:
            from gateway.run import _load_profile_secret_scope

            try:
                secrets = await asyncio.to_thread(_load_profile_secret_scope, owner.identity.authorization_home)
            except BaseException:
                change.close()
                raise
        # Load the secrets before the ownership check. An await between the check and
        # entering the scope would let the client or session change unnoticed.
        if cancelled.is_set():
            change.close()
            return {"error": f"{operation_label} interrupted"}
        if not owner.matches() or owner.identity is not None and owner.identity.runtime_home != owner.home:
            change.close()
            return {"error": "Matrix session or client ownership changed"}
        if owner.identity is None:
            return await change
        from gateway.run import _profile_runtime_scope

        with _profile_runtime_scope(owner.identity.authorization_home, secrets):
            return await change

    def cancel_change() -> None:
        if owner_task is not None:
            owner_task.cancel()

    operation = run_change()
    if owner_loop is asyncio.get_running_loop():
        pending = asyncio.create_task(operation)
    else:
        try:
            future = asyncio.run_coroutine_threadsafe(operation, owner_loop)
        except RuntimeError:
            operation.close()
            change.close()
            return json.dumps({"error": "Matrix gateway loop is unavailable"})
        pending = asyncio.wrap_future(future)

    deadline = _monotonic() + 30.0
    failure: str | None = None
    result: dict[str, Any] | None = None
    try:
        while not pending.done():
            if not owner_loop.is_running():
                if (
                    owner_task is not None
                    and owner_task.done()
                    and not owner_task.cancelled()
                ):
                    result = owner_task.result()
                    break
                failure = "Matrix gateway loop is unavailable"
                break
            if interrupted():
                failure = f"{operation_label} interrupted"
                break
            remaining = deadline - _monotonic()
            if remaining <= 0:
                failure = f"{operation_label} timed out"
                break
            await asyncio.wait({pending}, timeout=min(0.1, remaining))
        if failure is None and result is None:
            result = await pending
    except asyncio.CancelledError:
        cancelled.set()
        raise
    finally:
        if not pending.done():
            cancelled.set()
            try:
                owner_loop.call_soon_threadsafe(cancel_change)
            except RuntimeError:
                if owner_task is None:
                    operation.close()
                    change.close()
            # Cancelling the result future would finish before the owning task stops.
            completion = asyncio.gather(pending, return_exceptions=True)
            cancellation: asyncio.CancelledError | None = None
            while not completion.done():
                if not owner_loop.is_running():
                    pending.cancel()
                    break
                try:
                    await asyncio.wait({completion}, timeout=0.1)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            if cancellation is not None:
                raise cancellation
    if failure is None:
        return json.dumps(result, ensure_ascii=False)
    if not write_started.is_set():
        return json.dumps({"error": failure})
    return json.dumps({
        "error": f"{failure} after the change was sent to the homeserver",
        "outcome": "unknown",
        "next_step": next_step,
    })
