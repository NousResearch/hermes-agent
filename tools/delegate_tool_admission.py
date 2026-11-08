"""Admission adapters shared by model delegation, internal callers and lifecycle API."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
from concurrent.futures import Future

from tools.host_policy import HostPolicyError
from tools.worker_admission import AdmissionError, Reservation, reserve_workers

logger = logging.getLogger(__name__)
_BATCH: contextvars.ContextVar[dict | None] = contextvars.ContextVar("delegated_worker_reservations", default=None)


def _reserve(count, parent):
    from hermes_constants import get_hermes_home
    db_path = getattr(getattr(parent, "_session_db", None), "db_path", None)
    from pathlib import Path
    profile = str(Path(db_path).parent) if isinstance(db_path, (str, Path)) else str(get_hermes_home())
    session = getattr(parent, "session_id", "")
    return reserve_workers(count, profile=profile, session=session if isinstance(session, str) else "")


def _reservation(child):
    value = getattr(child, "_worker_reservation", None)
    return value if isinstance(value, Reservation) else None


def cancel_unstarted_child(child) -> None:
    reservation = _reservation(child)
    if reservation is not None:
        reservation.cancel_unstarted()


def discard_unstarted_child(parent, child) -> None:
    """Cancellation wins against claim before closing; never close a duplicate live run."""
    reservation = _reservation(child)
    if reservation is not None:
        try:
            if not reservation.cancel_unstarted():
                return
        except HostPolicyError:
            # Cancellation already fenced the local claim. Failed ledger I/O
            # retains capacity for repair, but must not strand child resources.
            logger.warning("Unstarted worker capacity retained pending authority repair", exc_info=True)
    from tools.delegate_tool_child_run import _close_child, _detach_child
    _detach_child(parent, child)
    _close_child(child, "Failed to close an unstarted admitted child")


def admission_metadata(error) -> dict:
    return error.metadata() if isinstance(error, AdmissionError) else {"code": "host_policy_invalid"}


@contextlib.contextmanager
def construction_batch(count, parent):
    reservations = _reserve(count, parent)
    batch = {"available": list(reservations), "children": []} if reservations else None
    token = _BATCH.set(batch)
    try:
        yield
    except BaseException:
        if batch is not None:
            for child in batch["children"]:
                if getattr(child, "_worker_dispatch_transferred", None) is not True:
                    discard_unstarted_child(parent, child)
        raise
    finally:
        _BATCH.reset(token)
        # Transferred claims belong to child lifecycle; unused construction slots do not.
        for reservation in batch["available"] if batch is not None else ():
            reservation.cancel_unstarted()


def admitted_construction(build):
    @functools.wraps(build)
    def construct(**kwargs):
        batch = _BATCH.get()
        slots = _reserve(1, kwargs.get("parent_agent")) if batch is None else batch["available"]
        if batch is not None and not slots:
            raise AdmissionError("Batch construction exceeded its prospective worker reservations")
        reservation = slots.pop(0) if slots else None
        try:
            child = build(**kwargs)
        except BaseException:
            if reservation is not None:
                reservation.cancel_unstarted()
            raise
        if reservation is not None:
            child._worker_reservation = reservation
            if batch is not None:
                batch["children"].append(child)
        return child
    return construct


def admitted_child(run):
    @functools.wraps(run)
    def execute(task_index, goal, child=None, parent_agent=None, **kwargs):
        reservation = _reservation(child)
        try:
            if reservation is None:
                slots = _reserve(1, parent_agent)
                reservation = slots[0] if slots else None
                if reservation is not None:
                    child._worker_reservation = reservation
            if reservation is not None:
                reservation.claim()
        except HostPolicyError as exc:
            from tools.delegate_tool_child_run import _fabricated_entry
            discard_unstarted_child(parent_agent, child)
            entry = _fabricated_entry(task_index, "error", str(exc), child)
            entry["admission"] = admission_metadata(exc)
            return entry
        try:
            token = _BATCH.set(None)
            try:
                return run(task_index, goal, child, parent_agent, **kwargs)
            finally:
                _BATCH.reset(token)
        finally:
            if reservation is not None:
                futures = [
                    value for name in ("_delegate_conversation_future", "_delegate_close_future")
                    if isinstance(value := getattr(child, name, None), Future)
                ]
                reservation.finish(futures)
    return execute


def build_and_dispatch(task_list, task_schemas, routes, *, parent_agent, creds, context, top_role,
                       max_children, max_iterations, task_images, background, refused):
    """Reserve a prospective batch before transcripts, child construction or dispatch."""
    from tools import delegate_tool as facade
    from tools.delegation_live_log import create_live_transcripts
    import time
    try:
        with construction_batch(sum(not route.error for route in routes), parent_agent):
            err = facade._oneshot_spawn_budget(parent_agent, sum(not route.error for route in routes))
            if err:
                return facade.tool_error(err)
            overall_start = time.monotonic()
            home = facade._parent_live_home(parent_agent)
            live_id, writers, paths = create_live_transcripts(
                task_list, context, model=creds.get("model"), provider=creds.get("provider"), home=home,
            )
            if refused:
                facade._settle_refused_transcripts(refused, live_id, writers, home)
            facade._announce_batch(parent_agent, len(task_list), live_id)
            origin = facade._capture_origin()
            children, err = facade._build_children(
                task_list, task_schemas, routes, top_role=top_role, max_iterations=max_iterations,
                parent_agent=parent_agent, live_deleg_id=live_id, live_writers=writers, task_images=task_images,
            )
            if err:
                return facade.tool_error(err)
            batch = facade._Batch(
                task_list, children, parent_agent, creds, context, top_role, max_children,
                live_id, writers, paths, *origin, overall_start, live_home=home, refused=refused,
            )
            return facade._run_batch(batch, background)
    except HostPolicyError as exc:
        return facade.tool_error(str(exc), admission=admission_metadata(exc))
