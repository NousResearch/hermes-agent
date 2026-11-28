"""Who sent a terminal command, and whether Hermes's command gate already approved it.

A terminal backend's ``execute()`` is the one chokepoint every Hermes caller reaches: the terminal
tool, a background spawn and its pollers and stop, execute_code's script ship and RPC helpers, the
file tools, media reads, result spills and Hermes's own probes. A backend that must treat those
differently (Hermes's own work one way, the model's commands another, a device job polled where it
started) used to read the Python call stack for private function names; it now reads
:func:`current_command_context` inside ``execute()``.

The context rides a :class:`contextvars.ContextVar` rather than a new ``execute()`` keyword:
``execute()`` has no ``**kwargs`` on every backend (``managed_modal.py`` is a fixed signature) and
plugin environments are duck types, so a new keyword would raise ``TypeError`` on the environments
that never opted in. Context already propagates to every place that needs it — ``run_bounded_sync``
copies context into its worker, ``asyncio.to_thread`` copies it, and execute_code's RPC threads run
under ``propagate_context_to_thread``. A background poller thread starts under a copy of its *caller's*
context, which may still carry the spawn's stamp, so the poller sets ``background_poll`` itself
instead of inheriting one.

``None`` means unknown, the default for anything that has not opted in (including a plugin's own
calls into its environment). Hermes sets the innermost context; a ``terminal`` tool call dispatched
from inside execute_code's RPC replaces ``"execute_code"`` for the duration of its own
``execute()`` and the reset on return puts it back, so no stamp leaks into a later poll or a
nested call.

Hermes's own commands: for its job and cleanup traffic (``background_spawn``, ``background_poll``,
``background_kill``, execute_code's removal of its own script folder) the stamp's ``command`` is the exact
text Hermes's code sends in that block. A backend that treats such a command specially (runs it where the
job started, or without asking) should also require ``command`` to equal what ``execute()`` received, so
anything else that runs inside the block is judged as an ordinary command.

Reading an approval: trust ``approval`` only when ``origin == "terminal"`` and ``approval.command``
is exactly the ``command`` argument ``execute()`` received (the gate saw that string before
``execute()`` rewrote anything), or for ``origin == "background_spawn"`` when ``inner_command``
matches it. Otherwise it is provenance only: ``execute_code`` stamps ``"execute_code:<kind>"`` for a
script gated as a whole, which is not a judgement about any one command inside it.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, Optional


@dataclass(frozen=True)
class CommandApproval:
    """Hermes's command gate decision for one command.

    ``command`` is the exact string the gate saw: for ``origin == "terminal"`` it equals
    ``execute()``'s ``command`` argument (``execute()`` may rewrite it later, on its way to the
    shell); for a background spawn it equals ``CommandContext.inner_command``.

    ``kind`` is how the command passed the gate; the values are documented on
    :class:`CommandContext`.
    """

    command: str
    kind: str


@dataclass(frozen=True)
class CommandContext:
    """One caller's reason for reaching a terminal environment.

    ``origin`` is the call site's label: ``"terminal"`` (a model/agent terminal tool call),
    ``"background_spawn"``/``"background_poll"``/``"background_kill"`` (Hermes's background job
    registry), ``"execute_code"`` (an execute_code script launch), ``"execute_code_rpc"`` (the
    files, probes and RPC traffic around such a script), ``"file_op"`` (read_file/write_file/
    patch/search_files), ``"media_read"`` (an image/video/attachment read), ``"spill"`` (an
    oversized tool result written out), ``"guard_probe"`` (Hermes's own pre-execution script read),
    ``"internal"`` (Hermes's own probes, e.g. the system prompt's backend probe).

    ``job_id`` is the ``ProcessSession.id`` of a background job (spawn, poll, kill). ``inner_command``
    is the command a background wrapper carries: the one the gate approved. ``approval.kind`` says
    how the gate let the command through. A human or Hermes's rules approved this exact command for
    ``"user"``, ``"smart"``, ``"pre_approved"``, ``"batch"`` (a decision collected ahead of the run
    by the batch approval flow) and ``"no_findings"`` (a scan ran and found nothing).
    Nothing judged it for ``"bypass"`` (yolo or ``approvals.mode: off``), ``"unattended"`` (no
    reviewer present under an approve policy), ``"forced"`` (the caller's ``force`` skipped the gate)
    and ``"gate_skipped"`` (the gate did not run, e.g. an isolated backend). ``"execute_code:<kind>"`` is a whole-script gate with one of those kinds.

    ``command`` is the exact text Hermes's own code sends in this block when it sends one fixed command
    (a background wrapper, poll, status read or kill; execute_code's removal of its own folder), else
    ``None`` (a model's command, or traffic Hermes does not pin).
    """

    origin: str
    job_id: Optional[str] = None
    inner_command: Optional[str] = None
    task_id: Optional[str] = None
    session_key: Optional[str] = None
    approval: Optional[CommandApproval] = None
    command: Optional[str] = None


_COMMAND_CONTEXT: contextvars.ContextVar[Optional[CommandContext]] = contextvars.ContextVar(
    "hermes_command_context", default=None
)


def current_command_context() -> Optional[CommandContext]:
    """The context of the command in flight, or ``None`` when the caller is unknown."""
    return _COMMAND_CONTEXT.get()


@contextmanager
def exact_command(command: str) -> Iterator[Optional[CommandContext]]:
    """The context in flight, naming *command* as the exact text sent in this block (unchanged when none)."""
    current = _COMMAND_CONTEXT.get()
    if current is None:
        yield None
        return
    token = _COMMAND_CONTEXT.set(replace(current, command=command))
    try:
        yield _COMMAND_CONTEXT.get()
    finally:
        _COMMAND_CONTEXT.reset(token)


@contextmanager
def command_context(origin: str, **fields: Any) -> Iterator[CommandContext]:
    """Announce the command in flight for the duration of the block; innermost wins."""
    ctx = CommandContext(origin=origin, **fields)
    token = _COMMAND_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _COMMAND_CONTEXT.reset(token)
