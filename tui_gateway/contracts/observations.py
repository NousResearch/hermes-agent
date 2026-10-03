"""Volatile shared-session display evidence; never a durable admission or ownership receipt."""

from typing import Literal

from .base import JsonValue, Result, WireEnum


class SharedSessionCapability(Result):
    version: Literal[1]
    socket_id: str


class ConnectionTurnSource(Result):
    kind: Literal["connection"]
    socket_id: str


class UnattributedTurnSource(Result):
    kind: Literal["mixed", "unknown"]


class TurnDescriptor(Result):
    """One runner invocation, scoped by replay epoch and runtime session."""

    id: str
    source: ConnectionTurnSource | UnattributedTurnSource


class InputOccurrence(Result):
    """A fresh accepted RPC occurrence; reused refs do not deduplicate work."""

    id: str
    ref: str | None = None


class SubmissionDisposition(WireEnum):
    starting = "starting"
    queued = "queued"
    merged = "merged"
    absorbed = "absorbed"
    steered = "steered"
    redirected = "redirected"
    unresolved = "unresolved"


class InputSubmission(Result):
    input_id: str
    ref: str | None = None
    disposition: SubmissionDisposition
    turn: TurnDescriptor | None = None


class ProjectedInput(Result):
    """The canonical user transcript projection; null at the containing field suppresses display."""

    role: Literal["user"]
    text: str
    display_kind: str | None = None
    display_metadata: JsonValue | None = None


class InputBatchProjection(Result):
    inputs: list[InputOccurrence]
    inputs_complete: bool


class InputObservation(InputBatchProjection):
    kind: Literal["steer", "redirect"]
    input: ProjectedInput | None
    offset: int


class InputOutcomeDisposition(WireEnum):
    cancelled = "cancelled"
    absorbed = "absorbed"
    failed_before_start = "failed_before_start"
    unresolved = "unresolved"
    terminal = "terminal"


class InputOutcome(Result):
    revision: int
    input: InputOccurrence
    disposition: InputOutcomeDisposition
    reason: str | None = None
    into_inputs: list[InputOccurrence] | None = None
    turn: TurnDescriptor | None = None
    status: Literal["complete", "error", "interrupted"] | None = None


class SubmissionState(Result):
    """Bounded evidence owned by the gateway queue; absence does not prove cancellation."""

    revision: int
    queued: list[InputBatchProjection]
    queued_complete: bool
    outcomes: list[InputOutcome]
    outcomes_truncated_before_revision: int | None
