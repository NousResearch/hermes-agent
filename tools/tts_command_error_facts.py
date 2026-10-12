"""Internal command-operation metadata on original native exceptions.

Events retain native exceptions for later classification. They must not enter
public error objects, serialized output or log records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CommandOperation(StrEnum):
    READ = "read"
    CLOSE = "close"
    PROCESS_CLEANUP = "process_cleanup"
    OUTPUT_DRAIN = "output_drain"
    NATIVE_EXIT = "native_exit"
    NATIVE_TIMEOUT = "native_timeout"


class CommandStream(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"


@dataclass(frozen=True, eq=False)
class CommandFailureEvent:
    error: BaseException = field(repr=False)
    operation: CommandOperation
    stream: CommandStream | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.error, BaseException) or not isinstance(
            self.operation, CommandOperation
        ):
            raise ValueError("invalid command failure event")
        if self.stream is not None and not isinstance(self.stream, CommandStream):
            raise ValueError("invalid command stream")


@dataclass(frozen=True)
class CommandFailureFacts:
    root_operation: CommandOperation | None = None
    root_stream: CommandStream | None = None
    cleanup_events: tuple[CommandFailureEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.root_operation is not None and not isinstance(
            self.root_operation, CommandOperation
        ):
            raise ValueError("invalid root command operation")
        if self.root_stream is not None and not isinstance(
            self.root_stream, CommandStream
        ):
            raise ValueError("invalid root command stream")
        if type(self.cleanup_events) is not tuple:
            raise ValueError("invalid command cleanup events")
        for event in self.cleanup_events:
            if type(event) is not CommandFailureEvent:
                raise ValueError("invalid command cleanup event")
            CommandFailureEvent.__post_init__(event)

    def with_cleanup(self, *events: CommandFailureEvent) -> CommandFailureFacts:
        combined = list(self.cleanup_events)
        for event in events:
            if not any(event is existing for existing in combined):
                combined.append(event)
        return CommandFailureFacts(
            self.root_operation, self.root_stream, tuple(combined)
        )


_ATTRIBUTE = "_hermes_command_failure_facts"


def command_failure_facts(error: BaseException) -> CommandFailureFacts | None:
    attributes = BaseException.__dict__["__dict__"].__get__(error)
    value = dict.get(attributes, _ATTRIBUTE)
    if type(value) is not CommandFailureFacts:
        return None
    try:
        return CommandFailureFacts(
            value.root_operation, value.root_stream, value.cleanup_events
        )
    except ValueError:
        return None


def set_command_failure_facts(error: BaseException, facts: CommandFailureFacts) -> None:
    if type(facts) is not CommandFailureFacts:
        raise ValueError("invalid command failure facts")
    validated = CommandFailureFacts(
        facts.root_operation, facts.root_stream, facts.cleanup_events
    )
    # Native exception storage avoids provider-defined attribute descriptors.
    attributes = BaseException.__dict__["__dict__"].__get__(error)
    dict.__setitem__(attributes, _ATTRIBUTE, validated)
