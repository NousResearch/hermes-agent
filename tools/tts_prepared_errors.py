"""Reduce native speech failures to safe typed facts."""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any

from tools.tts_command_error_facts import (
    CommandOperation,
    CommandStream,
    command_failure_facts,
)

from tools.tts_prepared_types import (
    SpeechCause,
    SpeechCauseType,
    SpeechCauseRelation,
    SpeechFailureCode,
)


@dataclass(frozen=True)
class _SpeechFailure:
    code: SpeechFailureCode
    causes: tuple[SpeechCause, ...] = ()


def _integer(value: object) -> int | None:
    return value if type(value) is int else None


def _status(value: object) -> int | None:
    status = _integer(value)
    return status if status is not None and 100 <= status <= 599 else None


def _retry_after(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        seconds = float(value)
    except (ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _is_native(error: BaseException, module: str, identity: str) -> bool:
    loaded = sys.modules.get(module)
    if not isinstance(loaded, ModuleType):
        return False
    native_type = getattr(loaded, identity, None)
    return isinstance(native_type, type) and isinstance(error, native_type)


def _http_cause(
    error: Any, identity: SpeechCauseType, relation: SpeechCauseRelation
) -> SpeechCause:
    response = error.response
    if response is None:
        return SpeechCause(identity, relation)
    return SpeechCause(
        identity,
        relation,
        _status(response.status_code),
        _retry_after(response.headers.get("retry-after")),
    )


def _aiohttp_cause(error: Any, relation: SpeechCauseRelation) -> SpeechCause:
    headers = error.headers
    retry_after = None
    if isinstance(headers, Mapping):
        retry_after = next(
            (
                value
                for key, value in headers.items()
                if isinstance(key, str) and key.casefold() == "retry-after"
            ),
            None,
        )
    return SpeechCause(
        SpeechCauseType.AIOHTTP_STATUS,
        relation,
        _status(error.status),
        _retry_after(retry_after),
    )


def _cause(error: BaseException, relation: SpeechCauseRelation) -> SpeechCause:
    if isinstance(error, subprocess.TimeoutExpired):
        return SpeechCause(
            SpeechCauseType.COMMAND_TIMEOUT,
            relation,
            operation=CommandOperation.NATIVE_TIMEOUT,
        )
    if isinstance(error, subprocess.CalledProcessError):
        return SpeechCause(
            SpeechCauseType.COMMAND_EXIT,
            relation,
            exit_code=_integer(error.returncode),
            operation=CommandOperation.NATIVE_EXIT,
        )
    if isinstance(error, TimeoutError):
        return SpeechCause(SpeechCauseType.TIMEOUT, relation)
    native_status_types = {
        "RateLimitError": SpeechCauseType.OPENAI_RATE_LIMIT,
        "AuthenticationError": SpeechCauseType.OPENAI_AUTHENTICATION,
        "PermissionDeniedError": SpeechCauseType.OPENAI_PERMISSION,
        "InternalServerError": SpeechCauseType.OPENAI_SERVER,
        "APIStatusError": SpeechCauseType.OPENAI_STATUS,
    }
    for identity, cause_type in native_status_types.items():
        if _is_native(error, "openai", identity):
            return _http_cause(error, cause_type, relation)
    if _is_native(error, "openai", "APITimeoutError"):
        return SpeechCause(SpeechCauseType.OPENAI_TIMEOUT, relation)
    if _is_native(error, "openai", "APIConnectionError"):
        return SpeechCause(SpeechCauseType.OPENAI_CONNECTION, relation)
    if _is_native(error, "aiohttp", "ClientResponseError"):
        return _aiohttp_cause(error, relation)
    if _is_native(error, "httpx", "HTTPStatusError"):
        return _http_cause(error, SpeechCauseType.HTTPX_STATUS, relation)
    if _is_native(error, "httpx", "TimeoutException"):
        return SpeechCause(SpeechCauseType.HTTPX_TIMEOUT, relation)
    if _is_native(error, "httpx", "NetworkError"):
        return SpeechCause(SpeechCauseType.HTTPX_NETWORK, relation)
    if _is_native(error, "requests.exceptions", "HTTPError"):
        return _http_cause(error, SpeechCauseType.REQUESTS_STATUS, relation)
    if _is_native(error, "requests.exceptions", "Timeout"):
        return SpeechCause(SpeechCauseType.REQUESTS_TIMEOUT, relation)
    if _is_native(error, "requests.exceptions", "ConnectionError"):
        return SpeechCause(SpeechCauseType.REQUESTS_CONNECTION, relation)
    if isinstance(error, ImportError):
        return SpeechCause(SpeechCauseType.IMPORT_ERROR, relation)
    if isinstance(error, OSError):
        return SpeechCause(SpeechCauseType.OS_ERROR, relation)
    if isinstance(error, ValueError):
        return SpeechCause(SpeechCauseType.CONFIGURATION_ERROR, relation)
    return SpeechCause(SpeechCauseType.UNKNOWN, relation)


def _capture_failure(error: BaseException) -> _SpeechFailure:
    causes = _capture_graph(error)
    primary = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        primary.append(_cause(current, SpeechCauseRelation.ROOT))
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        current = current.__context__ if not current.__suppress_context__ else None
    code = _failure_code(primary)
    return _SpeechFailure(code, causes)


@dataclass(frozen=True)
class _PendingCause:
    error: BaseException = field(repr=False)
    relation: SpeechCauseRelation
    parent_index: int | None = None
    operation: CommandOperation | None = None
    stream: CommandStream | None = None


def _capture_graph(error: BaseException) -> tuple[SpeechCause, ...]:
    causes: list[SpeechCause] = []
    canonical: dict[int, SpeechCause] = {}
    pending = [_PendingCause(error, SpeechCauseRelation.ROOT)]
    while pending:
        edge = pending.pop()
        current = edge.error
        facts = command_failure_facts(current)
        operation, stream = edge.operation, edge.stream
        if operation is None and facts is not None:
            operation, stream = facts.root_operation, facts.root_stream
        previous = canonical.get(id(current))
        first = previous is None
        if previous is None:
            native_errno = (
                OSError.__dict__["errno"].__get__(current)
                if isinstance(current, OSError)
                else None
            )
            previous = replace(
                _cause(current, SpeechCauseRelation.ROOT),
                context_suppressed=current.__suppress_context__,
                errno=_integer(native_errno),
                node_index=len(causes),
            )
            canonical[id(current)] = previous
        if operation is None:
            operation = previous.operation
        causes.append(
            replace(
                previous,
                relation=edge.relation,
                parent_index=edge.parent_index,
                operation=operation,
                stream=stream,
            )
        )
        if not first:
            continue
        children = []
        if current.__cause__ is not None:
            children.append(
                _PendingCause(
                    current.__cause__, SpeechCauseRelation.CAUSE, previous.node_index
                )
            )
        if current.__context__ is not None:
            children.append(
                _PendingCause(
                    current.__context__,
                    SpeechCauseRelation.CONTEXT,
                    previous.node_index,
                )
            )
        if facts is not None:
            children.extend(
                _PendingCause(
                    event.error,
                    SpeechCauseRelation.CLEANUP,
                    previous.node_index,
                    event.operation,
                    event.stream,
                )
                for event in facts.cleanup_events
            )
        pending.extend(reversed(children))
    return tuple(causes)


def _failure_code(causes: list[SpeechCause]) -> SpeechFailureCode:
    code = SpeechFailureCode.PROVIDER_ERROR
    for cause in causes:
        if cause.status_code in {401, 403}:
            code = SpeechFailureCode.AUTHENTICATION
            break
        if cause.status_code == 429:
            code = SpeechFailureCode.RATE_LIMIT
            break
        if cause.identity in {
            SpeechCauseType.TIMEOUT,
            SpeechCauseType.COMMAND_TIMEOUT,
            SpeechCauseType.OPENAI_TIMEOUT,
            SpeechCauseType.HTTPX_TIMEOUT,
            SpeechCauseType.REQUESTS_TIMEOUT,
        }:
            code = SpeechFailureCode.TIMEOUT
            break
        if cause.identity == SpeechCauseType.IMPORT_ERROR:
            code = SpeechFailureCode.DEPENDENCY_UNAVAILABLE
            break
        if cause.identity == SpeechCauseType.CONFIGURATION_ERROR:
            code = SpeechFailureCode.INVALID_CONFIGURATION
            break
    return code
