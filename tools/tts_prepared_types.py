"""Typed facts for one prepared native speech request."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Generic, TypeVar

from tools.tts_command_error_facts import CommandOperation, CommandStream


class SpeechFormat(StrEnum):
    MP3 = "mp3"
    WAV = "wav"


class SpeechRouteKind(StrEnum):
    BUILTIN = "builtin"
    COMMAND = "command"
    PLUGIN = "plugin"


class SpeechControl(StrEnum):
    MODEL = "model"
    VOICE = "voice"
    SPEED = "speed"
    INSTRUCTIONS = "instructions"


class SpeechFactSource(StrEnum):
    CONFIGURED = "configured"
    NATIVE_DEFAULT = "native_default"
    NATIVE_POLICY = "native_policy"
    DECLARED = "declared"
    UNKNOWN = "unknown"
    INAPPLICABLE = "inapplicable"
    OBSERVED = "observed"


_Value = TypeVar("_Value")


@dataclass(frozen=True)
class SpeechFact(Generic[_Value]):
    value: _Value | None
    source: SpeechFactSource


@dataclass(frozen=True)
class SpeechSettings:
    output_format: SpeechFormat = SpeechFormat.MP3
    model: str | None = None
    voice: str | None = None
    speed: float | None = None
    instructions: str | None = None


@dataclass(frozen=True)
class PreparedSpeechDeclaration:
    """A trusted provider or operator assertion, not an observed synthesis result.

    ``revision`` and ``identity_coverage`` describe only the opaque settings
    whose changes the declarer records. Native validation still rejects known
    rewrites, ignored controls and incompatible output formats.
    """

    revision: str
    identity_coverage: tuple[str, ...]
    prepared_native_text: bool
    formats: frozenset[SpeechFormat]
    controls: frozenset[SpeechControl]
    input_limit: int | None = None
    speed_min: float | None = None
    speed_max: float | None = None


@dataclass(frozen=True)
class SpeechResolution:
    """Passive facts; input limits use Python codepoints.

    ``fingerprint`` excludes instruction text and credentials. Synthesis compares
    the complete resolution, including settings and provenance, not the digest.
    """

    settings: SpeechSettings
    configured_provider: str
    kind: SpeechRouteKind
    engine: SpeechFact[str]
    model: SpeechFact[str]
    voice: SpeechFact[str]
    speed: SpeechFact[float]
    input_limit: SpeechFact[int]
    controls: frozenset[SpeechControl]
    controls_source: SpeechFactSource
    prepared_native_text: SpeechFact[bool]
    endpoint_origin: str | None
    declaration: PreparedSpeechDeclaration | None
    identity_coverage: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class PreparedSpeechRequest:
    text: str
    output_path: str
    expected_resolution: SpeechResolution


@dataclass(frozen=True)
class SpeechSynthesisResult:
    path: str
    output_format: SpeechFormat
    engine: SpeechFact[str]
    model: SpeechFact[str]
    voice: SpeechFact[str]
    speed: SpeechFact[float]


class SpeechFailureCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_CONFIGURATION = "invalid_configuration"
    UNKNOWN_CAPABILITY = "unknown_capability"
    UNSUPPORTED_CONTROL = "unsupported_control"
    RESOLUTION_CHANGED = "resolution_changed"
    INPUT_LIMIT = "input_limit"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    OUTPUT_MISSING = "output_missing"
    OUTPUT_FORMAT = "output_format"
    OUTPUT_PATH = "output_path"


class SpeechCauseType(StrEnum):
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    OS_ERROR = "os_error"
    IMPORT_ERROR = "import_error"
    CONFIGURATION_ERROR = "configuration_error"
    OPENAI_STATUS = "openai_status"
    OPENAI_RATE_LIMIT = "openai_rate_limit"
    OPENAI_AUTHENTICATION = "openai_authentication"
    OPENAI_PERMISSION = "openai_permission"
    OPENAI_SERVER = "openai_server"
    OPENAI_TIMEOUT = "openai_timeout"
    OPENAI_CONNECTION = "openai_connection"
    AIOHTTP_STATUS = "aiohttp_status"
    HTTPX_STATUS = "httpx_status"
    HTTPX_TIMEOUT = "httpx_timeout"
    HTTPX_NETWORK = "httpx_network"
    REQUESTS_STATUS = "requests_status"
    REQUESTS_TIMEOUT = "requests_timeout"
    REQUESTS_CONNECTION = "requests_connection"
    COMMAND_EXIT = "command_exit"
    COMMAND_TIMEOUT = "command_timeout"


class SpeechCauseRelation(StrEnum):
    ROOT = "root"
    CAUSE = "cause"
    CONTEXT = "context"
    CLEANUP = "cleanup"


@dataclass(frozen=True)
class SpeechCause:
    """One safe node occurrence and its incoming edge.

    ``node_index`` is the first tuple position for that native exception.
    Later occurrences reuse it, including cycles. ``parent_index`` refers
    to a previously declared canonical node. Stream and operation describe
    this occurrence; the remaining facts must agree across aliases.
    """

    identity: SpeechCauseType
    relation: SpeechCauseRelation = SpeechCauseRelation.ROOT
    status_code: int | None = None
    retry_after: float | None = None
    exit_code: int | None = None
    context_suppressed: bool = False
    errno: int | None = None
    stream: CommandStream | None = None
    operation: CommandOperation | None = None
    parent_index: int | None = None
    node_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SpeechCauseType) or not isinstance(
            self.relation, SpeechCauseRelation
        ):
            raise ValueError("invalid speech cause identity")
        if self.status_code is not None and (
            type(self.status_code) is not int or not 100 <= self.status_code <= 599
        ):
            raise ValueError("invalid speech status")
        if self.retry_after is not None and (
            type(self.retry_after) not in (int, float)
            or (type(self.retry_after) is float and not math.isfinite(self.retry_after))
            or self.retry_after < 0
        ):
            raise ValueError("invalid speech retry delay")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("invalid speech exit status")
        if type(self.context_suppressed) is not bool:
            raise ValueError("invalid speech context suppression")
        if self.errno is not None and type(self.errno) is not int:
            raise ValueError("invalid speech errno")
        if self.stream is not None and not isinstance(self.stream, CommandStream):
            raise ValueError("invalid speech command stream")
        if self.operation is not None and not isinstance(
            self.operation, CommandOperation
        ):
            raise ValueError("invalid speech command operation")
        for index in (self.parent_index, self.node_index):
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("invalid speech graph index")

    def validated_copy(self) -> SpeechCause:
        return SpeechCause(
            self.identity,
            self.relation,
            self.status_code,
            self.retry_after,
            self.exit_code,
            self.context_suppressed,
            self.errno,
            self.stream,
            self.operation,
            self.parent_index,
            self.node_index,
        )

    def _intrinsic_facts(
        self,
    ) -> tuple[SpeechCauseType, int | None, float | None, int | None, bool, int | None]:
        return (
            self.identity,
            self.status_code,
            self.retry_after,
            self.exit_code,
            self.context_suppressed,
            self.errno,
        )


def _validated_causes(causes: tuple[SpeechCause, ...]) -> tuple[SpeechCause, ...]:
    legacy = all(
        cause.node_index is None and cause.parent_index is None for cause in causes
    )
    canonical: dict[int, SpeechCause] = {}
    validated = []
    for position, original in enumerate(causes):
        cause = SpeechCause.validated_copy(original)
        if legacy:
            if cause.relation == SpeechCauseRelation.CLEANUP:
                raise ValueError("missing speech cleanup edge")
            cause = replace(
                cause,
                node_index=position,
                parent_index=None
                if cause.relation == SpeechCauseRelation.ROOT
                else position - 1,
            )
        index = cause.node_index
        if index is None:
            raise ValueError("missing speech node index")
        if cause.relation == SpeechCauseRelation.ROOT:
            if position != 0 or cause.parent_index is not None:
                raise ValueError("invalid speech root parent")
        elif cause.parent_index not in canonical:
            raise ValueError("undeclared speech cause parent")
        previous = canonical.get(index)
        if previous is not None:
            if cause._intrinsic_facts() != previous._intrinsic_facts():
                raise ValueError("inconsistent speech node facts")
        else:
            if index != position:
                raise ValueError("undeclared speech node index")
            canonical[index] = cause
        validated.append(cause)
    return tuple(validated)


class SpeechSynthesisError(Exception):
    """Safe failure facts. Serialization excludes traceback locals and native objects."""

    def __init__(self, code: SpeechFailureCode, causes: tuple[SpeechCause, ...] = ()):
        if not isinstance(code, SpeechFailureCode) or type(causes) is not tuple:
            raise ValueError("invalid speech failure")
        if any(type(cause) is not SpeechCause for cause in causes):
            raise ValueError("invalid speech causes")
        self.code = code
        self.causes = _validated_causes(causes)
        super().__init__(code.value)

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "causes": [
                {
                    "identity": cause.identity.value,
                    "relation": cause.relation.value,
                    "status_code": cause.status_code,
                    "retry_after": cause.retry_after,
                    "exit_code": cause.exit_code,
                    "context_suppressed": cause.context_suppressed,
                    "errno": cause.errno,
                    "stream": cause.stream.value if cause.stream is not None else None,
                    "operation": cause.operation.value
                    if cause.operation is not None
                    else None,
                    "parent_index": cause.parent_index,
                    "node_index": cause.node_index,
                }
                for cause in self.causes
            ],
        }
