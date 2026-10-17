"""Resolve configured speech and synthesize one exact prepared native chunk.

Resolution describes passive configuration and trusted declarations, not SDK
readiness or acoustic quality. Opaque declaration revisions cover only their
explicit fields. Canonical configuration reads may create normal home directories
and backups, and expand profile environment references. Drift validation precedes
output creation, auth refresh, SDK readiness, catalogue requests and provider
execution. Output format is observed from the generated container; codec
completeness and interruption remain the caller's responsibility.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
import traceback
from pathlib import Path
from typing import Any

from tools.tts_prepared_errors import _SpeechFailure, _capture_failure
from tools.tts_prepared_policy import _apply_settings, _resolve
from tools.tts_prepared_types import (
    PreparedSpeechRequest,
    SpeechFact,
    SpeechFactSource,
    SpeechFailureCode,
    SpeechFormat,
    SpeechResolution,
    SpeechRouteKind,
    SpeechSettings,
    SpeechSynthesisError,
    SpeechSynthesisResult,
)


logger = logging.getLogger(__name__)


def _config() -> dict[str, Any]:
    from hermes_cli.config import FailedConfigRead, load_config

    config = load_config()
    if isinstance(config, FailedConfigRead):
        raise SpeechSynthesisError(SpeechFailureCode.INVALID_CONFIGURATION)
    section = config.get("tts")
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise SpeechSynthesisError(SpeechFailureCode.INVALID_CONFIGURATION)
    return section


def _registered_plugin(config: dict[str, Any]):
    from agent import tts_registry
    from tools import tts_tool

    provider = tts_tool._get_provider(config)
    if provider in tts_tool.BUILTIN_TTS_PROVIDERS:
        return None
    return tts_registry._registry.get_provider(provider)


def _failure(error: Exception) -> _SpeechFailure:
    if type(error) is SpeechSynthesisError:
        try:
            safe = SpeechSynthesisError(error.code, error.causes)
        except (TypeError, ValueError):
            return _capture_failure(error)
        return _SpeechFailure(safe.code, safe.causes)
    return _capture_failure(error)


def _resolve_safely(settings: SpeechSettings) -> SpeechResolution | _SpeechFailure:
    try:
        config = _config()
        return _resolve(settings, config, plugin=_registered_plugin(config))
    except Exception as error:
        failure = _failure(error)
        safe = SpeechSynthesisError(failure.code, failure.causes)
        trace = traceback.TracebackException(
            SpeechSynthesisError,
            safe,
            error.__traceback__,
            lookup_lines=False,
            capture_locals=False,
        )
        positions = tuple(
            (frame.filename, frame.lineno, frame.name) for frame in trace.stack
        )
        logger.exception(
            "Prepared speech failed at %s",
            positions,
            exc_info=(SpeechSynthesisError, safe, None),
        )
        return failure


def resolve_prepared_speech(settings: SpeechSettings | None = None) -> SpeechResolution:
    """Inspect native configuration without SDK readiness, auth refresh or catalogue requests."""
    result = _resolve_safely(settings if settings is not None else SpeechSettings())
    if isinstance(result, _SpeechFailure):
        raise SpeechSynthesisError(result.code, result.causes)
    return result


def _run_prepared(
    request: PreparedSpeechRequest,
) -> SpeechSynthesisResult | _SpeechFailure:
    from tools import tts_tool as native
    from tools.audio_container import sniff_container
    from tools.tts_command_provider import _resolve_command_provider_config
    from tools.tts_tool_providers import TTS_RESPONSE_BODY_LIMIT_BYTES
    from tools.tts_tool_synthesis import _synthesize_single

    try:
        if not isinstance(request, PreparedSpeechRequest) or not isinstance(
            request.expected_resolution, SpeechResolution
        ):
            raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST)
        if not isinstance(request.text, str) or not request.text.strip():
            raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST)
        try:
            request.text.encode("utf-8")
        except UnicodeEncodeError:
            raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST) from None
        if not isinstance(request.output_path, str) or not request.output_path:
            raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST)
        config = _config()
        resolution = _resolve(
            request.expected_resolution.settings,
            config,
            plugin=_registered_plugin(config),
        )
        if resolution != request.expected_resolution:
            raise SpeechSynthesisError(SpeechFailureCode.RESOLUTION_CHANGED)
        if (
            resolution.input_limit.value is not None
            and len(request.text) > resolution.input_limit.value
        ):
            raise SpeechSynthesisError(SpeechFailureCode.INPUT_LIMIT)
        provider = resolution.engine.value
        if provider is None:
            raise SpeechSynthesisError(SpeechFailureCode.UNKNOWN_CAPABILITY)
        if (
            Path(request.output_path).suffix.lower()
            != "." + resolution.settings.output_format.value
        ):
            raise SpeechSynthesisError(SpeechFailureCode.OUTPUT_FORMAT)
        output, error = native._resolve_output_base(
            request.output_path, provider, None, False
        )
        if error:
            raise SpeechSynthesisError(SpeechFailureCode.OUTPUT_PATH)
        applied = _apply_settings(config, resolution)
        command = _resolve_command_provider_config(provider, applied)
        with tempfile.TemporaryDirectory(
            prefix=".prepared-speech-", dir=output.parent
        ) as owned:
            destination = Path(owned) / (
                "speech." + resolution.settings.output_format.value
            )
            attempt = _synthesize_single(
                request.text,
                str(destination),
                provider=provider,
                tts_config=applied,
                command_provider_config=command,
                instructions=resolution.settings.instructions,
                prepared_resolution=resolution,
            )
            if attempt.diagnostic is not None:
                raise SpeechSynthesisError(
                    attempt.failure_code or SpeechFailureCode.OUTPUT_MISSING
                )
            actual = Path(attempt.path)
            metadata = actual.lstat()
            if (
                actual.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or not actual.resolve().is_relative_to(Path(owned).resolve())
            ):
                raise SpeechSynthesisError(SpeechFailureCode.OUTPUT_PATH)
            if (
                metadata.st_size <= 0
                or metadata.st_size > TTS_RESPONSE_BODY_LIMIT_BYTES
            ):
                raise SpeechSynthesisError(SpeechFailureCode.OUTPUT_MISSING)
            with actual.open("rb") as audio:
                container = sniff_container(audio.read(12))
            if container != resolution.settings.output_format.value:
                raise SpeechSynthesisError(SpeechFailureCode.OUTPUT_FORMAT)
            os.replace(actual, output)
            return SpeechSynthesisResult(
                str(output),
                SpeechFormat(container),
                SpeechFact(attempt.engine, SpeechFactSource.OBSERVED),
                resolution.model,
                resolution.voice,
                resolution.speed,
            )
    except Exception as error:
        failure = _failure(error)
        safe = SpeechSynthesisError(failure.code, failure.causes)
        trace = traceback.TracebackException(
            SpeechSynthesisError,
            safe,
            error.__traceback__,
            lookup_lines=False,
            capture_locals=False,
        )
        positions = tuple(
            (frame.filename, frame.lineno, frame.name) for frame in trace.stack
        )
        logger.exception(
            "Prepared speech failed at %s",
            positions,
            exc_info=(SpeechSynthesisError, safe, None),
        )
        return failure


def synthesize_prepared_speech(request: PreparedSpeechRequest) -> SpeechSynthesisResult:
    """Submit one validated prepared chunk through the configured native dispatcher."""
    result = _run_prepared(request)
    if isinstance(result, _SpeechFailure):
        raise SpeechSynthesisError(result.code, result.causes)
    return result
