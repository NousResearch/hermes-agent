"""One native speech dispatch, shared by the tool and prepared synthesis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tools.tts_prepared_types import (
    SpeechFailureCode,
    SpeechResolution,
    SpeechSynthesisError,
)


@dataclass
class _SpeechDispatch:
    path: str
    provider: str
    engine: str
    diagnostic: str | None = None
    failure_code: SpeechFailureCode | None = None


def _synthesize_single(
    text: str,
    file_str: str,
    *,
    provider: str,
    tts_config: dict[str, Any],
    command_provider_config: dict[str, Any] | None,
    instructions: str | None,
    prepared_resolution: SpeechResolution | None = None,
    attempt: _SpeechDispatch | None = None,
) -> _SpeechDispatch:
    from tools import tts_tool as origin

    engine = provider
    if attempt is None:
        attempt = _SpeechDispatch(file_str, provider, engine)
    if command_provider_config is not None:
        origin.logger.info(
            "Generating speech with command TTS provider '%s'...", provider
        )
        command_kwargs = {}
        if prepared_resolution is not None and instructions is not None:
            command_kwargs["instructions"] = instructions
        file_str = origin._generate_command_tts(
            text,
            file_str,
            provider,
            command_provider_config,
            tts_config,
            **command_kwargs,
        )
    else:
        kwargs = (
            {"prepared_resolution": prepared_resolution}
            if prepared_resolution is not None
            else {}
        )
        plugin_path = None
        if provider not in origin.BUILTIN_TTS_PROVIDERS:
            plugin_path = origin._dispatch_to_plugin_provider(
                text, file_str, provider, tts_config, **kwargs
            )
        if plugin_path is not None:
            file_str = plugin_path
        else:
            if (
                prepared_resolution is not None
                and provider not in origin.BUILTIN_TTS_PROVIDERS
            ):
                raise SpeechSynthesisError(SpeechFailureCode.UNKNOWN_CAPABILITY)
            provider, error = origin._select_builtin_engine(provider)
            engine = provider if provider in origin._BUILTIN_DISPATCH else "edge"
            attempt.provider, attempt.engine = provider, engine
            if (
                prepared_resolution is not None
                and engine != prepared_resolution.engine.value
            ):
                raise SpeechSynthesisError(SpeechFailureCode.RESOLUTION_CHANGED)
            if error:
                return _SpeechDispatch(
                    file_str,
                    provider,
                    engine,
                    error,
                    failure_code=SpeechFailureCode.DEPENDENCY_UNAVAILABLE,
                )
            if prepared_resolution is not None and provider == "openai":
                origin._generate_openai_tts(
                    text,
                    file_str,
                    tts_config,
                    instructions=instructions,
                    prepared_resolution=prepared_resolution,
                )
            elif prepared_resolution is not None and provider == "deepinfra":
                origin._generate_deepinfra_tts(
                    text,
                    file_str,
                    tts_config,
                    prepared_resolution=prepared_resolution,
                )
            else:
                origin._synthesize_builtin(
                    provider, text, file_str, tts_config, instructions
                )
    if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
        return _SpeechDispatch(
            file_str,
            provider,
            engine,
            origin._error_json(
                f"TTS generation produced no output (provider: {provider})"
            ),
            failure_code=SpeechFailureCode.OUTPUT_MISSING,
        )
    return _SpeechDispatch(file_str, provider, engine)
