"""Passive native speech facts and trusted prepared-speech declarations."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import re
from dataclasses import asdict, replace
from typing import Any, NoReturn
from urllib.parse import urlsplit

from tools.tts_prepared_types import (
    PreparedSpeechDeclaration,
    SpeechControl,
    SpeechFact,
    SpeechFactSource,
    SpeechFailureCode,
    SpeechFormat,
    SpeechResolution,
    SpeechRouteKind,
    SpeechSettings,
    SpeechSynthesisError,
)


_SAFE_REVISION = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_KNOWN_CONTROLS = {
    "edge": (None, "voice", frozenset({SpeechControl.VOICE, SpeechControl.SPEED})),
    "openai": ("model", "voice", frozenset(SpeechControl)),
    "deepinfra": (
        "model",
        "voice",
        frozenset({SpeechControl.MODEL, SpeechControl.VOICE, SpeechControl.SPEED}),
    ),
    "elevenlabs": (
        "model_id",
        "voice_id",
        frozenset({SpeechControl.MODEL, SpeechControl.VOICE}),
    ),
    "minimax": (
        "model",
        "voice_id",
        frozenset({SpeechControl.MODEL, SpeechControl.VOICE, SpeechControl.SPEED}),
    ),
    "xai": (None, "voice_id", frozenset({SpeechControl.VOICE, SpeechControl.SPEED})),
    "mistral": (
        "model",
        "voice_id",
        frozenset({SpeechControl.MODEL, SpeechControl.VOICE}),
    ),
    "neutts": ("model", None, frozenset({SpeechControl.MODEL})),
    "piper": (None, "voice", frozenset({SpeechControl.VOICE})),
    "kittentts": (
        "model",
        "voice",
        frozenset({SpeechControl.MODEL, SpeechControl.VOICE, SpeechControl.SPEED}),
    ),
    "gemini": ("model", "voice", frozenset({SpeechControl.MODEL, SpeechControl.VOICE})),
}
_NATIVE_COVERAGE = (
    "configured_provider",
    "kind",
    "engine",
    "model",
    "voice",
    "speed",
    "input_limit",
    "controls",
    "prepared_native_text",
    "output_format",
    "endpoint_origin",
)


def _invalid() -> NoReturn:
    raise SpeechSynthesisError(SpeechFailureCode.INVALID_CONFIGURATION)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _string(value: Any) -> str | None:
    return (
        value if isinstance(value, str) and value and value == value.strip() else None
    )


def _validate_settings(settings: SpeechSettings) -> None:
    if not isinstance(settings, SpeechSettings) or not isinstance(
        settings.output_format, SpeechFormat
    ):
        raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST)
    if any(
        value is not None and _string(value) is None
        for value in (settings.model, settings.voice)
    ):
        raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST)
    if settings.speed is not None and _number(settings.speed) is None:
        raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST)
    if settings.instructions is not None and (
        not isinstance(settings.instructions, str) or not settings.instructions
    ):
        raise SpeechSynthesisError(SpeechFailureCode.INVALID_REQUEST)


def _declaration(value: Any) -> PreparedSpeechDeclaration | None:
    if value is None:
        return None
    if isinstance(value, PreparedSpeechDeclaration):
        declaration = value
    elif isinstance(value, dict):
        revision = value.get("revision")
        coverage = value.get("identity_coverage")
        formats = value.get("formats")
        controls = value.get("controls")
        if not all(isinstance(items, list) for items in (coverage, formats, controls)):
            _invalid()
        try:
            declaration = PreparedSpeechDeclaration(
                revision,
                tuple(coverage),
                value.get("prepared_native_text"),
                frozenset(SpeechFormat(item) for item in formats),
                frozenset(SpeechControl(item) for item in controls),
                value.get("input_limit"),
                value.get("speed_min"),
                value.get("speed_max"),
            )
        except (ValueError, TypeError):
            _invalid()
    else:
        _invalid()
    if not isinstance(declaration.revision, str) or not _SAFE_REVISION.fullmatch(
        declaration.revision
    ):
        _invalid()
    if not declaration.identity_coverage or any(
        not isinstance(item, str) or not _SAFE_REVISION.fullmatch(item)
        for item in declaration.identity_coverage
    ):
        _invalid()
    if type(declaration.prepared_native_text) is not bool:
        _invalid()
    if not all(
        isinstance(item, SpeechFormat) for item in declaration.formats
    ) or not all(isinstance(item, SpeechControl) for item in declaration.controls):
        _invalid()
    if declaration.input_limit is not None and (
        type(declaration.input_limit) is not int or declaration.input_limit <= 0
    ):
        _invalid()
    if any(
        bound is not None and _number(bound) is None
        for bound in (declaration.speed_min, declaration.speed_max)
    ):
        _invalid()
    if (
        declaration.speed_min is not None
        and declaration.speed_max is not None
        and declaration.speed_min > declaration.speed_max
    ):
        _invalid()
    return declaration


def _endpoint_origin(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _invalid()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        _invalid()
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        _invalid()
    try:
        port = parsed.port
    except ValueError:
        _invalid()
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}" + (f":{port}" if port is not None else "")


def _fact(value: Any, default: Any = None, *, applicable: bool = True):
    if not applicable:
        return SpeechFact(None, SpeechFactSource.INAPPLICABLE)
    if value is not None:
        if _string(value) is None:
            _invalid()
        return SpeechFact(value, SpeechFactSource.CONFIGURED)
    if default is not None:
        return SpeechFact(default, SpeechFactSource.NATIVE_DEFAULT)
    return SpeechFact(None, SpeechFactSource.UNKNOWN)


def _configured_fact(
    value: str | None, section: dict[str, Any], field: str | None, default: str | None
) -> SpeechFact[str]:
    if field is None:
        return SpeechFact(None, SpeechFactSource.INAPPLICABLE)
    if value is not None:
        return _fact(value)
    if field in section:
        if section[field] is None:
            return SpeechFact(None, SpeechFactSource.UNKNOWN)
        return _fact(section[field])
    return _fact(None, default)


def _defaults(provider: str) -> tuple[str | None, str | None]:
    from tools import tts_tool_local as local
    from tools import tts_tool_openai as openai
    from tools import tts_tool_providers as native

    defaults = {
        "edge": (None, native.DEFAULT_EDGE_VOICE),
        "openai": (openai.DEFAULT_OPENAI_MODEL, openai.DEFAULT_OPENAI_VOICE),
        "deepinfra": (None, openai.DEFAULT_DEEPINFRA_TTS_VOICE),
        "elevenlabs": (
            native.DEFAULT_ELEVENLABS_MODEL_ID,
            native.DEFAULT_ELEVENLABS_VOICE_ID,
        ),
        "minimax": (native.DEFAULT_MINIMAX_MODEL, native.DEFAULT_MINIMAX_VOICE_ID),
        "xai": (None, native.DEFAULT_XAI_VOICE_ID),
        "mistral": (
            native.DEFAULT_MISTRAL_TTS_MODEL,
            native.DEFAULT_MISTRAL_TTS_VOICE_ID,
        ),
        "piper": (None, local.DEFAULT_PIPER_VOICE),
        "kittentts": (local.DEFAULT_KITTENTTS_MODEL, local.DEFAULT_KITTENTTS_VOICE),
        "gemini": (native.DEFAULT_GEMINI_TTS_MODEL, native.DEFAULT_GEMINI_TTS_VOICE),
    }
    return defaults.get(provider, (None, None))


def _projection(resolution: SpeechResolution) -> dict[str, Any]:
    value = asdict(resolution)
    value.pop("fingerprint")
    value["settings"].pop("instructions")
    value["controls"] = sorted(resolution.controls)
    if resolution.declaration is not None:
        value["declaration"]["formats"] = sorted(resolution.declaration.formats)
        value["declaration"]["controls"] = sorted(resolution.declaration.controls)
    return value


def _fingerprint(resolution: SpeechResolution) -> SpeechResolution:
    encoded = json.dumps(
        _projection(resolution), sort_keys=True, separators=(",", ":")
    ).encode()
    return replace(resolution, fingerprint=hashlib.sha256(encoded).hexdigest())


def _validate_command_controls(
    section: dict[str, Any], requested: set[SpeechControl]
) -> None:
    from tools.tts_command_provider import render_command_template

    template = section.get("command")
    if not isinstance(template, str):
        _invalid()

    def forwards(field: str) -> bool:
        marker = "__PREPARED_CONTROL_PROBE__"
        rendered = render_command_template(template, {field: marker})
        return rendered.count(marker) > template.count(marker)

    if not any(
        forwards(field) for field in ("input_path", "text_path")
    ) or not forwards("output_path"):
        raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
    for control in requested:
        if not forwards(control.value):
            raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)


def _speed_fact(
    settings: SpeechSettings,
    section: dict[str, Any],
    config: dict[str, Any],
    provider: str,
    builtin: bool,
    controls: frozenset[SpeechControl],
    declaration: PreparedSpeechDeclaration | None,
) -> SpeechFact[float]:
    raw_speed = (
        settings.speed
        if settings.speed is not None
        else section.get("speed", config.get("speed"))
    )
    speed_value = (
        _number(raw_speed) if raw_speed is not None else 1.0 if builtin else None
    )
    if raw_speed is not None and speed_value is None:
        _invalid()
    if SpeechControl.SPEED not in controls:
        speed = SpeechFact(None, SpeechFactSource.INAPPLICABLE)
    else:
        speed = SpeechFact(
            speed_value,
            SpeechFactSource.CONFIGURED
            if raw_speed is not None
            else SpeechFactSource.NATIVE_DEFAULT
            if builtin
            else SpeechFactSource.UNKNOWN,
        )
    bounds = {
        "openai": (0.25, 4.0),
        "deepinfra": (0.25, 4.0),
        "xai": (0.7, 1.5),
        "edge": (0.25, 4.0),
    }.get(provider)
    if settings.speed is not None and declaration is not None:
        if declaration.speed_min is None or declaration.speed_max is None:
            raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
        bounds = (declaration.speed_min, declaration.speed_max)
    if speed_value is not None and SpeechControl.SPEED in controls:
        if bounds is not None and not bounds[0] <= speed_value <= bounds[1]:
            raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
        if provider == "edge" and speed_value != (
            1 + round((speed_value - 1) * 100) / 100
        ):
            raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
    return speed


def _resolve(
    settings: SpeechSettings, config: dict[str, Any], *, plugin=None
) -> SpeechResolution:
    from tools import tts_tool as native
    from tools.tts_command_provider import (
        _get_named_provider_config,
        _is_command_provider_config,
    )

    _validate_settings(settings)
    raw_provider = config.get("provider") or native.DEFAULT_PROVIDER
    if not isinstance(raw_provider, str):
        _invalid()
    configured = raw_provider.lower().strip()
    if not _SAFE_REVISION.fullmatch(configured):
        _invalid()
    provider = native._get_provider(config)
    section = config.get(provider) or {}
    named = _get_named_provider_config(config, provider)
    if not isinstance(section, dict):
        _invalid()
    builtin = provider in native.BUILTIN_TTS_PROVIDERS
    command = not builtin and _is_command_provider_config(named)
    kind = (
        SpeechRouteKind.BUILTIN
        if builtin
        else SpeechRouteKind.COMMAND
        if command
        else SpeechRouteKind.PLUGIN
    )
    declaration = None
    effective_config = _settings_config(config, provider, kind, settings)
    if not builtin:
        section = named if command else config
        operator = (
            _declaration(named.get("prepared")) if isinstance(named, dict) else None
        )
        static = (
            inspect.getattr_static(plugin, "prepared_speech", None)
            if plugin is not None
            else None
        )
        if static is not None and not isinstance(static, PreparedSpeechDeclaration):
            raise SpeechSynthesisError(SpeechFailureCode.UNKNOWN_CAPABILITY)
        supplied = _declaration(static)
        declaration = supplied or operator
        if supplied is not None and operator is not None and supplied != operator:
            raise SpeechSynthesisError(SpeechFailureCode.RESOLUTION_CHANGED)
        if declaration is None or not declaration.prepared_native_text:
            raise SpeechSynthesisError(SpeechFailureCode.UNKNOWN_CAPABILITY)
        if settings.output_format not in declaration.formats:
            raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
        controls = declaration.controls
        model_field, voice_field = "model", "voice"
        defaults = (None, None)
        native_limit = native._resolve_max_text_length(provider, effective_config)
        limit = SpeechFact(
            min(native_limit, declaration.input_limit)
            if declaration.input_limit
            else native_limit,
            SpeechFactSource.DECLARED
            if declaration.input_limit is not None
            and declaration.input_limit < native_limit
            else SpeechFactSource.NATIVE_POLICY,
        )
        source = SpeechFactSource.DECLARED
    else:
        model_field, voice_field, controls = _KNOWN_CONTROLS[provider]
        defaults = _defaults(provider)
        limit = SpeechFact(
            native._resolve_max_text_length(provider, effective_config),
            SpeechFactSource.NATIVE_POLICY,
        )
        source = SpeechFactSource.NATIVE_POLICY
        if provider == "gemini":
            raise SpeechSynthesisError(SpeechFailureCode.UNKNOWN_CAPABILITY)
        if settings.output_format == SpeechFormat.WAV and provider in {
            "edge",
            "elevenlabs",
            "minimax",
        }:
            raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
    if (
        provider == "minimax"
        and section.get("base_url")
        and "t2a_v2" not in section["base_url"]
    ):
        controls = controls - {SpeechControl.SPEED}
    model = _configured_fact(settings.model, section, model_field, defaults[0])
    if provider == "openai" and not _openai_instructions_supported(
        section.get("base_url") or "https://api.openai.com/v1", model.value
    ):
        controls = controls - {SpeechControl.INSTRUCTIONS}
    requested = {
        control
        for control in SpeechControl
        if getattr(settings, control.value) is not None
    }
    if not requested <= controls:
        raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
    if command:
        _validate_command_controls(section, requested)
    voice = _configured_fact(settings.voice, section, voice_field, defaults[1])
    speed = _speed_fact(
        settings, section, config, provider, builtin, controls, declaration
    )
    coverage = _NATIVE_COVERAGE + (declaration.identity_coverage if declaration else ())
    return _fingerprint(
        SpeechResolution(
            settings,
            configured,
            kind,
            SpeechFact(provider, source),
            model,
            voice,
            speed,
            limit,
            controls,
            source,
            SpeechFact(True, source),
            _endpoint_origin(section.get("base_url")),
            declaration,
            coverage,
            "",
        )
    )


def _settings_config(
    config: dict[str, Any],
    provider: str,
    kind: SpeechRouteKind,
    settings: SpeechSettings,
) -> dict[str, Any]:
    from tools.tts_command_provider import _get_named_provider_config

    copied = copy.deepcopy(config)
    model_field, voice_field, _ = _KNOWN_CONTROLS.get(
        provider, ("model", "voice", frozenset())
    )
    if kind == SpeechRouteKind.COMMAND:
        section = _get_named_provider_config(copied, provider)
    elif kind == SpeechRouteKind.PLUGIN:
        section = copied
    else:
        section = copied.setdefault(provider, {})
    for field, value in (
        (model_field, settings.model),
        (voice_field, settings.voice),
        ("speed", settings.speed),
    ):
        if field and value is not None:
            section[field] = value
    copied["output_format"] = settings.output_format.value
    if provider == "xai":
        section["auto_speech_tags"] = False
        section["speech_tags"] = False
        section["text_normalization"] = False
    if provider == "kittentts":
        section["clean_text"] = False
    return copied


def _apply_settings(
    config: dict[str, Any], resolution: SpeechResolution
) -> dict[str, Any]:
    provider = resolution.engine.value
    if provider is None:
        raise SpeechSynthesisError(SpeechFailureCode.UNKNOWN_CAPABILITY)
    return _settings_config(config, provider, resolution.kind, resolution.settings)


def _openai_instructions_supported(base_url: str, model: str | None) -> bool:
    parsed = urlsplit(base_url)
    return (
        _endpoint_origin(base_url) == "https://api.openai.com"
        and parsed.path.rstrip("/") == "/v1"
        and model == "gpt-4o-mini-tts"
    )


def _validate_openai_submission(
    resolution: SpeechResolution, base_url: str, model: str, voice: str, speed: float
) -> None:
    if (
        resolution.endpoint_origin is not None
        and _endpoint_origin(base_url) != resolution.endpoint_origin
    ):
        raise SpeechSynthesisError(SpeechFailureCode.RESOLUTION_CHANGED)
    if (
        resolution.settings.instructions is not None
        and not _openai_instructions_supported(base_url, model)
    ):
        raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
    for requested, actual in (
        (resolution.model.value, model),
        (resolution.voice.value, voice),
        (resolution.speed.value, speed),
    ):
        if requested is not None and requested != actual:
            raise SpeechSynthesisError(SpeechFailureCode.UNSUPPORTED_CONTROL)
