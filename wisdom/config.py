"""Configuration resolution for Hermes Wisdom Kernel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from wisdom.models import WisdomConfig


VALID_CAPTURE_MODES = {"off", "explicit"}
VALID_INTERPRETATION_MODES = {"deterministic", "llm"}
VALID_APPLICATION_MODES = {"deterministic", "llm"}


def default_db_path() -> Path:
    return get_hermes_home() / "wisdom" / "wisdom.db"


def default_wisdom_dir() -> Path:
    return default_db_path().parent


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _int_value(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _float_value(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _raw_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config.get("wisdom", config) or {})
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
    except Exception:
        loaded = {}
    if isinstance(loaded, dict):
        return dict(loaded.get("wisdom", {}) or {})
    return {}


def load_wisdom_config(config: dict[str, Any] | None = None) -> WisdomConfig:
    raw = _raw_config(config)

    enabled = _bool_value(raw.get("enabled"), True)
    capture_mode = str(raw.get("capture_mode") or "explicit").strip().lower()
    if capture_mode not in VALID_CAPTURE_MODES:
        capture_mode = "explicit"

    db_value = str(raw.get("db_path") or "").strip()
    db_path = Path(os.path.expanduser(db_value)) if db_value else default_db_path()

    interpretation = raw.get("interpretation", {}) if isinstance(raw.get("interpretation"), dict) else {}
    interpretation_mode = str(interpretation.get("mode") or "deterministic").strip().lower()
    if interpretation_mode not in VALID_INTERPRETATION_MODES:
        interpretation_mode = "deterministic"

    application = raw.get("application", {}) if isinstance(raw.get("application"), dict) else {}
    application_mode = str(application.get("mode") or "deterministic").strip().lower()
    if application_mode not in VALID_APPLICATION_MODES:
        application_mode = "deterministic"

    result = WisdomConfig(
        enabled=enabled,
        db_path=db_path,
        capture_mode=capture_mode,
        max_results=_int_value(raw.get("max_results"), 5, 1, 50),
        interpret_timeout_seconds=_float_value(
            raw.get("interpret_timeout_seconds"), 5.0, 0.5, 60.0
        ),
        interpretation_mode=interpretation_mode,
        application_mode=application_mode,
        apply_timeout_seconds=_float_value(
            application.get("timeout_seconds"),
            30.0,
            1.0,
            180.0,
        ),
    )
    return _apply_env_overrides(result)


def _apply_env_overrides(config: WisdomConfig) -> WisdomConfig:
    enabled = _bool_value(os.environ.get("HERMES_WISDOM_ENABLED"), config.enabled)
    db_path = config.db_path
    env_db = os.environ.get("HERMES_WISDOM_DB_PATH", "").strip()
    if env_db:
        db_path = Path(os.path.expanduser(env_db))

    capture_mode = os.environ.get("HERMES_WISDOM_CAPTURE_MODE", config.capture_mode).strip().lower()
    if capture_mode not in VALID_CAPTURE_MODES:
        capture_mode = config.capture_mode

    interpretation_mode = os.environ.get(
        "HERMES_WISDOM_INTERPRETATION_MODE", config.interpretation_mode
    ).strip().lower()
    if interpretation_mode not in VALID_INTERPRETATION_MODES:
        interpretation_mode = config.interpretation_mode

    application_mode = os.environ.get(
        "HERMES_WISDOM_APPLICATION_MODE", config.application_mode
    ).strip().lower()
    if application_mode not in VALID_APPLICATION_MODES:
        application_mode = config.application_mode

    return WisdomConfig(
        enabled=enabled,
        db_path=db_path,
        capture_mode=capture_mode,
        max_results=_int_value(os.environ.get("HERMES_WISDOM_MAX_RESULTS"), config.max_results, 1, 50),
        interpret_timeout_seconds=_float_value(
            os.environ.get("HERMES_WISDOM_INTERPRET_TIMEOUT"),
            config.interpret_timeout_seconds,
            0.5,
            60.0,
        ),
        interpretation_mode=interpretation_mode,
        application_mode=application_mode,
        apply_timeout_seconds=_float_value(
            os.environ.get("HERMES_WISDOM_APPLY_TIMEOUT"),
            config.apply_timeout_seconds,
            1.0,
            180.0,
        ),
    )
