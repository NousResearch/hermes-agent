"""Gateway-safe wrappers for Wisdom Kernel."""

from __future__ import annotations

import logging
from typing import Any

from wisdom.capture import capture_text
from wisdom.classify import detect_explicit_trigger
from wisdom.commands import WisdomCommandContext, can_natural_capture, handle_wisdom_command
from wisdom.config import load_wisdom_config
from wisdom.db import WisdomDB
from wisdom.redaction import detect_secret_like_text, redact_for_log
from wisdom.render import render_capture, render_error


logger = logging.getLogger(__name__)


async def handle_gateway_command(event: Any, gateway: Any) -> str:
    try:
        source = getattr(event, "source", None)
        source_kind = _source_kind(source, "command")
        config = load_wisdom_config()
        db = WisdomDB(config.db_path)
        context = WisdomCommandContext(
            channel=_platform_name(source) or "gateway",
            source_kind=source_kind,
            session_key=_session_key(gateway, source),
            message_ref=getattr(event, "message_id", None),
        )
        return handle_wisdom_command(event.get_command_args(), context=context, config=config, db=db)
    except Exception as exc:
        logger.warning("Wisdom command failed: %s", redact_for_log(str(exc)))
        return render_error()


async def maybe_capture_gateway_event(
    event: Any,
    source: Any,
    *,
    session_key: object | None = None,
    gateway: Any | None = None,
) -> str | None:
    try:
        text = getattr(event, "text", "") or ""
        if not text or text.lstrip().startswith("/"):
            return None
        trigger = detect_explicit_trigger(text)
        if trigger is None:
            return None
        if detect_secret_like_text(text):
            return None

        config = load_wisdom_config()
        db = WisdomDB(config.db_path)
        if not can_natural_capture(db, config):
            return None

        outcome = capture_text(
            text,
            channel=_platform_name(source) or "gateway",
            source_kind=_source_kind(source, "natural"),
            session_key=session_key if session_key is not None else _session_key(gateway, source),
            message_ref=getattr(event, "message_id", None),
            metadata={"platform": _platform_name(source) or "gateway"},
            config=config,
            db=db,
            cleaned_text=trigger.cleaned_text,
            require_enabled=True,
        )
        if outcome.status == "captured" and outcome.capture:
            return render_capture(outcome.capture)
        return None
    except Exception as exc:
        logger.warning("Wisdom natural capture failed open: %s", redact_for_log(str(exc)))
        return None


def _platform_name(source: Any) -> str:
    platform = getattr(source, "platform", None)
    return str(getattr(platform, "value", platform) or "").lower()


def _source_kind(source: Any, suffix: str) -> str:
    platform = _platform_name(source) or "gateway"
    return f"{platform}_{suffix}"


def _session_key(gateway: Any, source: Any) -> object | None:
    if gateway is None or source is None:
        return None
    try:
        return gateway._session_key_for_source(source)
    except Exception:
        return None
