from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_SECRET_PATTERNS = (
    r"telegram_bot_token",
    r"openrouter_api_key",
    r"google_client_secret",
    r"onepassword_token",
    r"Authorization",
    r"Bearer [A-Za-z0-9\-_\.]+",
    r"api[_-]?key",
    r"secret",
)


def _sanitize(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = re.sub(pattern, "***REDACTED***", text, flags=re.IGNORECASE)
    return text


class TelegramDeliveryError(Exception):
    pass


def _get_home_channel() -> str:
    return os.environ.get("HERMES_TELEGRAM_HOME_CHANNEL") or "telegram"


def _call_send_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Use the Hermes send_message command path
        from hermes_cli.send_cmd import send_command
    except Exception as exc:
        raise TelegramDeliveryError(f"send path unavailable: {exc}") from exc
    try:
        result = send_command(json.dumps(payload))
    except Exception as exc:
        raise TelegramDeliveryError(f"send runtime error: {exc}") from exc
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    return result if isinstance(result, dict) else {"raw": str(result)}


async def _send_with_retry(payload: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    last = None
    for attempt in range(max_retries + 1):
        try:
            result = _call_send_message(payload)
            sanitized = {k: (_sanitize(str(v)) if isinstance(v, str) else v) for k, v in result.items()}
            logger.info("Telegram send attempt %d succeeded: %s", attempt + 1, sanitized)
            return result
        except TelegramDeliveryError as exc:
            last = exc
            if attempt < max_retries:
                logger.warning("Telegram send failed attempt %d/%d: %s", attempt + 1, max_retries + 1, _sanitize(str(exc)))
                await asyncio.sleep(2 ** attempt)
            else:
                break
    raise last or TelegramDeliveryError("unknown telegram send failure")


async def send_text(message: str, target: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "action": "send",
        "target": target or _get_home_channel(),
        "message": _sanitize(message),
    }
    return await _send_with_retry(payload)


async def deliver_brief(brief: Dict[str, Any], target: Optional[str] = None) -> Dict[str, Any]:
    message = _render_brief(brief)
    try:
        if len(message) > 3800:
            first = message[:3800]
            rest = message[3800:]
            a1 = await send_text(first, target=target)
            a2 = await send_text(rest or "(continued)", target=target)
            return {"success": True, "chunks": 2, "results": [a1, a2], "duplicate_key": None}
        result = await send_text(message, target=target)
        return {"success": True, "chunks": 1, "results": [result], "duplicate_key": None}
    except TelegramDeliveryError as exc:
        sanitized = _sanitize(str(exc))
        logger.error("Brief delivery failed: %s", sanitized)
        return {"success": False, "error": sanitized}


def _render_brief(brief: Dict[str, Any]) -> str:
    lines = [
        "📋 *Executive Briefing*",
        f"*Execution ID:* `{brief.get('execution_id')}`",
        f"*Generated:* `{brief.get('generated_at')}`",
        "",
    ]

    executive_summary = brief.get("executive_summary")
    if executive_summary:
        lines += [
            "*Executive Summary*",
            f"{executive_summary.get('text', '')} [`{executive_summary.get('provenance', '')}`]",
            "",
        ]

    section_emojis = {
        "calendar": "📅",
        "blackgold_priority_items": "💼",
        "deal_pipeline": "🔁",
        "approval_queue": "✅",
        "banking_and_gci": "🏦",
        "portfolio_alerts": "📊",
        "market_intelligence": "🌐",
        "ai_workforce_status": "🤖",
        "infrastructure_health": "⚙️",
        "failed_jobs": "❌",
        "financial_alerts": "💸",
        "weather": "🌦",
        "scripture": "📖",
        "top_executive_priorities": "🎯",
    }
    for section in brief.get("sections", []):
        block = brief.get(section, {})
        data = block.get("data")
        if data in (None, {}, "") and block.get("provenance") == "unavailable":
            lines.append(f"{section_emojis.get(section, '•')} *{section.replace('_', ' ').title()}* [`unavailable`]")
            continue
        provenance = block.get("provenance", "unavailable")
        lines.append(f"{section_emojis.get(section, '•')} *{section.replace('_', ' ').title()}* [`{provenance}`]")
        lines.append(_format_value(data))
        lines.append("")

    lines.append("_Provenance: live | cached | synthetic | pending | unavailable_")
    return "\n".join(lines)


def _format_value(data: Any) -> str:
    if data is None:
        return "_unavailable_"
    if isinstance(data, dict):
        out = []
        for key, value in data.items():
            out.append(f"• {key}: {_format_value(value)}")
        return "\n".join(out)
    if isinstance(data, list):
        return "\n".join([f"- {_format_value(item)}" for item in data])
    return str(data)
