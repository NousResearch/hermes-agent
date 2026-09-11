"""Trusted product notifications; not part of the model-facing message schema."""

import html
import inspect
import logging
import re

from agent.secret_scope import get_secret
from tools.send_message_senders import _error, _sanitize_error_text, _send_telegram

logger = logging.getLogger("tools.send_message_tool")


def telegram_notification_markup(*, action_button_rows=None, action_buttons=None, url_buttons=None):
    """Build an inline keyboard from already validated internal controls."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    rows = action_button_rows or [[button] for button in (action_buttons or url_buttons or [])]
    if not rows:
        return None
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            str(button["label"]),
            **({"url": str(button["url"])} if button.get("url") else {"callback_data": str(button["callback_data"])}),
        ) for button in row]
        for row in rows if row
    ])


async def try_telegram_rich_notification(bot, chat_id, rich_html, thread_kwargs, *, disable_link_previews=False):
    """Return a receipt or error; None permits fallback only after definite rejection."""
    raw_request = getattr(bot, "do_api_request", None)
    if not inspect.iscoroutinefunction(raw_request):
        return None
    payload = {"chat_id": chat_id, "rich_message": {"html": rich_html}, **thread_kwargs}
    if disable_link_previews:
        payload["link_preview_options"] = {"is_disabled": True}
    try:
        result = await raw_request("sendRichMessage", api_kwargs=payload)
    except Exception as exc:
        error_text = str(exc).lower()
        permanent_rejection = (
            exc.__class__.__name__.lower() in {"badrequest", "endpointnotfound"}
            or getattr(exc, "error_code", None) in {400, 404}
            or isinstance(exc, (AttributeError, TypeError, NotImplementedError))
            or (("method" in error_text or "endpoint" in error_text)
                and ("not found" in error_text or "does not exist" in error_text))
            or "unsupported" in error_text
            or "not implemented" in error_text
        )
        if not permanent_rejection:
            logger.warning("Telegram rich notification failed; not resending to avoid a duplicate: %s",
                           _sanitize_error_text(exc))
            return _error(f"Telegram rich notification failed: {_sanitize_error_text(exc)}")
        logger.debug("Telegram rejected rich notification; falling back to inline keyboard: %s",
                     _sanitize_error_text(exc))
        return None
    if isinstance(result, dict):
        message_id = result.get("message_id")
        if message_id is None:
            message_id = (result.get("result") or {}).get("message_id")
    else:
        message_id = getattr(result, "message_id", None)
    return {"success": True, "message_id": str(message_id) if message_id is not None else None}


def send_telegram_notification_pane(
    *,
    message: str,
    button_rows: list[list[dict[str, str]]],
    items: list[dict[str, object]] | None = None,
) -> dict:
    """Send an internal notification to the Telegram home chat with trusted actions.

    This is intentionally not part of the model-facing ``send_message`` schema.
    Product notifications can offer trusted managed actions and Portal deep
    links without giving an arbitrary tool caller a second, Telegram-specific
    message surface. Rows are explicit so each skill keeps its action and link
    together. Bot API 10.3 clients render the trusted controls directly inside
    each rich-message item; the compact inline keyboard is retained as a safe
    fallback for older Bot API servers and clients.
    """
    from tools.send_message_tool import prepare_send_message_platforms

    prepare_send_message_platforms()
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        platform = Platform.TELEGRAM
        pconfig = config.platforms.get(platform)
        if not pconfig or not pconfig.enabled:
            return _error("Telegram is not configured")
        home = config.get_home_channel(platform)
        if not home:
            return _error("Telegram has no configured home channel")

        safe_rows: list[list[dict[str, str]]] = []
        safe_rows_by_item: list[list[dict[str, str]]] = []
        button_count = 0
        for row in button_rows[:8]:
            safe_row: list[dict[str, str]] = []
            for button in row[:2]:
                if button_count >= 16:
                    break
                label = str(button.get("label") or "").strip()[:32]
                url = str(button.get("url") or "").strip()
                if label and re.fullmatch(r"https?://[^\s]+", url):
                    safe_row.append({"label": label, "url": url})
                    button_count += 1
                    continue
                callback_data = str(button.get("callback_data") or "").strip()
                if (
                    label
                    and len(callback_data.encode("utf-8")) <= 64
                    and re.fullmatch(
                        r"wi:(?:plan:(?:install|update):[A-Za-z0-9_-]+|"
                        r"confirm:(?:install|update):w(?:ip|up)_[a-f0-9]+|cancel)",
                        callback_data,
                    )
                ):
                    safe_row.append({
                        "label": label,
                        "callback_data": callback_data,
                    })
                    button_count += 1
            safe_rows_by_item.append(safe_row)
            if safe_row:
                safe_rows.append(safe_row)

        rich_message_html = None
        if items:
            rich_item_parts: list[str] = []
            for item, item_buttons in zip(items[:8], safe_rows_by_item):
                heading = html.escape(str(item.get("heading") or "").strip())
                detail = html.escape(str(item.get("detail") or "").strip())
                detail = detail.replace("\r\n", "\n").replace("\r", "\n")
                detail = detail.replace("\n", "<br/>")
                if not heading or not detail:
                    continue
                controls: list[str] = []
                for button in item_buttons:
                    label = html.escape(button["label"])
                    if button.get("url"):
                        controls.append(
                            '<tg-button type="url" url="'
                            f'{html.escape(button["url"], quote=True)}">'
                            f"{label}</tg-button>"
                        )
                    else:
                        controls.append(
                            '<tg-button type="callback_data" style="primary" data="'
                            f'{html.escape(button["callback_data"], quote=True)}">'
                            f"{label}</tg-button>"
                        )
                control_html = (
                    f'<tg-button-row align="left">'
                    f"{' '.join(controls)}"
                    "</tg-button-row>"
                    if controls
                    else ""
                )
                rich_item_parts.append(
                    f"<p><b>{heading}</b><br/>{detail}</p>{control_html}"
                )
            if rich_item_parts:
                item_count = len(rich_item_parts)
                rich_parts = [
                    # Telegram owns rich-message bubble sizing and exposes no
                    # width/min-width control. Keep a stable, naturally wider
                    # product heading so short skill names and one-word actions
                    # do not collapse these notifications into narrow cards.
                    "<h3>Hermes Collective Wisdom</h3>",
                    (
                        f"<p>{item_count} new "
                        f"{'notification' if item_count == 1 else 'notifications'}</p>"
                    ),
                    *rich_item_parts,
                ]
                rich_message_html = "".join(rich_parts)

        from model_tools import _run_async

        result = _run_async(
            _send_telegram(
                pconfig.token,
                home.chat_id,
                message,
                disable_link_previews=True,
                action_button_rows=safe_rows,
                rich_message_html=rich_message_html,
            )
        )
        return result if isinstance(result, dict) else {"success": bool(result)}
    except Exception as exc:
        return _error(f"Telegram notification failed: {exc}")


def send_slack_wisdom_notification_pane(
    *,
    message: str,
    button_rows: list[list[dict[str, str]]],
    items: list[dict[str, object]],
) -> dict:
    """Send a trusted Collective Wisdom Block Kit pane to Slack's home chat.

    This internal surface deliberately accepts only Wisdom's compact managed
    callback grammar and HTTPS Portal links. It is not exposed through the
    model-facing ``send_message`` tool.
    """
    from tools.send_message_tool import prepare_send_message_platforms

    prepare_send_message_platforms()
    try:
        from gateway.config import Platform, load_gateway_config
        from model_tools import _run_async

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.SLACK)
        if not pconfig or not pconfig.enabled:
            return _error("Slack is not configured")
        home = config.get_home_channel(Platform.SLACK)
        if not home:
            return _error("Slack has no configured home channel")

        blocks: list[dict[str, object]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Hermes Collective Wisdom",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{len(items[:8])} new "
                        f"{'update' if len(items[:8]) == 1 else 'updates'}"
                    ),
                },
            },
        ]
        for item_index, (item, row) in enumerate(
            zip(items[:8], button_rows[:8])
        ):
            heading = str(item.get("heading") or "").strip()[:150]
            detail = str(item.get("detail") or "").strip()[:2600]
            if not heading or not detail:
                continue
            safe_heading = heading.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe_detail = detail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{safe_heading}*\n{safe_detail}",
                    },
                }
            )
            buttons: list[dict[str, object]] = []
            for button_index, button in enumerate(row[:2]):
                label = str(button.get("label") or "").strip()[:75]
                if not label:
                    continue
                element: dict[str, object] = {
                    "type": "button",
                    "text": {"type": "plain_text", "text": label, "emoji": True},
                    "action_id": (
                        f"hermes_wisdom_feed_{item_index}_{button_index}"
                    ),
                }
                url = str(button.get("url") or "").strip()
                # These are trusted product links assembled by Wisdom, not
                # model-authored buttons. Permit HTTP as well so the local
                # demo Portal retains the same View action as production.
                if re.fullmatch(r"https?://[^\s]+", url):
                    element["url"] = url
                    element["value"] = "wisdom:portal"
                else:
                    callback_data = str(button.get("callback_data") or "").strip()
                    if not re.fullmatch(
                        r"wi:plan:(?:install|update):[A-Za-z0-9_-]+",
                        callback_data,
                    ):
                        continue
                    element["value"] = callback_data
                    element["style"] = "primary"
                buttons.append(element)
            if buttons:
                blocks.append({"type": "actions", "elements": buttons})

        async def send() -> dict:
            try:
                from slack_sdk.web.async_client import AsyncWebClient
            except ImportError:
                return _error("slack_sdk is not installed")
            raw_token = getattr(pconfig, "token", None) or get_secret(
                "SLACK_BOT_TOKEN", ""
            )
            tokens = [
                token.strip()
                for token in str(raw_token or "").split(",")
                if token.strip()
            ]
            if not tokens:
                return _error("Slack bot token is not configured")
            last_error = "unknown"
            for token in tokens:
                try:
                    client = AsyncWebClient(token=token)
                    if home.thread_id:
                        response = await client.chat_postMessage(
                            channel=home.chat_id,
                            text=message[:39000],
                            blocks=blocks[:50],
                            unfurl_links=False,
                            unfurl_media=False,
                            thread_ts=str(home.thread_id),
                        )
                    else:
                        response = await client.chat_postMessage(
                            channel=home.chat_id,
                            text=message[:39000],
                            blocks=blocks[:50],
                            unfurl_links=False,
                            unfurl_media=False,
                        )
                    payload = getattr(response, "data", response)
                    if isinstance(payload, dict) and payload.get("ok", True):
                        return {
                            "success": True,
                            "message_id": payload.get("ts"),
                            "private": str(home.chat_id).startswith("D"),
                        }
                    if isinstance(payload, dict):
                        last_error = str(payload.get("error") or last_error)
                except Exception as exc:
                    last_error = str(exc)
            return _error(f"Slack notification failed: {last_error}")

        result = _run_async(send())
        return result if isinstance(result, dict) else {"success": bool(result)}
    except Exception as exc:
        return _error(f"Slack notification failed: {exc}")
