"""Standalone per-platform senders and error helpers for send_message."""

import asyncio
import contextlib
import logging
import os
import re
import time

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.send_message_tool")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".m2a", ".wav", ".m4a", ".flac"}
_VOICE_EXTS = {".ogg", ".opus"}
_TELEGRAM_SEND_AUDIO_EXTS = {".mp3", ".m4a"}  # sendAudio accepts only these; other audio -> sendVoice / document
# Captionable on the media bubble; voice/audio notes excluded (a caption there reads as a separate label).
_CAPTIONABLE_EXTS = _IMAGE_EXTS | _VIDEO_EXTS | {".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".xlsx", ".zip"}
# Native caption limits (chars): Telegram caps photo/video at 1024; one conservative shared ceiling elsewhere.
_TELEGRAM_CAPTION_LIMIT = 1024
_DEFAULT_CAPTION_LIMIT = 4096


def _media_caption_split(text, media_files, *, max_caption_len):
    """Single chokepoint deciding whether text rides on the media bubble as its caption:
    ``(caption, "")`` only for exactly one captionable file (not a voice/audio note) whose
    text fits ``max_caption_len``, else ``(None, text)`` — multi-file caption→file association
    is ambiguous. Length is codepoints (never under-counts Telegram's UTF-16 units for BMP
    text); the Telegram sender re-checks the *formatted* caption since escaping inflates it."""
    stripped = (text or "").strip()
    media = media_files or []
    if (not stripped or len(media) != 1 or len(stripped) > max_caption_len or media[0][1]
            or os.path.splitext(media[0][0])[1].lower() not in _CAPTIONABLE_EXTS):
        return None, text
    return stripped, ""


_URL_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig)=)([^&#\s]+)", re.IGNORECASE)
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"\b(access_token|api[_-]?key|auth[_-]?token|signature|sig)\s*=\s*([^\s,;]+)", re.IGNORECASE)


def _sanitize_error_text(text) -> str:
    """Redact secrets from error text before surfacing it to users/models."""
    redacted = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}***", redact_sensitive_text(text))
    return _GENERIC_SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=***", redacted)


def _error(message: str) -> dict:
    """Build a standardized error payload with redacted content."""
    return {"error": _sanitize_error_text(message)}


def _success(platform: str, chat_id, warnings=None, **fields) -> dict:
    """Standard success payload; ``warnings`` is only included when non-empty."""
    return {"success": True, "platform": platform, "chat_id": chat_id, **fields,
            **({"warnings": warnings} if warnings else {})}


_NO_DELIVERABLE = "No deliverable text or media remained after processing MEDIA tags"

_TELEGRAM_TRANSIENT_MARKERS = ("bad gateway", "502", "too many requests", "429", "service unavailable", "503",
                               "gateway timeout", "504")


def _telegram_retry_delay(exc: Exception, attempt: int) -> float | None:
    """Retry delay in seconds, or None when final: honours ``retry_after``; timeouts are
    never retried (the send may have gone through); 5xx/429 back off exponentially."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            return 1.0
    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return None
    return float(2 ** attempt) if any(marker in text for marker in _TELEGRAM_TRANSIENT_MARKERS) else None


async def _send_telegram_message_with_retry(bot, *, attempts: int = 3, **kwargs):
    """``bot.send_message`` with bounded retries on transient failures."""
    for attempt in range(attempts):
        try:
            return await bot.send_message(**kwargs)
        except Exception as exc:
            delay = _telegram_retry_delay(exc, attempt)
            if delay is None or attempt >= attempts - 1:
                raise
            logger.warning("Transient Telegram send failure (attempt %d/%d), retrying in %.1fs: %s",
                           attempt + 1, attempts, delay, _sanitize_error_text(exc))
            await asyncio.sleep(delay)


def _is_telegram_thread_not_found(error: Exception) -> bool:
    """Mirror of the gateway adapter's ``_is_thread_not_found_error``.

    Matches the gateway adapter's ``_is_thread_not_found_error`` for the standalone ``_send_telegram`` path
    (issue #27012).
    """
    return "thread not found" in str(error).lower()


def _telegram_bot(token):
    """Bot honouring TELEGRAM_PROXY (standalone sends time out where api.telegram.org is
    blocked); falls back to a direct connection."""
    from telegram import Bot
    try:
        from gateway.platforms.base import resolve_proxy_url
        proxy = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org"])
        if not proxy:
            return Bot(token=token)
        from telegram.request import HTTPXRequest
        logger.info("send_message: standalone Telegram send routed through proxy %s", proxy)
        return Bot(token=token, request=HTTPXRequest(proxy=proxy), get_updates_request=HTTPXRequest(proxy=proxy))
    except Exception as proxy_err:
        logger.warning("send_message: failed to attach Telegram proxy (%s), falling back to direct connection", proxy_err)
    return Bot(token=token)


def _telegram_thread_kwargs(thread_id):
    """Topic id -> ``message_thread_id`` kwargs. Forum "General" is thread "1" inbound but
    the Bot API rejects message_thread_id=1, so it maps to no thread — same as the adapter."""
    if thread_id is None:
        return {}
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
        effective = TelegramAdapter._message_thread_id_for_send(str(thread_id))
    except Exception:  # adapter import failed (python-telegram-bot missing): explicit mapping
        effective = None if str(thread_id) == "1" else int(thread_id)
    return {} if effective is None else {"message_thread_id": effective}


def _strip_mdv2_safe(text):
    """Strip MarkdownV2 escapes for the plain-text fallback; identity if unavailable."""
    try:
        from plugins.platforms.telegram.adapter import _strip_mdv2
        return _strip_mdv2(text)
    except Exception:
        return text


def _adapter_media_method(ext, voice, force_document=False):
    """``(adapter method, kind)``: document when forced, else image / video / voice by
    extension (``voice`` already folds in the caller's audio rule)."""
    if force_document:
        return "send_document", "document"
    if ext in _IMAGE_EXTS:
        return "send_image_file", "image"
    if ext in _VIDEO_EXTS:
        return "send_video", "video"
    return ("send_voice", "audio") if voice else ("send_document", "document")


async def _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **kwargs):
    """Bot API media method by extension: photo (unless forced document), video, voice note,
    sendAudio (MP3/M4A only), else document."""
    kind = next((k for exts, k in ((() if force_document else _IMAGE_EXTS, "photo"), (_VIDEO_EXTS, "video"),
                                    (_VOICE_EXTS if is_voice else (), "voice"), (_TELEGRAM_SEND_AUDIO_EXTS, "audio"))
                 if ext in exts), "document")
    return await getattr(bot, f"send_{kind}")(chat_id=chat_id, **{kind: f}, **kwargs)


async def _telegram_send_text_chunk(bot, chat_id, chunk, parse_mode, has_html, text_kwargs):
    """One text chunk with adapter-matching fallbacks: thread-not-found -> retry without
    ``message_thread_id`` (dropped from ``text_kwargs`` for later chunks too); parse failure
    -> plain text."""
    async def send(text, mode):
        return await _send_telegram_message_with_retry(bot, chat_id=chat_id, text=text, parse_mode=mode, **text_kwargs)
    try:
        return await send(chunk, parse_mode)
    except Exception as md_error:
        # Thread not found — retry without message_thread_id so the message still delivers (matching the
        # gateway adapter's fallback behaviour, issue #27012).
        if _is_telegram_thread_not_found(md_error) and text_kwargs.get("message_thread_id") is not None:
            logger.warning("Thread %s not found in _send_telegram, retrying without message_thread_id",
                           text_kwargs.pop("message_thread_id"))
            return await send(chunk, parse_mode)
        err_text = str(md_error).lower()
        if "parse" in err_text or "markdown" in err_text or "html" in err_text:
            logger.warning("Parse mode %s failed in _send_telegram, falling back to plain text: %s",
                           parse_mode, _sanitize_error_text(md_error))
            return await send(chunk if has_html else _strip_mdv2_safe(chunk), None)
        raise


async def _telegram_send_one_media(bot, chat_id, media_path, is_voice, *, caption, parse_mode, has_html,
                                   thread_kwargs, force_document):
    """Upload one file with adapter-matching fallbacks (thread-not-found -> no
    ``message_thread_id``; caption parse failure -> plain caption); retries re-seek the file."""
    ext = os.path.splitext(media_path)[1].lower()
    voice_note = ext in _VOICE_EXTS and is_voice
    # ``caption`` is only set for a single captionable file, so this never double-captions
    # a multi-file send or a voice note.
    media_kwargs = {**thread_kwargs, **({"caption": caption, "parse_mode": parse_mode}
                                        if caption is not None and not voice_note else {})}
    if voice_note or ext in _TELEGRAM_SEND_AUDIO_EXTS:
        with contextlib.suppress(Exception):
            from plugins.platforms.telegram.adapter import _probe_voice_duration_seconds
            duration = await asyncio.to_thread(_probe_voice_duration_seconds, media_path)
            if duration is not None:
                media_kwargs["duration"] = duration
    with open(media_path, "rb") as f:
        try:
            return await _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **media_kwargs)
        except Exception as media_err:
            err_text = str(media_err).lower()
            if _is_telegram_thread_not_found(media_err) and media_kwargs.get("message_thread_id"):
                logger.warning("Thread %s not found for media send, retrying without message_thread_id",
                               media_kwargs.pop("message_thread_id"))
            elif media_kwargs.get("parse_mode") and ("parse" in err_text or "caption" in err_text):
                logger.warning("Caption parse failed for media send, retrying plain: %s",
                               _sanitize_error_text(media_err))
                media_kwargs.pop("parse_mode", None)
                if not has_html and media_kwargs.get("caption"):
                    media_kwargs["caption"] = _strip_mdv2_safe(media_kwargs["caption"])
            else:
                raise
            f.seek(0)
            return await _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **media_kwargs)


def _telegram_format(message):
    """``(formatted, parse_mode, has_html)``: text already containing HTML tags is sent as
    HTML; otherwise Markdown -> MarkdownV2 via the adapter's ``format_message``."""
    from telegram.constants import ParseMode
    if re.search(r'<[a-zA-Z/][^>]*>', message):
        return message, ParseMode.HTML, True
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
        return TelegramAdapter.__new__(TelegramAdapter).format_message(message), ParseMode.MARKDOWN_V2, False
    except Exception:
        return message, ParseMode.MARKDOWN_V2, False  # formatting unavailable: send as-is


async def _send_telegram(token, chat_id, message, media_files=None, thread_id=None, disable_link_previews=False, force_document=False):
    """One-shot Telegram Bot API send; parse failures fall back to plain text."""
    try:
        formatted, send_parse_mode, _has_html = _telegram_format(message)
        bot = _telegram_bot(token)
        from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
        from gateway.platforms.base import BasePlatformAdapter, utf16_len
        # Telegram accepts a numeric chat_id OR an @username string; never force-int.
        # See #13206.
        int_chat_id = normalize_telegram_chat_id(chat_id)
        media_files = media_files or []
        thread_kwargs = _telegram_thread_kwargs(thread_id)
        # disable_web_page_preview is only valid for send_message, not media sends.
        text_kwargs = {**thread_kwargs, **({"disable_web_page_preview": True} if disable_link_previews else {})}
        last_msg, warnings, _tg_caption = None, [], None
        # MEDIA caption rides on the bubble as its *formatted* caption; formatting can inflate a
        # raw <1024 string past Telegram's cap, so re-check in UTF-16 units.
        _cap, _ = _media_caption_split(message, media_files, max_caption_len=_TELEGRAM_CAPTION_LIMIT)
        if _cap is not None and utf16_len(formatted) <= _TELEGRAM_CAPTION_LIMIT:
            _tg_caption, formatted = formatted, ""  # suppress the separate text send below
        # Chunk *after* formatting, in UTF-16 units: escaping can push a raw-<4096 message over.
        for chunk in BasePlatformAdapter.truncate_message(formatted, 4096, len_fn=utf16_len) if formatted.strip() else ():
            last_msg = await _telegram_send_text_chunk(bot, int_chat_id, chunk, send_parse_mode, _has_html, text_kwargs)
        for media_path, is_voice in media_files:
            if not os.path.exists(media_path):
                warnings.append(f"Media file not found, skipping: {media_path}")
                logger.warning(warnings[-1])
                # Caption mode suppressed the text send; the file is gone, so deliver the words alone.
                if _tg_caption is not None and last_msg is None:
                    try:
                        last_msg = await _send_telegram_message_with_retry(
                            bot, chat_id=int_chat_id, text=_tg_caption, parse_mode=send_parse_mode, **text_kwargs)
                        _tg_caption = None  # delivered — don't re-caption a later file
                    except Exception as _cap_err:
                        logger.warning("Telegram caption-fallback send failed for missing media: %s",
                                       _sanitize_error_text(_cap_err))
                continue
            try:
                last_msg = await _telegram_send_one_media(
                    bot, int_chat_id, media_path, is_voice, caption=_tg_caption, parse_mode=send_parse_mode,
                    has_html=_has_html, thread_kwargs=thread_kwargs, force_document=force_document)
            except Exception as e:
                warnings.append(_sanitize_error_text(f"Failed to send media {media_path}: {e}"))
                logger.error(warnings[-1])
        if last_msg is None:
            return {"error": _NO_DELIVERABLE, **({"warnings": warnings} if warnings else {})}
        return _success("telegram", chat_id, warnings, message_id=str(last_msg.message_id))
    except ImportError:
        return {"error": "python-telegram-bot not installed. Run: pip install python-telegram-bot"}
    except Exception as e:
        return _error(f"Telegram send failed: {e}")


def _live_adapter(platform, *, lookup_failed_warning=None):
    """``(runner, adapter)`` for the in-process gateway; ``(None, None)`` standalone (cron);
    ``(runner, None)`` when the lookup fails — logged when a warning is given, never silently
    swallowed (a silent fall-through could recreate a reconnect storm)."""
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is None:
        return None, None
    try:
        return runner, runner.adapters.get(platform)
    except Exception:
        if lookup_failed_warning:
            logger.warning(lookup_failed_warning, exc_info=True)
        return runner, None


def _plugin_standalone_sender(platform_name, *, label=None, discover=True):
    """``(standalone_sender_fn, None)`` for a registered plugin or ``(None, error_dict)``;
    ``discover`` runs the idempotent plugin scan first."""
    from gateway.platform_registry import platform_registry
    if discover:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    entry = platform_registry.get(platform_name)
    if entry is None or entry.standalone_sender_fn is None:
        return None, {"error": f"{label or platform_name} plugin not registered or missing standalone_sender_fn"}
    return entry.standalone_sender_fn, None


async def _registry_standalone_send(platform_name, pconfig, chat_id, message, thread_id=None):
    """One-shot text send through a plugin's ``standalone_sender_fn``."""
    sender, err = _plugin_standalone_sender(platform_name)
    return err or await sender(pconfig, chat_id, message, thread_id=thread_id)


async def _resolve_slack_user_target(token, chat_id):
    """Resolve ``user:U...`` / ``user_name:<handle>`` to a D... DM conversation (chat.postMessage
    needs a conversation ID); ``user_name:`` goes through users.list first (stable handle match
    only); other ids pass through. ``(chat_id, None)`` or ``(None, error_dict)``."""
    if not (chat_id.startswith("user:") or chat_id.startswith("user_name:")):
        return chat_id, None
    try:
        import aiohttp
    except ImportError:
        return None, {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url())
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            async def post_api(method, payload):
                async with session.post(f"https://slack.com/api/{method}", headers=headers, json=payload,
                                        **_req_kw) as resp:
                    return await resp.json()

            if chat_id.startswith("user_name:"):
                name = chat_id[len("user_name:"):]
                query = name.strip().lstrip("@").lower()
                matches, cursor = [], None
                for _page in range(20):
                    data = await post_api("users.list", {"limit": 200, **({"cursor": cursor} if cursor else {})})
                    if not data.get("ok"):
                        return None, _error(f"Slack users.list error: {data.get('error', 'unknown')}")
                    # Stable handle only: display/real names are mutable and non-unique.
                    matches += [m for m in data.get("members", []) if not (m.get("deleted") or m.get("is_bot"))
                                and str(m.get("name", "")).strip().lower() == query]
                    cursor = (data.get("response_metadata") or {}).get("next_cursor")
                    if not cursor:
                        break
                if not matches:
                    return None, _error(f"Could not resolve Slack user '@{name}'.")
                if len(matches) > 1:
                    return None, _error(f"Slack user '@{name}' matched multiple Slack users. Use a Slack user ID instead.")
                chat_id = f"user:{matches[0].get('id')}"
            opened = await post_api("conversations.open", {"users": chat_id[len("user:"):]})
            if not opened.get("ok"):
                return None, _error(f"Slack conversations.open error: {opened.get('error', 'unknown')}. "
                                    "Check bot permissions (im:write).")
            dm_id = (opened.get("channel") or {}).get("id")
            return (dm_id, None) if dm_id else (None, _error("Slack conversations.open did not return a DM channel ID"))
    except Exception as e:
        return None, _error(f"Slack DM resolution failed: {e}")


async def _signal_send_batch(post, scheduler, rl, idx, n_batches, att_batch, batch_message):
    """One Signal batch under the scheduler with rate-limit retries: None on success, False when
    retries were exhausted (batch lost), error dict for a non-rate-limit RPC error."""
    n, max_attempts = len(att_batch), rl.SIGNAL_RATE_LIMIT_MAX_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        try:
            await scheduler.acquire(n)
            _rpc_t0 = time.monotonic()
            data = await post(att_batch, batch_message)
            if "error" not in data:
                await scheduler.report_rpc_duration(time.monotonic() - _rpc_t0, n)
                return None
            err = data["error"]
            if not rl._is_signal_rate_limit_error(err):
                return _error(f"Signal RPC error on batch {idx + 1}/{n_batches}: {err}")
            server_retry_after = rl._extract_retry_after_seconds(err)
            scheduler.feedback(server_retry_after, n)
            retry_after_label = f"{server_retry_after:.0f}s" if server_retry_after else "unknown"
            if attempt >= max_attempts:
                logger.error("Signal: rate-limit retries exhausted on batch %d/%d (%d attachments lost, "
                             "server retry_after=%s)", idx + 1, n_batches, n, retry_after_label)
                return False
            logger.warning("Signal: rate-limited on batch %d/%d (attempt %d/%d, server retry_after=%s); "
                           "scheduler will pace the retry",
                           idx + 1, n_batches, attempt, max_attempts, retry_after_label)
        except Exception as e:
            if attempt >= max_attempts:
                logger.error("Signal: send error on batch %d/%d after %d attempts: %s",
                             idx + 1, n_batches, attempt, str(e))
                return False
            logger.warning("Signal: transient error on batch %d/%d (attempt %d/%d): %s; will retry",
                           idx + 1, n_batches, attempt, max_attempts, str(e))


async def _send_signal(extra, chat_id, message, media_files=None):
    """signal-cli JSON-RPC send; attachments go in SIGNAL_MAX_ATTACHMENTS_PER_MSG batches metered
    by the process-wide SignalAttachmentScheduler (shared with the gateway adapter's rate-limit state)."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}
    from gateway.platforms import signal_rate_limit as rl
    from gateway.platforms.signal_format import markdown_to_signal
    try:
        http_url, account = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/"), extra.get("account", "")
        if not account:
            return {"error": "Signal account not configured"}
        valid_media = media_files or []
        attachment_paths = []
        for media_path, _is_voice in valid_media:
            if os.path.exists(media_path):
                attachment_paths.append(media_path)
            else:
                logger.warning("Signal media file not found, skipping: %s", media_path)
        # No attachments still means one (text-only) batch; text rides on batch #0 only.
        per_batch = rl.SIGNAL_MAX_ATTACHMENTS_PER_MSG
        att_batches = [attachment_paths[i:i + per_batch] for i in range(0, len(attachment_paths), per_batch)] or [[]]
        n_batches, (plain_text, text_styles) = len(att_batches), markdown_to_signal(message)
        recipient = {"groupId": chat_id[6:]} if chat_id.startswith("group:") else {"recipient": [chat_id]}

        async def _rpc_send(text, *, id_prefix, timeout, attachments=None, styled=False):
            params = {"account": account, "message": text, **recipient}
            if styled and text and text_styles:
                params["textStyle" if len(text_styles) == 1 else "textStyles"] = (
                    text_styles[0] if len(text_styles) == 1 else text_styles)
            if attachments:
                params["attachments"] = attachments
            payload = {"jsonrpc": "2.0", "method": "send", "params": params,
                       "id": f"{id_prefix}_{int(time.time() * 1000)}"}
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(f"{http_url}/api/v1/rpc", json=payload)

        async def _post(batch_attachments, batch_message):
            resp = await _rpc_send(batch_message, id_prefix="send", attachments=batch_attachments, styled=True,
                                   timeout=rl._signal_send_timeout(len(batch_attachments)))
            resp.raise_for_status()
            return resp.json()
        scheduler = rl.get_scheduler()
        logger.info("send_message Signal: scheduler state=%s, %d attachment(s) in %d batch(es)",
                    scheduler.state(), len(attachment_paths), n_batches)
        failed_batches: list[int] = []
        for idx, att_batch in enumerate(att_batches):
            n = len(att_batch)
            if n > 0 and (estimated := scheduler.estimate_wait(n)) >= rl.SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                # Best-effort one-shot RPC for a user-facing pacing notice.
                try:
                    await _rpc_send(f"(More images coming — pausing ~{rl._format_wait(estimated)} "
                                    f"for Signal rate limit, batch {idx + 1}/{n_batches}.)", id_prefix="notice", timeout=30.0)
                except Exception as _e:
                    logger.warning("Signal: inline notice failed: %s", _e)
            outcome = await _signal_send_batch(_post, scheduler, rl, idx, n_batches, att_batch,
                                               plain_text if idx == 0 else "")
            if outcome is False:
                failed_batches.append(idx + 1)
            elif outcome is not None:
                return outcome
        warnings = []
        if len(attachment_paths) < len(valid_media):
            warnings.append("Some media files were skipped (not found on disk)")
        if failed_batches:
            warnings.append(f"Signal rate-limited {len(failed_batches)} batch(es) "
                            f"(#{', #'.join(str(b) for b in failed_batches)})")
        if failed_batches and len(failed_batches) == n_batches:
            return _error(f"Signal: every batch ({n_batches}) hit rate limit; no attachments delivered")
        # Result-safe chat identifier for tool transcripts/log consumers.
        return _success("signal", "group:***" if str(chat_id).startswith("group:") else chat_id, warnings)
    except Exception as e:
        return _error(f"Signal send failed: {e}")


# "ephemeral connect (may re-init E2EE per send, see #46310)",
async def _send_matrix_via_adapter(pconfig, chat_id, message, media_files=None, thread_id=None):
    """Matrix adapter send (native media preserved). Prefer the live gateway adapter's persistent
    olm/megolm session: ephemeral per-send connects re-init E2EE and claim one-time keys, which
    under bursts exhausts recipient OTKs and silently drops messages — ephemeral is cron-only.

    When a live gateway adapter is available (i.e. the tool runs inside a running gateway), the persistent
    connection is reused — one olm/megolm session for all sends. This avoids per-message E2EE re-init storms
    that exhaust recipient OTKs and silently drop messages (issue #46310).
    """
    media_files = media_files or []
    metadata = {"thread_id": thread_id} if thread_id else None
    from gateway.config import Platform
    _, live_adapter = _live_adapter(Platform.MATRIX, lookup_failed_warning=(
        "Matrix: live gateway adapter lookup failed; falling back to an "
        "ephemeral connect (may re-init E2EE per send)"))
    if live_adapter is not None:
        # Owned by the gateway — must NOT be disconnected (return before the ephemeral ``finally``).
        return await _matrix_send_core(live_adapter, chat_id, message, media_files, metadata)
    try:
        from plugins.platforms.matrix.adapter import MatrixAdapter
    except ImportError:
        return {"error": "Matrix dependencies not installed. Run: pip install 'mautrix[encryption]'"}
    adapter = MatrixAdapter(pconfig)
    try:
        if not await adapter.connect():
            return _error("Matrix connect failed")
        return await _matrix_send_core(adapter, chat_id, message, media_files, metadata)
    except Exception as e:
        return _error(f"Matrix send failed: {e}")
    finally:
        with contextlib.suppress(Exception):
            await adapter.disconnect()


async def _matrix_send_core(adapter, chat_id, message, media_files, metadata):
    """Core send logic shared by live and ephemeral Matrix adapters."""
    last_result = None
    if message.strip():
        last_result = await adapter.send(chat_id, message, metadata=metadata)
        if not last_result.success:
            return _error(f"Matrix send failed: {last_result.error}")
    for media_path, is_voice in media_files:
        if not os.path.exists(media_path):
            return _error(f"Media file not found: {media_path}")
        ext = os.path.splitext(media_path)[1].lower()
        method, _ = _adapter_media_method(ext, (ext in _VOICE_EXTS and is_voice) or ext in _AUDIO_EXTS)
        last_result = await getattr(adapter, method)(chat_id, media_path, metadata=metadata)
        if not last_result.success:
            return _error(f"Matrix media send failed: {last_result.error}")
    return {"error": _NO_DELIVERABLE} if last_result is None else _success("matrix", chat_id, message_id=last_result.message_id)


def _gateway_platform_module(name, *, unavailable, unmet):
    """``(gateway.platforms.<name>, None)`` once its ``check_<name>_requirements`` passes, else ``(None, error)``."""
    import importlib
    try:
        module = importlib.import_module(f"gateway.platforms.{name}")
    except ImportError:
        return None, {"error": unavailable}
    return (module, None) if getattr(module, f"check_{name}_requirements")() else (None, {"error": unmet})


async def _send_weixin(pconfig, chat_id, message, media_files=None):
    """Send via Weixin iLink using the native adapter helper."""
    wx, err = _gateway_platform_module("weixin", unavailable="Weixin adapter not available.",
                                       unmet="Weixin requirements not met. Need aiohttp + cryptography.")
    if err:
        return err
    try:
        return await wx.send_weixin_direct(extra=pconfig.extra, token=pconfig.token, chat_id=chat_id,
                                           message=message, media_files=media_files)
    except Exception as e:
        return _error(f"Weixin send failed: {e}")


async def _send_bluebubbles(extra, chat_id, message):
    """Send via BlueBubbles iMessage server using the adapter's REST API."""
    bb, err = _gateway_platform_module("bluebubbles", unavailable="BlueBubbles adapter not available.",
                                       unmet="BlueBubbles requirements not met (need aiohttp + httpx).")
    if err:
        return err
    try:
        from gateway.config import PlatformConfig
        adapter = bb.BlueBubblesAdapter(PlatformConfig(extra=extra))
        if not await adapter.connect():
            return _error("BlueBubbles: failed to connect to server")
        try:
            result = await adapter.send(chat_id, message)
        finally:
            await adapter.disconnect()
        if not result.success:
            return _error(f"BlueBubbles send failed: {result.error}")
        return _success("bluebubbles", chat_id, message_id=result.message_id)
    except Exception as e:
        return _error(f"BlueBubbles send failed: {e}")


async def _send_qqbot(pconfig, chat_id, message):
    """Send via the QQ Bot Open Platform REST API (no WebSocket needed)."""
    try:
        import httpx
    except ImportError:
        return _error("QQBot direct send requires httpx. Run: pip install httpx")

    # Profile-scoped lookup so a multiplex profile never borrows another's QQ credentials.
    from gateway.config import _getenv
    extra = pconfig.extra or {}
    appid = extra.get("app_id") or _getenv("QQ_APP_ID", "")
    secret = pconfig.token or extra.get("client_secret") or _getenv("QQ_CLIENT_SECRET", "")
    if not appid or not secret:
        return _error("QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured.")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            token_resp = await client.post("https://bots.qq.com/app/getAppAccessToken",
                                           json={"appId": str(appid), "clientSecret": str(secret)})
            if token_resp.status_code != 200:
                return _error(f"QQBot token request failed: {token_resp.status_code}")
            access_token = token_resp.json().get("access_token")
            if not access_token:
                return _error("QQBot: no access_token in response")

            # Separate endpoints for guild channels, C2C (private) and groups; first 2xx wins.
            headers = {"Authorization": f"QQBot {access_token}", "Content-Type": "application/json"}
            payload = {"content": message[:4000], "msg_type": 0}
            endpoints = (("channel", f"https://api.sgroup.qq.com/channels/{chat_id}/messages"),
                         ("c2c", f"https://api.sgroup.qq.com/v2/users/{chat_id}/messages"),
                         ("group", f"https://api.sgroup.qq.com/v2/groups/{chat_id}/messages"))
            statuses = []
            for kind, url in endpoints:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in {200, 201}:
                    return _success("qqbot", chat_id, message_id=resp.json().get("id"))
                statuses.append(f"{kind}={resp.status_code}")
            return _error(f"QQBot send failed: {' '.join(statuses)}")
    except Exception as e:
        return _error(f"QQBot send failed: {e}")


async def _send_yuanbao(chat_id, message, media_files=None):
    """Send via the running Yuanbao adapter's persistent WebSocket (no throwaway client possible)."""
    try:
        from gateway.platforms.yuanbao import YuanbaoAdapter
    except ImportError:
        return _error("Yuanbao adapter module not available.")
    adapter = YuanbaoAdapter.get_active()
    if adapter is None:
        return _error("Yuanbao adapter is not running. Start the gateway with yuanbao platform enabled first.")
    try:
        return await adapter._outbound.sender.send_direct(chat_id, message, media_files)
    except Exception as e:
        return _error(f"Yuanbao send failed: {e}")
