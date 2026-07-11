"""Tool for sending persistent Telegram thesis review cards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Tuple

from tools.registry import registry, tool_error


SEND_REVIEW_CARD_SCHEMA = {
    "name": "send_review_card",
    "description": "Send a persistent Telegram thesis-review card with Accept/Deny/Rescore/Skip buttons.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Telegram target, e.g. telegram:-1003915682412:3930"},
            "card_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["evidence", "expert", "startup", "job"]},
            "thesis": {"type": "string", "description": "For evidence: one target thesis. For expert/source cards: comma-separated relevant theses, or 'Generalist / cross-thesis watchlist'; this is relevance/routing, not ownership."},
            "body": {"type": "string"},
            "person": {"type": "string"},
            "url": {"type": "string"},
            "source": {"type": "string"},
        },
        "required": ["target", "card_id", "kind", "thesis", "body"],
    },
}


def _check_send_review_card() -> bool:
    return True


def _callback_safe_card_id(card_id: str, kind: str) -> str:
    """Keep Telegram callback_data within the 64-byte Bot API limit.

    Callback data is shaped like `rq:a:<card_id>`, so card_id must be short.
    Preserve already-short IDs; compact long cron-generated slugs deterministically.
    Original evidence/expert IDs should live in the card body.
    """
    card_id = str(card_id or "").strip()
    if len(f"rq:a:{card_id}".encode("utf-8")) <= 64:
        return card_id
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", card_id.lower()).strip("-")[:18] or "card"
    digest = hashlib.sha1(card_id.encode("utf-8")).hexdigest()[:10]
    kind_key = str(kind).lower()
    prefix = "ex" if kind_key == "expert" else "su" if kind_key == "startup" else "jb" if kind_key == "job" else "ev"
    return f"rq-{prefix}-{slug}-{digest}"


_TWEET_ID_RE = re.compile(r"(?:x\.com|twitter\.com)/(?:i/web/status/|[^\s/]+/status/)(\d+)")


def _extract_tweet_ids(*parts: str) -> List[str]:
    seen: set[str] = set()
    ids: List[str] = []
    for part in parts:
        for tweet_id in _TWEET_ID_RE.findall(str(part or "")):
            if tweet_id not in seen:
                seen.add(tweet_id)
                ids.append(tweet_id)
    return ids


def _validate_x_post_links(*parts: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate embedded X post links before sending Review Queue cards.

    Review Queue cards are action items; dead X links make them unusable. If an
    x.com/twitter.com status URL is present, require X API hydration to find it.
    """
    tweet_ids = _extract_tweet_ids(*parts)
    if not tweet_ids:
        return True, {"checked": 0, "valid_ids": [], "invalid_ids": []}
    endpoint = (
        "/2/tweets?ids=" + ",".join(tweet_ids)
        + "&tweet.fields=created_at,author_id,public_metrics,entities,referenced_tweets"
    )
    env = {**os.environ, "HOME": os.environ.get("HOME") or "/root"}
    if env.get("HOME") in {"", "/"}:
        env["HOME"] = "/root"
    try:
        proc = subprocess.run(
            ["xurl", endpoint],
            cwd="/usr/local/lib/hermes-agent" if os.path.exists("/usr/local/lib/hermes-agent") else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
        )
    except Exception as exc:
        return False, {"checked": len(tweet_ids), "error": f"xurl validation failed: {exc}", "invalid_ids": tweet_ids}
    if proc.returncode != 0:
        return False, {
            "checked": len(tweet_ids),
            "error": "xurl validation failed",
            "stderr": proc.stderr[-500:],
            "stdout": proc.stdout[-500:],
            "invalid_ids": tweet_ids,
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return False, {"checked": len(tweet_ids), "error": f"invalid xurl JSON: {exc}", "invalid_ids": tweet_ids}
    valid_ids = {str(item.get("id")) for item in (payload.get("data") or []) if item.get("id")}
    invalid_ids = [tid for tid in tweet_ids if tid not in valid_ids]
    errors = []
    for err in payload.get("errors") or []:
        errors.append({k: err.get(k) for k in ("value", "title", "detail", "type") if err.get(k)})
    return not invalid_ids, {
        "checked": len(tweet_ids),
        "valid_ids": sorted(valid_ids),
        "invalid_ids": invalid_ids,
        "errors": errors,
    }


def send_review_card_tool(args: Dict[str, Any], **kw) -> str:
    target = str(args.get("target") or "").strip()
    if not target.startswith("telegram:"):
        return tool_error("send_review_card currently requires a telegram:<chat_id>:<thread_id> target")
    try:
        from gateway.config import Platform, load_gateway_config
        from gateway.platforms.telegram import TelegramAdapter, Bot
        from model_tools import _run_async
        from tools.send_message_tool import _parse_target_ref

        chat_id, thread_id, explicit = _parse_target_ref("telegram", target.split(":", 1)[1])
        if not explicit or not chat_id:
            return tool_error("Invalid Telegram target. Use telegram:<chat_id>:<thread_id>.")

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.TELEGRAM)
        if not pconfig or not pconfig.enabled or not pconfig.token:
            return tool_error("Telegram is not configured.")

        adapter = TelegramAdapter(pconfig)
        adapter._bot = Bot(token=pconfig.token)
        safe_card_id = _callback_safe_card_id(str(args["card_id"]), str(args["kind"]))
        links_ok, link_validation = _validate_x_post_links(
            str(args.get("url") or ""),
            str(args.get("body") or ""),
            str(args.get("source") or ""),
        )
        if not links_ok:
            return json.dumps({
                "error": "x_post_link_invalid",
                "message": "Review card not sent because at least one X post link is unavailable/not found.",
                "card_id": safe_card_id,
                "original_card_id": str(args["card_id"]),
                "validation": link_validation,
            })
        result = _run_async(
            adapter.send_review_card(
                chat_id=str(chat_id),
                card_id=safe_card_id,
                kind=str(args["kind"]),
                thesis=str(args["thesis"]),
                body=str(args["body"]),
                person=str(args.get("person") or ""),
                url=str(args.get("url") or ""),
                source=str(args.get("source") or ""),
                metadata={"thread_id": str(thread_id or "")},
            )
        )
        if result.success:
            return json.dumps({
                "success": True,
                "platform": "telegram",
                "chat_id": str(chat_id),
                "thread_id": str(thread_id or ""),
                "message_id": result.message_id,
                "card_id": safe_card_id,
                "original_card_id": str(args["card_id"]),
            })
        return json.dumps({"error": result.error or "send_review_card failed"})
    except Exception as e:
        return json.dumps({"error": f"send_review_card failed: {e}"})


registry.register(
    name="send_review_card",
    toolset="review_queue",
    schema=SEND_REVIEW_CARD_SCHEMA,
    handler=send_review_card_tool,
    check_fn=_check_send_review_card,
    emoji="🗳️",
)
