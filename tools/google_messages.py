"""Read-only Google Messages for Web checker tools.

Milestone 1 is intentionally conservative:

* use a dedicated persistent browser profile by default;
* open/navigate only to https://messages.google.com/web;
* report pairing/login status;
* extract the visible conversation list only;
* never open individual threads and never send messages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry

GOOGLE_MESSAGES_URL = "https://messages.google.com/web"
DEFAULT_PROFILE_SUBDIR = Path("browser-profiles") / "google-messages"
_TOOLSET = "google_messages"

_TIMESTAMP_RE = re.compile(
    r"^(?:"
    r"\d{1,2}:\d{2}(?:\s?[AP]M)?|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day)?|"
    r"(?:Yesterday|Today)|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:,?\s+\d{4})?"
    r")$",
    re.IGNORECASE,
)

_PAIRING_HINT_RE = re.compile(
    r"(qr code|pair(?:ing)?|scan the code|use messages on the web)",
    re.IGNORECASE,
)

_LOGIN_REQUIRED_RE = re.compile(
    r"(sign in|google account|choose an account|accounts\.google\.com/(?:.*signin|.*sign-in))",
    re.IGNORECASE,
)


def _default_profile_path() -> Path:
    """Return the profile directory dedicated to Google Messages for Web."""
    return get_hermes_home() / DEFAULT_PROFILE_SUBDIR


def _resolve_profile_path(profile_path: Optional[str]) -> Path:
    if profile_path:
        return Path(profile_path).expanduser()
    return _default_profile_path()


def check_google_messages_requirements() -> bool:
    """Return True when local Playwright is importable.

    Chromium browser installation is validated lazily by Playwright at runtime so
    the tool can return a clear setup error instead of disappearing because the
    Python package is installed but browser bits are missing.
    """
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return True


def _clean_lines(text: str) -> List[str]:
    seen: set[str] = set()
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _looks_like_timestamp(line: str) -> bool:
    return bool(_TIMESTAMP_RE.match(line.strip()))


def _detect_unread(raw: Dict[str, Any], text: str) -> bool:
    attrs = " ".join(
        str(raw.get(key) or "")
        for key in ("aria_label", "class_name", "data_unread", "role")
    )
    haystack = f"{attrs}\n{text}".lower()
    if re.search(r"\bunread\b", haystack):
        return True
    if str(raw.get("data_unread") or "").lower() in {"true", "1", "yes"}:
        return True
    return False


def _parse_conversation_item(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one raw DOM candidate into a conversation preview.

    The Google Messages Web DOM is private and changes over time, so this parser
    accepts loose text/ARIA/class candidates and returns best-effort fields.
    """
    text = str(raw.get("text") or raw.get("inner_text") or "")
    aria = str(raw.get("aria_label") or "")
    lines = _clean_lines(text or aria)
    if not lines:
        return None

    sender: Optional[str] = None
    timestamp: Optional[str] = None
    snippet_parts: List[str] = []

    for line in lines:
        lower = line.lower()
        if lower in {"you", "me", "sent", "delivered", "read"}:
            snippet_parts.append(line)
            continue
        if timestamp is None and _looks_like_timestamp(line):
            timestamp = line
            continue
        if sender is None:
            sender = line
            continue
        snippet_parts.append(line)

    if sender is None and aria:
        sender = aria.split(",", 1)[0].strip() or None

    snippet = " ".join(snippet_parts).strip() or None
    return {
        "sender": sender,
        "timestamp": timestamp,
        "snippet": snippet,
        "unread": _detect_unread(raw, text or aria),
    }


def _parse_conversation_items(raw_items: Iterable[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    conversations: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for raw in raw_items:
        item = _parse_conversation_item(raw)
        if not item or not item.get("sender"):
            continue
        key = (item.get("sender"), item.get("timestamp"), item.get("snippet"))
        if key in seen:
            continue
        seen.add(key)
        conversations.append(item)
        if len(conversations) >= limit:
            break
    return conversations


def _classify_status(url: str, body_text: str, raw_items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    raw_list = list(raw_items)
    url_text = url or ""
    body = body_text or ""
    pairing_required = bool(_PAIRING_HINT_RE.search(body))
    login_required = bool(_LOGIN_REQUIRED_RE.search(f"{url_text}\n{body}"))
    conversations = [] if pairing_required or login_required else _parse_conversation_items(raw_list, limit=5)
    logged_in = bool(conversations) or "conversations" in url_text.lower()
    if login_required:
        state = "login_required"
    elif pairing_required:
        state = "pairing_required"
    elif logged_in:
        state = "ready"
    else:
        state = "unknown"
    return {
        "state": state,
        "pairing_required": pairing_required,
        "login_required": login_required,
        "ready": logged_in and not pairing_required and not login_required,
        "conversation_candidates": len(raw_list),
    }


def _conversation_extract_script() -> str:
    # Kept as a string so tests can focus on parser logic without Playwright.
    return r"""
() => {
  const selectors = [
    'mws-conversation-list-item',
    '[data-e2e-conversation-list-item]',
    'a[href*="/web/conversations"]',
    'mws-conversation-list [role="button"]',
    'mws-conversation-list a'
  ];
  const nodes = [];
  for (const selector of selectors) {
    document.querySelectorAll(selector).forEach((node) => {
      if (!nodes.includes(node)) nodes.push(node);
    });
  }
  return nodes.slice(0, 80).map((node) => ({
    text: node.innerText || node.textContent || '',
    aria_label: node.getAttribute('aria-label') || '',
    class_name: node.getAttribute('class') || '',
    data_unread: node.getAttribute('data-unread') || node.getAttribute('aria-current') || '',
    role: node.getAttribute('role') || ''
  }));
}
"""


def _with_page(profile_path: Path, headless: bool, timeout_ms: int):
    """Launch Google Messages with a persistent context and return Playwright objects.

    Caller must close the context and stop Playwright after successful return.
    Partial launch/navigation failures are cleaned up here before re-raising.
    """
    from playwright.sync_api import sync_playwright

    profile_path.mkdir(parents=True, exist_ok=True)
    try:
        profile_path.chmod(0o700)
    except OSError:
        pass
    pw = None
    context = None
    try:
        pw = sync_playwright().start()
        context = pw.chromium.launch_persistent_context(
            str(profile_path),
            headless=headless,
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(GOOGLE_MESSAGES_URL, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
        except Exception:
            pass
        return pw, context, page
    except Exception:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass
        raise


def google_messages_status(
    profile_path: Optional[str] = None,
    headless: bool = False,
    timeout_ms: int = 15000,
) -> str:
    """Open Google Messages for Web and report pairing/login status.

    This is read-only. It navigates to the web app but does not click a
    conversation and does not send or type anything.
    """
    resolved_profile = _resolve_profile_path(profile_path)
    try:
        pw, context, page = _with_page(resolved_profile, headless, timeout_ms)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "error": str(exc),
            "profile_path": str(resolved_profile),
            "setup_hint": "Install Playwright and Chromium: pip install playwright && python -m playwright install chromium",
        })
    try:
        body_text = page.locator("body").inner_text(timeout=min(timeout_ms, 5000))
        raw_items = page.evaluate(_conversation_extract_script())
        status = _classify_status(page.url, body_text, raw_items)
        return json.dumps({
            "ok": True,
            "url": page.url,
            "profile_path": str(resolved_profile),
            **status,
            "safety": "read-only status check; no thread opened and no message sent",
        })
    finally:
        context.close()
        pw.stop()


def google_messages_conversations(
    limit: int = 20,
    profile_path: Optional[str] = None,
    headless: bool = False,
    timeout_ms: int = 15000,
) -> str:
    """Return visible Google Messages conversation-list previews only.

    The tool never opens individual threads. Snippets are whatever the list UI
    already exposes; opening a thread is deliberately out of scope for milestone 1.
    """
    limit = max(1, min(int(limit or 20), 50))
    resolved_profile = _resolve_profile_path(profile_path)
    try:
        pw, context, page = _with_page(resolved_profile, headless, timeout_ms)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "error": str(exc),
            "profile_path": str(resolved_profile),
            "setup_hint": "Install Playwright and Chromium: pip install playwright && python -m playwright install chromium",
        })
    try:
        body_text = page.locator("body").inner_text(timeout=min(timeout_ms, 5000))
        raw_items = page.evaluate(_conversation_extract_script())
        status = _classify_status(page.url, body_text, raw_items)
        conversations = _parse_conversation_items(raw_items, limit=limit)
        return json.dumps({
            "ok": True,
            "url": page.url,
            "profile_path": str(resolved_profile),
            **status,
            "count": len(conversations),
            "conversations": conversations,
            "safety": "conversation-list extraction only; no thread opened and no message sent",
        })
    finally:
        context.close()
        pw.stop()


registry.register(
    name="google_messages_status",
    toolset=_TOOLSET,
    schema={
        "name": "google_messages_status",
        "description": (
            "Read-only Google Messages for Web status/pairing check. Opens "
            "messages.google.com/web using a dedicated persistent profile and "
            "reports whether manual QR pairing/login appears needed. Does not "
            "open threads or send messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_path": {"type": "string", "description": "Optional persistent browser profile path. Defaults to the active Hermes profile home under browser-profiles/google-messages."},
                "headless": {"type": "boolean", "description": "Run browser headlessly. Defaults to false so QR pairing is visible."},
                "timeout_ms": {"type": "integer", "description": "Navigation/detection timeout in milliseconds."},
            },
        },
    },
    handler=lambda args, **kw: google_messages_status(
        profile_path=args.get("profile_path"),
        headless=bool(args.get("headless", False)),
        timeout_ms=int(args.get("timeout_ms", 15000)),
    ),
    check_fn=check_google_messages_requirements,
    description="Read-only Google Messages for Web pairing/status check",
    emoji="💬",
)

registry.register(
    name="google_messages_conversations",
    toolset=_TOOLSET,
    schema={
        "name": "google_messages_conversations",
        "description": (
            "Read-only Google Messages for Web conversation-list extraction. "
            "Returns visible sender/name, timestamp, snippet, and defensively "
            "detected unread-ish flag. Never opens individual threads and never sends messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum conversation previews to return, capped at 50."},
                "profile_path": {"type": "string", "description": "Optional persistent browser profile path. Defaults to the active Hermes profile home under browser-profiles/google-messages."},
                "headless": {"type": "boolean", "description": "Run browser headlessly. Defaults to false so pairing problems are visible."},
                "timeout_ms": {"type": "integer", "description": "Navigation/extraction timeout in milliseconds."},
            },
        },
    },
    handler=lambda args, **kw: google_messages_conversations(
        limit=int(args.get("limit", 20)),
        profile_path=args.get("profile_path"),
        headless=bool(args.get("headless", False)),
        timeout_ms=int(args.get("timeout_ms", 15000)),
    ),
    check_fn=check_google_messages_requirements,
    description="Read-only Google Messages for Web conversation previews",
    emoji="💬",
)
