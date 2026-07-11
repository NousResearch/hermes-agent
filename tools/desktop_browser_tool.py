"""Safe protocol helpers for driving the visible Desktop browser webview.

The model keeps using Hermes' existing ``browser_*`` tool schemas.  When a
Desktop bridge callback is configured, these helpers translate the renderer's
same-webview snapshot into opaque refs and bind every action to a page URL and
an element fingerprint.  Raw DOM indexes are never exposed as an execution
contract.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, Callable


DESKTOP_BROWSER_TOOL_NAMES = frozenset(
    {
        "browser_back",
        "browser_click",
        "browser_navigate",
        "browser_press",
        "browser_scroll",
        "browser_snapshot",
        "browser_type",
    }
)

_FINGERPRINT_KEYS = (
    "ariaLabel",
    "hermesRef",
    "href",
    "id",
    "name",
    "placeholder",
    "role",
    "tag",
    "text",
    "type",
)
_RISKY_ACTION_WORDS = (
    "发送",
    "发布",
    "评论",
    "私信",
    "回复",
    "点赞",
    "关注",
    "提交",
    "删除",
    "确认",
    "支付",
    "下单",
    "send",
    "publish",
    "comment",
    "reply",
    "like",
    "follow",
    "submit",
    "delete",
    "confirm",
    "pay",
)
_DOUYIN_HOST_SUFFIXES = ("douyin.com", "jinritemai.com", "iesdouyin.com")


class DesktopBrowserProtocolError(ValueError):
    """The Desktop browser request cannot be executed safely."""


def _clean(value: Any, limit: int = 2_000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _multiline(value: Any, limit: int = 200_000) -> str:
    text = str(value or "").replace("\x00", "").replace("\r\n", "\n")
    return "\n".join(_clean(line, 20_000) for line in text.splitlines()).strip()[:limit]


def _fingerprint(element: object) -> dict[str, str]:
    if not isinstance(element, dict):
        return {}
    return {
        key: cleaned
        for key in _FINGERPRINT_KEYS
        if (cleaned := _clean(element.get(key)))
    }


def _element_line(ref: str, fingerprint: dict[str, str]) -> str:
    tag = fingerprint.get("tag", "element")
    text = fingerprint.get("text", "")
    attributes = []
    for key, label in (
        ("role", "role"),
        ("ariaLabel", "aria-label"),
        ("placeholder", "placeholder"),
        ("type", "type"),
        ("href", "href"),
    ):
        if value := fingerprint.get(key):
            attributes.append(f'{label}={json.dumps(value, ensure_ascii=False)}')
    suffix = f" {json.dumps(text, ensure_ascii=False)}" if text else ""
    attrs = f" {' '.join(attributes)}" if attributes else ""
    return f"{ref} {tag}{attrs}{suffix}"


def normalize_desktop_browser_snapshot(
    raw: object,
    *,
    max_chars: int = 8_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build model-facing snapshot text and a private ref/fingerprint map."""
    if not isinstance(raw, dict):
        raise DesktopBrowserProtocolError("Desktop browser snapshot returned no data")
    if raw.get("ok") is not True:
        raise DesktopBrowserProtocolError(
            _clean(raw.get("error")) or "Desktop browser snapshot failed"
        )

    url = _clean(raw.get("url"), 8_000)
    if not url:
        raise DesktopBrowserProtocolError("Desktop browser snapshot is missing its page URL")

    refs: dict[str, dict[str, str]] = {}
    element_lines: list[str] = []
    elements = raw.get("elements") if isinstance(raw.get("elements"), list) else []
    for element in elements:
        if isinstance(element, dict) and element.get("visible") is False:
            continue
        fingerprint = _fingerprint(element)
        if not fingerprint:
            continue
        ref = f"@e{len(refs)}"
        refs[ref] = fingerprint
        element_lines.append(_element_line(ref, fingerprint))

    title = _clean(raw.get("title"), 4_000)
    body = _multiline(raw.get("text"), 200_000)
    headings = []
    for heading in raw.get("headings") if isinstance(raw.get("headings"), list) else []:
        if not isinstance(heading, dict):
            continue
        text = _clean(heading.get("text"), 2_000)
        if text:
            try:
                level = max(1, min(6, int(heading.get("level") or 1)))
            except (TypeError, ValueError):
                level = 1
            headings.append(f"H{level}: {text}")

    table_blocks = []
    tables = raw.get("tables") if isinstance(raw.get("tables"), list) else []
    for table in tables:
        if not isinstance(table, dict):
            continue
        lines = []
        caption = _clean(table.get("caption"), 1_000)
        lines.append(f"Table: {caption}" if caption else "Table")
        headers = table.get("headers") if isinstance(table.get("headers"), list) else []
        if headers:
            lines.append(" | ".join(_clean(cell, 500) for cell in headers))
        for row in table.get("rows") if isinstance(table.get("rows"), list) else []:
            if isinstance(row, list):
                lines.append(" | ".join(_clean(cell, 500) for cell in row))
        table_blocks.append("\n".join(lines))

    sections = [f"Title: {title}" if title else "", f"URL: {url}"]
    if headings:
        sections.append("Headings:\n" + "\n".join(headings))
    if element_lines:
        sections.append("Interactive elements:\n" + "\n".join(element_lines))
    if table_blocks:
        sections.append("Tables:\n" + "\n\n".join(table_blocks))
    if body:
        sections.append("Page text:\n" + body)
    snapshot_text = "\n\n".join(section for section in sections if section)

    from tools.browser_tool import _redact_browser_output, _truncate_snapshot

    result = _redact_browser_output(
        {
            "success": True,
            "url": url,
            "title": title,
            "snapshot": _truncate_snapshot(snapshot_text, max_chars=max_chars),
            "element_count": len(refs),
        }
    )
    state = {"url": url, "title": title, "refs": refs}
    return result, state


def build_desktop_browser_action(
    function_name: str,
    args: dict[str, Any] | None,
    snapshot_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Translate a browser tool call into a URL-bound same-webview action."""
    if not snapshot_state or not snapshot_state.get("url"):
        raise DesktopBrowserProtocolError(
            "Take a new browser_snapshot before acting in the Desktop browser."
        )

    tool_args = args or {}
    action: dict[str, Any] = {"expectedUrl": snapshot_state["url"]}

    if function_name in {"browser_click", "browser_type"}:
        ref = _clean(tool_args.get("ref"), 100)
        if ref and not ref.startswith("@"):
            ref = f"@{ref}"
        target = (snapshot_state.get("refs") or {}).get(ref)
        if not target:
            raise DesktopBrowserProtocolError(
                f"Element reference {ref or '(missing)'} is not present in the latest Desktop snapshot. "
                "Take a new browser_snapshot and retry with a current ref."
            )
        action.update({"kind": "click" if function_name == "browser_click" else "type", "target": target})
        if function_name == "browser_type":
            action["text"] = str(tool_args.get("text") or "")
        return action

    if function_name == "browser_scroll":
        direction = str(tool_args.get("direction") or "")
        if direction not in {"up", "down"}:
            raise DesktopBrowserProtocolError(
                f"Invalid direction {direction!r}. Use 'up' or 'down'."
            )
        action.update({"amount": 500, "direction": direction, "kind": "scroll"})
        return action

    if function_name == "browser_press":
        key = _clean(tool_args.get("key"), 100) or "Enter"
        action.update({"key": key, "kind": "press"})
        return action

    raise DesktopBrowserProtocolError(
        f"Unsupported Desktop browser action tool: {function_name}"
    )


def desktop_browser_approval_reason(action: dict[str, Any]) -> str | None:
    """Return a user-facing reason when an action may mutate external state."""
    kind = str(action.get("kind") or "")
    if kind == "press" and str(action.get("key") or "").lower() in {"enter", "return"}:
        return "按下 Enter 可能提交评论、私信、表单或发布操作"
    if kind != "click":
        return None

    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    searchable = " ".join(str(target.get(key) or "") for key in _FINGERPRINT_KEYS).lower()
    matched_word = next((word for word in _RISKY_ACTION_WORDS if word in searchable), None)
    if matched_word:
        return f"点击“{target.get('text') or target.get('ariaLabel') or matched_word}”可能触发外部写操作"

    try:
        host = (urllib.parse.urlsplit(str(action.get("expectedUrl") or "")).hostname or "").lower()
    except ValueError:
        host = ""
    tag = str(target.get("tag") or "").lower()
    if host and any(host == suffix or host.endswith(f".{suffix}") for suffix in _DOUYIN_HOST_SUFFIXES) and tag != "a":
        label = target.get("text") or target.get("ariaLabel") or target.get("role") or tag or "控件"
        return f"点击抖音页面控件“{label}”可能改变账号或商家数据"
    return None


def validate_desktop_browser_navigation(url: object) -> tuple[str, str | None]:
    """Apply the local-browser navigation safety floor without opening a second browser."""
    raw = str(url or "").strip()
    if not raw:
        return "", "Navigation URL is empty"
    if raw == "about:blank":
        return raw, None
    if not re.match(r"^[a-zA-Z][a-zA-Z\d+.-]*:", raw):
        raw = f"https://{raw}"

    from agent.redact import _PREFIX_RE
    from tools.url_safety import is_always_blocked_url, normalize_url_for_request
    from tools.website_policy import check_website_access

    decoded = urllib.parse.unquote(raw)
    if _PREFIX_RE.search(raw) or _PREFIX_RE.search(decoded):
        return "", (
            "Blocked: URL contains what appears to be an API key or token. "
            "Secrets must not be sent in URLs."
        )

    normalized = normalize_url_for_request(raw)
    try:
        parsed = urllib.parse.urlsplit(normalized)
    except ValueError:
        return "", "Blocked: URL is invalid"
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return "", "Blocked: Desktop browser navigation only supports HTTP and HTTPS URLs"
    if not parsed.path:
        normalized = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, "/", parsed.query, parsed.fragment)
        )
    if is_always_blocked_url(normalized):
        return "", "Blocked: URL targets a cloud metadata endpoint"

    blocked = check_website_access(normalized)
    if blocked:
        return "", str(blocked.get("message") or "Blocked by website policy")
    return normalized, None


def execute_desktop_browser_tool(
    function_name: str,
    function_args: dict[str, Any],
    callback: Callable[[str, dict[str, Any]], object],
) -> str:
    """Execute an existing browser tool through a configured Desktop bridge."""
    try:
        result = callback(function_name, function_args)
        from tools.browser_tool import _redact_browser_output

        if isinstance(result, str):
            return _redact_browser_output(result)

        return json.dumps(_redact_browser_output(result), ensure_ascii=False)
    except DesktopBrowserProtocolError as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Desktop browser bridge failed: {exc}"},
            ensure_ascii=False,
        )
