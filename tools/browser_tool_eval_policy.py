"""browser_console(expression=...) policy: SSRF guard gating, private-URL probes,
and the opt-in sensitive-primitive denylist.

Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle.
"""

import re
from typing import Optional
from utils import is_truthy_value
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_session as _session


def _eval_ssrf_guard_active(effective_task_id: str) -> bool:
    """Return True when eval-driven private-network access must be guarded.

    Same gating as ``browser_navigate`` / ``browser_snapshot`` / ``browser_vision``: the SSRF guard
    only matters for non-local backends (cloud browser, or a containerized terminal whose
    browser-on-host reaches networks the terminal can't); skipped for local sidecars / ``allow_private_urls``.
    """
    _bt = _origin()
    return not _cloud._is_local_backend() and not _bt._is_local_sidecar_key(effective_task_id) and not _cloud._allow_private_urls()


def _url_blocked(_bt, url: str) -> bool:
    """True when ``url`` hits the always-blocked cloud-metadata floor or fails the SSRF guard."""
    return _bt._is_always_blocked_url(url) or not _bt._is_safe_url(url)


# URL-shaped literals embedded in a JS expression (http/https only). fetch/XHR/navigate
# to a private host never updates ``location.href``, so the post-eval page-URL recheck
# can't see it; pre-screen the literals instead.
_JS_URL_LITERAL_RE = re.compile(r"""https?://[^\s'"`)\]<>]+""", re.IGNORECASE)


def _expression_targets_private_url(expression: str) -> Optional[str]:
    """Return the first private/always-blocked ``http(s)://`` literal in a JS expression (best-effort), else None."""
    _bt = _origin()
    literals = _JS_URL_LITERAL_RE.findall(expression) if isinstance(expression, str) else []
    return next((c for c in (m.rstrip(".,;") for m in literals) if _url_blocked(_bt, c)), None)


def _current_page_private_url(effective_task_id: str) -> Optional[str]:
    """Return the current page URL when it targets a private/internal address (e.g. after a prior
    ``location.href = '...'`` eval). Fail-open on probe failure, matching the snapshot/vision guards."""
    _bt = _origin()
    try:
        url_result = _session._run_browser_command(effective_task_id, "eval", ["window.location.href"], timeout=5, _engine_override="auto")
        if url_result.get("success"):
            current_url = url_result.get("data", {}).get("result", "").strip().strip('"').strip("'")
            if current_url and _url_blocked(_bt, current_url):
                return current_url
    except Exception as exc:
        _bt.logger.debug("_current_page_private_url: probe failed (%s)", exc)
    return None


_RISKY_BROWSER_EVAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdocument\s*\.\s*cookie\b", re.I), "document.cookie"),
    (re.compile(r"\b(?:localStorage|sessionStorage)\b", re.I), "web storage"),
    (re.compile(r"\bindexedDB\b", re.I), "IndexedDB"),
    (re.compile(r"\bcaches\s*\.\s*(?:open|match|keys)\b", re.I), "Cache Storage"),
    (re.compile(r"\bnavigator\s*\.\s*(?:clipboard|credentials|serviceWorker)\b", re.I), "navigator sensitive API"),
    (re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(", re.I), "network request"),
    (re.compile(r"\bnavigator\s*\.\s*sendBeacon\s*\(", re.I), "network beacon"),
    (re.compile(r"\bdocument\s*\.\s*forms\b.*\bvalue\b", re.I | re.S), "form value extraction"),
    (re.compile(r"\bquerySelector(?:All)?\s*\([^)]*(?:input|textarea|password)[^)]*\).*\bvalue\b", re.I | re.S), "form value extraction"),
)

_JS_STRING_LITERAL_RE = re.compile(r"""'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`""", re.S)


_SENSITIVE_BROWSER_EVAL_TOKENS: tuple[tuple[str, str], ...] = (
    ("cookie", "document.cookie"),
    ("localStorage", "web storage"), ("sessionStorage", "web storage"),
    ("indexedDB", "IndexedDB"), ("caches", "Cache Storage"),
    ("clipboard", "navigator sensitive API"), ("credentials", "navigator sensitive API"),
    ("serviceWorker", "navigator sensitive API"),
    ("fetch", "network request"), ("XMLHttpRequest", "network request"),
    ("WebSocket", "network request"), ("EventSource", "network request"),
    ("sendBeacon", "network beacon"),
)


def _browser_eval_flag(key: str) -> bool:
    """Read boolean ``browser.<key>`` (default False) through the origin's config reader."""
    _bt = _origin()
    return _bt._browser_cfg(key, False, lambda v: is_truthy_value(v, default=False), f"browser.{key} from config")


def _allow_unsafe_browser_evaluate() -> bool:
    """Whether sensitive browser JS evaluation is explicitly allowed (overrides ``restrict_evaluate``)."""
    return _browser_eval_flag("allow_unsafe_evaluate")


def _restrict_browser_evaluate() -> bool:
    """Whether the sensitive-primitive eval denylist is enabled (off by default).

    It blocks the *names* of common primitives (``fetch``, ``cookie``, ``querySelector(...input...)``),
    not actual exfiltration, so it also blocks much legitimate DOM extraction; egress is still gated
    by the SSRF/private-URL guards in ``_browser_eval`` regardless. Opt in via
    ``browser.restrict_evaluate: true`` (e.g. hostile pages with a logged-in profile).
    """
    return _browser_eval_flag("restrict_evaluate")


def _decode_js_string_literal(literal: str) -> str:
    """Best-effort decode of a JS string literal (not a parser): normalizes escapes like ``"co\\x6fkie"``."""
    if len(literal) < 2:
        return literal
    try:
        return bytes(literal[1:-1], "utf-8").decode("unicode_escape")
    except Exception:
        return literal[1:-1]


def _decoded_js_string_literals(expression: str) -> list[str]:
    return [_decode_js_string_literal(match.group(0)) for match in _JS_STRING_LITERAL_RE.finditer(expression)]


def _sensitive_browser_eval_token_reason(expression: str) -> Optional[str]:
    """Risk reason for direct or quoted sensitive primitives: direct spellings alone miss
    ``document["cookie"]`` / ``globalThis["fetch"]``, so tokens are also matched inside the decoded,
    concatenated string literals (catches ``document["coo" + "kie"]``)."""
    literals = "".join(_decoded_js_string_literals(expression)).lower()
    return next((reason for token, reason in _SENSITIVE_BROWSER_EVAL_TOKENS
                 if re.search(rf"\b{re.escape(token)}\b", expression, re.I) or token.lower() in literals), None)


def _risky_browser_eval_reason(expression: str) -> Optional[str]:
    """Return a human-readable reason if a JS expression uses risky primitives."""
    if not expression:
        return None
    hit = next((reason for pattern, reason in _RISKY_BROWSER_EVAL_PATTERNS if pattern.search(expression)), None)
    return hit or _sensitive_browser_eval_token_reason(expression)


def _enforce_browser_eval_policy(expression: str) -> Optional[str]:
    """Block sensitive browser JS evaluation when the opt-in denylist is on (opt-in because it gates on
    primitive *names*; private-address egress is enforced separately in ``_browser_eval``)."""
    if not _restrict_browser_evaluate() or _allow_unsafe_browser_evaluate():
        return None
    reason = _risky_browser_eval_reason(expression)
    if not reason:
        return None
    return ("Blocked: browser_console(expression=...) tried to use sensitive browser "
            f"JavaScript primitive ({reason}) while browser.restrict_evaluate is "
            "enabled. Use browser_snapshot/browser_get_images/browser_console "
            "without expression for normal inspection, or set "
            "browser.restrict_evaluate: false in config.yaml to allow programmatic evaluation.")


def _camofox_current_page_private_url(tab_id: str, user_id: str) -> Optional[str]:
    """Camofox analogue of ``_current_page_private_url`` (evaluate endpoint instead of the CLI). Fail-open
    on probe failure, matching the snapshot/vision guards — do not make fail-closed without the sibling."""
    _bt = _origin()
    try:
        from tools.browser_camofox import _post
        data = _post(f"/tabs/{tab_id}/evaluate", body={"expression": "window.location.href", "userId": user_id})
        current_url = str(data.get("result") if isinstance(data, dict) else data or "")
        current_url = current_url.strip().strip('"').strip("'")
        if current_url and _url_blocked(_bt, current_url):
            return current_url
    except Exception as exc:
        _bt.logger.debug("_camofox_current_page_private_url: probe failed (%s)", exc)
    return None
