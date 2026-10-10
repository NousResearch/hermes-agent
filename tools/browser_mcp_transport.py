"""MCP transport for the browser tool family against hosted gateway backends.

Hosted gateway backends serve API-key callers MCP-only: REST `/tabs` returns
403 `rest_not_available`. This module maps the hermes browser verbs onto the
the gateway's MCP tool surface (`browser_navigate/snapshot/click/type/press/
scroll/back/close/evaluate/screenshot`) with the stateless call shape proven
by the centipede plugin's `mcp_transport.py`:

* every call = one `tools/call` POST to `<backend>/mcp`
* headers: `Authorization: Bearer <key>`, `Mcp-Method: tools/call`,
  `Mcp-Name: <tool>`, full-SSE Accept
* no initialize handshake / session id — the gateway treats each POST as an
  isolated MCP exchange and binds tabs to the caller via the API key

Result text is parsed back into the same JSON shapes the REST verbs return,
so `browser_camofox.py` keeps its external contract (`success`, `snapshot`,
`element_count`, `url`, ...).

`browser_navigate` creates a NEW server-side tab per call on this transport.
The gateway now echoes the created tab id in the navigate payload (``Tab: <id>``
line, wolverine SEK-MT lineage); the transport records it per hermes task session
(`_TAB_REGISTRY`) and every verb forwards the recorded `tab_id` explicitly — no
"most recent tab for the key" implicit targeting, which cross-task interleaves
could misdeliver. Callers may still pass explicit tab ids, which win over the
registry.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import requests

from agent.secret_scope import get_secret
from tools.registry import tool_error

_MCP_TIMEOUT_DEFAULT = 120  # navigate w/ captcha-solve budget lives under this
_MCP_PROTOCOL_VERSION = "2026-07-28"
_REF_RE = re.compile(r"\[([a-zA-Z0-9_]+)\]")

# hermes task session -> gateway tab id. Populated from the navigate payload's
# ``Tab: <id>`` line; consulted by every verb so calls target the task's own tab
# instead of the key's most-recent one (session-isolation fix, PR #135861 review P1-1).
_TAB_REGISTRY: dict[str, str] = {}


def _task_key(task_id: Optional[str]) -> str:
    return task_id or "default"


def set_task_tab(task_id: Optional[str], tab_id: str) -> None:
    """Record the gateway tab a task session owns (after navigate)."""
    if tab_id:
        _TAB_REGISTRY[_task_key(task_id)] = tab_id


def get_task_tab(task_id: Optional[str], explicit: Optional[str] = None) -> Optional[str]:
    """Explicit tab id wins; else the task's recorded tab; else None."""
    if explicit:
        return explicit
    return _TAB_REGISTRY.get(_task_key(task_id))


def drop_task_tab(task_id: Optional[str]) -> None:
    _TAB_REGISTRY.pop(_task_key(task_id), None)


_TAB_LINE_RE = re.compile(r"^Tab:\s*(\S+)\s*$", re.MULTILINE)


def _backend_url() -> str:
    # Generic MCP-browser backend first (any MCP server that speaks the browser-tool surface),
    # then the Camofox-compat gateway lane. `browser.mcp_url` config beats env names.
    try:
        from tools.tool_backend_helpers import browser_mcp_backend_url
        url = browser_mcp_backend_url()
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    url = (get_secret("CAMOFOX_URL", "") or "").rstrip("/")
    return url


def _is_gateway_backend() -> bool:
    """True when the browser backend is a remote MCP gateway (MCP-only, topologically remote).

    Covers the Camofox-compat gateway lane AND the generic ``browser.mcp_url`` lane
    (any MCP browser server configured by URL+key). A non-loopback URL with an auth
    key = remote = MCP-only + SSRF-guard-active.
    """
    url = _backend_url()
    if not url:
        return False
    host = url.split("//", 1)[-1].split("/", 1)[0]
    loopback = host.split(":")[0] in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")
    if (get_secret("BROWSER_MCP_API_KEY", "") or "").strip():
        # Generic browser-MCP lane (browser.mcp_url / BROWSER_MCP_URL): URL+key = remote lane.
        return not loopback
    # Credential lane: CAMOFOX_API_KEY (the community compat gateway uses it too).
    key = (get_secret("CAMOFOX_API_KEY", "") or "").strip()
    gateway = bool(not loopback and key)
    _log_gateway_resolution_once(gateway)
    return gateway


_gateway_resolution_logged = False


def _log_gateway_resolution_once(gateway: bool) -> None:
    """Observability: one INFO naming the resolved browser backend."""
    global _gateway_resolution_logged
    if _gateway_resolution_logged:
        return
    _gateway_resolution_logged = True
    import logging
    logging.getLogger(__name__).info(
        "Browser backend resolved: %s",
        "hosted gateway (MCP transport)" if gateway else "local/community (REST)")


def _call(tool: str, arguments: dict[str, Any], timeout: Optional[int] = None,
          task_id: Optional[str] = None, tab_id: Optional[str] = None) -> dict[str, Any]:
    """One stateless MCP tools/call. Returns the raw JSON-RPC envelope (parsed)."""
    base = _backend_url()
    host = base.split("//", 1)[-1].split("/", 1)[0]
    loopback = host.split(":")[0] in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")
    if not loopback and not base.startswith("https://"):
        # P2 (PR #135861 review): the bearer key must not cross a non-loopback
        # network in cleartext. Loopback http stays allowed for local dev/sidecars.
        raise RuntimeError(
            f"browser backend URL must be HTTPS for non-loopback endpoints (got {base})")
    effective_tab = get_task_tab(task_id, tab_id)
    if effective_tab and effective_tab not in arguments:
        arguments = {**arguments, "tab_id": effective_tab}
    url = f"{base}/mcp"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": _MCP_PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
            },
            "name": tool,
            "arguments": arguments,
        },
    }
    key = (get_secret("BROWSER_MCP_API_KEY", "") or "").strip() or (
        get_secret("CAMOFOX_API_KEY", "") or ""
    ).strip()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Mcp-Method": "tools/call",
        "Mcp-Name": tool,
    }
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=timeout or _MCP_TIMEOUT_DEFAULT)
    except requests.exceptions.Timeout:
        # Cold-boot warm-up: the first call after an idle server boot can take
        # up to ~60s. Surface the retry guidance, not a bare timeout.
        raise RuntimeError(
            "browser call timed out — likely the hosted browser was warming up after "
            "an idle period: retry once, steady-state calls are sub-second"
        ) from None
    if resp.status_code == 401:
        raise PermissionError("browser backend rejected the API key (401) — key expired or revoked")
    resp.raise_for_status()
    raw = resp.text.strip()
    # SSE frame or plain JSON — accept both (gateway content-negotiates).
    if raw.startswith("data:") or "\ndata:" in raw:
        # SSE frames: split on LF ONLY. str.splitlines() also splits on U+0085/U+2028/U+2029,
        # which appear inside JSON string values (page text) and would truncate the line.
        last = [ln[5:].strip() for ln in raw.split("\n") if ln.startswith("data:")][-1]
        env = json.loads(last)
    else:
        env = json.loads(raw)
    if env.get("error"):
        raise RuntimeError(env["error"].get("message", "MCP error"))
    return env.get("result", {})


def _result_text(result: dict[str, Any]) -> str:
    content = result.get("content") or []
    return content[0].get("text", "") if content else ""


def _result_ok(result: dict[str, Any]) -> bool:
    return result.get("isError") is not True


def _mcp_error(result: dict[str, Any], fallback: str) -> str:
    return tool_error(_result_text(result) or fallback, success=False)


def _extract_snapshot(text: str) -> tuple[str, int]:
    """The navigate/snapshot payloads embed the snapshot in text form. Parse
    the trailing snapshot block into `(snapshot, ref_count)`."""
    if "\nSnapshot:\n" in text:
        snap = text.split("\nSnapshot:\n", 1)[1]
    elif text.lstrip().startswith(("-", "heading", "link", "img", "button", "text ", "[e")):
        snap = text
    else:
        snap = ""
    refs = set(_REF_RE.findall(snap))
    return snap, len(refs)


# ---- Verb mappings (same return shapes as the REST paths in browser_camofox.py) ----

def mcp_navigate(url: str, timeout_secs: int = 30, task_id: Optional[str] = None) -> str:
    try:
        result = _call("browser_navigate", {"url": url, "timeout_secs": timeout_secs})
        if not _result_ok(result):
            return _mcp_error(result, "navigate failed")
        text = _result_text(result)
        tab_match = _TAB_LINE_RE.search(text)
        if tab_match:
            set_task_tab(task_id, tab_match.group(1))
        snap, n_refs = _extract_snapshot(text)
        out: dict[str, Any] = {"success": True, "url": url, "title": "",
                               "snapshot": snap, "element_count": n_refs}
        status_line = next((l for l in text.splitlines() if l.startswith("Status:")), None)
        if status_line:
            out["status"] = status_line.split(":", 1)[1].strip()
        return json.dumps(out)
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def mcp_snapshot(task_id: Optional[str] = None) -> str:
    try:
        result = _call("browser_snapshot", {}, task_id=task_id)
        if not _result_ok(result):
            return _mcp_error(result, "snapshot failed")
        text = _result_text(result)
        snap, n_refs = _extract_snapshot(text)
        return json.dumps({"success": True, "snapshot": snap, "element_count": n_refs})
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def mcp_click(ref: str, task_id: Optional[str] = None) -> str:
    clean = ref.lstrip("@")
    try:
        result = _call("browser_click", {"ref": clean, "return_snapshot": True}, task_id=task_id)
        if not _result_ok(result):
            return _mcp_error(result, f"element not found or click failed for ref {clean}")
        text = _result_text(result)
        snap, _ = _extract_snapshot(text)
        out: dict[str, Any] = {"success": True, "clicked": clean}
        if snap:
            out["snapshot"], out["element_count"] = snap, len(set(_REF_RE.findall(snap)))
        return json.dumps(out)
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def mcp_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    clean = ref.lstrip("@")
    from agent.display import redact_browser_typed_text_for_display, redact_tool_args_for_display
    try:
        result = _call("browser_type", {"ref": clean, "text": text, "return_snapshot": False}, task_id=task_id)
        display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {}).get("text", text)
        if not _result_ok(result):
            err = _mcp_error(result, f"type failed for ref {clean}")
            # tool_error payloads reach history — redact the echoed text there too.
            return redact_browser_typed_text_for_display(err, text)
        # P1-4 (PR #135861 review): never echo the raw typed text back (REST path redacts).
        return json.dumps(redact_browser_typed_text_for_display(
            {"success": True, "typed": display_text, "element": clean}, text))
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return redact_browser_typed_text_for_display(
            tool_error(str(exc), success=False), text)


def mcp_press(key: str, task_id: Optional[str] = None) -> str:
    try:
        result = _call("browser_press", {"key": key, "return_snapshot": False}, task_id=task_id)
        if not _result_ok(result):
            return _mcp_error(result, f"press failed for key {key}")
        return json.dumps({"success": True, "pressed": key})
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def mcp_scroll(direction: str, task_id: Optional[str] = None) -> str:
    try:
        result = _call("browser_scroll", {"direction": direction, "return_snapshot": False}, task_id=task_id)
        if not _result_ok(result):
            return _mcp_error(result, f"scroll failed ({direction})")
        return json.dumps({"success": True, "scrolled": direction})
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def mcp_back(task_id: Optional[str] = None, back_guard: bool = False) -> str:
    """``back_guard=True`` (browser_tool caller): after back, the result URL passes the
    same private-address recheck the REST path applies (review parity break — history
    can land on an intranet/metadata page the navigate preflight never saw)."""
    try:
        result = _call("browser_back", {"return_snapshot": False}, task_id=task_id)
        if not _result_ok(result):
            return _mcp_error(result, "back failed")
        text = _result_text(result)
        # best-effort url extraction from the text payload
        match = re.search(r"https?://\S+", text)
        url = match.group(0).rstrip(".,;") if match else ""
        if back_guard:
            from tools.browser_tool_eval_policy import _url_blocked
            from tools.browser_tool_origin import origin_module as _origin_fn
            if url and _url_blocked(_origin_fn(), url):
                return json.dumps({"success": False, "error": (
                    "Blocked: page URL targets a private or internal address "
                    f"({url}). Browser history navigation (back) landed on this address "
                    "in this browser mode.")}, ensure_ascii=False)
        return json.dumps({"success": True, "url": url})
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def mcp_close(task_id: Optional[str] = None) -> str:
    try:
        result = _call("browser_close", {}, task_id=task_id)
        drop_task_tab(task_id)
        if not _result_ok(result):
            return _mcp_error(result, "close failed")
        return json.dumps({"success": True, "closed": True})
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return json.dumps({"success": True, "closed": True, "warning": str(exc)})


def mcp_evaluate(expression: str, timeout_secs: int = 10, task_id: Optional[str] = None,
                 postprocess: bool = False) -> str:
    """``postprocess=True`` routes the result through the shared eval postprocessor
    (post-eval private-URL recheck + forced redaction) — the browser_tool caller uses
    this; internal guard probes use raw evaluation and must not recurse into guards."""
    try:
        result = _call("browser_evaluate", {"expression": expression, "timeout_secs": timeout_secs}, task_id=task_id)
        if not _result_ok(result):
            return _mcp_error(result, "evaluate failed")
        raw_text = _result_text(result)
        try:
            parsed = json.loads(raw_text) if raw_text.lstrip().startswith(("{", "[", '"')) else raw_text
        except ValueError:
            parsed = raw_text
        out = {"success": True, "result": parsed if isinstance(parsed, str) else json.dumps(parsed)}
        if postprocess:
            from tools.browser_tool import _eval_result_or_blocked, _parse_eval_value
            return _eval_result_or_blocked(task_id or "default", _parse_eval_value(out["result"]), {})
        return json.dumps(out)
    except (requests.RequestException, RuntimeError, PermissionError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def mcp_screenshot_saves_path() -> bool:
    """Screenshot over MCP returns base64 in text; call sites expecting a raw
    response body should decode. (Currently only browser_vision uses it.)"""
    return True


def mcp_screenshot_b64(task_id: Optional[str] = None) -> Optional[bytes]:
    try:
        result = _call("browser_screenshot", {"format": "png"}, task_id=task_id)
        if not _result_ok(result):
            logging.getLogger(__name__).error("mcp_screenshot_b64: backend returned not-ok: %s", str(result)[:300])
            return None
        text = _result_text(result)
        import base64 as _b64
        # Payload may be raw base64 or a data-URI.
        if text.startswith("data:"):
            text = text.split(",", 1)[-1]
        return _b64.b64decode(text)
    except Exception:
        # Never silently swallow: browser_vision surfaces a generic "MCP screenshot failed"
        # and the real cause (cold-boot warm-up timeout, decode error, 401...) died here.
        logging.getLogger(__name__).exception("mcp_screenshot_b64 failed")
        return None