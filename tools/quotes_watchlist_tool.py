#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quotes watchlist / preferences management for Hermes Web UI.

This tool is designed for skills/subagents to maintain the /hermes/quotes watchlist
without relying on frontend-only state.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tools.registry import registry, tool_error


def _webui_base() -> str:
    return str(os.getenv("HERMES_WEB_UI_BASE", "http://127.0.0.1:8648")).strip().rstrip("/")


def _has_webui_base() -> bool:
    b = _webui_base()
    return bool(b and (b.startswith("http://") or b.startswith("https://")))


def _http_json(method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout_sec: float = 10.0) -> Dict[str, Any]:
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = Request(
        url,
        data=data,
        method=str(method).upper(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        try:
            raw = e.read() or b""
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = str(e)
        raise RuntimeError(f"HTTP {getattr(e, 'code', '?')}: {text[:500]}")
    except URLError as e:
        raise RuntimeError(f"Network error: {e}")


QUOTES_WATCHLIST_SCHEMA = {
    "name": "quotes_watchlist",
    "description": "管理 Hermes Web UI 的 /hermes/quotes 自选股与刷新设置（落盘）。支持 list/add/remove/get_prefs/set_prefs/replace。",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型：list | add | remove | replace | get_prefs | set_prefs",
            },
            "code": {
                "type": "string",
                "description": "股票代码（6位数字）。用于 add/remove。",
            },
            "watchlist": {
                "type": "array",
                "items": {"type": "string"},
                "description": "用于 replace：完整替换自选股列表（6位数字数组）。",
            },
            "auto_refresh": {
                "type": "boolean",
                "description": "用于 set_prefs：是否自动刷新。",
            },
            "refresh_interval_seconds": {
                "type": "integer",
                "description": "用于 set_prefs：刷新间隔（秒）。",
                "minimum": 1,
                "maximum": 3600,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "HTTP 超时秒数，默认 10。",
                "default": 10,
            },
        },
        "required": ["action"],
    },
}


def _handler(args: Dict[str, Any], **_kw: Any) -> str:
    if not _has_webui_base():
        return tool_error("HERMES_WEB_UI_BASE 未配置且默认值不可用", success=False)

    base = _webui_base()
    action = str(args.get("action", "")).strip().lower()
    timeout = float(args.get("timeout_seconds") or 10.0)

    try:
        if action == "list":
            return json.dumps(_http_json("GET", f"{base}/api/hermes/quotes/watchlist", timeout_sec=timeout), ensure_ascii=False)
        if action == "add":
            code = str(args.get("code", "")).strip()
            return json.dumps(_http_json("POST", f"{base}/api/hermes/quotes/watchlist/add", {"code": code}, timeout_sec=timeout), ensure_ascii=False)
        if action == "remove":
            code = str(args.get("code", "")).strip()
            return json.dumps(_http_json("DELETE", f"{base}/api/hermes/quotes/watchlist/{code}", timeout_sec=timeout), ensure_ascii=False)
        if action == "replace":
            watchlist = args.get("watchlist", [])
            return json.dumps(_http_json("PUT", f"{base}/api/hermes/quotes/prefs", {"watchlist": watchlist}, timeout_sec=timeout), ensure_ascii=False)
        if action == "get_prefs":
            return json.dumps(_http_json("GET", f"{base}/api/hermes/quotes/prefs", timeout_sec=timeout), ensure_ascii=False)
        if action == "set_prefs":
            payload: Dict[str, Any] = {}
            if "auto_refresh" in args:
                payload["auto_refresh"] = bool(args.get("auto_refresh"))
            if "refresh_interval_seconds" in args:
                payload["refresh_interval_seconds"] = int(args.get("refresh_interval_seconds"))
            return json.dumps(_http_json("PUT", f"{base}/api/hermes/quotes/prefs", payload, timeout_sec=timeout), ensure_ascii=False)

        return tool_error(f"未知 action: {action}", success=False)
    except Exception as e:
        return tool_error(f"quotes_watchlist failed: {str(e)}", success=False)


registry.register(
    name="quotes_watchlist",
    toolset="quotes",
    schema=QUOTES_WATCHLIST_SCHEMA,
    handler=_handler,
    check_fn=_has_webui_base,
    # Optional: defaults to http://127.0.0.1:8648
    requires_env=[],
    emoji="📈",
    max_result_size_chars=100_000,
)

