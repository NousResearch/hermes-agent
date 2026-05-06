#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoTraders / WinAPI relay tools for Hermes Agent.

This integrates the Windows-only tqcenter (and full Evo upstream APIs) into
the native hermes-agent tool registry by calling a WinAPI HTTP relay.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tools.registry import registry, tool_error


def _get_base() -> str:
    # Do not hardcode org-internal IPs here; require explicit opt-in via env.
    return str(os.getenv("EVOTRADERS_WINAPI_BASE", "")).strip().rstrip("/")


def _use_proxy_prefix() -> bool:
    v = str(os.getenv("EVOTRADERS_USE_PROXY_PREFIX", "1")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def _has_base() -> bool:
    b = _get_base()
    return bool(b and (b.startswith("http://") or b.startswith("https://")))


def _thsquant_base() -> str:
    # Prefer explicit env. Otherwise infer host from EVOTRADERS_WINAPI_BASE and use :19090
    env = str(os.getenv("THSQUANT_BASE", "")).strip().rstrip("/")
    if env:
        return env
    try:
        from urllib.parse import urlparse

        p = urlparse(_get_base() or "")
        host = p.hostname or ""
        if host:
            return f"http://{host}:19090"
    except Exception:
        pass
    return ""


def _has_thsquant_base() -> bool:
    b = _thsquant_base()
    return bool(b and (b.startswith("http://") or b.startswith("https://")))


def _decode_bytes(raw: bytes) -> str:
    # Win-side may respond with GBK text; tolerate both.
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def _http_json(method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout_sec: float = 20.0) -> Dict[str, Any]:
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        method=str(method).upper(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    t0 = time.time()
    try:
        with urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            raw = resp.read()
            status = int(getattr(resp, "status", 200))
    except HTTPError as e:
        raw = e.read()
        status = int(getattr(e, "code", 500) or 500)
        return {
            "ok": False,
            "status_code": status,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "error": "http_error",
            "raw": _decode_bytes(raw)[:2000],
            "url": url,
        }
    except URLError as e:
        return {
            "ok": False,
            "status_code": 0,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "error": f"url_error:{e}",
            "url": url,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "status_code": 0,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "error": f"request_failed:{e}",
            "url": url,
        }

    text = _decode_bytes(raw)
    try:
        obj = json.loads(text) if text.strip() else {}
    except Exception:
        obj = {"raw": text[:2000]}
    return {
        "ok": status < 400,
        "status_code": status,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "data": obj,
        "url": url,
    }


def _payload_ok(resp: Dict[str, Any]) -> bool:
    """Best-effort success detection across wrapper and payload layers."""
    if not isinstance(resp, dict):
        return False
    if not bool(resp.get("ok")):
        return False
    data = resp.get("data")
    # Some endpoints return {"ok": false, ...} in payload with HTTP 200.
    if isinstance(data, dict) and ("ok" in data):
        return bool(data.get("ok"))
    return True


def _post_with_retry(url: str, body: Dict[str, Any], timeout_sec: float, attempts: int = 2) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    for i in range(max(1, int(attempts))):
        last = _http_json("POST", url, body=body, timeout_sec=float(timeout_sec))
        if _payload_ok(last):
            return last
        # simple short backoff for transient bridge hiccups
        if i < attempts - 1:
            time.sleep(0.15 * (i + 1))
    return last


def _request_with_candidates(
    method: str,
    paths: list[str],
    *,
    body: Optional[Dict[str, Any]] = None,
    query: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 20.0,
) -> Dict[str, Any]:
    base = _get_base()
    q = query if isinstance(query, dict) else {}
    qs = urlencode({k: "" if v is None else str(v) for k, v in q.items()}, doseq=True)
    attempts = []
    for p in paths:
        raw = "/" + str(p or "").lstrip("/")
        full = f"{base}{raw}"
        if qs:
            full = f"{full}?{qs}"
        r = _http_json(method, full, body=body, timeout_sec=float(timeout_sec))
        attempts.append(r)
        if _payload_ok(r):
            r["endpoint_path"] = raw
            return r
    return {"ok": False, "error": "all_candidates_failed", "attempts": attempts}


def evotraders_health(timeout_sec: float = 5.0) -> str:
    if not _has_base():
        return tool_error(
            "EVOTRADERS_WINAPI_BASE is not configured. Set EVOTRADERS_WINAPI_BASE like: http://192.168.x.x:18880",
            success=False,
        )
    base = _get_base()
    out = _http_json("GET", f"{base}/v1/health", timeout_sec=float(timeout_sec))
    return json.dumps(out, ensure_ascii=False)


def evotraders_tq_call(method: str, params: Optional[Dict[str, Any]] = None, timeout_sec: float = 30.0) -> str:
    if not _has_base():
        return tool_error(
            "EVOTRADERS_WINAPI_BASE is not configured. Set EVOTRADERS_WINAPI_BASE like: http://192.168.x.x:18880",
            success=False,
        )
    m = str(method or "").strip()
    if not m:
        return tool_error("method is required", success=False)
    base = _get_base()
    body = params if isinstance(params, dict) else {}
    direct_url = f"{base}/v1/tq/{m}"
    proxy_tq_url = f"{base}/v1/proxy/v1/tq/{m}"

    # 1) direct tq endpoint (with retry)
    direct = _post_with_retry(direct_url, body=body, timeout_sec=float(timeout_sec), attempts=2)
    if _payload_ok(direct):
        direct["relay"] = "direct_tq"
        return json.dumps(direct, ensure_ascii=False)

    # 2) explicit proxy tq fallback (with retry)
    proxy = _post_with_retry(proxy_tq_url, body=body, timeout_sec=float(timeout_sec), attempts=2)
    proxy["relay"] = "proxy_tq_fallback"
    if _payload_ok(proxy):
        return json.dumps(proxy, ensure_ascii=False)

    # 3) structured merged error so model can reason about fallback status
    merged = {
        "ok": False,
        "status_code": proxy.get("status_code", direct.get("status_code", 0)),
        "error": "tq_call_failed_after_fallback",
        "method": m,
        "params": body,
        "attempts": [
            {"relay": "direct_tq", "status_code": direct.get("status_code"), "error": direct.get("error", "")},
            {"relay": "proxy_tq_fallback", "status_code": proxy.get("status_code"), "error": proxy.get("error", "")},
        ],
        "direct": direct,
        "proxy": proxy,
    }
    return json.dumps(merged, ensure_ascii=False)


def evotraders_proxy_call(
    subpath: str,
    http_method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 30.0,
) -> str:
    if not _has_base():
        return tool_error(
            "EVOTRADERS_WINAPI_BASE is not configured. Set EVOTRADERS_WINAPI_BASE like: http://192.168.x.x:18880",
            success=False,
        )
    sp = str(subpath or "").strip().lstrip("/")
    if not sp:
        return tool_error("subpath is required (e.g. 'v1/quant/runtime/metrics')", success=False)
    base = _get_base()
    prefix = "/v1/proxy" if _use_proxy_prefix() else ""
    out = _http_json(str(http_method), f"{base}{prefix}/{sp}", body=body, timeout_sec=float(timeout_sec))
    return json.dumps(out, ensure_ascii=False)


def evotraders_market_mainline(max_age_sec: int = 60, timeout_sec: float = 20.0) -> str:
    """Convenience helper for A-share mainline context.

    Tries canonical market sentiment endpoint through proxy first, then direct path.
    """
    if not _has_base():
        return tool_error(
            "EVOTRADERS_WINAPI_BASE is not configured. Set EVOTRADERS_WINAPI_BASE like: http://192.168.x.x:18880",
            success=False,
        )
    base = _get_base()
    q = f"max_age_sec={max(1, int(max_age_sec))}"
    urls = [
        f"{base}/v1/proxy/v1/market/sentiment-context?{q}",
        f"{base}/v1/market/sentiment-context?{q}",
    ]
    attempts = []
    for idx, u in enumerate(urls):
        r = _http_json("GET", u, body=None, timeout_sec=float(timeout_sec))
        attempts.append(r)
        if _payload_ok(r):
            r["relay"] = "proxy_first" if idx == 0 else "direct_fallback"
            return json.dumps(r, ensure_ascii=False)
    return json.dumps(
        {
            "ok": False,
            "error": "market_mainline_fetch_failed",
            "attempts": attempts,
        },
        ensure_ascii=False,
    )


def evotraders_wenda_query(
    message: str,
    page_no: int = 1,
    page_size: int = 30,
    rang: str = "A",
    timeout_sec: float = 25.0,
) -> str:
    if not _has_base():
        return tool_error("EVOTRADERS_WINAPI_BASE is not configured", success=False)
    msg = str(message or "").strip()
    if not msg:
        return tool_error("message is required", success=False)
    body = {
        "message": msg,
        "pageNo": str(max(1, int(page_no))),
        "pageSize": str(max(1, int(page_size))),
        "rang": str(rang or "A").strip() or "A",
    }
    # Prefer Evo function-chain method first (not UI endpoint).
    fn_first = _call_json(
        evotraders_tq_call,
        method="tdx_wenda_query_tool",
        params={
            "message": msg,
            "rang": body["rang"],
            "page_no": body["pageNo"],
            "page_size": body["pageSize"],
            "timeout_sec": float(timeout_sec),
        },
        timeout_sec=float(timeout_sec),
    )
    if _payload_ok(fn_first):
        fn_first["relay"] = "function_chain_tdx_wenda_query_tool"
        return json.dumps(fn_first, ensure_ascii=False)

    out = _request_with_candidates(
        "POST",
        paths=[
            "/v1/proxy/v1/hermes/tools/wenda-query",
            "/v1/proxy/v1/hot-standby/tools/wenda-query",
            "/v1/hermes/tools/wenda-query",
            "/v1/hot-standby/tools/wenda-query",
        ],
        body=body,
        timeout_sec=float(timeout_sec),
    )
    out["fallback_from"] = "function_chain_tdx_wenda_query_tool"
    return json.dumps(out, ensure_ascii=False)


def evotraders_indicator_select(
    query: str,
    topk: int = 10,
    timeout_sec: float = 20.0,
) -> str:
    if not _has_base():
        return tool_error("EVOTRADERS_WINAPI_BASE is not configured", success=False)
    q = str(query or "").strip()
    if not q:
        return tool_error("query is required", success=False)
    # Prefer Evo function-chain method first (not UI endpoint).
    fn_first = _call_json(
        evotraders_tq_call,
        method="tdx_indicator_select_tool",
        params={
            "message": q,
            "rang": "AG",
            "timeout_sec": float(timeout_sec),
        },
        timeout_sec=float(timeout_sec),
    )
    if _payload_ok(fn_first):
        fn_first["relay"] = "function_chain_tdx_indicator_select_tool"
        return json.dumps(fn_first, ensure_ascii=False)

    out = _request_with_candidates(
        "POST",
        paths=[
            "/v1/proxy/v1/hermes/tools/indicator-select",
            "/v1/proxy/v1/hot-standby/tools/indicator-select",
            "/v1/hermes/tools/indicator-select",
            "/v1/hot-standby/tools/indicator-select",
        ],
        body={"q": q, "query": q, "topk": max(1, int(topk))},
        timeout_sec=float(timeout_sec),
    )
    out["fallback_from"] = "function_chain_tdx_indicator_select_tool"
    return json.dumps(out, ensure_ascii=False)


def evotraders_tqlex_public_call(
    entry: str,
    params_json: str = "[]",
    full_body_json: str = "",
    timeout_sec: float = 30.0,
    persist: bool = False,
) -> str:
    if not _has_base():
        return tool_error("EVOTRADERS_WINAPI_BASE is not configured", success=False)
    ent = str(entry or "").strip()
    if not ent:
        return tool_error("entry is required", success=False)
    # Exact reuse of Evo function chain: get_tqlex_public_data_tool(...)
    out = _call_json(
        evotraders_tq_call,
        method="get_tqlex_public_data_tool",
        params={
            "entry": ent,
            "params_json": str(params_json or "[]"),
            "full_body_json": str(full_body_json or ""),
            "timeout_sec": float(timeout_sec),
            "persist": bool(persist),
        },
        timeout_sec=float(timeout_sec),
    )
    out["relay"] = "function_chain_get_tqlex_public_data_tool"
    return json.dumps(out, ensure_ascii=False)


def evotraders_iwencai_query(
    query: str,
    page: int = 1,
    limit: int = 20,
    timeout_sec: float = 25.0,
) -> str:
    if not _has_base():
        return tool_error("EVOTRADERS_WINAPI_BASE is not configured", success=False)
    q = str(query or "").strip()
    if not q:
        return tool_error("query is required", success=False)
    out = _request_with_candidates(
        "POST",
        paths=[
            "/v1/proxy/v1/hot-standby/skills/iwencai-store/connectivity-check",
            "/v1/hot-standby/skills/iwencai-store/connectivity-check",
        ],
        body={"query": q, "page": str(max(1, int(page))), "limit": str(max(1, int(limit)))},
        timeout_sec=float(timeout_sec),
    )
    return json.dumps(out, ensure_ascii=False)


def evotraders_ths_bigorder(
    code: str,
    trade_date: str = "",
    timeout_sec: float = 20.0,
) -> str:
    if not _has_base():
        return tool_error("EVOTRADERS_WINAPI_BASE is not configured", success=False)
    c = str(code or "").strip()
    if not c:
        return tool_error("code is required", success=False)
    q = {"code": c}
    if str(trade_date or "").strip():
        q["date"] = str(trade_date).strip()
    out = _request_with_candidates(
        "GET",
        paths=[
            "/v1/proxy/api/ths/bigorder/json",
            "/api/ths/bigorder/json",
        ],
        query=q,
        timeout_sec=float(timeout_sec),
    )
    return json.dumps(out, ensure_ascii=False)


def evotraders_trade_query(kind: str = "account", timeout_sec: float = 20.0) -> str:
    if not _has_base():
        return tool_error("EVOTRADERS_WINAPI_BASE is not configured", success=False)
    k = str(kind or "account").strip().lower()
    if k == "account":
        paths = ["/v1/proxy/v1/trade/account", "/v1/trade/account"]
        body = {}
    elif k == "positions":
        paths = ["/v1/proxy/v1/trade/positions", "/v1/trade/positions"]
        body = {}
    elif k == "orders":
        paths = ["/v1/proxy/v1/trade/orders", "/v1/trade/orders", "/v1/proxy/v1/trade/orders/today", "/v1/trade/orders/today"]
        body = {}
    elif k == "orders_history":
        paths = ["/v1/proxy/v1/trade/orders/history", "/v1/trade/orders/history"]
        body = {}
    else:
        return tool_error("kind must be one of: account, positions, orders, orders_history", success=False)
    # Evo trade mainline is mostly POST /v1/trade/*. Use POST first, GET fallback for compatibility.
    out = _request_with_candidates("POST", paths=paths, body=body, timeout_sec=float(timeout_sec))
    if not _payload_ok(out):
        out = _request_with_candidates("GET", paths=paths, timeout_sec=float(timeout_sec))
    out["kind"] = k
    return json.dumps(out, ensure_ascii=False)


def evotraders_trade_buy(
    stock_code: str,
    price: float,
    order_volume: int,
    account_id: int = -1,
    price_type: int = 0,
    confirm: bool = False,
    timeout_sec: float = 20.0,
) -> str:
    """Place buy order through TQ trade method."""
    if not confirm:
        return tool_error("buy order blocked: set confirm=true to execute", success=False)
    params = {
        "account_id": int(account_id),
        "stock_code": str(stock_code or "").strip(),
        "order_type": 0,  # buy
        "order_volume": int(order_volume),
        "price_type": int(price_type),
        "price": float(price),
    }
    if not params["stock_code"]:
        return tool_error("stock_code is required", success=False)
    out = _call_json(evotraders_tq_call, method="order_stock", params=params, timeout_sec=float(timeout_sec))
    out["trade_action"] = "buy"
    return json.dumps(out, ensure_ascii=False)


def evotraders_trade_sell(
    stock_code: str,
    price: float,
    order_volume: int,
    account_id: int = -1,
    price_type: int = 0,
    confirm: bool = False,
    timeout_sec: float = 20.0,
) -> str:
    """Place sell order through TQ trade method."""
    if not confirm:
        return tool_error("sell order blocked: set confirm=true to execute", success=False)
    params = {
        "account_id": int(account_id),
        "stock_code": str(stock_code or "").strip(),
        "order_type": 1,  # sell
        "order_volume": int(order_volume),
        "price_type": int(price_type),
        "price": float(price),
    }
    if not params["stock_code"]:
        return tool_error("stock_code is required", success=False)
    out = _call_json(evotraders_tq_call, method="order_stock", params=params, timeout_sec=float(timeout_sec))
    out["trade_action"] = "sell"
    return json.dumps(out, ensure_ascii=False)


def evotraders_trade_cancel(
    stock_code: str,
    order_id: str,
    account_id: int = -1,
    confirm: bool = False,
    timeout_sec: float = 20.0,
) -> str:
    """Cancel order through TQ trade method."""
    if not confirm:
        return tool_error("cancel order blocked: set confirm=true to execute", success=False)
    sc = str(stock_code or "").strip()
    oid = str(order_id or "").strip()
    if not sc:
        return tool_error("stock_code is required", success=False)
    if not oid:
        return tool_error("order_id is required", success=False)
    params = {
        "account_id": int(account_id),
        "stock_code": sc,
        "order_id": oid,
    }
    out = _call_json(evotraders_tq_call, method="cancel_stock_order", params=params, timeout_sec=float(timeout_sec))
    out["trade_action"] = "cancel"
    return json.dumps(out, ensure_ascii=False)


def evotraders_trade_verify_bundle(
    timeout_sec: float = 25.0,
) -> str:
    """One-shot trade verification bundle to avoid false 'empty position' conclusion."""
    account = _call_json(evotraders_trade_query, kind="account", timeout_sec=float(timeout_sec))
    positions = _call_json(evotraders_trade_query, kind="positions", timeout_sec=float(timeout_sec))
    orders = _call_json(evotraders_trade_query, kind="orders", timeout_sec=float(timeout_sec))
    orders_history = _call_json(evotraders_trade_query, kind="orders_history", timeout_sec=float(timeout_sec))

    def _ok(x: Dict[str, Any]) -> bool:
        return bool(isinstance(x, dict) and x.get("ok"))

    def _extract_rows(x: Dict[str, Any]) -> int:
        if not isinstance(x, dict):
            return 0
        if isinstance(x.get("data"), list):
            return len(x.get("data") or [])
        if isinstance(x.get("data"), dict):
            return len(x.get("data") or {})
        return int(x.get("rows", 0) or 0)

    account_ok = _ok(account)
    positions_ok = _ok(positions)
    orders_ok = _ok(orders)
    history_ok = _ok(orders_history)
    positions_rows = _extract_rows(positions)

    if positions_ok and positions_rows > 0:
        status = "has_positions"
        reason = "positions_non_empty"
    elif positions_ok and positions_rows == 0 and account_ok:
        status = "likely_empty_positions"
        reason = "positions_empty_but_account_ok"
    elif (not positions_ok) and (account_ok or orders_ok or history_ok):
        status = "positions_call_failed_but_trade_link_alive"
        reason = "partial_trade_chain_ok"
    else:
        status = "trade_chain_unhealthy"
        reason = "all_or_most_calls_failed"

    out = {
        "ok": True,
        "schema": "evotraders_trade_verify_bundle.v1",
        "status": status,
        "reason": reason,
        "summary": {
            "account_ok": account_ok,
            "positions_ok": positions_ok,
            "orders_ok": orders_ok,
            "orders_history_ok": history_ok,
            "positions_rows": positions_rows,
        },
        "results": {
            "account": account,
            "positions": positions,
            "orders": orders,
            "orders_history": orders_history,
        },
    }
    return json.dumps(out, ensure_ascii=False)


def evotraders_thsquant_health(timeout_sec: float = 6.0) -> str:
    base = _thsquant_base()
    if not base:
        return tool_error("THSQUANT_BASE is not configured (e.g. http://192.168.100.168:19090)", success=False)
    out = _http_json("GET", f"{base}/v1/system/health", body=None, timeout_sec=float(timeout_sec))
    return json.dumps(out, ensure_ascii=False)


def evotraders_thsquant_trade_query(kind: str = "positions", timeout_sec: float = 20.0) -> str:
    base = _thsquant_base()
    if not base:
        return tool_error("THSQUANT_BASE is not configured (e.g. http://192.168.100.168:19090)", success=False)
    k = str(kind or "positions").strip().lower()
    if k == "account":
        path = "/v1/trade/account"
        body = {}
    elif k == "positions":
        path = "/v1/trade/positions"
        body = {}
    elif k == "orders_today":
        path = "/v1/trade/orders/today"
        body = {}
    elif k == "orders_history":
        path = "/v1/trade/orders/history"
        body = {}
    else:
        return tool_error("kind must be one of: account, positions, orders_today, orders_history", success=False)
    out = _http_json("POST", f"{base}{path}", body=body, timeout_sec=float(timeout_sec))
    out["kind"] = k
    return json.dumps(out, ensure_ascii=False)


def evotraders_thsquant_trade_verify_bundle(timeout_sec: float = 25.0) -> str:
    """One-shot thsQuant verification bundle (health + account/positions/orders)."""
    health = _call_json(evotraders_thsquant_health, timeout_sec=min(8.0, float(timeout_sec)))
    account = _call_json(evotraders_thsquant_trade_query, kind="account", timeout_sec=float(timeout_sec))
    positions = _call_json(evotraders_thsquant_trade_query, kind="positions", timeout_sec=float(timeout_sec))
    orders_today = _call_json(evotraders_thsquant_trade_query, kind="orders_today", timeout_sec=float(timeout_sec))
    orders_history = _call_json(evotraders_thsquant_trade_query, kind="orders_history", timeout_sec=float(timeout_sec))

    def _ok(x: Dict[str, Any]) -> bool:
        return bool(isinstance(x, dict) and x.get("ok"))

    def _rows(x: Dict[str, Any]) -> int:
        if not isinstance(x, dict):
            return 0
        if isinstance(x.get("data"), list):
            return len(x.get("data") or [])
        if isinstance(x.get("data"), dict):
            return len(x.get("data") or {})
        return int(x.get("rows", 0) or 0)

    positions_ok = _ok(positions)
    positions_rows = _rows(positions)
    if positions_ok and positions_rows > 0:
        status = "has_positions"
    elif positions_ok and positions_rows == 0:
        status = "positions_empty"
    else:
        status = "positions_unavailable"

    out = {
        "ok": True,
        "schema": "evotraders_thsquant_trade_verify_bundle.v1",
        "base": _thsquant_base(),
        "status": status,
        "summary": {
            "health_ok": _ok(health),
            "account_ok": _ok(account),
            "positions_ok": positions_ok,
            "positions_rows": positions_rows,
            "orders_today_ok": _ok(orders_today),
            "orders_history_ok": _ok(orders_history),
        },
        "results": {
            "health": health,
            "account": account,
            "positions": positions,
            "orders_today": orders_today,
            "orders_history": orders_history,
        },
    }
    return json.dumps(out, ensure_ascii=False)


def _contains_any(text: str, tokens: list[str]) -> bool:
    t = str(text or "").lower()
    return any(str(x).lower() in t for x in tokens)


def _default_routing_path() -> Path:
    # Local hermes-side routing config (preferred).
    return Path(__file__).resolve().parents[1] / "agent" / "prompt_presets" / "evotraders_implicit_routing.json"


def _evotraders_repo_routing_path() -> Path:
    # Optional fallback to source-of-truth routing in evotraders repo.
    return Path(__file__).resolve().parents[2] / "evotraders" / "skills" / "preinstalled-skills" / "implicit-routing.json"


def _load_implicit_rules() -> list[dict]:
    env_path = str(os.getenv("EVOTRADERS_IMPLICIT_ROUTING_PATH", "")).strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(_default_routing_path())
    candidates.append(_evotraders_repo_routing_path())
    for p in candidates:
        try:
            if not p.is_file():
                continue
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                rows = raw.get("rules", [])
            else:
                rows = raw
            if isinstance(rows, list):
                out = []
                for r in rows:
                    if isinstance(r, dict) and str(r.get("rule_id", "")).strip():
                        out.append(_normalize_implicit_rule(r))
                if out:
                    return out
        except Exception:
            continue
    return []


def _normalize_implicit_rule(rule: dict) -> dict:
    r = dict(rule or {})
    rid = str(r.get("rule_id", "")).strip()
    if not rid:
        return r
    r["rule_id"] = rid
    r["enabled"] = bool(r.get("enabled", True))
    r["name"] = str(r.get("name", rid)).strip()
    r["description"] = str(r.get("description", "")).strip()
    r["priority"] = int(r.get("priority", 0) or 0)
    r["weight"] = float(r.get("weight", 1.0) or 1.0)
    r["min_score"] = float(r.get("min_score", 0.0) or 0.0)
    r["requires_ticker"] = bool(r.get("requires_ticker", False))
    r["regex_keywords"] = [str(x) for x in (r.get("regex_keywords") or []) if str(x).strip()]
    r["keywords"] = [str(x) for x in (r.get("keywords") or []) if str(x).strip()]
    r["negative_keywords"] = [str(x) for x in (r.get("negative_keywords") or []) if str(x).strip()]
    return r


def _query_has_ticker(query: str) -> bool:
    q = str(query or "")
    # A-share ticker forms: 000001 / 000001.SZ / SH600000
    if re.search(r"\b\d{6}(?:\.(?:SZ|SH|BJ))?\b", q, re.IGNORECASE):
        return True
    if re.search(r"\b(?:SZ|SH|BJ)\d{6}\b", q, re.IGNORECASE):
        return True
    return False


def _extract_ticker(query: str) -> str:
    q = str(query or "")
    m = re.search(r"\b(\d{6})(?:\.(SZ|SH|BJ))?\b", q, re.IGNORECASE)
    if m:
        code = m.group(1)
        exch = (m.group(2) or "").upper()
        return f"{code}.{exch}" if exch else code
    m2 = re.search(r"\b(SZ|SH|BJ)(\d{6})\b", q, re.IGNORECASE)
    if m2:
        return f"{m2.group(2)}.{m2.group(1).upper()}"
    return ""


def _is_stock_analysis_query(query_lower: str) -> bool:
    ql = str(query_lower or "")
    if not _query_has_ticker(ql):
        return False
    positive = [
        "分析", "怎么看", "看法", "建议", "诊断", "走势", "趋势", "支撑", "压力",
        "买点", "卖点", "仓位", "止损", "目标价", "复盘", "交易计划",
        "帮我看", "看一下", "适不适合", "值得买", "能不能买",
    ]
    negative = [
        "选股", "筛选", "板块", "行业", "概念", "新闻", "公告", "研报", "宏观",
        "指数", "问财", "iwencai",
    ]
    return _contains_any(ql, positive) and (not _contains_any(ql, negative))


def _is_explicit_order_intent(
    query_lower: str,
    *,
    stock_code: str,
    price: float,
    order_volume: int,
    confirm: bool,
) -> bool:
    ql = str(query_lower or "")
    strong_order_tokens = [
        "下单", "委托", "提交订单", "立即买入", "立即卖出", "执行买入", "执行卖出",
        "buy now", "sell now", "place order", "submit order",
    ]
    if _contains_any(ql, strong_order_tokens):
        return True
    has_order_params = bool(str(stock_code or "").strip()) and float(price or 0.0) > 0 and int(order_volume or 0) > 0
    if bool(confirm) and has_order_params:
        return True
    return False


def _infer_trade_kind_from_query(query_lower: str) -> str:
    ql = str(query_lower or "")
    if _contains_any(ql, ["持仓", "positions", "position"]):
        return "positions"
    if _contains_any(ql, ["历史委托", "orders_history", "history order", "history"]):
        return "orders_history"
    if _contains_any(ql, ["委托", "成交", "orders", "order"]):
        return "orders"
    return "account"


def _match_rule(query: str, rule: dict) -> bool:
    q = str(query or "")
    ql = q.lower()
    negatives = [str(x) for x in (rule.get("negative_keywords") or []) if str(x).strip()]
    if _contains_any(ql, negatives):
        return False
    keywords = [str(x) for x in (rule.get("keywords") or []) if str(x).strip()]
    if keywords and _contains_any(ql, keywords):
        return True
    regexes = [str(x) for x in (rule.get("regex_keywords") or []) if str(x).strip()]
    for pat in regexes:
        try:
            if re.search(pat, q, re.IGNORECASE):
                return True
        except Exception:
            continue
    return False


def _pick_best_rule(query: str, rules: list[dict]) -> Optional[dict]:
    if not rules:
        return None

    scored: list[tuple[float, dict]] = []
    for r in rules:
        if not bool(r.get("enabled", True)):
            continue
        if not _match_rule(query, r):
            continue
        # score = weight + priority bias
        wt = float(r.get("weight", 1.0) or 1.0)
        pr = float(r.get("priority", 0) or 0)
        score = wt + pr * 0.001
        min_score = float(r.get("min_score", 0) or 0)
        if score < min_score:
            continue
        if bool(r.get("requires_ticker", False)) and (not _query_has_ticker(query)):
            continue
        scored.append((score, r))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _call_json(fn, **kwargs) -> Dict[str, Any]:
    try:
        raw = fn(**kwargs)
        obj = json.loads(raw) if isinstance(raw, str) else {}
        return obj if isinstance(obj, dict) else {"ok": False, "error": "non_dict_response"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"route_call_failed:{e}"}


def _is_tqlex_learning_query(query: str) -> bool:
    q = str(query or "").lower()
    tokens = [
        "tqlex",
        "entry",
        "params",
        "wendaquery",
        "infoselectv2",
        "tdxshare",
        "问达",
        "指标查询",
        "参数模板",
    ]
    return any(t in q for t in tokens)


def _tqlex_skill_hint_payload() -> Dict[str, Any]:
    return {
        "skill_autoload": True,
        "recommended_skill_ids": ["tqlex-function-playbook"],
        "skill_path": "skills/quant/tqlex-function-playbook/SKILL.md",
        "quick_templates": {
            "wenda_query": {
                "tool": "evotraders_wenda_query",
                "args": {"message": "涨停", "rang": "AG", "page_no": 1, "page_size": 30},
            },
            "indicator_select": {
                "tool": "evotraders_indicator_select",
                "args": {"query": "上证指数 今日涨跌家数 市场情绪", "topk": 10},
            },
            "tqlex_params": {
                "tool": "evotraders_tqlex_public_call",
                "args": {
                    "entry": "TdxSharePCCW.tdxf10_gg_ybpj",
                    "params_json": "[\"000001\",\"yzyq\"]",
                },
            },
        },
    }


def evotraders_route_and_call(
    query: str,
    prefer_local_tdx: bool = False,
    disable_iwencai: bool = False,
    disable_wenda: bool = False,
    confirm: bool = False,
    stock_code: str = "",
    price: float = 0.0,
    order_volume: int = 0,
    order_id: str = "",
    account_id: int = -1,
    price_type: int = 0,
    timeout_sec: float = 25.0,
) -> str:
    """Intent router aligned to Evo implicit-routing core rules."""
    if not _has_base():
        return tool_error("EVOTRADERS_WINAPI_BASE is not configured", success=False)
    q = str(query or "").strip()
    if not q:
        return tool_error("query is required", success=False)

    ql = q.lower()
    force_no_iwencai = disable_iwencai or _contains_any(ql, ["不要问财", "不用问财"])
    force_no_wenda = disable_wenda or _contains_any(ql, ["不要问达", "不用问达"])
    prefer_local = prefer_local_tdx or _contains_any(ql, ["优先本地", "本地通达信"])

    _route_t0 = time.time()
    route_meta: Dict[str, Any] = {
        "route_version": "v1",
        "query": q,
        "matched_rule_id": "",
        "selected_tool": "",
        "fallback_chain": [],
        "route_flags": {
            "prefer_local_tdx": bool(prefer_local),
            "disable_iwencai": bool(force_no_iwencai),
            "disable_wenda": bool(force_no_wenda),
            "confirm": bool(confirm),
        },
    }
    extracted_ticker = _extract_ticker(q)
    if extracted_ticker:
        route_meta["extracted_ticker"] = extracted_ticker
    if _is_tqlex_learning_query(q):
        route_meta["learning_hint"] = _tqlex_skill_hint_payload()

    rules = _load_implicit_rules()
    matched_rule = _pick_best_rule(q, rules)
    explicit_order_intent = _is_explicit_order_intent(
        ql,
        stock_code=stock_code,
        price=price,
        order_volume=order_volume,
        confirm=confirm,
    )
    if matched_rule:
        _rid = str(matched_rule.get("rule_id", "")).strip()
        if _rid in {"rh-trade-buy-core", "rh-trade-sell-core"} and (not explicit_order_intent) and _is_stock_analysis_query(ql):
            matched_rule = None

    if matched_rule:
        rid = str(matched_rule.get("rule_id", "")).strip()
        route_meta["matched_rule_id"] = rid

        if rid == "ths-trade-core":
            route_meta["selected_tool"] = "evotraders_thsquant_trade_verify_bundle"
            out = _call_json(evotraders_thsquant_trade_verify_bundle, timeout_sec=timeout_sec)
            return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

        if rid == "trade-query-core":
            route_meta["selected_tool"] = "evotraders_trade_verify_bundle"
            out = _call_json(evotraders_trade_verify_bundle, timeout_sec=timeout_sec)
            return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

        if rid.startswith("iwencai-") and (not prefer_local) and (not force_no_iwencai):
            route_meta["selected_tool"] = "evotraders_iwencai_query"
            out = _call_json(evotraders_iwencai_query, query=q, page=1, limit=20, timeout_sec=timeout_sec)
            if (not _payload_ok(out)) and (not force_no_wenda):
                route_meta["fallback_chain"].append("evotraders_wenda_query")
                out = _call_json(evotraders_wenda_query, message=q, timeout_sec=timeout_sec)
            return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

        if rid == "tdx-wenda-screener-core" and (not force_no_wenda):
            route_meta["selected_tool"] = "evotraders_wenda_query"
            out = _call_json(evotraders_wenda_query, message=q, timeout_sec=timeout_sec)
            return json.dumps({**route_meta, "result": out}, ensure_ascii=False)

        if rid == "lianban-tier-analysis-core":
            route_meta["selected_tool"] = "evotraders_market_mainline"
            steps = [
                {"tool": "evotraders_market_mainline", "result": _call_json(evotraders_market_mainline, max_age_sec=60, timeout_sec=timeout_sec)}
            ]
            if not force_no_wenda:
                steps.append({"tool": "evotraders_wenda_query", "result": _call_json(evotraders_wenda_query, message="涨停", timeout_sec=timeout_sec)})
                steps.append({"tool": "evotraders_wenda_query", "result": _call_json(evotraders_wenda_query, message="连板", timeout_sec=timeout_sec)})
            return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": {"ok": True, "steps": steps}}, ensure_ascii=False)

    # Rule: trade buy/sell/cancel
    if _contains_any(ql, ["买入", "下单买", "buy"]) and explicit_order_intent:
        route_meta["matched_rule_id"] = "trade-buy-core"
        route_meta["selected_tool"] = "evotraders_trade_buy"
        out = _call_json(
            evotraders_trade_buy,
            stock_code=stock_code,
            price=float(price),
            order_volume=int(order_volume),
            account_id=int(account_id),
            price_type=int(price_type),
            confirm=bool(confirm),
            timeout_sec=timeout_sec,
        )
        return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

    if _contains_any(ql, ["卖出", "下单卖", "sell"]) and explicit_order_intent:
        route_meta["matched_rule_id"] = "trade-sell-core"
        route_meta["selected_tool"] = "evotraders_trade_sell"
        out = _call_json(
            evotraders_trade_sell,
            stock_code=stock_code,
            price=float(price),
            order_volume=int(order_volume),
            account_id=int(account_id),
            price_type=int(price_type),
            confirm=bool(confirm),
            timeout_sec=timeout_sec,
        )
        return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

    if _contains_any(ql, ["撤单", "取消委托", "cancel order", "cancel"]):
        route_meta["matched_rule_id"] = "trade-cancel-core"
        route_meta["selected_tool"] = "evotraders_trade_cancel"
        out = _call_json(
            evotraders_trade_cancel,
            stock_code=stock_code,
            order_id=order_id,
            account_id=int(account_id),
            confirm=bool(confirm),
            timeout_sec=timeout_sec,
        )
        return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

    # Rule: trade account/positions/orders
    if _contains_any(ql, ["持仓", "账户", "委托", "成交", "orders", "positions", "account"]):
        route_meta["matched_rule_id"] = "trade-query-core"
        route_meta["selected_tool"] = "evotraders_trade_verify_bundle"
        out = _call_json(evotraders_trade_verify_bundle, timeout_sec=timeout_sec)
        return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

    # Rule: enforce quote-first path for single-stock analysis.
    if _is_stock_analysis_query(ql):
        route_meta["matched_rule_id"] = "stock-analysis-evidence-core"
        route_meta["selected_tool"] = "evotraders_tq_call"
        route_meta["route_flags"]["enforce_quote_first"] = True
        stock = extracted_ticker or "000001.SZ"
        steps = [
            {
                "tool": "evotraders_tq_call",
                "method": "get_market_snapshot",
                "result": _call_json(
                    evotraders_tq_call,
                    method="get_market_snapshot",
                    params={"stock_code": stock},
                    timeout_sec=timeout_sec,
                ),
            },
            {
                "tool": "evotraders_tq_call",
                "method": "get_more_info",
                "result": _call_json(
                    evotraders_tq_call,
                    method="get_more_info",
                    params={"stock_code": stock},
                    timeout_sec=timeout_sec,
                ),
            },
        ]
        if (not force_no_wenda) and (not _payload_ok(steps[0]["result"])):
            route_meta["fallback_chain"].append("evotraders_wenda_query")
            steps.append(
                {
                    "tool": "evotraders_wenda_query",
                    "result": _call_json(evotraders_wenda_query, message=q, timeout_sec=timeout_sec),
                }
            )
        return json.dumps(
            {
                **route_meta,
                "route_latency_ms": int((time.time() - _route_t0) * 1000),
                "result": {"ok": True, "stock_code": stock, "steps": steps},
            },
            ensure_ascii=False,
        )

    # Rule: THS bigorder / fund flow
    if _contains_any(ql, ["大单", "资金流", "净流入", "同花顺资金"]):
        route_meta["matched_rule_id"] = "ths-bigorder-core"
        route_meta["selected_tool"] = "evotraders_ths_bigorder"
        code = "000001"
        out = _call_json(evotraders_ths_bigorder, code=code, trade_date="", timeout_sec=timeout_sec)
        if not _payload_ok(out) and not prefer_local:
            route_meta["fallback_chain"].append("evotraders_iwencai_query")
            out = _call_json(evotraders_iwencai_query, query=q, page=1, limit=20, timeout_sec=timeout_sec)
        return json.dumps({**route_meta, "result": out}, ensure_ascii=False)

    # Rule: mainline / lianban tier
    if _contains_any(ql, ["主线", "情绪周期", "连板梯队", "连板结构", "打板", "跟随"]):
        route_meta["matched_rule_id"] = "mainline-lianban-core"
        route_meta["selected_tool"] = "evotraders_market_mainline"
        mainline = _call_json(evotraders_market_mainline, max_age_sec=60, timeout_sec=timeout_sec)
        steps = [{"tool": "evotraders_market_mainline", "result": mainline}]
        if not force_no_wenda:
            steps.append({"tool": "evotraders_wenda_query", "result": _call_json(evotraders_wenda_query, message="涨停", timeout_sec=timeout_sec)})
            steps.append({"tool": "evotraders_wenda_query", "result": _call_json(evotraders_wenda_query, message="连板", timeout_sec=timeout_sec)})
        steps.append({"tool": "evotraders_indicator_select", "result": _call_json(evotraders_indicator_select, query="主线 强势板块 资金流入", topk=10, timeout_sec=timeout_sec)})
        return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": {"ok": True, "steps": steps}}, ensure_ascii=False)

    # Rule: iwencai priority for screener/sector/news/event/etc
    iwencai_intent = _contains_any(
        ql,
        ["问财", "选股", "筛选", "板块", "行业", "概念", "题材", "新闻", "公告", "事件", "研报", "宏观", "指数"],
    )
    if iwencai_intent and (not prefer_local) and (not force_no_iwencai):
        route_meta["matched_rule_id"] = "iwencai-priority-core"
        route_meta["selected_tool"] = "evotraders_iwencai_query"
        out = _call_json(evotraders_iwencai_query, query=q, page=1, limit=20, timeout_sec=timeout_sec)
        if (not _payload_ok(out)) and (not force_no_wenda):
            route_meta["fallback_chain"].append("evotraders_wenda_query")
            out = _call_json(evotraders_wenda_query, message=q, timeout_sec=timeout_sec)
        return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

    # Rule: wenda fallback / explicit wenda
    if (not force_no_wenda) and _contains_any(ql, ["问达", "wendaquery", "涨停", "连板", "炸板", "1进2", "2进3", "3进4", "4进5"]):
        route_meta["matched_rule_id"] = "tdx-wenda-screener-core"
        route_meta["selected_tool"] = "evotraders_wenda_query"
        out = _call_json(evotraders_wenda_query, message=q, timeout_sec=timeout_sec)
        return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)

    # Default: local-leaning market snapshot chain
    route_meta["matched_rule_id"] = "default-local-tq"
    route_meta["selected_tool"] = "evotraders_tq_call"
    out = _call_json(evotraders_tq_call, method="get_market_snapshot", params={"stock_code": "000001.SZ"}, timeout_sec=timeout_sec)
    return json.dumps({**route_meta, "route_latency_ms": int((time.time() - _route_t0) * 1000), "result": out}, ensure_ascii=False)


def evotraders_route_dry_run(
    query: str,
    prefer_local_tdx: bool = False,
    disable_iwencai: bool = False,
    disable_wenda: bool = False,
    confirm: bool = False,
    stock_code: str = "",
    price: float = 0.0,
    order_volume: int = 0,
    order_id: str = "",
    account_id: int = -1,
    price_type: int = 0,
) -> str:
    q = str(query or "").strip()
    if not q:
        return tool_error("query is required", success=False)
    ql = q.lower()
    force_no_iwencai = disable_iwencai or _contains_any(ql, ["不要问财", "不用问财"])
    force_no_wenda = disable_wenda or _contains_any(ql, ["不要问达", "不用问达"])
    prefer_local = prefer_local_tdx or _contains_any(ql, ["优先本地", "本地通达信"])

    rules = _load_implicit_rules()
    matched = _pick_best_rule(q, rules)
    explicit_order_intent = _is_explicit_order_intent(
        ql,
        stock_code=stock_code,
        price=price,
        order_volume=order_volume,
        confirm=confirm,
    )
    if matched:
        _rid = str(matched.get("rule_id", "")).strip()
        if _rid in {"rh-trade-buy-core", "rh-trade-sell-core"} and (not explicit_order_intent) and _is_stock_analysis_query(ql):
            matched = None
    route = {
        "ok": True,
        "schema": "evotraders_route_dry_run.v1",
        "query": q,
        "flags": {
            "prefer_local_tdx": bool(prefer_local),
            "disable_iwencai": bool(force_no_iwencai),
            "disable_wenda": bool(force_no_wenda),
            "confirm": bool(confirm),
        },
        "matched_rule": None,
        "decision": {"selected_tool": "", "fallback_chain": []},
        "rule_count": len(rules),
    }
    extracted_ticker = _extract_ticker(q)
    if extracted_ticker:
        route["extracted_ticker"] = extracted_ticker
    if matched:
        rid = str(matched.get("rule_id", "")).strip()
        route["matched_rule"] = {
            "rule_id": rid,
            "name": str(matched.get("name", "")).strip(),
            "description": str(matched.get("description", "")).strip(),
            "priority": matched.get("priority"),
            "weight": matched.get("weight"),
            "min_score": matched.get("min_score"),
            "requires_ticker": matched.get("requires_ticker"),
        }
        if rid in {"ths-trade-core", "trade-query-core"}:
            route["decision"]["selected_tool"] = "evotraders_trade_verify_bundle"
        elif rid.startswith("iwencai-") and (not prefer_local) and (not force_no_iwencai):
            route["decision"]["selected_tool"] = "evotraders_iwencai_query"
            if not force_no_wenda:
                route["decision"]["fallback_chain"] = ["evotraders_wenda_query"]
        elif rid == "tdx-wenda-screener-core" and (not force_no_wenda):
            route["decision"]["selected_tool"] = "evotraders_wenda_query"
        elif rid == "lianban-tier-analysis-core":
            route["decision"]["selected_tool"] = "evotraders_market_mainline"
            chain = ["evotraders_indicator_select"]
            if not force_no_wenda:
                chain = ["evotraders_wenda_query", "evotraders_wenda_query"] + chain
            route["decision"]["fallback_chain"] = chain
    if not route["decision"]["selected_tool"]:
        if _contains_any(ql, ["买入", "下单买", "buy"]) and explicit_order_intent:
            route["decision"]["selected_tool"] = "evotraders_trade_buy"
            route["decision"]["decision_args"] = {
                "stock_code": stock_code,
                "price": float(price),
                "order_volume": int(order_volume),
                "account_id": int(account_id),
                "price_type": int(price_type),
                "confirm": bool(confirm),
            }
        elif _contains_any(ql, ["卖出", "下单卖", "sell"]) and explicit_order_intent:
            route["decision"]["selected_tool"] = "evotraders_trade_sell"
            route["decision"]["decision_args"] = {
                "stock_code": stock_code,
                "price": float(price),
                "order_volume": int(order_volume),
                "account_id": int(account_id),
                "price_type": int(price_type),
                "confirm": bool(confirm),
            }
        elif _contains_any(ql, ["撤单", "取消委托", "cancel order", "cancel"]):
            route["decision"]["selected_tool"] = "evotraders_trade_cancel"
            route["decision"]["decision_args"] = {
                "stock_code": stock_code,
                "order_id": order_id,
                "account_id": int(account_id),
                "confirm": bool(confirm),
            }
        elif _contains_any(ql, ["持仓", "账户", "委托", "orders", "positions", "account"]):
            route["decision"]["selected_tool"] = "evotraders_trade_verify_bundle"
        elif _is_stock_analysis_query(ql):
            stock = extracted_ticker or "000001.SZ"
            route["matched_rule"] = {
                "rule_id": "stock-analysis-evidence-core",
                "name": "Single-stock evidence first",
                "description": "enforce quote-first chain for single stock analysis",
                "priority": 0,
                "weight": 1.0,
                "min_score": 0.0,
                "requires_ticker": True,
            }
            route["decision"]["selected_tool"] = "evotraders_tq_call"
            route["decision"]["decision_args"] = {"stock_code": stock}
            route["decision"]["fallback_chain"] = ["evotraders_tq_call:get_more_info"]
        elif _contains_any(ql, ["大单", "资金流", "净流入"]):
            route["decision"]["selected_tool"] = "evotraders_ths_bigorder"
        else:
            route["decision"]["selected_tool"] = "evotraders_tq_call"
    return json.dumps(route, ensure_ascii=False)


def check_evotraders_requirements() -> bool:
    return _has_base()


EVOTRADERS_HEALTH_SCHEMA = {
    "name": "evotraders_health",
    "description": "Check EvoTraders WinAPI relay health (/v1/health). Requires EVOTRADERS_WINAPI_BASE.",
    "parameters": {
        "type": "object",
        "properties": {
            "timeout_sec": {"type": "number", "description": "HTTP timeout seconds", "default": 5.0},
        },
        "required": [],
    },
}

EVOTRADERS_TQ_CALL_SCHEMA = {
    "name": "evotraders_tq_call",
    "description": "Call tqcenter (or upstream Evo fallback) via WinAPI relay: POST /v1/tq/{method}.",
    "parameters": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "description": "tq method name, e.g. get_market_snapshot, formula_kline_series"},
            "params": {"type": "object", "description": "JSON body params passed through to tq/upstream", "default": {}},
            "timeout_sec": {"type": "number", "description": "HTTP timeout seconds", "default": 30.0},
        },
        "required": ["method"],
    },
}

EVOTRADERS_PROXY_CALL_SCHEMA = {
    "name": "evotraders_proxy_call",
    "description": "Proxy any upstream Evo endpoint through WinAPI relay (/v1/proxy/*). Useful for broad API surface.",
    "parameters": {
        "type": "object",
        "properties": {
            "subpath": {"type": "string", "description": "Path under proxy, e.g. 'v1/quant/runtime/metrics' or 'v1/health'"},
            "http_method": {"type": "string", "description": "HTTP method: GET/POST/PUT/PATCH/DELETE", "default": "GET"},
            "body": {"type": "object", "description": "Optional JSON body for write calls", "default": None},
            "timeout_sec": {"type": "number", "description": "HTTP timeout seconds", "default": 30.0},
        },
        "required": ["subpath"],
    },
}

EVOTRADERS_MARKET_MAINLINE_SCHEMA = {
    "name": "evotraders_market_mainline",
    "description": "Get A-share market mainline sentiment/context from Evo upstream (proxy-first, direct fallback).",
    "parameters": {
        "type": "object",
        "properties": {
            "max_age_sec": {"type": "integer", "description": "Cache tolerance seconds", "default": 60},
            "timeout_sec": {"type": "number", "description": "HTTP timeout seconds", "default": 20.0},
        },
        "required": [],
    },
}

EVOTRADERS_WENDA_QUERY_SCHEMA = {
    "name": "evotraders_wenda_query",
    "description": "Run TQLEX wendaQuery (A-share screener) via Evo relay.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Natural-language screener query"},
            "page_no": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 30},
            "rang": {"type": "string", "description": "Universe range, default A", "default": "A"},
            "timeout_sec": {"type": "number", "default": 25.0},
        },
        "required": ["message"],
    },
}

EVOTRADERS_INDICATOR_SELECT_SCHEMA = {
    "name": "evotraders_indicator_select",
    "description": "Natural-language indicator screener through Evo relay.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "topk": {"type": "integer", "default": 10},
            "timeout_sec": {"type": "number", "default": 20.0},
        },
        "required": ["query"],
    },
}

EVOTRADERS_IWENCAI_QUERY_SCHEMA = {
    "name": "evotraders_iwencai_query",
    "description": "Query iwencai data path (through connectivity-check endpoint) via Evo relay.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "page": {"type": "integer", "default": 1},
            "limit": {"type": "integer", "default": 20},
            "timeout_sec": {"type": "number", "default": 25.0},
        },
        "required": ["query"],
    },
}

EVOTRADERS_THS_BIGORDER_SCHEMA = {
    "name": "evotraders_ths_bigorder",
    "description": "Fetch THS bigorder summary by code/date via relay.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Stock code, e.g. 000001"},
            "trade_date": {"type": "string", "description": "YYYY-MM-DD optional", "default": ""},
            "timeout_sec": {"type": "number", "default": 20.0},
        },
        "required": ["code"],
    },
}

EVOTRADERS_TRADE_QUERY_SCHEMA = {
    "name": "evotraders_trade_query",
    "description": "Query trade account/positions/orders endpoints through relay.",
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": "account|positions|orders|orders_history",
                "default": "account",
            },
            "timeout_sec": {"type": "number", "default": 20.0},
        },
        "required": [],
    },
}

EVOTRADERS_TRADE_BUY_SCHEMA = {
    "name": "evotraders_trade_buy",
    "description": "Place BUY order via Evo TQ trade method (order_stock). Requires confirm=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "stock_code": {"type": "string", "description": "Stock code, e.g. 600000.SH"},
            "price": {"type": "number", "description": "Order price"},
            "order_volume": {"type": "integer", "description": "Order volume, e.g. 100"},
            "account_id": {"type": "integer", "default": -1},
            "price_type": {"type": "integer", "default": 0},
            "confirm": {"type": "boolean", "default": False, "description": "Must be true to execute"},
            "timeout_sec": {"type": "number", "default": 20.0},
        },
        "required": ["stock_code", "price", "order_volume", "confirm"],
    },
}

EVOTRADERS_TRADE_SELL_SCHEMA = {
    "name": "evotraders_trade_sell",
    "description": "Place SELL order via Evo TQ trade method (order_stock). Requires confirm=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "stock_code": {"type": "string", "description": "Stock code, e.g. 600000.SH"},
            "price": {"type": "number", "description": "Order price"},
            "order_volume": {"type": "integer", "description": "Order volume, e.g. 100"},
            "account_id": {"type": "integer", "default": -1},
            "price_type": {"type": "integer", "default": 0},
            "confirm": {"type": "boolean", "default": False, "description": "Must be true to execute"},
            "timeout_sec": {"type": "number", "default": 20.0},
        },
        "required": ["stock_code", "price", "order_volume", "confirm"],
    },
}

EVOTRADERS_TRADE_CANCEL_SCHEMA = {
    "name": "evotraders_trade_cancel",
    "description": "Cancel an order via Evo TQ trade method (cancel_stock_order). Requires confirm=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "stock_code": {"type": "string", "description": "Stock code, e.g. 600000.SH"},
            "order_id": {"type": "string", "description": "Broker order id/contract id"},
            "account_id": {"type": "integer", "default": -1},
            "confirm": {"type": "boolean", "default": False, "description": "Must be true to execute"},
            "timeout_sec": {"type": "number", "default": 20.0},
        },
        "required": ["stock_code", "order_id", "confirm"],
    },
}

EVOTRADERS_TRADE_VERIFY_BUNDLE_SCHEMA = {
    "name": "evotraders_trade_verify_bundle",
    "description": "Verify trade chain by querying account/positions/orders/orders_history in one shot.",
    "parameters": {
        "type": "object",
        "properties": {
            "timeout_sec": {"type": "number", "default": 25.0},
        },
        "required": [],
    },
}

EVOTRADERS_THSQUANT_HEALTH_SCHEMA = {
    "name": "evotraders_thsquant_health",
    "description": "Check thsQuant service health: GET /v1/system/health (THSQUANT_BASE).",
    "parameters": {
        "type": "object",
        "properties": {"timeout_sec": {"type": "number", "default": 6.0}},
        "required": [],
    },
}

EVOTRADERS_THSQUANT_TRADE_QUERY_SCHEMA = {
    "name": "evotraders_thsquant_trade_query",
    "description": "Query thsQuant trade tables via POST /v1/trade/* (THSQUANT_BASE).",
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": "account|positions|orders_today|orders_history",
                "default": "positions",
            },
            "timeout_sec": {"type": "number", "default": 20.0},
        },
        "required": ["kind"],
    },
}

EVOTRADERS_THSQUANT_TRADE_VERIFY_BUNDLE_SCHEMA = {
    "name": "evotraders_thsquant_trade_verify_bundle",
    "description": "Verify thsQuant trade chain: health + account + positions + orders_today + orders_history.",
    "parameters": {
        "type": "object",
        "properties": {"timeout_sec": {"type": "number", "default": 25.0}},
        "required": [],
    },
}

EVOTRADERS_TQLEX_PUBLIC_CALL_SCHEMA = {
    "name": "evotraders_tqlex_public_call",
    "description": "Call Evo function-chain get_tqlex_public_data_tool (TQLEX public data) via /v1/tq method relay.",
    "parameters": {
        "type": "object",
        "properties": {
            "entry": {"type": "string", "description": "TQLEX entry, e.g. JNLPSE:wendaQuery"},
            "params_json": {"type": "string", "description": "JSON array string for Params", "default": "[]"},
            "full_body_json": {"type": "string", "description": "Optional full JSON body string", "default": ""},
            "timeout_sec": {"type": "number", "default": 30.0},
            "persist": {"type": "boolean", "default": False},
        },
        "required": ["entry"],
    },
}

EVOTRADERS_ROUTE_AND_CALL_SCHEMA = {
    "name": "evotraders_route_and_call",
    "description": "Implicit-rule router aligned to Evo routing. Auto-selects tool and fallback chain by intent.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "User intent text"},
            "prefer_local_tdx": {"type": "boolean", "default": False},
            "disable_iwencai": {"type": "boolean", "default": False},
            "disable_wenda": {"type": "boolean", "default": False},
            "confirm": {"type": "boolean", "default": False, "description": "Required true for buy/sell/cancel execution"},
            "stock_code": {"type": "string", "description": "Stock code for buy/sell/cancel, e.g. 600000.SH"},
            "price": {"type": "number", "description": "Order price for buy/sell"},
            "order_volume": {"type": "integer", "description": "Order volume for buy/sell"},
            "order_id": {"type": "string", "description": "Order id for cancel"},
            "account_id": {"type": "integer", "default": -1},
            "price_type": {"type": "integer", "default": 0},
            "timeout_sec": {"type": "number", "default": 25.0},
        },
        "required": ["query"],
    },
}

EVOTRADERS_ROUTE_DRY_RUN_SCHEMA = {
    "name": "evotraders_route_dry_run",
    "description": "Dry-run implicit routing decision. Returns matched rule and selected tool without executing remote calls.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "User intent text"},
            "prefer_local_tdx": {"type": "boolean", "default": False},
            "disable_iwencai": {"type": "boolean", "default": False},
            "disable_wenda": {"type": "boolean", "default": False},
            "confirm": {"type": "boolean", "default": False},
            "stock_code": {"type": "string"},
            "price": {"type": "number"},
            "order_volume": {"type": "integer"},
            "order_id": {"type": "string"},
            "account_id": {"type": "integer", "default": -1},
            "price_type": {"type": "integer", "default": 0}
        },
        "required": ["query"],
    },
}


registry.register(
    name="evotraders_health",
    toolset="evotraders",
    schema=EVOTRADERS_HEALTH_SCHEMA,
    handler=lambda args, **kw: evotraders_health(timeout_sec=args.get("timeout_sec", 5.0)),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_tq_call",
    toolset="evotraders",
    schema=EVOTRADERS_TQ_CALL_SCHEMA,
    handler=lambda args, **kw: evotraders_tq_call(
        method=args.get("method", ""),
        params=args.get("params") if isinstance(args.get("params"), dict) else {},
        timeout_sec=args.get("timeout_sec", 30.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_proxy_call",
    toolset="evotraders",
    schema=EVOTRADERS_PROXY_CALL_SCHEMA,
    handler=lambda args, **kw: evotraders_proxy_call(
        subpath=args.get("subpath", ""),
        http_method=args.get("http_method", "GET"),
        body=args.get("body") if isinstance(args.get("body"), dict) else None,
        timeout_sec=args.get("timeout_sec", 30.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_market_mainline",
    toolset="evotraders",
    schema=EVOTRADERS_MARKET_MAINLINE_SCHEMA,
    handler=lambda args, **kw: evotraders_market_mainline(
        max_age_sec=args.get("max_age_sec", 60),
        timeout_sec=args.get("timeout_sec", 20.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_wenda_query",
    toolset="evotraders",
    schema=EVOTRADERS_WENDA_QUERY_SCHEMA,
    handler=lambda args, **kw: evotraders_wenda_query(
        message=args.get("message", ""),
        page_no=args.get("page_no", 1),
        page_size=args.get("page_size", 30),
        rang=args.get("rang", "A"),
        timeout_sec=args.get("timeout_sec", 25.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_indicator_select",
    toolset="evotraders",
    schema=EVOTRADERS_INDICATOR_SELECT_SCHEMA,
    handler=lambda args, **kw: evotraders_indicator_select(
        query=args.get("query", ""),
        topk=args.get("topk", 10),
        timeout_sec=args.get("timeout_sec", 20.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_iwencai_query",
    toolset="evotraders",
    schema=EVOTRADERS_IWENCAI_QUERY_SCHEMA,
    handler=lambda args, **kw: evotraders_iwencai_query(
        query=args.get("query", ""),
        page=args.get("page", 1),
        limit=args.get("limit", 20),
        timeout_sec=args.get("timeout_sec", 25.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_ths_bigorder",
    toolset="evotraders",
    schema=EVOTRADERS_THS_BIGORDER_SCHEMA,
    handler=lambda args, **kw: evotraders_ths_bigorder(
        code=args.get("code", ""),
        trade_date=args.get("trade_date", ""),
        timeout_sec=args.get("timeout_sec", 20.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_trade_query",
    toolset="evotraders",
    schema=EVOTRADERS_TRADE_QUERY_SCHEMA,
    handler=lambda args, **kw: evotraders_trade_query(
        kind=args.get("kind", "account"),
        timeout_sec=args.get("timeout_sec", 20.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_trade_buy",
    toolset="evotraders",
    schema=EVOTRADERS_TRADE_BUY_SCHEMA,
    handler=lambda args, **kw: evotraders_trade_buy(
        stock_code=args.get("stock_code", ""),
        price=float(args.get("price", 0)),
        order_volume=int(args.get("order_volume", 0)),
        account_id=int(args.get("account_id", -1)),
        price_type=int(args.get("price_type", 0)),
        confirm=bool(args.get("confirm", False)),
        timeout_sec=float(args.get("timeout_sec", 20.0)),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_trade_sell",
    toolset="evotraders",
    schema=EVOTRADERS_TRADE_SELL_SCHEMA,
    handler=lambda args, **kw: evotraders_trade_sell(
        stock_code=args.get("stock_code", ""),
        price=float(args.get("price", 0)),
        order_volume=int(args.get("order_volume", 0)),
        account_id=int(args.get("account_id", -1)),
        price_type=int(args.get("price_type", 0)),
        confirm=bool(args.get("confirm", False)),
        timeout_sec=float(args.get("timeout_sec", 20.0)),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_trade_cancel",
    toolset="evotraders",
    schema=EVOTRADERS_TRADE_CANCEL_SCHEMA,
    handler=lambda args, **kw: evotraders_trade_cancel(
        stock_code=args.get("stock_code", ""),
        order_id=args.get("order_id", ""),
        account_id=int(args.get("account_id", -1)),
        confirm=bool(args.get("confirm", False)),
        timeout_sec=float(args.get("timeout_sec", 20.0)),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_trade_verify_bundle",
    toolset="evotraders",
    schema=EVOTRADERS_TRADE_VERIFY_BUNDLE_SCHEMA,
    handler=lambda args, **kw: evotraders_trade_verify_bundle(
        timeout_sec=float(args.get("timeout_sec", 25.0)),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_thsquant_health",
    toolset="evotraders",
    schema=EVOTRADERS_THSQUANT_HEALTH_SCHEMA,
    handler=lambda args, **kw: evotraders_thsquant_health(timeout_sec=float(args.get("timeout_sec", 6.0))),
    check_fn=_has_thsquant_base,
    emoji="📈",
)

registry.register(
    name="evotraders_thsquant_trade_query",
    toolset="evotraders",
    schema=EVOTRADERS_THSQUANT_TRADE_QUERY_SCHEMA,
    handler=lambda args, **kw: evotraders_thsquant_trade_query(
        kind=args.get("kind", "positions"),
        timeout_sec=float(args.get("timeout_sec", 20.0)),
    ),
    check_fn=_has_thsquant_base,
    emoji="📈",
)

registry.register(
    name="evotraders_thsquant_trade_verify_bundle",
    toolset="evotraders",
    schema=EVOTRADERS_THSQUANT_TRADE_VERIFY_BUNDLE_SCHEMA,
    handler=lambda args, **kw: evotraders_thsquant_trade_verify_bundle(timeout_sec=float(args.get("timeout_sec", 25.0))),
    check_fn=_has_thsquant_base,
    emoji="📈",
)

registry.register(
    name="evotraders_tqlex_public_call",
    toolset="evotraders",
    schema=EVOTRADERS_TQLEX_PUBLIC_CALL_SCHEMA,
    handler=lambda args, **kw: evotraders_tqlex_public_call(
        entry=args.get("entry", ""),
        params_json=args.get("params_json", "[]"),
        full_body_json=args.get("full_body_json", ""),
        timeout_sec=args.get("timeout_sec", 30.0),
        persist=bool(args.get("persist", False)),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_route_and_call",
    toolset="evotraders",
    schema=EVOTRADERS_ROUTE_AND_CALL_SCHEMA,
    handler=lambda args, **kw: evotraders_route_and_call(
        query=args.get("query", ""),
        prefer_local_tdx=bool(args.get("prefer_local_tdx", False)),
        disable_iwencai=bool(args.get("disable_iwencai", False)),
        disable_wenda=bool(args.get("disable_wenda", False)),
        confirm=bool(args.get("confirm", False)),
        stock_code=args.get("stock_code", ""),
        price=float(args.get("price", 0.0) or 0.0),
        order_volume=int(args.get("order_volume", 0) or 0),
        order_id=args.get("order_id", ""),
        account_id=int(args.get("account_id", -1) or -1),
        price_type=int(args.get("price_type", 0) or 0),
        timeout_sec=args.get("timeout_sec", 25.0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

registry.register(
    name="evotraders_route_dry_run",
    toolset="evotraders",
    schema=EVOTRADERS_ROUTE_DRY_RUN_SCHEMA,
    handler=lambda args, **kw: evotraders_route_dry_run(
        query=args.get("query", ""),
        prefer_local_tdx=bool(args.get("prefer_local_tdx", False)),
        disable_iwencai=bool(args.get("disable_iwencai", False)),
        disable_wenda=bool(args.get("disable_wenda", False)),
        confirm=bool(args.get("confirm", False)),
        stock_code=args.get("stock_code", ""),
        price=float(args.get("price", 0.0) or 0.0),
        order_volume=int(args.get("order_volume", 0) or 0),
        order_id=args.get("order_id", ""),
        account_id=int(args.get("account_id", -1) or -1),
        price_type=int(args.get("price_type", 0) or 0),
    ),
    check_fn=check_evotraders_requirements,
    emoji="📈",
)

