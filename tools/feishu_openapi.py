"""Shared Feishu/Lark Open Platform helpers for Hermes tools.

This module intentionally stays independent from ``gateway.platforms.feishu``.
The gateway adapter owns messaging transport; this helper owns standalone
Open Platform business API calls used by tools such as Docx, Bitable, Drive,
and Wiki.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

FEISHU_DOMAIN = "https://open.feishu.cn"
LARK_DOMAIN = "https://open.larksuite.com"


class FeishuOpenAPIError(RuntimeError):
    """Raised when a Feishu Open Platform request returns a non-zero code."""

    def __init__(self, message: str, *, code: int | None = None, data: dict | None = None):
        super().__init__(message)
        self.code = code
        self.data = data or {}


@dataclass(frozen=True)
class FeishuOpenAPISettings:
    app_id: str
    app_secret: str
    domain_name: str = "feishu"

    @property
    def base_url(self) -> str:
        return LARK_DOMAIN if self.domain_name.lower() == "lark" else FEISHU_DOMAIN


def load_settings() -> FeishuOpenAPISettings:
    """Load Feishu settings from environment variables."""

    return FeishuOpenAPISettings(
        app_id=os.getenv("FEISHU_APP_ID", "").strip(),
        app_secret=os.getenv("FEISHU_APP_SECRET", "").strip(),
        domain_name=os.getenv("FEISHU_DOMAIN", os.getenv("FEISHU_DOMAIN_NAME", "feishu")).strip() or "feishu",
    )


def check_feishu_openapi_requirements() -> bool:
    """Return whether lark_oapi and required credentials are available."""

    settings = load_settings()
    if not settings.app_id or not settings.app_secret:
        return False
    try:
        import lark_oapi  # noqa: F401
    except ImportError:
        return False
    return True


def get_feishu_client(client: Any | None = None) -> Any:
    """Return an existing client or build a lark_oapi client from env settings."""

    if client is not None:
        return client

    settings = load_settings()
    if not settings.app_id or not settings.app_secret:
        raise FeishuOpenAPIError("FEISHU_APP_ID and FEISHU_APP_SECRET are required")

    try:
        import lark_oapi as lark
    except ImportError as exc:
        raise FeishuOpenAPIError("lark_oapi not installed. Run: pip install 'hermes-agent[feishu]'") from exc

    builder = lark.Client.builder().app_id(settings.app_id).app_secret(settings.app_secret)
    # Python SDK versions expose different builder options. Prefer the domain
    # setter when present, otherwise keep SDK defaults.
    if hasattr(builder, "domain"):
        builder = builder.domain(settings.base_url)
    elif hasattr(builder, "open_base_url"):
        builder = builder.open_base_url(settings.base_url)
    return builder.build()


def _http_method(method: str) -> Any:
    from lark_oapi.core.enum import HttpMethod

    method = method.upper()
    return {
        "GET": HttpMethod.GET,
        "POST": HttpMethod.POST,
        "PUT": HttpMethod.PUT,
        "PATCH": getattr(HttpMethod, "PATCH", HttpMethod.PUT),
        "DELETE": HttpMethod.DELETE,
    }.get(method, HttpMethod.GET)


def _normalize_queries(queries: dict[str, Any] | Iterable[tuple[str, Any]] | None) -> list[tuple[str, str]] | None:
    if not queries:
        return None
    items = queries.items() if isinstance(queries, dict) else queries
    result: list[tuple[str, str]] = []
    for key, value in items:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        result.append((str(key), str(value)))
    return result or None


def _response_data(response: Any) -> dict[str, Any]:
    raw = getattr(response, "raw", None)
    if raw is not None and hasattr(raw, "content"):
        content = raw.content
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        try:
            parsed = json.loads(content or "{}")
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, json.JSONDecodeError):
            pass

    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return {"code": getattr(response, "code", 0), "msg": getattr(response, "msg", ""), "data": data}
    if data is not None:
        if hasattr(data, "to_dict"):
            payload = data.to_dict()
        elif hasattr(data, "__dict__"):
            payload = vars(data)
        else:
            payload = {"value": data}
        return {"code": getattr(response, "code", 0), "msg": getattr(response, "msg", ""), "data": payload}

    return {"code": getattr(response, "code", 0), "msg": getattr(response, "msg", ""), "data": {}}


def request_json(
    method: str,
    uri: str,
    *,
    paths: dict[str, Any] | None = None,
    queries: dict[str, Any] | Iterable[tuple[str, Any]] | None = None,
    body: dict[str, Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Execute a Feishu Open Platform request and return its ``data`` object.

    Raises ``FeishuOpenAPIError`` for SDK import/config errors and non-zero
    Feishu response codes. Tool handlers catch this and convert to JSON errors.
    """

    client = get_feishu_client(client)

    try:
        from lark_oapi import AccessTokenType
        from lark_oapi.core.model.base_request import BaseRequest
    except ImportError as exc:
        raise FeishuOpenAPIError("lark_oapi not installed. Run: pip install 'hermes-agent[feishu]'") from exc

    builder = (
        BaseRequest.builder()
        .http_method(_http_method(method))
        .uri(uri)
        .token_types({AccessTokenType.TENANT})
    )
    if paths:
        builder = builder.paths({str(k): str(v) for k, v in paths.items()})
    norm_queries = _normalize_queries(queries)
    if norm_queries:
        builder = builder.queries(norm_queries)
    if body is not None:
        builder = builder.body(body)

    response = client.request(builder.build())
    payload = _response_data(response)
    code = payload.get("code", getattr(response, "code", 0))
    if code not in (0, None):
        msg = payload.get("msg") or getattr(response, "msg", "unknown error")
        raise FeishuOpenAPIError(f"Feishu API request failed: code={code} msg={msg}", code=code, data=payload)
    data = payload.get("data", {})
    return data if isinstance(data, dict) else {"value": data}