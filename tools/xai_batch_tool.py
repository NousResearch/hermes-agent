"""xAI Batch API tool — process many chat completions asynchronously.

Wraps xAI's Batch API (``POST /v1/batches`` + add-requests + poll + paginated
results) behind a single ``xai_batch_chat`` entry point. Intended for offline /
asynchronous workflows where dozens-to-thousands of chat completions can be
submitted together at reduced rate-limit pressure.

Lifecycle implemented
---------------------
1. ``POST /v1/batches`` — create empty batch with a name.
2. ``POST /v1/batches/{batch_id}/requests`` — add chat-completion requests
   inline (this tool only supports inline mode — file upload is for huge
   batches and would require Files API plumbing that doesn't exist yet).
3. (optional, ``wait=True``) ``GET /v1/batches/{batch_id}`` — poll the state
   counters until ``num_pending == 0`` or ``max_wait_seconds`` is exceeded.
4. ``GET /v1/batches/{batch_id}/results?limit=...&pagination_token=...`` —
   page through the results, return them in submission order.

Reference
---------
- https://docs.x.ai/docs/guides/batch-api
- Most batches complete within 24 hours per xAI docs; default wait budget is
  86 400 seconds.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from tools.xai_http import hermes_xai_user_agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
DEFAULT_MODEL = "grok-4.3"
DEFAULT_MAX_WAIT_SECONDS = 86_400        # 24h — matches xAI's "most batches" SLA
DEFAULT_POLL_INTERVAL_SECONDS = 30.0     # 30s polls keep load light over 24h
DEFAULT_RESULTS_PAGE_LIMIT = 100
DEFAULT_HTTP_TIMEOUT_SECONDS = 30


def _config_section() -> Dict[str, Any]:
    """Read the ``xai_batch:`` block from the active Hermes config, if any."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return {}
    section = cfg.get("xai_batch") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def _resolve(name: str, default: Any) -> Any:
    section = _config_section()
    val = section.get(name)
    if val is None or (isinstance(val, str) and not val.strip()):
        return default
    return val


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class XaiBatchError(RuntimeError):
    """Raised when a Batch API call fails or times out."""


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

def xai_batch_chat(
    requests: List[Dict[str, Any]],
    *,
    name: Optional[str] = None,
    model: Optional[str] = None,
    wait: bool = True,
    max_wait_seconds: Optional[int] = None,
    poll_interval_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Submit a list of chat completions to xAI's Batch API.

    Parameters
    ----------
    requests : list of dicts
        Each dict represents one chat completion. Recognized keys:

        - ``prompt`` (str, required if ``messages`` not given): user message.
        - ``messages`` (list of dicts, optional): full conversation, overrides
          ``prompt`` + ``system`` when provided.
        - ``system`` (str, optional): system prompt prepended when only
          ``prompt`` is given.
        - ``model`` (str, optional): per-request override; defaults to the
          tool-level ``model``.
        - ``request_id`` (str, optional): caller-provided ID for matching
          results back; auto-generated if absent.
        - ``extra_body`` (dict, optional): extra fields merged into the body.

    name : str, optional
        Display name for the batch. Defaults to ``"hermes-xai-batch-<uuid>"``.
    model : str, optional
        Default model for any request without its own ``model`` key. Defaults
        to config ``xai_batch.model`` then ``"grok-4.3"``.
    wait : bool
        When True (default) the call blocks until all requests are done (or
        ``max_wait_seconds`` is exceeded) and returns paginated results.
        When False, returns immediately after submission with the ``batch_id``.
    max_wait_seconds : int, optional
        Polling budget in seconds. Defaults to config or 86 400 (24h).
    poll_interval_seconds : float, optional
        Sleep between polls. Defaults to config or 30s.

    Returns
    -------
    dict
        - When ``wait=True``: ``{"batch_id", "state", "results"}`` where
          ``results`` is a list of ``{"request_id", "response", "error"}``
          in submission order.
        - When ``wait=False``: ``{"batch_id", "state"}`` only — the caller
          is responsible for polling later.

    Raises
    ------
    XaiBatchError
        On HTTP errors during submit / add / poll / retrieve, or when
        ``max_wait_seconds`` elapses before the batch finishes.
    """
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        raise XaiBatchError("XAI_API_KEY is not set")

    resolved_model = model or _resolve("model", DEFAULT_MODEL)
    resolved_max_wait = int(max_wait_seconds or _resolve("max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS))
    resolved_poll_interval = float(poll_interval_seconds or _resolve("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS))
    if resolved_max_wait <= 0:
        raise ValueError("max_wait_seconds must be positive")
    if resolved_poll_interval <= 0:
        raise ValueError("poll_interval_seconds must be positive")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }
    base_url = DEFAULT_BASE_URL.rstrip("/")

    # ------- 1. Build the batch_requests payload (ordered, with IDs) -------
    batch_name = name or f"hermes-xai-batch-{uuid.uuid4().hex[:12]}"
    submission: List[Dict[str, Any]] = []
    request_ids: List[str] = []
    for i, req in enumerate(requests):
        rid = req.get("request_id") or f"req-{i:05d}-{uuid.uuid4().hex[:6]}"
        request_ids.append(rid)
        body = _build_chat_body(req, default_model=resolved_model)
        submission.append({
            "custom_id": rid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    # ------- 2. Create the empty batch + add requests inline -------
    batch_id = _create_batch(base_url, headers, name=batch_name)
    _add_requests(base_url, headers, batch_id, submission)

    # ------- 3. Optionally poll until done -------
    if not wait:
        state = _get_state(base_url, headers, batch_id)
        return {"batch_id": batch_id, "state": state}

    state = _poll_until_done(
        base_url=base_url,
        headers=headers,
        batch_id=batch_id,
        max_wait=resolved_max_wait,
        poll_interval=resolved_poll_interval,
    )

    # ------- 4. Retrieve paginated results, sort by submission order -------
    raw_results = _fetch_all_results(base_url, headers, batch_id)
    by_id = {r.get("custom_id") or r.get("batch_request_id"): r for r in raw_results}
    ordered: List[Dict[str, Any]] = []
    for rid in request_ids:
        item = by_id.get(rid, {})
        ordered.append({
            "request_id": rid,
            "response": item.get("response") or item.get("batch_result", {}).get("response"),
            "error": item.get("error"),
        })

    return {
        "batch_id": batch_id,
        "state": state,
        "results": ordered,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chat_body(req: Dict[str, Any], *, default_model: str) -> Dict[str, Any]:
    """Translate a tool-level request dict to the xAI chat-completions body."""
    if "messages" in req and isinstance(req["messages"], list):
        messages = list(req["messages"])
    else:
        prompt = req.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("each request needs either 'messages' or a non-empty 'prompt'")
        messages = []
        if req.get("system"):
            messages.append({"role": "system", "content": req["system"]})
        messages.append({"role": "user", "content": prompt})

    body: Dict[str, Any] = {
        "model": req.get("model") or default_model,
        "messages": messages,
    }
    extra = req.get("extra_body") or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            body[k] = v
    return body


def _request(method: str, url: str, headers: Dict[str, str], **kw: Any) -> httpx.Response:
    try:
        resp = httpx.request(method, url, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS, **kw)
    except httpx.HTTPError as exc:
        raise XaiBatchError(f"{method} {url} failed: {exc}") from exc
    if resp.status_code >= 400:
        raise XaiBatchError(f"{method} {url} returned {resp.status_code}: {resp.text[:300]}")
    return resp


def _create_batch(base_url: str, headers: Dict[str, str], *, name: str) -> str:
    resp = _request(
        "POST",
        f"{base_url}/batches",
        headers,
        json={"name": name},
    )
    payload = resp.json() if resp.text else {}
    batch_id = payload.get("batch_id") or payload.get("id")
    if not isinstance(batch_id, str) or not batch_id:
        raise XaiBatchError(f"create-batch response missing batch_id: {payload!r}")
    return batch_id


def _add_requests(
    base_url: str,
    headers: Dict[str, str],
    batch_id: str,
    submission: List[Dict[str, Any]],
) -> None:
    """Add inline requests to a batch.

    Wrap each tool-level submission entry in the ``batch_request`` envelope
    expected by the API.
    """
    payload = {
        "batch_requests": [
            {
                "batch_request_id": entry["custom_id"],
                "batch_request": {
                    "method": entry["method"],
                    "url": entry["url"],
                    "body": entry["body"],
                },
            }
            for entry in submission
        ]
    }
    _request(
        "POST",
        f"{base_url}/batches/{batch_id}/requests",
        headers,
        json=payload,
    )


def _get_state(base_url: str, headers: Dict[str, str], batch_id: str) -> Dict[str, Any]:
    resp = _request("GET", f"{base_url}/batches/{batch_id}", headers)
    payload = resp.json() if resp.text else {}
    state = payload.get("state") if isinstance(payload, dict) else None
    return state if isinstance(state, dict) else payload


def _poll_until_done(
    *,
    base_url: str,
    headers: Dict[str, str],
    batch_id: str,
    max_wait: int,
    poll_interval: float,
) -> Dict[str, Any]:
    started = time.monotonic()
    while True:
        if time.monotonic() - started > max_wait:
            raise XaiBatchError(
                f"batch {batch_id!r} not done after {max_wait}s — aborting poll"
            )
        state = _get_state(base_url, headers, batch_id)
        pending = state.get("num_pending")
        if isinstance(pending, int) and pending == 0:
            return state
        time.sleep(poll_interval)


def _fetch_all_results(
    base_url: str,
    headers: Dict[str, str],
    batch_id: str,
) -> List[Dict[str, Any]]:
    """Walk paginated results, returning a flat list of result entries."""
    out: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"limit": DEFAULT_RESULTS_PAGE_LIMIT}
        if page_token:
            params["pagination_token"] = page_token
        resp = _request(
            "GET",
            f"{base_url}/batches/{batch_id}/results",
            headers,
            params=params,
        )
        payload = resp.json() if resp.text else {}
        page = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(page, list):
            out.extend(page)
        page_token = payload.get("pagination_token") if isinstance(payload, dict) else None
        if not page_token:
            break
    return out


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def check_xai_batch_requirements() -> "tuple[bool, str]":
    """Tool gate: only available when XAI_API_KEY is set."""
    if os.environ.get("XAI_API_KEY", "").strip():
        return True, ""
    return False, "XAI_API_KEY environment variable is not set"


XAI_BATCH_SCHEMA = {
    "name": "xai_batch_chat",
    "description": (
        "Submit many chat completions to xAI's Batch API in one call. "
        "Use for offline / async workflows where dozens-to-thousands of "
        "completions can be processed together (reduced pricing, higher rate "
        "limits, up to 24h SLA). With wait=true (default) the call blocks "
        "until results are ready and returns them in submission order."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "description": (
                    "List of request dicts. Each dict needs at least 'prompt' "
                    "(str) or 'messages' (list). Optional: 'system', 'model', "
                    "'request_id', 'extra_body'."
                ),
                "items": {"type": "object"},
            },
            "name": {
                "type": "string",
                "description": "Display name for the batch (default: auto-generated).",
            },
            "model": {
                "type": "string",
                "description": (
                    "Default model for requests that don't specify their own. "
                    "Defaults to config xai_batch.model or 'grok-4.3'."
                ),
            },
            "wait": {
                "type": "boolean",
                "description": (
                    "When true (default), block until all requests complete and "
                    "return results. When false, return immediately with the "
                    "batch_id for caller-driven polling."
                ),
            },
            "max_wait_seconds": {
                "type": "integer",
                "description": (
                    "Polling budget in seconds. Defaults to config or 86400 (24h)."
                ),
            },
            "poll_interval_seconds": {
                "type": "number",
                "description": (
                    "Sleep between polls in seconds. Defaults to config or 30."
                ),
            },
        },
        "required": ["requests"],
    },
}


def _handle_xai_batch_tool_call(args: Dict[str, Any], **_kw: Any) -> Dict[str, Any]:
    """Bridge from registry handler signature to xai_batch_chat()."""
    return xai_batch_chat(
        requests=args.get("requests") or [],
        name=args.get("name"),
        model=args.get("model"),
        wait=bool(args.get("wait", True)),
        max_wait_seconds=args.get("max_wait_seconds"),
        poll_interval_seconds=args.get("poll_interval_seconds"),
    )


# Self-register at import time. The ``registry`` symbol must come from
# ``tools.registry`` (the ToolRegistry singleton), not the module itself —
# tools/registry.py:_is_registry_register_call also matches the literal name.
from tools.registry import registry  # noqa: E402

registry.register(
    name="xai_batch_chat",
    toolset="xai_batch",
    schema=XAI_BATCH_SCHEMA,
    handler=_handle_xai_batch_tool_call,
    check_fn=check_xai_batch_requirements,
    emoji="📦",
)
