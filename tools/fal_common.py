"""Shared FAL.ai SDK plumbing: lazy import, managed-gateway sync client, small helpers.

Holds the stateless atoms that every FAL-backed tool needs:

* :func:`import_fal_client` — lazy import + ``pm.ensure_import`` so
  ``fal_client`` isn't pulled at cold start (it added ~64 ms per CLI
  invocation when imported eagerly).
* :class:`_ManagedFalSyncClient` — wrapper that drives a Nous-managed
  fal-queue gateway through the standard ``fal_client.SyncClient``
  primitives.
* :func:`_normalize_fal_queue_url_format`, :func:`_extract_http_status`
  — small helpers used by both the managed client wrapper and
  ``_submit_fal_request``.

Stateful pieces (cache globals, ``_managed_fal_client*`` selectors,
``_submit_fal_request``) intentionally stay on
:mod:`tools.image_generation_tool`. That module is the patch target for
existing test suites (``tests/tools/test_image_generation.py``,
``tests/tools/test_managed_media_gateways.py``) and for the
``plugins/image_gen/fal/`` plugin's ``_it`` indirection — moving the
caches here would silently defeat ``monkeypatch.setattr(image_tool,
"_managed_fal_client", None)`` because the lookups would go against
``fal_common``'s namespace instead. See the per-rule walkthrough at
issue #26241 for details.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

# A 429 whose Retry-After is longer than this is reported, not waited out inside a tool call.
MANAGED_FAL_RATE_LIMIT_RETRY_CAP_SECONDS = 30.0


def import_fal_client() -> Any:
    """Import ``fal_client`` (via ``pm`` when available) and return
    the module reference.

    Callers cache the result on their own module global so tests can monkeypatch it.
    """
    try:
        from pm import ensure_import as _lazy_ensure
    except ImportError:
        # pm itself unavailable (externally-managed env, partial install) —
        # the plain import below is the authority on availability.
        pass
    else:
        try:
            _lazy_ensure("fal")
        except ImportError:
            pass  # same authority rule: let the plain import decide
        except Exception as exc:  # noqa: BLE001 — pm surfaces install hints
            raise ImportError(str(exc))
    import fal_client  # type: ignore  # noqa: WPS433 — intentionally lazy
    return fal_client


def _normalize_fal_queue_url_format(queue_run_origin: str) -> str:
    normalized_origin = str(queue_run_origin or "").strip().rstrip("/")
    if not normalized_origin:
        raise ValueError("Managed FAL queue origin is required")
    return f"{normalized_origin}/"


def _extract_http_status(exc: BaseException) -> Optional[int]:
    """HTTP status from httpx (``.response.status_code``) or fal_client (``.status_code``) exceptions, else None."""
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _managed_fal_billing_error(exc: BaseException, what: str) -> Optional[str]:
    """Human-readable tail for a Nous managed-gateway ``BILLING_ERROR`` response, else None.

    ``what`` names the rejected thing ("model", "endpoint"); the wording is shared by the image
    and video callers so the two surfaces never drift.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 — diagnostics must not mask the provider error
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict) or error.get("code") != "BILLING_ERROR":
        return None
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    upstream = details.get("upstreamPayload") if isinstance(details.get("upstreamPayload"), dict) else {}
    code = upstream.get("code") or details.get("chargeIntentErrorCode") or "billing_error"
    detail = upstream.get("error") or "Nous Portal rejected the charge authorization"
    return (
        f"{error.get('message') or 'Charge authorization failed'} (BILLING_ERROR; {code}: {detail}). "
        "This is a Nous Portal billing configuration issue, not a missing local API key. "
        f"The managed route cannot run this {what} until Nous enables its billing meter; "
        "a direct FAL_KEY is an optional bypass."
    )


def _managed_fal_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Seconds the managed gateway asked us to wait after a 429: the ``Retry-After`` header,
    else the body's ``error.retryAfter``; None when the status is not 429 or neither is present."""
    response = getattr(exc, "response", None)
    if response is None or _extract_http_status(exc) != 429:
        return None
    headers = getattr(response, "headers", None)
    raw = headers.get("Retry-After") if headers is not None and hasattr(headers, "get") else None
    if raw is None:
        try:
            error = response.json().get("error")
        except Exception:  # noqa: BLE001 — a non-JSON 429 body simply has no hint
            return None
        raw = error.get("retryAfter") if isinstance(error, dict) else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _managed_fal_rate_limit_message(what: str, name: str, retry_after: Optional[float]) -> str:
    hint = f"retry after {retry_after:g}s" if retry_after is not None else "no Retry-After given"
    return (
        f"Nous Subscription gateway rate-limited {what} '{name}' (HTTP 429; {hint}). "
        "The model is enabled — retry later instead of switching models or setting FAL_KEY."
    )


def submit_managed_fal_with_rate_limit_retry(
    submit: Callable[[Dict[str, str]], Any], *, what: str, name: str,
):
    """Call ``submit(headers)`` with a fresh ``x-idempotency-key``; on a 429 whose Retry-After
    fits the cap, wait it out (interrupt-aware) and resubmit ONCE under a new key.

    A second 429, or one with an unknown/too-long Retry-After, raises ValueError naming the
    rate limit — it must never fall through to the callers' "model may not be enabled" 4xx text,
    which sends agents off to switch models. Every other exception propagates untouched.
    """
    from tools.interrupt import is_interrupted
    for attempt in (1, 2):
        try:
            return submit({"x-idempotency-key": str(uuid.uuid4())})
        except Exception as exc:
            if _extract_http_status(exc) != 429:
                raise
            retry_after = _managed_fal_retry_after_seconds(exc)
            if attempt == 2 or retry_after is None or retry_after > MANAGED_FAL_RATE_LIMIT_RETRY_CAP_SECONDS:
                raise ValueError(_managed_fal_rate_limit_message(what, name, retry_after)) from exc
            deadline = time.monotonic() + retry_after
            while time.monotonic() < deadline:
                if is_interrupted():
                    raise ValueError(_managed_fal_rate_limit_message(what, name, retry_after)) from exc
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def _require(value: Any, what: str) -> Any:
    if value is None:
        raise RuntimeError(f"{what} is required for managed FAL gateway mode")
    return value


class _ManagedFalSyncClient:
    """Drives a Nous-managed fal-queue gateway via ``fal_client.SyncClient`` primitives; carries
    its own ``fal_client`` reference so the caller decides which (possibly test-patched) module is used."""

    def __init__(self, fal_client: Any, *, key: str, queue_run_origin: str):
        sync_client_class = _require(getattr(fal_client, "SyncClient", None), "fal_client.SyncClient")
        client_module = _require(getattr(fal_client, "client", None), "fal_client.client")
        self._queue_url_format = _normalize_fal_queue_url_format(queue_run_origin)
        self._sync_client = sync_client_class(key=key)
        self._http_client = _require(getattr(self._sync_client, "_client", None), "fal_client.SyncClient._client")
        self._maybe_retry_request = getattr(client_module, "_maybe_retry_request", None)
        self._raise_for_status = getattr(client_module, "_raise_for_status", None)
        if self._maybe_retry_request is None or self._raise_for_status is None:
            raise RuntimeError("fal_client.client request helpers are required for managed FAL gateway mode")
        self._request_handle_class = _require(
            getattr(client_module, "SyncRequestHandle", None), "fal_client.client.SyncRequestHandle")
        self._add_hint_header = getattr(client_module, "add_hint_header", None)
        self._add_priority_header = getattr(client_module, "add_priority_header", None)
        self._add_timeout_header = getattr(client_module, "add_timeout_header", None)

    def submit(
        self, application: str, arguments: Dict[str, Any], *, path: str = "",
        hint: Optional[str] = None, webhook_url: Optional[str] = None, priority: Any = None,
        headers: Optional[Dict[str, str]] = None, start_timeout: Optional[Union[int, float]] = None,
    ):
        url = self._queue_url_format + application
        if path:
            url += "/" + path.lstrip("/")
        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})
        request_headers = dict(headers or {})
        if hint is not None and self._add_hint_header is not None:
            self._add_hint_header(hint, request_headers)
        if priority is not None:
            if self._add_priority_header is None:
                raise RuntimeError("fal_client.client.add_priority_header is required for priority requests")
            self._add_priority_header(priority, request_headers)
        if start_timeout is not None:
            if self._add_timeout_header is None:
                raise RuntimeError("fal_client.client.add_timeout_header is required for timeout requests")
            self._add_timeout_header(start_timeout, request_headers)
        request_kwargs = {
            "json": arguments,
            "timeout": getattr(self._sync_client, "default_timeout", 120.0),
            "headers": request_headers,
        }
        # The Nous gateway currently records a keyed submission before billing
        # authorization finishes, but cannot replay the resulting error. The
        # SDK's automatic 409 retry therefore replaces the real billing error
        # with an idempotency conflict. Make one attempt when the caller supplied
        # a key; an ambiguous transport failure is safer than a possible duplicate
        # generation or a masked entitlement failure.
        has_idempotency_key = any(
            str(key).lower() == "x-idempotency-key" for key in request_headers
        )
        if has_idempotency_key:
            response = self._http_client.request("POST", url, **request_kwargs)
        else:
            response = self._maybe_retry_request(self._http_client, "POST", url, **request_kwargs)
        self._raise_for_status(response)
        data = response.json()
        return self._request_handle_class(
            request_id=data["request_id"], response_url=data["response_url"],
            status_url=data["status_url"], cancel_url=data["cancel_url"], client=self._http_client)
