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

from typing import Any, Dict, Optional, Union
from urllib.parse import urlencode


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
        response = self._maybe_retry_request(
            self._http_client, "POST", url, json=arguments,
            timeout=getattr(self._sync_client, "default_timeout", 120.0), headers=request_headers)
        self._raise_for_status(response)
        data = response.json()
        return self._request_handle_class(
            request_id=data["request_id"], response_url=data["response_url"],
            status_url=data["status_url"], cancel_url=data["cancel_url"], client=self._http_client)
