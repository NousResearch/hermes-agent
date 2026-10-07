"""Shared FAL.ai SDK plumbing: lazy import and small helpers.

Holds the stateless atoms that every FAL-backed tool needs:

* :func:`import_fal_client` — lazy import + ``pm.ensure_import`` so
  ``fal_client`` isn't pulled at cold start (it added ~64 ms per CLI
  invocation when imported eagerly).
"""

from __future__ import annotations

from typing import Any, Optional


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
        raise ValueError("FAL queue origin is required")
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
