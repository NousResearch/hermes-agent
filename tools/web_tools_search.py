"""Shared search acquisition below model-tool formatting and above provider dispatch."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from pydantic import TypeAdapter, ValidationError

from agent.web_acquisition import (
    WebAttempt,
    WebAttemptFailure,
    WebAttemptRoute,
    WebAttemptSuccess,
    WebSearchFailure,
    WebSearchHit,
    WebSearchRequest,
    WebSearchResult,
    WebSearchSuccess,
)
from agent.web_acquisition_errors import (
    WebAcquisitionError,
    WebCancelledError,
    WebConfigurationError,
    WebInvalidResponseError,
    WebUnclassifiedFailure,
    acquisition_error,
    failure_from_data,
)
from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger("tools.web_tools")
_ATTEMPTS = TypeAdapter(tuple[WebAttempt, ...])


@dataclass(frozen=True)
class SearchExchange:
    outcome: WebSearchResult
    presentation: dict[str, Any]
    presentation_error: str | None = None


def search_failure(response: dict[str, Any]) -> WebAcquisitionError | None:
    """Validate the provider's success flag, hits and optional structured failure."""
    try:
        return _validated_search_failure(response)
    except WebInvalidResponseError as exc:
        return exc
    except (ValidationError, KeyError, TypeError, AttributeError) as exc:
        return WebInvalidResponseError(cause=exc)


def _validated_search_failure(response: dict[str, Any]) -> WebAcquisitionError | None:
    if type(response.get("success")) is not bool:
        raise WebInvalidResponseError()
    _ring_attempts(response)
    serving = response.get("served_provider")
    if serving is not None and (not isinstance(serving, str) or not serving):
        raise WebInvalidResponseError()
    if not response["success"]:
        return (
            failure_from_data(response["failure"])
            if "failure" in response
            else WebUnclassifiedFailure()
        )
    search_hits(response)
    return None


def checked_search_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        _validated_search_failure(response)
        return response
    except WebInvalidResponseError as exc:
        return _failed_response(exc)
    except (ValidationError, KeyError, TypeError, AttributeError) as exc:
        return _failed_response(WebInvalidResponseError(cause=exc))


def search_hits(response: dict[str, Any]) -> tuple[WebSearchHit, ...]:
    data = response.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("web"), list):
        raise WebInvalidResponseError()
    hits = []
    for row in data["web"]:
        if not isinstance(row, dict):
            raise WebInvalidResponseError()
        fields = {
            key: row[key]
            for key in ("url", "title", "description", "position")
            if key in row
        }
        hits.append(WebSearchHit.model_validate(fields))
    return tuple(hits)


def search_attempts(
    response: dict[str, Any],
    provider: str | None,
    route: WebAttemptRoute,
) -> tuple[WebAttempt, ...]:
    """Preserve keyless-ring receipts without inventing an unperformed selected attempt."""
    failure = search_failure(response)
    attempts = _ring_attempts(response)
    if attempts is not None:
        if route == "keyless-rescue":
            return tuple(
                attempt.model_copy(update={"route": route}) for attempt in attempts
            )
        return attempts
    if failure is not None:
        return (
            WebAttemptFailure(
                provider=provider, route=route, failure=failure.to_failure()
            ),
        )
    return (WebAttemptSuccess(provider=provider, route=route),)


def _ring_attempts(response: dict[str, Any]) -> tuple[WebAttempt, ...] | None:
    raw = response.get("attempts")
    if raw is None:
        return None
    try:
        if not isinstance(raw, list):
            raise WebInvalidResponseError()
        attempts = _ATTEMPTS.validate_python(tuple(raw), strict=True)
    except (TypeError, ValidationError) as exc:
        raise WebInvalidResponseError(cause=exc) from exc
    cancelled = (
        "failure" in response
        and failure_from_data(response["failure"]).kind == "cancelled"
    )
    if (
        (not attempts and not cancelled)
        or len(attempts) > 4
        or any(attempt.route != "keyless" for attempt in attempts)
    ):
        raise WebInvalidResponseError()
    if any(isinstance(attempt, WebAttemptSuccess) for attempt in attempts[:-1]):
        raise WebInvalidResponseError()
    if (
        attempts
        and not cancelled
        and isinstance(attempts[-1], WebAttemptSuccess) != response.get("success")
    ):
        raise WebInvalidResponseError()
    return attempts


def _failed_response(
    error: WebAcquisitionError, diagnostic: str | None = None
) -> dict[str, Any]:
    return {
        "success": False,
        "error": diagnostic or error.diagnostic,
        "failure": error.to_failure(),
    }


def _provider_search(
    provider: WebSearchProvider, query: str, limit: int
) -> tuple[dict[str, Any], str | None]:
    try:
        return checked_search_response(provider.search(query, limit)), None
    except Exception as exc:  # noqa: BLE001 — classify at the native acquisition boundary
        return _failed_response(
            acquisition_error(exc), str(exc)
        ), f"Error searching web: {exc}"


def _cancelled_search(provider: str, attempts: list[WebAttempt]) -> SearchExchange:
    error = WebCancelledError()
    return SearchExchange(
        WebSearchFailure(
            selected_provider=provider,
            failure=error.to_failure(),
            attempts=tuple(attempts),
        ),
        _failed_response(error, "Interrupted"),
    )


def _memoized_search(
    provider: WebSearchProvider, request: WebSearchRequest
) -> SearchExchange:
    from tools.web_result_cache import (
        bucket_limit,
        cache_enabled,
        search_memo,
        slice_search_response,
    )
    from tools.web_tools_rescue import (
        _managed_search_fallback,
        _rescue_eligible,
        _rescue_search,
    )
    from tools.interrupt import is_interrupted

    attempts: list[WebAttempt] = []
    cache_status = "miss" if cache_enabled() else "disabled"
    presentation_error = None

    def _paid_search() -> tuple[dict[str, Any], bool, str | None]:
        fetch_limit = bucket_limit(request.limit)
        response, raised = _provider_search(provider, request.query, fetch_limit)
        attempts.extend(search_attempts(response, provider.name, "selected"))
        if is_interrupted():
            return _failed_response(WebCancelledError(), "Interrupted"), False, None
        primary = search_failure(response)
        if primary is None:
            return response, False, raised
        if primary.scope in {"input", "policy"} or isinstance(
            primary, WebCancelledError
        ):
            return response, False, raised
        fallback = _managed_search_fallback(
            provider,
            str(response.get("error", "")),
            request.query,
            fetch_limit,
            attempts=attempts,
        )
        if is_interrupted():
            return _failed_response(WebCancelledError(), "Interrupted"), True, None
        if fallback is not None:
            fallback.setdefault("data", {})["backend_failure"] = primary.to_failure()
            return fallback, True, None
        if not _rescue_eligible(provider, "search"):
            return response, False, raised
        rescued = _rescue_search(
            provider.name,
            str(response.get("error", "")),
            request.query,
            fetch_limit,
            attempts=attempts,
        )
        if is_interrupted():
            return _failed_response(WebCancelledError(), "Interrupted"), True, None
        if rescued.get("success"):
            rescued.setdefault("data", {})["backend_failure"] = primary.to_failure()
        else:
            rescued["failure"] = primary.to_failure()
        return rescued, True, None

    def _cached_response() -> dict[str, Any] | None:
        cached = search_memo.lookup(provider.name, request.query, request.limit)
        return cached if cached is not None and search_failure(cached) is None else None

    response = _cached_response()
    if is_interrupted():
        return _cancelled_search(provider.name, attempts)
    if response is not None:
        cache_status = "hit"
    if response is None:
        with search_memo.flight_lock(provider.name, request.query, request.limit):
            if is_interrupted():
                return _cancelled_search(provider.name, attempts)
            response = _cached_response()
            if is_interrupted():
                return _cancelled_search(provider.name, attempts)
            if response is not None:
                cache_status = "hit"
            if response is None:
                response, rescued, presentation_error = _paid_search()
                if not rescued:
                    search_memo.store(
                        provider.name, request.query, request.limit, response
                    )
    if is_interrupted():
        return _cancelled_search(provider.name, attempts)
    response = slice_search_response(response, request.limit)
    failure = search_failure(response)
    if failure is not None:
        return SearchExchange(
            WebSearchFailure(
                selected_provider=provider.name,
                failure=failure.to_failure(),
                attempts=tuple(attempts),
            ),
            response,
            presentation_error,
        )
    data = response["data"]
    served = response.get("served_provider") or data.get(
        "served_by",
        "firecrawl" if data.get("fallback_from") == "managed_primary" else None,
    )
    successful_attempts = [
        attempt for attempt in attempts if isinstance(attempt, WebAttemptSuccess)
    ]
    if successful_attempts:
        served = successful_attempts[-1].provider
    if served is None and not data.get("rescued_from"):
        served = provider.name
    outcome = WebSearchSuccess(
        selected_provider=provider.name,
        served_provider=served,
        cache=cache_status,
        hits=search_hits(response),
        attempts=tuple(attempts),
    )
    return SearchExchange(outcome, response)


def acquire_search(request: WebSearchRequest) -> SearchExchange:
    """Resolve the native route before cache access and acquisition."""
    from agent.web_search_registry import get_active_search_provider, get_provider
    from tools import web_tools
    from tools.interrupt import is_interrupted
    from tools.tool_backend_helpers import selection_exists
    from tools.web_tools_extract import _no_provider_error, _strict_selection_error

    backend = None
    try:
        if is_interrupted():
            error = WebCancelledError()
            return SearchExchange(
                WebSearchFailure(failure=error.to_failure()),
                _failed_response(error, "Interrupted"),
            )
        web_tools._ensure_web_plugins_loaded()
        backend = web_tools._get_search_backend()
        provider = get_provider(backend) if backend else None
        if provider is None or not provider.supports_search():
            if provider is None and backend and selection_exists("web"):
                error = WebConfigurationError()
                diagnostic = _strict_selection_error("search", backend)
                return SearchExchange(
                    WebSearchFailure(
                        selected_provider=backend, failure=error.to_failure()
                    ),
                    _failed_response(error, diagnostic),
                )
            provider = get_active_search_provider()
        if provider is None:
            error = WebConfigurationError()
            diagnostic = _no_provider_error(
                "search",
                "No web search provider configured. Run `hermes tools` to set one up.",
            )
            return SearchExchange(
                WebSearchFailure(
                    selected_provider=backend or None, failure=error.to_failure()
                ),
                _failed_response(error, diagnostic),
            )
        logger.info(
            "Web search via %s: '%s' (limit: %d)",
            provider.name,
            request.query,
            request.limit,
        )
        return _memoized_search(provider, request)
    except Exception as exc:  # noqa: BLE001 — safe facts at the programmatic boundary
        error = acquisition_error(exc)
        return SearchExchange(
            WebSearchFailure(
                selected_provider=backend or None, failure=error.to_failure()
            ),
            _failed_response(error, str(exc)),
            f"Error searching web: {exc}",
        )
