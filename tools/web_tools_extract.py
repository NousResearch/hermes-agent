"""web_extract helpers: URL validation, provider resolution, cache-aware dispatch.

Order of controls (each is a gate, never skipped by a cache hit): secret-URL
refusal -> SSRF filter (in web_tools.web_extract_tool) -> provider resolution
(strict selection) -> per-URL website policy -> disk cache -> vendor call with
one-shot keyless rescue. Logs under the origin (tools.web_tools) logger.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import threading
import json
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from agent.web_acquisition import (
    WebAttempt, WebAttemptFailure, WebAttemptRoute, WebAttemptSuccess,
    WebExtractCapabilities, WebExtractFailure, WebExtractPage, WebExtractPageFailure, WebExtractPageSuccess,
    WebExtractRequest, WebExtractResult, WebExtractSuccess,
)
from agent.web_acquisition_errors import (
    ContentCoverage, WebAcquisitionError, WebAttributionInvalidError, WebCancelledError,
    WebCapabilityUnsupportedError, WebConfigurationError, WebContentCoverageError,
    WebFailureData, WebInvalidResponseError, WebPrivateAddressDeniedError,
    WebResultMissingError, WebSecretUrlDeniedError, WebTimeoutError,
    WebUnclassifiedFailure, WebWebsitePolicyDeniedError, acquisition_error, failure_from_data,
)
from agent.web_search_provider import WebSearchProvider

from tools.tool_backend_helpers import selection_error, selection_exists
from tools.url_safety import normalize_url_for_request
from tools.web_tools_rescue import _rescue_eligible

logger = logging.getLogger("tools.web_tools")

_NO_RESULT_ERROR = "Extract backend returned no result for this URL"
_DEFAULT_EXTRACT_TIMEOUT_S = 120.0
_EXTRACT_BACKENDS_HINT = "firecrawl, tavily, keenable, exa, or parallel."
_INVALID_ITEM_ERROR = (
    "Invalid URL item at index {}: expected a URL string or an object with a string 'url' or 'href' field"
)


def _web_extract_url(value: Any) -> Optional[str]:
    """URL from a model-supplied extract item (str, or dict with ``url``/``href``); None if unusable.

    Models sometimes forward a whole search result instead of its URL, hence the dict form. Never
    stringify arbitrary objects into misleading fetch targets.
    """
    if isinstance(value, dict):
        value = value.get("url") or value.get("href")
    return (value.strip() or None) if isinstance(value, str) else None


def _disabled_plugin_error(capability: str, disabled_key: str) -> str:
    """Error text when the configured backend's bundled plugin is disabled in config."""
    vendor = disabled_key.split("/", 1)[-1]
    return (
        f"web.{capability}_backend is set to '{vendor}', but its plugin ('{disabled_key}') is disabled "
        f"in config. Re-enable it with `hermes plugins enable {disabled_key}` "
        "(or remove it from plugins.disabled)."
    )


def _no_provider_error(capability: str, fallback: str) -> str:
    """Error when no provider resolved: point at a disabled bundled plugin if that is the real cause."""
    from agent.web_search_registry import _disabled_web_plugin_for
    disabled_key = _disabled_web_plugin_for(capability=capability)
    return _disabled_plugin_error(capability, disabled_key) if disabled_key else fallback


def _strict_selection_error(capability: str, backend: str) -> str:
    """Error for a stored-but-unregistered backend: name the disabled plugin, else the bad selection.
    Strict selection never silently switches to whatever the availability walk finds."""
    failure = f"no registered web {capability} provider has that name"
    return _no_provider_error(capability, selection_error("web", f"'{backend}'", failure))


def _result_entry(url: str, error: Optional[str]) -> dict[str, Any]:
    return {"url": url, "title": "", "content": "", "error": error}


def _extract_error_json(error: str) -> str:
    return json.dumps({"success": False, "error": error}, ensure_ascii=False)


def _refuse_all(error: str):
    """Whole-call refusal tuple for ``_validate_extract_urls`` (exfiltration prevention)."""
    return None, None, None, json.dumps({"success": False, "error": error})


def _merge_in_order(
    total: int, fixed: dict[int, dict], fetch_positions: list[int], fetch_urls: list[str], results: list[dict]
) -> list[dict]:
    """Rebuild a ``total``-long result list: *fixed* entries by position, fetched *results* at
    *fetch_positions* (a short provider list yields ``_NO_RESULT_ERROR`` entries for the rest)."""
    merged = dict(fixed)
    for pos, position in enumerate(fetch_positions):
        missing = _result_entry(fetch_urls[pos], _NO_RESULT_ERROR)
        merged[position] = results[pos] if pos < len(results) else missing
    return [merged[i] for i in range(total)]


def _validate_extract_urls(urls: list[Any]):
    """Normalize model-supplied items and block URLs carrying secrets (percent-encoded forms are unquoted
    and checked too). Returns ``(normalized_urls, normalized_indices, invalid_urls, blocked_json)``;
    ``blocked_json`` is a whole-call refusal (exfiltration prevention) or None."""
    from agent.redact import _PREFIX_RE
    from urllib.parse import unquote

    normalized_urls, normalized_indices, invalid_urls = [], [], {}
    for index, item in enumerate(urls):
        _url = _web_extract_url(item)
        if _url is None:
            invalid_urls[index] = _result_entry("", _INVALID_ITEM_ERROR.format(index))
            continue
        normalized_url = normalize_url_for_request(_url)
        if any(_PREFIX_RE.search(c) for c in (_url, unquote(_url), normalized_url, unquote(normalized_url))):
            return _refuse_all(
                "Blocked: URL contains what appears to be an API key or token. "
                "Secrets must not be sent in URLs."
            )
        normalized_urls.append(normalized_url)
        normalized_indices.append(index)
    return normalized_urls, normalized_indices, invalid_urls, None


def _resolve_extract_provider(backend: str):
    """Resolve the extract provider for *backend*; returns ``(provider, error_json)``.

    A registered search-only backend is a typed error (never a silent switch). An unregistered name with
    a stored web selection is a strict-selection error; with no selection, fall through to the walk.
    """
    from agent.web_search_registry import get_active_extract_provider, get_provider as _wsp_get_provider
    provider = _wsp_get_provider(backend) if backend else None
    if provider is not None and provider.supports_extract():
        return provider, None
    if provider is not None:
        return None, _extract_error_json(
            f"{provider.display_name} is a search-only backend and cannot extract URL content. "
            "Set web.extract_backend to " + _EXTRACT_BACKENDS_HINT
        )
    if backend and selection_exists("web"):
        return None, _extract_error_json(_strict_selection_error("extract", backend))
    provider = get_active_extract_provider()
    if provider is None:
        fallback = "No web extract provider configured. Set web.extract_backend to " + _EXTRACT_BACKENDS_HINT
        return None, _extract_error_json(_no_provider_error("extract", fallback))
    return provider, None


def _extract_timeout_seconds() -> float:
    """Wall-clock cap for one provider ``extract()`` dispatch (``web.extract_timeout``, default 120s).

    A hanging backend (server keeps the response open without finishing) otherwise stalls the
    tool call indefinitely. 0 or a negative value disables the cap.
    """
    from tools.web_tools import _load_web_config
    try:
        return float(_load_web_config().get("extract_timeout", _DEFAULT_EXTRACT_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_EXTRACT_TIMEOUT_S


class _ExtractMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    sourceURL: str | None = None


class _ExtractRow(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    url: str = Field(min_length=1)
    requested_url: str | None = Field(default=None, min_length=1)
    title: str = ""
    content: str = ""
    raw_content: str | None = None
    metadata: _ExtractMetadata | None = None
    error: str | None = None
    failure: WebFailureData | None = None
    coverage: ContentCoverage = "unknown"
    served_provider: str | None = Field(default=None, min_length=1)
    attempts: list[WebAttempt] | None = Field(default=None, max_length=4)

    @field_validator("failure", mode="before")
    @classmethod
    def _safe_failure(cls, value: object) -> WebFailureData | None:
        return None if value is None else failure_from_data(value).to_failure()


_ROWS = TypeAdapter(list[_ExtractRow])


@dataclass(frozen=True)
class _PageExchange:
    outcome: WebExtractPage
    presentation: dict[str, Any]


@dataclass(frozen=True)
class ExtractExchange:
    outcome: WebExtractResult
    presentation: list[dict[str, Any]]
    presentation_error: str | None = None


def _page_failure(
    url: str, error: WebAcquisitionError, provider: str | None = None,
    route: WebAttemptRoute = "selected", *, attempted: bool = True,
    diagnostic: str | None = None, attempts: tuple[WebAttempt, ...] | None = None,
) -> _PageExchange:
    if attempts is None:
        attempts = (WebAttemptFailure(provider=provider, route=route, failure=error.to_failure()),) if attempted else ()
    return _PageExchange(
        WebExtractPageFailure(requested_url=url, failure=error.to_failure(), attempts=attempts),
        {**_result_entry(url, diagnostic or error.diagnostic), "failure": error.to_failure()},
    )


def _row_identity(row: _ExtractRow, requested: set[str]) -> str:
    if row.requested_url is not None:
        if row.requested_url not in requested:
            raise WebAttributionInvalidError()
        return row.requested_url
    source = row.metadata.sourceURL if row.metadata is not None else None
    identity = row.requested_url or source or row.url
    if identity not in requested:
        raise WebAttributionInvalidError()
    if any(value in requested and value != identity for value in (row.url, source)):
        raise WebAttributionInvalidError()
    return identity


def _row_attempts(row: _ExtractRow, provider: str | None, route: WebAttemptRoute, error: WebAcquisitionError | None) -> tuple[WebAttempt, ...]:
    if row.attempts is None:
        if error is not None:
            return (WebAttemptFailure(provider=provider, route=route, failure=error.to_failure()),)
        return (WebAttemptSuccess(provider=provider, route=route),)
    attempts = tuple(row.attempts)
    cancelled = error is not None and error.kind == "cancelled"
    unattempted = error is not None and error.kind in {"cancelled", "unsupported-capability"}
    if not attempts and not unattempted:
        raise WebInvalidResponseError()
    if any(attempt.route != "keyless" for attempt in attempts):
        raise WebInvalidResponseError()
    if any(isinstance(attempt, WebAttemptSuccess) for attempt in attempts[:-1]):
        raise WebInvalidResponseError()
    if attempts and not cancelled and isinstance(attempts[-1], WebAttemptSuccess) != (error is None):
        raise WebInvalidResponseError()
    if route == "keyless-rescue":
        return tuple(attempt.model_copy(update={"route": route}) for attempt in attempts)
    return attempts


async def _page_controls(url: str) -> WebAcquisitionError | None:
    from tools import web_tools
    from tools.website_policy import check_website_access
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return WebCancelledError()
    _, _, _, blocked = _validate_extract_urls([url])
    if blocked is not None:
        return WebSecretUrlDeniedError()
    try:
        if not await web_tools.async_is_safe_url(url):
            return WebPrivateAddressDeniedError()
        if check_website_access(url) is not None:
            return WebWebsitePolicyDeniedError()
    except Exception as exc:  # noqa: BLE001 — controls must not silently permit a failed check
        return WebConfigurationError(cause=exc)
    return WebCancelledError() if is_interrupted() else None


async def _checked_pages(
    raw: object, urls: list[str], request: WebExtractRequest, provider: str | None,
    route: WebAttemptRoute = "selected", *, cache: Literal["hit", "miss", "disabled"] = "miss",
) -> dict[str, _PageExchange]:
    try:
        attributed = checked_extract_rows(raw, urls)
        results: dict[str, _PageExchange] = {}
        for url in urls:
            row = attributed.get(url)
            if row is None:
                results[url] = _page_failure(url, WebResultMissingError(), provider, route)
                continue
            error = failure_from_data(row.failure) if row.failure is not None else WebUnclassifiedFailure() if row.error else None
            attempts = () if cache == "hit" else _row_attempts(row, provider, route, error)
            if error is not None:
                results[url] = _page_failure(url, error, attempts=attempts, diagnostic=row.error)
                continue
            content = row.raw_content if row.raw_content is not None else row.content
            if row.coverage == "full" and not content.strip():
                results[url] = _page_failure(url, WebResultMissingError(), provider, route, attempts=attempts if row.attempts is not None else None)
                continue
            control = await _page_controls(row.url)
            if control is not None:
                results[url] = _page_failure(url, control, attempts=attempts)
                continue
            if request.required_coverage == "full" and row.coverage != "full":
                results[url] = _page_failure(url, WebContentCoverageError(row.coverage), provider, route, attempts=attempts if row.attempts is not None else None)
                continue
            successful = [attempt for attempt in attempts if isinstance(attempt, WebAttemptSuccess)]
            serving = row.served_provider
            if cache != "hit":
                serving = successful[-1].provider if successful else row.served_provider or provider
            outcome = WebExtractPageSuccess(
                requested_url=url, resolved_url=row.url, title=row.title, content=content,
                coverage=row.coverage, served_provider=serving, cache=cache, attempts=attempts,
            )
            presentation = {"url": row.url, "requested_url": url, "title": row.title, "content": content,
                            "raw_content": content, "coverage": row.coverage, "error": None}
            if cache == "hit":
                presentation["cached"] = True
            results[url] = _PageExchange(outcome, presentation)
        return results
    except (ValidationError, WebInvalidResponseError) as exc:
        error = WebInvalidResponseError(cause=exc)
    except WebAttributionInvalidError as exc:
        error = exc
    return {url: _page_failure(url, error, provider, route) for url in urls}


def checked_extract_rows(raw: object, urls: list[str]) -> dict[str, _ExtractRow]:
    """Validate provider rows and attribute each row without using response order."""
    try:
        rows = _ROWS.validate_python(raw, strict=True)
    except ValidationError as exc:
        raise WebInvalidResponseError(cause=exc) from exc
    attributed: dict[str, _ExtractRow] = {}
    requested = set(urls)
    for row in rows:
        url = _row_identity(row, requested)
        if url in attributed:
            raise WebAttributionInvalidError()
        attributed[url] = row
    return attributed


def _extract_kwargs(provider: WebSearchProvider, request: WebExtractRequest) -> dict[str, Any]:
    declare = getattr(provider, "extract_capabilities", None)
    capabilities = declare() if declare is not None else WebExtractCapabilities()
    if request.format is not None and request.format not in capabilities.formats:
        raise WebCapabilityUnsupportedError(f"extract-format-{request.format}")
    if request.max_characters is not None and not capabilities.max_characters:
        raise WebCapabilityUnsupportedError("extract-max-characters")
    kwargs: dict[str, Any] = {"format": request.format}
    if request.max_characters is not None:
        kwargs["max_chars"] = request.max_characters
    parameters = inspect.signature(provider.extract).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    if request.max_characters is not None and "max_chars" not in parameters:
        raise WebCapabilityUnsupportedError("extract-max-characters")
    return {key: value for key, value in kwargs.items() if key in parameters}


async def _call_extract(extract, urls: list[str], kwargs: dict[str, Any], timeout: float) -> object:
    from tools.interrupt import acting_for_tid
    token = acting_for_tid.set(acting_for_tid.get() or threading.get_ident())
    try:
        async def invoke() -> object:
            if inspect.iscoroutinefunction(extract):
                return await extract(urls, **kwargs)
            return await asyncio.to_thread(extract, urls, **kwargs)

        if timeout <= 0:
            return await invoke()
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline:
                return await invoke()
        except TimeoutError as exc:
            if deadline.expired():
                raise WebTimeoutError(timeout, cause=exc) from exc
            raise
    finally:
        acting_for_tid.reset(token)


async def _provider_pages(provider: WebSearchProvider, urls: list[str], request: WebExtractRequest) -> dict[str, _PageExchange]:
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {url: _page_failure(url, WebCancelledError(), attempted=False) for url in urls}
    timeout = _extract_timeout_seconds()
    try:
        kwargs = _extract_kwargs(provider, request)
    except WebAcquisitionError as exc:
        return {url: _page_failure(url, exc, attempted=False) for url in urls}
    try:
        raw = await _call_extract(provider.extract, urls, kwargs, timeout)
    except Exception as exc:  # noqa: BLE001 — classify before tool presentation discards the cause
        error = acquisition_error(exc)
        diagnostic = str(exc)
        if isinstance(error, WebTimeoutError) and error.timeout_seconds is not None:
            diagnostic = f"Extract timed out after {error.timeout_seconds:.0f}s via {provider.name}"
        return {url: _page_failure(url, error, provider.name, diagnostic=diagnostic) for url in urls}
    return await _checked_pages(raw, urls, request, provider.name)


async def _rescue_pages(provider: WebSearchProvider, urls: list[str], request: WebExtractRequest, primary: dict[str, _PageExchange]) -> dict[str, _PageExchange]:
    from plugins.web.keyless_mcp import extract_with_failover
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {url: _page_failure(url, WebCancelledError(), attempts=primary[url].outcome.attempts) for url in urls}
    if not all(isinstance(page.outcome, WebExtractPageFailure) for page in primary.values()):
        return primary
    eligible = []
    for url in urls:
        outcome = primary[url].outcome
        if isinstance(outcome, WebExtractPageFailure) and outcome.attempts and outcome.failure["scope"] == "provider" and outcome.failure["kind"] != "cancelled":
            eligible.append(url)
    if not eligible or not _rescue_eligible(provider, "extract"):
        return primary
    kwargs: dict[str, Any] = {"max_chars": request.max_characters} if request.max_characters is not None else {}
    if request.format is not None:
        kwargs["format"] = request.format
    try:
        raw = await _call_extract(lambda urls, **options: extract_with_failover("exa", urls, **options), eligible, kwargs, _extract_timeout_seconds())
        alternative = await _checked_pages(raw, eligible, request, None, "keyless-rescue")
    except Exception as exc:  # noqa: BLE001 — preserve the primary failure and the rescue cause
        alternative = {url: _page_failure(url, acquisition_error(exc), route="keyless-rescue") for url in eligible}
    merged = dict(primary)
    for url, rescued in alternative.items():
        original = primary[url]
        attempts = original.outcome.attempts + rescued.outcome.attempts
        if is_interrupted():
            merged[url] = _page_failure(url, WebCancelledError(), attempts=attempts)
            continue
        if isinstance(rescued.outcome, WebExtractPageFailure):
            merged[url] = _PageExchange(original.outcome.model_copy(update={"attempts": attempts}), original.presentation)
            continue
        presentation = dict(rescued.presentation)
        presentation["metadata"] = {"rescued_from": provider.name, "backend_error": original.presentation.get("error")}
        merged[url] = _PageExchange(rescued.outcome.model_copy(update={"attempts": attempts}), presentation)
    return merged


async def _acquire_provider_pages(provider: WebSearchProvider, urls: list[str], request: WebExtractRequest, *, use_cache: bool, lookup_cache: bool = True) -> dict[str, _PageExchange]:
    from tools.interrupt import is_interrupted
    from tools.web_result_cache import cache_enabled, extract_cache_get, extract_cache_put
    results: dict[str, _PageExchange] = {}
    fetch_urls: list[str] = []
    capability_error: WebAcquisitionError | None = None
    try:
        _extract_kwargs(provider, request)
    except WebAcquisitionError as exc:
        capability_error = exc
    cache_status = "miss" if cache_enabled() else "disabled"
    for url in dict.fromkeys(urls):
        control = await _page_controls(url)
        if control is not None:
            results[url] = _page_failure(url, control, attempted=False)
            continue
        if capability_error is not None:
            results[url] = _page_failure(url, capability_error, attempted=False)
            continue
        hit = extract_cache_get(url, format=request.format, provider=provider.name, max_chars=request.max_characters, required_coverage=request.required_coverage) if use_cache and lookup_cache else None
        if hit is not None:
            results.update(await _checked_pages([hit], [url], request, provider.name, cache="hit"))
            continue
        fetch_urls.append(url)
    if not fetch_urls:
        return results
    primary = await _provider_pages(provider, fetch_urls, request)
    fetched = await _rescue_pages(provider, fetch_urls, request, primary)
    for url, page in fetched.items():
        if is_interrupted():
            results[url] = _page_failure(url, WebCancelledError(), attempts=page.outcome.attempts)
            continue
        if isinstance(page.outcome, WebExtractPageSuccess):
            outcome = page.outcome.model_copy(update={"cache": cache_status})
            page = _PageExchange(outcome, page.presentation)
            rescued = any(attempt.route == "keyless-rescue" for attempt in outcome.attempts)
            if use_cache and not rescued:
                extract_cache_put(url, outcome.content, outcome.title, format=request.format, provider=provider.name,
                                  max_chars=request.max_characters, resolved_url=outcome.resolved_url,
                                  coverage=outcome.coverage, served_provider=outcome.served_provider)
        results[url] = page
    return results


async def _dispatch_extract(provider, fetch_urls: list[str], format: Optional[str]) -> list[dict]:
    request = WebExtractRequest.model_validate({"urls": tuple(fetch_urls), "format": format})
    results = await _acquire_provider_pages(provider, fetch_urls, request, use_cache=True, lookup_cache=False)
    return [results[url].presentation for url in fetch_urls]


async def _extract_safe_urls(provider, safe_urls: list[str], format: Optional[str]) -> list[dict]:
    request = WebExtractRequest.model_validate({"urls": tuple(safe_urls), "format": format})
    results = await _acquire_provider_pages(provider, safe_urls, request, use_cache=True)
    return [results[url].presentation for url in safe_urls]


async def acquire_extract(request: WebExtractRequest) -> ExtractExchange:
    """Apply native controls and provider routing once for both acquisition consumers."""
    from agent.web_search_registry import get_provider
    from tools import web_tools
    from tools.interrupt import is_interrupted
    selected = None
    try:
        _, _, _, blocked = _validate_extract_urls(list(request.urls))
        if blocked is not None:
            return ExtractExchange(WebExtractFailure(failure=WebSecretUrlDeniedError().to_failure()), [], blocked)
        if is_interrupted():
            return ExtractExchange(WebExtractFailure(failure=WebCancelledError().to_failure()), [])
        web_tools._ensure_web_plugins_loaded()
        selected = web_tools._get_extract_backend()
        provider, diagnostic = _resolve_extract_provider(selected)
        if provider is None:
            registered = get_provider(selected) if selected else None
            error = WebCapabilityUnsupportedError("extract") if registered is not None else WebConfigurationError()
            return ExtractExchange(WebExtractFailure(selected_provider=selected or None, failure=error.to_failure()), [], diagnostic)
        normalized = [normalize_url_for_request(url.strip()) for url in request.urls]
        request = request.model_copy(update={"urls": tuple(normalized)})
        results = await _acquire_provider_pages(provider, normalized, request, use_cache=True)
        return ExtractExchange(
            WebExtractSuccess(selected_provider=provider.name, results=tuple(results[url].outcome for url in normalized)),
            [results[url].presentation for url in normalized],
        )
    except Exception as exc:  # noqa: BLE001 — return safe facts to programmatic consumers
        return ExtractExchange(WebExtractFailure(selected_provider=selected or None, failure=acquisition_error(exc).to_failure()), [], _extract_error_json(f"Error extracting content: {exc}"))
