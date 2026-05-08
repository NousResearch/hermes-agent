"""Reusable Microsoft Graph REST client helpers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from tools.microsoft_graph_auth import GraphCredentials, MicrosoftGraphTokenProvider


DEFAULT_GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


class MicrosoftGraphClientError(RuntimeError):
    """Base class for Graph client failures."""


class MicrosoftGraphAPIError(MicrosoftGraphClientError):
    """Raised when a Graph API request fails."""

    def __init__(
        self,
        status_code: int,
        method: str,
        url: str,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        payload: Any = None,
    ) -> None:
        self.status_code = status_code
        self.method = method
        self.url = url
        self.retry_after_seconds = retry_after_seconds
        self.payload = payload
        super().__init__(
            f"Microsoft Graph API error {status_code} for {method} {url}: {message}"
        )


class MicrosoftGraphClient:
    """Minimal async Microsoft Graph client with retries and pagination."""

    def __init__(
        self,
        token_provider: MicrosoftGraphTokenProvider,
        *,
        base_url: str = DEFAULT_GRAPH_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        user_agent: str = "Hermes-Agent/graph-client",
    ) -> None:
        self.token_provider = token_provider
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self.user_agent = user_agent

    @classmethod
    def from_env(cls, **kwargs: Any) -> "MicrosoftGraphClient":
        credentials = GraphCredentials.from_env()
        provider = MicrosoftGraphTokenProvider(credentials)
        return cls(provider, **kwargs)

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = await self._request("GET", path, params=params, headers=headers)
        return self._decode_json(response)

    async def post_json(
        self,
        path: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = await self._request("POST", path, json_body=json_body, headers=headers)
        return self._decode_json(response)

    async def patch_json(
        self,
        path: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = await self._request("PATCH", path, json_body=json_body, headers=headers)
        if response.status_code == 204 or not response.content:
            return {}
        return self._decode_json(response)

    async def delete(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        response = await self._request("DELETE", path, headers=headers)
        if response.status_code == 204 or not response.content:
            return {"deleted": True, "status_code": response.status_code}
        return self._decode_json(response)

    async def iterate_pages(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        next_url: str | None = self._resolve_url(path)
        next_params = dict(params or {})
        while next_url:
            response = await self._request(
                "GET",
                next_url,
                params=next_params or None,
                headers=headers,
            )
            payload = self._decode_json(response)
            if not isinstance(payload, dict):
                raise MicrosoftGraphClientError(
                    f"Expected paginated Graph response dict, got {type(payload).__name__}."
                )
            yield payload
            next_url = payload.get("@odata.nextLink")
            next_params = {}

    async def collect_paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[Any]:
        items: list[Any] = []
        async for page in self.iterate_pages(path, params=params, headers=headers):
            value = page.get("value")
            if isinstance(value, list):
                items.extend(value)
        return items

    async def download_to_file(
        self,
        path: str,
        destination: str | Path,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int = 65536,
    ) -> dict[str, Any]:
        """Download a Graph resource to disk, streaming the response body.

        The body is written chunk-by-chunk via ``response.aiter_bytes`` with
        the ``httpx.AsyncClient`` kept open for the duration of the iteration,
        so recordings and other large artifacts do not need to fit in memory.
        """
        url = self._resolve_url(path)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(target.suffix + ".part")

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            token = await self.token_provider.get_access_token(
                force_refresh=attempt > 0 and self._should_refresh_token(last_error)
            )
            request_headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            }
            if headers:
                request_headers.update(headers)

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    transport=self._transport,
                ) as client:
                    async with client.stream(
                        "GET",
                        url,
                        headers=request_headers,
                    ) as response:
                        if response.status_code >= 400:
                            # Materialize error body so we can surface a meaningful
                            # message; error bodies are small.
                            await response.aread()
                            api_error = self._build_api_error("GET", url, response)
                            last_error = api_error

                            if (
                                response.status_code == 401
                                and attempt < self.max_retries
                            ):
                                self.token_provider.clear_cache()
                                await self._sleep(
                                    self._retry_delay(response, attempt)
                                )
                                attempt += 1
                                continue

                            if (
                                self._should_retry(response)
                                and attempt < self.max_retries
                            ):
                                await self._sleep(
                                    self._retry_delay(response, attempt)
                                )
                                attempt += 1
                                continue

                            raise api_error

                        content_type = response.headers.get("content-type")
                        with tmp_target.open("wb") as handle:
                            async for chunk in response.aiter_bytes(
                                chunk_size=chunk_size
                            ):
                                if chunk:
                                    handle.write(chunk)
            except httpx.HTTPError as exc:
                last_error = exc
                tmp_target.unlink(missing_ok=True)
                if attempt >= self.max_retries:
                    raise MicrosoftGraphClientError(
                        f"Microsoft Graph download failed for GET {url}: {exc}"
                    ) from exc
                await self._sleep(self._retry_delay(None, attempt))
                attempt += 1
                continue

            os.replace(tmp_target, target)
            return {
                "path": str(target),
                "size_bytes": target.stat().st_size,
                "content_type": content_type,
            }

        tmp_target.unlink(missing_ok=True)
        raise MicrosoftGraphClientError(
            f"Microsoft Graph download exhausted retries for GET {url}."
        )

    async def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = self._resolve_url(path_or_url)
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            token = await self.token_provider.get_access_token(
                force_refresh=attempt > 0 and self._should_refresh_token(last_error)
            )
            request_headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            }
            if json_body is not None:
                request_headers["Content-Type"] = "application/json"
            if headers:
                request_headers.update(headers)

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    transport=self._transport,
                ) as client:
                    response = await client.request(
                        method,
                        url,
                        params=params,
                        json=json_body,
                        headers=request_headers,
                    )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise MicrosoftGraphClientError(
                        f"Microsoft Graph request failed for {method} {url}: {exc}"
                    ) from exc
                await self._sleep(self._retry_delay(None, attempt))
                attempt += 1
                continue

            if response.status_code < 400:
                return response

            api_error = self._build_api_error(method, url, response)
            last_error = api_error

            if response.status_code == 401 and attempt < self.max_retries:
                self.token_provider.clear_cache()
                await self._sleep(self._retry_delay(response, attempt))
                attempt += 1
                continue

            if self._should_retry(response) and attempt < self.max_retries:
                await self._sleep(self._retry_delay(response, attempt))
                attempt += 1
                continue

            raise api_error

        raise MicrosoftGraphClientError(
            f"Microsoft Graph request exhausted retries for {method} {url}."
        )

    def _resolve_url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        path = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"
        return f"{self.base_url}{path}"

    @staticmethod
    def _decode_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise MicrosoftGraphClientError(
                "Microsoft Graph response was not valid JSON for "
                f"{response.request.method} {response.request.url}"
            ) from exc

    @staticmethod
    def _should_retry(response: httpx.Response | None) -> bool:
        if response is None:
            return True
        return response.status_code == 429 or 500 <= response.status_code < 600

    @staticmethod
    def _should_refresh_token(error: Exception | None) -> bool:
        return isinstance(error, MicrosoftGraphAPIError) and error.status_code == 401

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return min(8.0, 0.5 * (2 ** attempt))

    @staticmethod
    def _build_api_error(
        method: str,
        url: str,
        response: httpx.Response,
    ) -> MicrosoftGraphAPIError:
        payload: Any = None
        message = response.text.strip() or "unknown error"
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                inner_message = error.get("message")
                if code and inner_message:
                    message = f"{code}: {inner_message}"
                elif inner_message:
                    message = str(inner_message)
            elif isinstance(error, str):
                message = error

        retry_after: float | None = None
        header_value = response.headers.get("Retry-After")
        if header_value:
            try:
                retry_after = float(header_value)
            except ValueError:
                retry_after = None

        return MicrosoftGraphAPIError(
            response.status_code,
            method,
            url,
            message,
            retry_after_seconds=retry_after,
            payload=payload,
        )
