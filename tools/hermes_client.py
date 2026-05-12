#!/usr/bin/env python3
"""
Hermes REST Client — Python SDK for the Hermes Agent REST API.

Provides a typed client for the OpenAI-compatible Hermes REST API.
"""

import json
import os
from typing import Any, Dict, List, Optional, Iterator


class HermesClient:
    """Python client for the Hermes Agent REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8642",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("HERMES_API_KEY", "")
        self.timeout = timeout
        self._session = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import urllib.request
        import urllib.error

        url = f"{self.base_url}{path}"
        data = None
        headers = self._headers()
        if json_data := kwargs.pop("_json", None):
            data = json.dumps(json_data).encode("utf-8")
        if extra_headers := kwargs.pop("_headers", None):
            headers.update(extra_headers)

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"error": {"message": body, "type": "http_error", "code": str(e.code)}}
        except urllib.error.URLError as e:
            return {"error": {"message": str(e.reason), "type": "connection_error"}}

    # -- Health ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Basic health check."""
        return self._request("GET", "/health")

    # -- Models ------------------------------------------------------------------

    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        return self._request("GET", "/v1/models")

    # -- Chat Completions --------------------------------------------------------

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a chat completion."""
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        headers = self._headers()
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        return self._request("POST", "/v1/chat/completions", _json=body, _headers=headers if session_id else None)

    def chat_completions_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream chat completion deltas."""
        import urllib.request
        import urllib.error

        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        headers = self._headers()
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id

        url = f"{self.base_url}/v1/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                buffer = ""
                while True:
                    chunk = resp.read(1).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    buffer += chunk
                    if buffer.endswith("\n\n"):
                        for line in buffer.strip().split("\n"):
                            if line.startswith("data: "):
                                content = line[6:]
                                if content == "[DONE]":
                                    return
                                try:
                                    data = json.loads(content)
                                    delta = data.get("choices", [{}])[0].get("delta", {})
                                    text = delta.get("content", "")
                                    if text:
                                        yield text
                                except json.JSONDecodeError:
                                    pass
                        buffer = ""
        except urllib.error.HTTPError as e:
            yield f"Error: {e.read().decode('utf-8', errors='replace')}"

    # -- Responses ---------------------------------------------------------------

    def create_response(
        self,
        model: str,
        input_data: Any,
        instructions: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Create a response (OpenAI /v1/responses compatible)."""
        body: Dict[str, Any] = {"model": model, "input": input_data, "stream": stream}
        if instructions:
            body["instructions"] = instructions
        return self._request("POST", "/v1/responses", json=body)

    def get_response(self, response_id: str) -> Dict[str, Any]:
        """Get a response by ID."""
        return self._request("GET", f"/v1/responses/{response_id}")

    def delete_response(self, response_id: str) -> Dict[str, Any]:
        """Delete a response by ID."""
        return self._request("DELETE", f"/v1/responses/{response_id}")

    # -- Runs --------------------------------------------------------------------

    def create_run(self, **kwargs: Any) -> Dict[str, Any]:
        """Create a batch run."""
        return self._request("POST", "/v1/runs", json=kwargs)

    def list_runs(self) -> Dict[str, Any]:
        """List all runs."""
        return self._request("GET", "/v1/runs")

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get a run by ID."""
        return self._request("GET", f"/v1/runs/{run_id}")

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        """Stop a run."""
        return self._request("POST", f"/v1/runs/{run_id}/stop")


def check_hermes_client_requirements() -> bool:
    """No external dependencies required."""
    return True


HERMES_CLIENT_SCHEMA = {
    "name": "hermes_client",
    "description": (
        "Python SDK client for the Hermes Agent REST API.\n\n"
        "Provides typed access to health, models, chat completions, responses, and runs.\n"
        "Usage: client = HermesClient(base_url='http://localhost:8642')\n"
        "       client.chat_completions(model='default', messages=[{'role': 'user', 'content': 'Hello'}])"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": "Base URL of the Hermes REST API",
                "default": "http://localhost:8642",
            },
            "api_key": {
                "type": "string",
                "description": "API key for authentication (or HERMES_API_KEY env var)",
            },
        },
    },
}


from tools.registry import registry

registry.register(
    name="hermes_client",
    toolset="api",
    schema=HERMES_CLIENT_SCHEMA,
    handler=lambda args, **kw: json.dumps({
        "success": True,
        "client": "HermesClient",
        "base_url": args.get("base_url", "http://localhost:8642"),
        "api_key_configured": bool(args.get("api_key") or os.getenv("HERMES_API_KEY")),
        "usage": "from tools.hermes_client import HermesClient; client = HermesClient(); client.health()",
    }),
    check_fn=check_hermes_client_requirements,
    emoji="🔌",
)