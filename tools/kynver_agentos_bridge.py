"""Kynver AgentOS bridge for Hermes operating-state tools.

This module is intentionally small and side-effect free.  It does not register
or override any Hermes tools by itself; higher-level adapters can import it when
``kynver_tools`` is enabled and fall back to local Hermes stores otherwise.

Scope: memory, task/todo, session lifecycle, skills, and plan/progress APIs.
Do not use this bridge for local machine-control tools such as terminal, file,
browser, or media tools.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

_DEFAULT_API_URL = "https://www.kynver.com"
_DEFAULT_AGENT_OS_SLUG = "ghost"
_DEFAULT_TIMEOUT_SECONDS = 120.0


class KynverAgentOSError(RuntimeError):
    """Raised for Kynver AgentOS bridge failures.

    The message is safe to surface to the model/user: bearer tokens and common
    key patterns are redacted before construction.
    """


@dataclass(frozen=True)
class KynverAgentOSConfig:
    """Runtime config for Kynver AgentOS API calls."""

    api_url: str = _DEFAULT_API_URL
    api_key: str = ""
    slug: str = _DEFAULT_AGENT_OS_SLUG
    timeout: float = _DEFAULT_TIMEOUT_SECONDS

    @property
    def enabled(self) -> bool:
        """True when enough auth exists for write/read AgentOS calls."""

        return bool(self.api_url and self.api_key and self.slug)


def _active_env_path() -> Path:
    """Return the active Hermes profile env path."""

    return get_hermes_home() / ".env"


def _load_profile_env(path: Path | None = None) -> dict[str, str]:
    """Load key/value pairs from the active profile .env without mutating os.environ."""

    env_path = path or _active_env_path()
    out: dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return out
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip().strip('"\'')
    return out


def load_kynver_agentos_config(env: Mapping[str, str] | None = None) -> KynverAgentOSConfig:
    """Load Kynver AgentOS config from process env over active profile .env.

    Hermes MCP subprocesses use launcher scripts that source ``~/.hermes/.env``;
    this bridge mirrors that convention without copying secrets into
    ``config.yaml``.
    """

    profile_env = _load_profile_env()
    merged: dict[str, str] = dict(profile_env)
    merged.update(dict(env or os.environ))

    api_url = (merged.get("KYNVER_API_URL") or _DEFAULT_API_URL).strip().rstrip("/")
    api_key = (merged.get("KYNVER_API_KEY") or "").strip()
    slug = (merged.get("KYNVER_AGENT_OS_SLUG") or _DEFAULT_AGENT_OS_SLUG).strip()
    raw_timeout = (merged.get("KYNVER_FETCH_TIMEOUT_MS") or "").strip()
    timeout = _DEFAULT_TIMEOUT_SECONDS
    if raw_timeout:
        try:
            timeout = max(1.0, float(raw_timeout) / 1000.0)
        except ValueError:
            timeout = _DEFAULT_TIMEOUT_SECONDS
    return KynverAgentOSConfig(api_url=api_url, api_key=api_key, slug=slug, timeout=timeout)


def _redact(text: str) -> str:
    """Redact bearer/API-key shaped values from errors."""

    import re

    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", text)
    redacted = re.sub(r"(?i)(api[_-]?key|token|secret|password)=([^\s&]+)", r"\1=[REDACTED]", redacted)
    return redacted[:2000]


class KynverAgentOSClient:
    """Small REST client for Kynver AgentOS endpoints."""

    def __init__(self, config: KynverAgentOSConfig | None = None):
        self.config = config or load_kynver_agentos_config()

    def api_path(self, path: str, *, slug: str | None = None) -> str:
        """Build a safe ``/api/agent-os/{slug}/...`` path."""

        if "\x00" in path or ".." in path.split("?")[0].split("/"):
            raise KynverAgentOSError("Invalid AgentOS API path")
        clean = path if path.startswith("/") else f"/{path}"
        if clean.startswith("/api/"):
            return clean
        if clean.startswith("/agent-os/"):
            return f"/api{clean}"
        effective_slug = urllib.parse.quote((slug or self.config.slug or _DEFAULT_AGENT_OS_SLUG).strip())
        return f"/api/agent-os/{effective_slug}{clean}"

    def request(self, method: str, path: str, body: Any | None = None, *, slug: str | None = None) -> Any:
        """Call Kynver and return parsed JSON/text.

        Raises ``KynverAgentOSError`` with redacted messages on failure.
        """

        if not self.config.enabled:
            raise KynverAgentOSError(
                "Kynver AgentOS is not configured: set KYNVER_API_URL, "
                "KYNVER_API_KEY, and KYNVER_AGENT_OS_SLUG in the active Hermes profile .env."
            )
        url = f"{self.config.api_url}{self.api_path(path, slug=slug)}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": "hermes-forge-kynver-agentos-bridge/0.1",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as res:  # nosec B310 - user-configured trusted Kynver origin
                payload = res.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise KynverAgentOSError(_redact(f"Kynver AgentOS HTTP {exc.code}: {detail}")) from exc
        except Exception as exc:
            raise KynverAgentOSError(_redact(f"Kynver AgentOS request failed: {exc}")) from exc
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload

    def get(self, path: str, *, slug: str | None = None) -> Any:
        return self.request("GET", path, slug=slug)

    def post(self, path: str, body: Any, *, slug: str | None = None) -> Any:
        return self.request("POST", path, body, slug=slug)

    def patch(self, path: str, body: Any, *, slug: str | None = None) -> Any:
        return self.request("PATCH", path, body, slug=slug)

    def delete(self, path: str, *, slug: str | None = None) -> Any:
        return self.request("DELETE", path, slug=slug)


def agentos_available(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the active profile has enough Kynver config for calls."""

    return load_kynver_agentos_config(env).enabled
