#!/usr/bin/env python3
"""Native UTCP (Universal Tool Calling Protocol) tool provider.

UTCP providers publish a *manual* — a list of tool definitions plus the
call template (transport, URL, auth) needed to invoke each one.  This module
connects to those providers directly, registers their tools in the Hermes
tool registry, and dispatches model tool calls through the official
``utcp`` Python SDK.  No MCP server sits in the middle.

Configuration is read from ``~/.hermes/config.yaml`` under the ``utcp`` key:

    utcp:
      enabled: true
      providers:
        - name: weather
          call_template_type: http
          url: https://weather.example.com/.well-known/utcp
          variables:
            API_KEY: env:WEATHER_UTCP_API_KEY
          headers:
            Authorization: "Bearer ${API_KEY}"

Design notes
------------
* **Optional dependency.** ``utcp`` / ``utcp-http`` are imported lazily.  With
  the package absent this module is importable and inert, exactly like
  ``tools/mcp_tool.py`` without the ``mcp`` package.
* **One client per provider.** Each provider gets its own ``UtcpClient`` (and
  therefore its own transport instances, variable scope, and failure domain),
  so a broken or slow provider cannot take its siblings down.
* **Provider isolation.** Discovery runs per provider and failures are
  recorded per provider; the other providers still register their tools.
* **Credentials never enter the model context.** Secrets come from
  ``.env`` via ``env:NAME`` references, are substituted inside the SDK, and
  every error string is scrubbed before it reaches the agent.
* **Credential scoping.** The SDK resolves ``${VAR}`` references inside a
  provider namespace (``<provider>_<VAR>``).  Before a template reaches the
  SDK we require every reference to be declared in the provider's
  ``variables:`` block, so a malicious manual cannot reference (and exfiltrate)
  an unrelated environment variable such as ``OPENAI_API_KEY``.

Public entry points::

    discover_utcp_tools()          # register tools for every enabled provider
    refresh_utcp_provider(name)    # re-discover one provider (or all)
    get_utcp_status()              # provider-level status for CLI/UI
    get_utcp_tools()               # discovered tool inventory
    shutdown_utcp_providers()      # close clients and stop the loop thread
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fnmatch
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Registry/wire name prefix for UTCP-provided tools.  Mirrors the
# ``mcp__<server>__<tool>`` convention so models and provider-side name
# validators see one consistent shape for external toolsets.
UTCP_TOOL_NAME_PREFIX = "utcp__"
_NAME_DELIM = "__"

# Transports Hermes enables by default.  These are the transports the
# official SDK ships as stable HTTP plugins and are the only ones that cannot
# execute local code or read local files.  CLI/file/socket/MCP transports are
# deliberately excluded: enabling them would let a provider config run
# commands or read the filesystem, which is out of scope for this provider.
DEFAULT_ALLOWED_TRANSPORTS: tuple[str, ...] = ("http", "sse", "streamable_http")

DEFAULT_TOOL_TIMEOUT = 60.0
MIN_TOOL_TIMEOUT = 1.0
MAX_TOOL_TIMEOUT = 600.0
DEFAULT_CONNECT_TIMEOUT = 15.0
MAX_CONNECT_TIMEOUT = 120.0
DEFAULT_MAX_RESULT_CHARS = 200_000
DEFAULT_MAX_CONCURRENT_CALLS = 4
MAX_MAX_CONCURRENT_CALLS = 64
DEFAULT_MAX_ARG_BYTES = 256 * 1024

# Consecutive-failure circuit breaker, same shape as the MCP provider's:
# after N consecutive failures a provider short-circuits for a cooldown so a
# dead endpoint cannot burn the agent's whole turn budget.
FAILURE_THRESHOLD = 5
FAILURE_COOLDOWN_SECONDS = 60.0

_VARIABLE_REF_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)")
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-+/=]{8,}"),
    re.compile(r"(?i)\b(?:sk|pk|ghp|gho|ghs|xox[abp])[-_][A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)\b([a-z0-9_]*(?:api[_-]?key|token|secret|password))\b\s*[=:]\s*[\"']?([^\s\"'&,;]{6,})"),
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UtcpConfigError(ValueError):
    """Invalid ``utcp`` configuration.  Message is user-facing."""


class UtcpDependencyError(RuntimeError):
    """The optional ``utcp`` SDK is not installed."""


class UtcpDiscoveryError(RuntimeError):
    """A provider's manual could not be fetched or parsed."""


class UtcpProviderUnavailable(RuntimeError):
    """The provider has no live client (discovery failed or it was closed)."""


class UtcpTimeoutError(TimeoutError):
    """A UTCP call exceeded its configured timeout."""


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtcpSettings:
    """Process-wide ``utcp`` settings."""

    enabled: bool = False
    discovery_timeout: float = 1.5
    single_query_discovery_timeout: float = 15.0
    allowed_transports: tuple[str, ...] = DEFAULT_ALLOWED_TRANSPORTS
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS
    max_arg_bytes: int = DEFAULT_MAX_ARG_BYTES


@dataclass(frozen=True)
class UtcpProviderConfig:
    """A validated, self-contained provider definition.

    ``template`` is the raw UTCP call template (``${VAR}`` placeholders kept
    intact) handed to the SDK.  ``env_refs`` maps each declared variable to the
    environment variable that supplies it: values are read at discovery time, so
    a provider can be configured before its secret exists, and rotating a secret
    needs no config edit.
    """

    name: str
    call_template_type: str
    template: dict[str, Any]
    env_refs: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout: float = DEFAULT_TOOL_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @property
    def namespace(self) -> str:
        """Variable namespace the SDK applies to this provider's manual."""
        return self.name.replace("_", "__")

    @property
    def toolset(self) -> str:
        return f"utcp-{self.name}"

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.env_refs))

    def missing_env(self) -> list[str]:
        """Environment variables this provider declares but that are unset."""
        return sorted(
            env_name
            for env_name in self.env_refs.values()
            if not os.environ.get(env_name)
        )

    def resolve_variables(self) -> dict[str, str]:
        """Read declared variables from the environment.

        Raises :class:`UtcpConfigError` listing every missing variable at once,
        so a misconfigured provider produces one actionable error.
        """
        missing = self.missing_env()
        if missing:
            raise UtcpConfigError(
                f"utcp provider '{self.name}': unset environment variable(s) "
                f"{', '.join(missing)} (add them to ~/.hermes/.env)"
            )
        return {
            name: os.environ[env_name] for name, env_name in self.env_refs.items()
        }

    def sdk_variables(self) -> dict[str, str]:
        """Variables keyed the way ``UtcpClient`` looks them up."""
        return {
            f"{self.namespace}_{key}": value
            for key, value in self.resolve_variables().items()
        }


@dataclass
class ConfigProblem:
    """A provider entry that failed validation (reported, never registered)."""

    name: str
    message: str


@dataclass(frozen=True)
class UtcpToolSpec:
    """One discovered UTCP tool, normalized for the Hermes registry."""

    provider: str
    utcp_name: str  # name inside the UTCP client: "<provider>.<tool>"
    raw_name: str
    registry_name: str
    description: str
    parameters: dict[str, Any]
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return default


def _as_float(value: Any, default: float, *, minimum: float, maximum: float, label: str) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning("utcp: %s must be a number; using %s", label, default)
        return default
    if parsed < minimum or parsed > maximum:
        logger.warning(
            "utcp: %s=%s out of range [%s, %s]; clamping", label, parsed, minimum, maximum
        )
        return min(max(parsed, minimum), maximum)
    return parsed


def _as_int(value: Any, default: int, *, minimum: int, maximum: int, label: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("utcp: %s must be an integer; using %s", label, default)
        return default
    if parsed < minimum or parsed > maximum:
        logger.warning(
            "utcp: %s=%s out of range [%s, %s]; clamping", label, parsed, minimum, maximum
        )
        return min(max(parsed, minimum), maximum)
    return parsed


def _sanitize_component(value: str) -> str:
    """Reduce a name to ``[A-Za-z0-9_]`` for provider-side tool-name validators."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "").strip())
    return re.sub(r"_+", "_", cleaned).strip("_")


def utcp_registry_tool_name(provider: str, tool: str) -> str:
    """Deterministic, collision-resistant registry name for a UTCP tool.

    ``utcp__<provider>__<tool>``.  Long names are truncated with a stable
    digest suffix so two long tool names cannot silently collapse onto the
    same registry entry.
    """
    provider_part = _sanitize_component(provider)[:32] or "provider"
    tool_part = _sanitize_component(tool)[:48] or "tool"
    name = f"{UTCP_TOOL_NAME_PREFIX}{provider_part}{_NAME_DELIM}{tool_part}"
    if len(name) > 64:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:55].rstrip('_')}_{digest}"
    return name


def _is_loopback_host(host: str) -> bool:
    """True for hosts that resolve to the local machine (no DNS lookup)."""
    host = (host or "").strip().strip("[]").rstrip(".").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"}


def validate_provider_url(url: str, provider: str) -> None:
    """Enforce the transport trust boundary before any request is made.

    The SDK applies the same rule (HTTPS, or plain HTTP to loopback).  Hermes
    checks it at config load as well so users get a clear, early error instead
    of a discovery failure at agent startup.
    """
    if not isinstance(url, str) or not url.strip():
        raise UtcpConfigError(f"utcp provider '{provider}': url is required")
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise UtcpConfigError(
            f"utcp provider '{provider}': url must use http(s), got '{scheme or 'none'}'"
        )
    if "@" in (parsed.netloc or ""):
        raise UtcpConfigError(
            f"utcp provider '{provider}': url must not embed credentials; "
            "use variables/auth instead"
        )
    host = parsed.hostname or ""
    if not host:
        raise UtcpConfigError(f"utcp provider '{provider}': url has no host")
    if "{" in host or "}" in host:
        raise UtcpConfigError(
            f"utcp provider '{provider}': url host must not be a template placeholder"
        )
    if scheme == "http" and not _is_loopback_host(host):
        raise UtcpConfigError(
            f"utcp provider '{provider}': plain http is only allowed for loopback hosts; "
            f"use https for '{host}'"
        )


def _iter_variable_refs(value: Any, *, _skip_refs: bool = True) -> Iterable[str]:
    """Yield `${VAR}` / `$VAR` names referenced anywhere in a JSON value."""
    if isinstance(value, str):
        if _skip_refs and re.search(r"\$ref(?![a-zA-Z0-9_])", value):
            return
        for match in _VARIABLE_REF_RE.finditer(value):
            yield next(group for group in match.groups() if group is not None)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_variable_refs(item, _skip_refs=_skip_refs)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_variable_refs(item, _skip_refs=_skip_refs)


def _unscoped_variable_refs(value: Any, provider: UtcpProviderConfig) -> list[str]:
    """Return references that are not covered by the provider's own variables.

    The SDK namespaces every lookup (``<provider>_<VAR>``), so a reference is
    resolvable only when the provider declared ``VAR``.  Anything else is
    reported: letting it through would mean a provider-supplied manual that
    says ``${OPENAI_API_KEY}`` either fails at call time or, in an
    unnamespaced substitution path, leaks an unrelated secret.
    """
    declared = {f"{provider.namespace}_{key}" for key in provider.env_refs}
    allowed = declared | set(provider.env_refs)
    return sorted({ref for ref in _iter_variable_refs(value) if ref not in allowed})


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _utcp_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        from utils import env_var_enabled

        if env_var_enabled("HERMES_SAFE_MODE"):
            return {}
        config = load_config() or {}
    except Exception as exc:  # pragma: no cover - defensive: config must never crash startup
        logger.debug("utcp: could not read config: %s", exc)
        return {}
    section = config.get("utcp")
    if section is None:
        return {}
    if not isinstance(section, dict):
        logger.warning("utcp: config section must be a mapping; ignoring")
        return {}
    return section


def load_utcp_settings(section: Optional[dict[str, Any]] = None) -> UtcpSettings:
    section = _utcp_section() if section is None else section
    transports = section.get("allowed_transports")
    if isinstance(transports, str):
        transports = [transports]
    if not isinstance(transports, (list, tuple)) or not transports:
        transports = list(DEFAULT_ALLOWED_TRANSPORTS)
    normalized: list[str] = []
    for item in transports:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
    if not normalized:
        normalized = list(DEFAULT_ALLOWED_TRANSPORTS)
    return UtcpSettings(
        enabled=_as_bool(section.get("enabled"), False),
        discovery_timeout=_as_float(
            section.get("discovery_timeout"), 1.5, minimum=0.1, maximum=120.0,
            label="discovery_timeout",
        ),
        single_query_discovery_timeout=_as_float(
            section.get("single_query_discovery_timeout"), 15.0, minimum=0.1, maximum=300.0,
            label="single_query_discovery_timeout",
        ),
        allowed_transports=tuple(normalized),
        max_result_chars=_as_int(
            section.get("max_result_chars"), DEFAULT_MAX_RESULT_CHARS,
            minimum=1_000, maximum=10_000_000, label="max_result_chars",
        ),
        max_concurrent_calls=_as_int(
            section.get("max_concurrent_calls"), DEFAULT_MAX_CONCURRENT_CALLS,
            minimum=1, maximum=MAX_MAX_CONCURRENT_CALLS, label="max_concurrent_calls",
        ),
        max_arg_bytes=_as_int(
            section.get("max_arg_bytes"), DEFAULT_MAX_ARG_BYTES,
            minimum=1_024, maximum=10_000_000, label="max_arg_bytes",
        ),
    )


_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")


def _parse_env_refs(entry: dict[str, Any], name: str) -> dict[str, str]:
    """Parse ``variables: {NAME: env:ENV_VAR}`` into ``{NAME: ENV_VAR}``.

    Only ``env:`` references are accepted, and the value is **not** read here:
    keeping secrets in ``.env`` (and only the *name* in ``config.yaml``) means
    UTCP credentials follow the same storage rule as every other Hermes
    credential, and a provider can be configured before its secret exists.
    Missing variables surface as a per-provider discovery error.
    """
    raw = entry.get("variables")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise UtcpConfigError(
            f"utcp provider '{name}': variables must be a mapping of NAME -> env:ENV_VAR"
        )
    refs: dict[str, str] = {}
    for key, ref in raw.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", key):
            raise UtcpConfigError(
                f"utcp provider '{name}': variable name {key!r} must contain only "
                "letters, digits and underscores"
            )
        if not isinstance(ref, str) or not ref.startswith("env:"):
            raise UtcpConfigError(
                f"utcp provider '{name}': variables.{key} must be 'env:ENV_VAR_NAME' "
                "(put the secret in .env, never in config.yaml)"
            )
        env_name = ref[len("env:"):].strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", env_name or ""):
            raise UtcpConfigError(
                f"utcp provider '{name}': variables.{key} names an invalid environment variable"
            )
        refs[key] = env_name
    return refs


def _build_provider_config(
    entry: Any,
    settings: UtcpSettings,
    *,
    index: int = 0,
) -> UtcpProviderConfig:
    if not isinstance(entry, dict):
        raise UtcpConfigError(f"utcp.providers[{index}] must be a mapping")

    name = entry.get("name")
    if not isinstance(name, str) or not _PROVIDER_NAME_RE.match(name.strip()):
        raise UtcpConfigError(
            f"utcp.providers[{index}]: name must match [A-Za-z0-9_]{{1,40}} "
            "(it becomes part of every tool name)"
        )
    name = name.strip()

    call_template_type = entry.get("call_template_type") or entry.get("transport") or "http"
    if not isinstance(call_template_type, str):
        raise UtcpConfigError(f"utcp provider '{name}': call_template_type must be a string")
    call_template_type = call_template_type.strip()
    if call_template_type not in settings.allowed_transports:
        raise UtcpConfigError(
            f"utcp provider '{name}': transport '{call_template_type}' is not allowed; "
            f"allowed transports: {', '.join(settings.allowed_transports)}"
        )

    url = entry.get("url")
    validate_provider_url(url, name)

    env_refs = _parse_env_refs(entry, name)

    # Everything that is not Hermes-specific provider metadata is handed to
    # the SDK verbatim, so new SDK template fields keep working without a
    # Hermes release.
    control_keys = {
        "name", "enabled", "timeout", "connect_timeout", "max_result_chars",
        "max_concurrent_calls", "variables", "transport", "tools",
    }
    template: dict[str, Any] = {
        key: value for key, value in entry.items() if key not in control_keys
    }
    template["name"] = name
    template["call_template_type"] = call_template_type

    headers = template.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise UtcpConfigError(f"utcp provider '{name}': headers must map strings to strings")
        for key, value in headers.items():
            if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
                raise UtcpConfigError(
                    f"utcp provider '{name}': header {key!r} contains CR/LF "
                    "(header injection)"
                )

    tools_filter = entry.get("tools") or {}
    if not isinstance(tools_filter, dict):
        raise UtcpConfigError(f"utcp provider '{name}': tools must be a mapping")

    def _patterns(value: Any, label: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return tuple(value)
        raise UtcpConfigError(
            f"utcp provider '{name}': tools.{label} must be a string or list of strings"
        )

    provider = UtcpProviderConfig(
        name=name,
        call_template_type=call_template_type,
        template=template,
        env_refs=env_refs,
        enabled=_as_bool(entry.get("enabled"), True),
        timeout=_as_float(
            entry.get("timeout"), DEFAULT_TOOL_TIMEOUT,
            minimum=MIN_TOOL_TIMEOUT, maximum=MAX_TOOL_TIMEOUT, label=f"{name}.timeout",
        ),
        connect_timeout=_as_float(
            entry.get("connect_timeout"), DEFAULT_CONNECT_TIMEOUT,
            minimum=1.0, maximum=MAX_CONNECT_TIMEOUT, label=f"{name}.connect_timeout",
        ),
        max_result_chars=_as_int(
            entry.get("max_result_chars"), settings.max_result_chars,
            minimum=1_000, maximum=10_000_000, label=f"{name}.max_result_chars",
        ),
        max_concurrent_calls=_as_int(
            entry.get("max_concurrent_calls"), settings.max_concurrent_calls,
            minimum=1, maximum=MAX_MAX_CONCURRENT_CALLS,
            label=f"{name}.max_concurrent_calls",
        ),
        include=_patterns(tools_filter.get("include"), "include"),
        exclude=_patterns(tools_filter.get("exclude"), "exclude"),
    )

    # Pre-flight the credential scope of the provider's own template: an
    # undeclared reference here would be resolved (or fail) inside the SDK
    # before we ever see the manual.
    unknown = _unscoped_variable_refs(template, provider)
    if unknown:
        raise UtcpConfigError(
            f"utcp provider '{name}': template references undeclared variable(s) "
            f"{', '.join(unknown)}; declare each one under variables: with an env: reference"
        )
    return provider


def validate_provider_entry(entry: Any, settings: UtcpSettings) -> UtcpProviderConfig:
    """Validate one provider entry without touching global state.

    Public wrapper used by ``hermes utcp add`` so the CLI and the runtime
    enforce exactly the same rules.
    """
    return _build_provider_config(entry, settings)


def load_utcp_config(
    section: Optional[dict[str, Any]] = None,
) -> tuple[UtcpSettings, list[UtcpProviderConfig], list[ConfigProblem]]:
    """Load and validate the ``utcp`` config section.

    Returns ``(settings, providers, problems)``.  Invalid providers are
    reported instead of aborting the whole feature, so one typo cannot hide
    every other provider's tools.
    """
    section = _utcp_section() if section is None else section
    settings = load_utcp_settings(section)
    raw_providers = section.get("providers") or []
    providers: list[UtcpProviderConfig] = []
    problems: list[ConfigProblem] = []
    if not isinstance(raw_providers, (list, tuple)):
        problems.append(ConfigProblem("<config>", "utcp.providers must be a list"))
        return settings, providers, problems

    seen: dict[str, int] = {}
    for index, entry in enumerate(raw_providers):
        fallback_name = ""
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            fallback_name = entry["name"]
        try:
            provider = _build_provider_config(entry, settings, index=index)
        except UtcpConfigError as exc:
            problems.append(ConfigProblem(fallback_name or f"#{index}", str(exc)))
            continue
        if provider.name in seen:
            problems.append(
                ConfigProblem(
                    provider.name,
                    f"utcp provider '{provider.name}' is defined twice "
                    f"(indexes {seen[provider.name]} and {index}); the second is ignored",
                )
            )
            continue
        seen[provider.name] = index
        providers.append(provider)
    return settings, providers, problems


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    return str(value)


def _dump_result(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default)


def normalize_input_schema(raw: Any) -> dict[str, Any]:
    """Normalize a UTCP/JSON-Schema tool input into a tool-calling schema.

    Repairs the shapes that provider APIs reject (missing ``type``, dangling
    ``required`` entries, legacy ``definitions`` refs) without changing the
    tool's meaning.  A non-object input schema is not representable as a
    function parameter object and is rejected by the caller.
    """
    if raw is None:
        return {"type": "object", "properties": {}}
    if hasattr(raw, "model_dump"):
        try:
            raw = raw.model_dump(by_alias=True, exclude_none=True)
        except Exception:  # pragma: no cover - pydantic model that refuses dumping
            raw = {}
    if not isinstance(raw, dict) or not raw:
        return {"type": "object", "properties": {}}

    schema = json.loads(json.dumps(raw, default=_json_default))

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return
        if isinstance(node.get("definitions"), dict) and "$defs" not in node:
            node["$defs"] = node.pop("definitions")
        for key, value in list(node.items()):
            if key == "properties" and isinstance(value, dict):
                for prop in value.values():
                    _walk(prop)
            elif isinstance(value, (dict, list)):
                _walk(value)

    _walk(schema)
    schema.pop("$schema", None)
    schema.pop("$id", None)

    if schema.get("type") is None:
        schema["type"] = "object"
    properties = schema.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, dict):
        raise UtcpConfigError("tool input schema has non-object 'properties'")
    schema["properties"] = properties
    required = schema.get("required")
    if isinstance(required, list):
        pruned = [name for name in required if isinstance(name, str) and name in properties]
        if pruned:
            schema["required"] = pruned
        else:
            schema.pop("required", None)
    return schema


def _truncate_description(text: str, limit: int = 1024) -> str:
    cleaned = _CONTROL_CHARS_RE.sub(" ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _matches_filter(tool_name: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    if include:
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in include)
    if exclude:
        return not any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in exclude)
    return True


def _tool_specs_from_manual(
    provider: UtcpProviderConfig,
    tools: Sequence[Any],
) -> tuple[list[UtcpToolSpec], list[str]]:
    """Convert SDK ``Tool`` objects into registry-ready specs.

    Returns ``(specs, rejected)`` where ``rejected`` describes tools that were
    skipped (bad schema, undeclared credential reference, name collision).
    """
    rejected: list[str] = []
    by_name: dict[str, list[UtcpToolSpec]] = {}

    for tool in tools:
        # `UtcpClient.register_manual` renames tools to "<manual>.<tool>";
        # keep the provider-local name for identity and filters, and use the
        # fully qualified name for dispatch.
        registered_name = str(getattr(tool, "name", "") or "").strip()
        if not registered_name:
            rejected.append("tool with an empty name")
            continue
        prefix = f"{provider.name}."
        raw_name = registered_name.removeprefix(prefix)
        if not raw_name:
            rejected.append("tool with an empty name")
            continue
        if not _matches_filter(raw_name, provider.include, provider.exclude):
            logger.debug("utcp '%s': tool '%s' filtered by config", provider.name, raw_name)
            continue

        # A discovered tool's template is provider-supplied data.  Refuse it
        # when it references an undeclared variable: this is the guard that
        # stops a hostile manual from pointing ${OPENAI_API_KEY} at its own
        # endpoint.
        try:
            from utcp.data.call_template import CallTemplateSerializer

            template_dict = CallTemplateSerializer().to_dict(tool.tool_call_template)
        except Exception as exc:
            rejected.append(f"{raw_name}: unreadable call template ({exc})")
            continue

        tool_transport = template_dict.get("call_template_type")
        if tool_transport != provider.call_template_type:
            # The SDK enforces the same rule via allowed_communication_protocols;
            # checking here makes the error attributable to a specific tool.
            rejected.append(
                f"{raw_name}: transport '{tool_transport}' does not match the "
                f"provider transport '{provider.call_template_type}'"
            )
            continue

        unknown = _unscoped_variable_refs(template_dict, provider)
        if unknown:
            rejected.append(
                f"{raw_name}: references undeclared variable(s) {', '.join(unknown)}"
            )
            continue

        try:
            parameters = normalize_input_schema(getattr(tool, "inputs", None))
        except UtcpConfigError as exc:
            rejected.append(f"{raw_name}: {exc}")
            continue
        if parameters.get("type") != "object":
            rejected.append(
                f"{raw_name}: input schema type '{parameters.get('type')}' is not "
                "representable as tool-call parameters (object required)"
            )
            continue

        spec = UtcpToolSpec(
            provider=provider.name,
            utcp_name=f"{provider.name}.{raw_name}",
            raw_name=raw_name,
            registry_name=utcp_registry_tool_name(provider.name, raw_name),
            description=_truncate_description(
                getattr(tool, "description", "") or f"UTCP tool {raw_name} from {provider.name}"
            ),
            parameters=parameters,
            tags=tuple(str(tag) for tag in (getattr(tool, "tags", None) or ())),
        )
        by_name.setdefault(spec.registry_name, []).append(spec)

    specs: list[UtcpToolSpec] = []
    for registry_name, group in by_name.items():
        if len(group) > 1:
            rejected.append(
                f"{registry_name}: {len(group)} provider tools normalize to the same "
                "registry name; skipping all of them"
            )
            continue
        specs.append(group[0])
    return specs, rejected


# ---------------------------------------------------------------------------
# Argument validation & result normalization
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(parameters: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Cheap structural validation against the tool's input schema.

    Not a full JSON-Schema implementation: it catches the mistakes a model
    actually makes (missing required field, wrong primitive type, unknown
    field on a closed schema) without pulling in a validator dependency.
    """
    problems: list[str] = []
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    schema = parameters or {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []

    for name in required:
        if name not in args:
            problems.append(f"missing required argument '{name}'")

    closed = schema.get("additionalProperties") is False
    for name, value in args.items():
        prop = properties.get(name)
        if prop is None:
            if closed:
                problems.append(f"unknown argument '{name}'")
            continue
        expected = prop.get("type")
        if isinstance(expected, str) and expected in _TYPE_MAP:
            py_type = _TYPE_MAP[expected]
            # bool is a subclass of int — reject it where a number is expected.
            if expected in {"integer", "number"} and isinstance(value, bool):
                problems.append(f"argument '{name}' must be {expected}, got boolean")
            elif not isinstance(value, py_type):
                problems.append(
                    f"argument '{name}' must be {expected}, got {type(value).__name__}"
                )
        enum = prop.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            problems.append(f"argument '{name}' must be one of {enum!r}")
        if value is None and "null" not in str(prop.get("type", "")):
            if name in required:
                problems.append(f"argument '{name}' must not be null")
    return problems


def serialize_result(result: Any) -> Any:
    """Preserve structured UTCP results.

    Dicts/lists/numbers/bools pass through untouched (the agent serializes
    them to JSON once), strings pass through as text, anything exotic falls
    back to ``str``.
    """
    if result is None or isinstance(result, (dict, list, str, int, float, bool)):
        return result
    if isinstance(result, (bytes, bytearray)):
        return bytes(result).decode("utf-8", errors="replace")
    return str(result)


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


def _secret_values(provider: UtcpProviderConfig) -> list[str]:
    values: list[str] = []
    for env_name in provider.env_refs.values():
        value = os.environ.get(env_name)
        if isinstance(value, str) and len(value) >= 4:
            values.append(value)
    auth = provider.template.get("auth")
    if isinstance(auth, dict):
        for key, value in auth.items():
            if isinstance(value, str) and any(
                marker in key.lower() for marker in ("key", "secret", "password", "token")
            ) and len(value) >= 4:
                values.append(value)
    return values


def redact_secrets(text: str, provider: Optional[UtcpProviderConfig] = None) -> str:
    """Strip configured secrets and obvious credential shapes from *text*."""
    if not text:
        return text
    scrubbed = text
    for secret in _secret_values(provider) if provider else []:
        scrubbed = scrubbed.replace(secret, "***")
    scrubbed = _SECRET_PATTERNS[0].sub(r"\1 ***", scrubbed)
    scrubbed = _SECRET_PATTERNS[1].sub("***", scrubbed)
    scrubbed = _SECRET_PATTERNS[2].sub(r"\1=***", scrubbed)
    return scrubbed


def _safe_error_text(exc: BaseException, provider: Optional[UtcpProviderConfig]) -> str:
    message = str(exc) or exc.__class__.__name__
    return redact_secrets(_CONTROL_CHARS_RE.sub(" ", message), provider).strip()[:800]


def classify_error(exc: BaseException) -> str:
    """Map an SDK/transport exception onto a stable error class for the model."""
    name = type(exc).__name__
    if isinstance(exc, (asyncio.TimeoutError, concurrent.futures.TimeoutError)) or name in {
        "TimeoutError", "ServerTimeoutError",
    }:
        return "timeout"
    if name == "UtcpVariableNotFound" or "variable not found" in str(exc).lower():
        return "auth"
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        if status in (401, 403):
            return "auth"
        if status == 404:
            return "not_found"
        return "provider_error"
    if name in {"ClientConnectorError", "ServerDisconnectedError", "ClientConnectionError"}:
        return "provider_unavailable"
    if name in {"ClientResponseError", "ClientError"}:
        return "provider_error"
    if name in {"ClientPayloadError"}:
        return "malformed_response"
    if isinstance(exc, UtcpProviderUnavailable):
        return "provider_unavailable"
    if isinstance(exc, UtcpDiscoveryError):
        return "discovery_failed"
    if isinstance(exc, ValueError) and "tool not found" in str(exc).lower():
        return "not_found"
    return "protocol_error"


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


@dataclass
class _ProviderState:
    config: UtcpProviderConfig
    client: Any = None
    specs: list[UtcpToolSpec] = field(default_factory=list)
    registered: list[str] = field(default_factory=list)
    semaphore: Any = None
    connected: bool = False
    error: Optional[str] = None
    last_discovery: Optional[float] = None
    discovery_ms: Optional[float] = None
    consecutive_failures: int = 0
    breaker_opened_at: float = 0.0


_lock = threading.RLock()
_discovery_lock = threading.Lock()
_states: dict[str, _ProviderState] = {}
_config_problems: list[ConfigProblem] = []
_settings: UtcpSettings = UtcpSettings()
_discovery_started = False
_discovery_thread: Optional[threading.Thread] = None

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Start (once) the background event loop that owns every UTCP client."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run, name="hermes-utcp-loop", daemon=True)
        thread.start()
        _loop = loop
        _loop_thread = thread
        return loop


def _run(coro: Any, *, timeout: Optional[float] = None) -> Any:
    """Run *coro* on the UTCP loop and block the caller for its result."""
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise UtcpTimeoutError("UTCP operation timed out") from exc


def _import_utcp() -> None:
    """Import the optional SDK, preparing its extra on first use.

    Follows the ``pm.extras`` contract every opt-in backend uses: when the
    anchor modules are missing, ``ensure_import("utcp")`` syncs the venv with
    the extra enabled (interactively, only to a TTY). A failed prep becomes an
    actionable message instead of a bare ImportError.
    """
    try:
        import utcp
        import utcp_http

        return
    except ImportError:
        pass

    try:
        from pm.extras import ensure_import

        ensure_import("utcp")
    except Exception as exc:
        hint = "hermes pm install --extra utcp"
        try:
            from pm.extras import install_hint

            hint = install_hint("utcp")
        except Exception:
            pass
        raise UtcpDependencyError(
            "UTCP support requires the optional 'utcp' extra. Install it with:\n"
            f"  {hint}\n({exc})"
        ) from exc

    try:
        import utcp
        import utcp_http
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise UtcpDependencyError(
            "UTCP support requires the optional 'utcp' extra. Install it with:\n"
            "  hermes pm install --extra utcp"
        ) from exc


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def _create_client(provider: UtcpProviderConfig) -> tuple[Any, Any]:
    from utcp.data.call_template import CallTemplateSerializer
    from utcp.utcp_client import UtcpClient

    _import_utcp()
    # Validate the template with the SDK's own model so unsupported fields are
    # rejected up front with the SDK's message instead of failing at call time.
    template = CallTemplateSerializer().validate_dict(dict(provider.template))

    client = await UtcpClient.create(
        root_dir=os.getcwd(),
        config={"variables": provider.sdk_variables()},
    )
    return client, template


async def _discover_provider(
    state: _ProviderState,
) -> tuple[Any, list[UtcpToolSpec], list[str]]:
    """Create a client for one provider and fetch its manual."""
    provider = state.config
    client = None
    try:
        client, template = await _create_client(provider)
        result = await client.register_manual(template)
        if not getattr(result, "success", False):
            errors = getattr(result, "errors", None) or ["unknown discovery error"]
            raise UtcpDiscoveryError(redact_secrets(str(errors[0]), provider))
        manual = getattr(result, "manual", None)
        tools = list(getattr(manual, "tools", None) or [])
        specs, rejected = _tool_specs_from_manual(provider, tools)
        for reason in rejected:
            logger.warning("utcp '%s': skipping tool — %s", provider.name, reason)
        return client, specs, rejected
    except BaseException:
        if client is not None:
            try:
                await client.close()
            except Exception:
                logger.debug("utcp '%s': client close after failed discovery failed", provider.name, exc_info=True)
        raise


def _register_state_tools(state: _ProviderState) -> list[str]:
    """Register (or re-register) a provider's discovered tools."""
    from tools.registry import registry

    registered: list[str] = []
    for spec in state.specs:
        existing = registry.get_toolset_for_tool(spec.registry_name)
        if existing and existing != state.config.toolset:
            logger.warning(
                "utcp '%s': tool '%s' (%s) collides with an existing tool in toolset "
                "'%s'; keeping the existing tool",
                state.config.name,
                spec.raw_name,
                spec.registry_name,
                existing,
            )
            continue
        registry.register(
            name=spec.registry_name,
            toolset=state.config.toolset,
            schema={
                "name": spec.registry_name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
            handler=_make_tool_handler(state, spec),
            check_fn=_make_check_fn(state.config.name),
            is_async=False,
            description=spec.description,
            emoji="🔌",
            max_result_size_chars=state.config.max_result_chars,
        )
        if registry.get_toolset_for_tool(spec.registry_name) != state.config.toolset:
            logger.error(
                "utcp '%s': registration of '%s' was rejected by the registry",
                state.config.name, spec.registry_name,
            )
            continue
        registered.append(spec.registry_name)

    if registered:
        # Alias so `utcp-<provider>` and the bare provider name both resolve as
        # toolsets (`--toolsets utcp-weather` / `--toolsets weather`), matching
        # the MCP server-alias convention.
        registry.register_toolset_alias(state.config.name, state.config.toolset)
    return registered


def _deregister_state_tools(state: _ProviderState) -> None:
    from tools.registry import registry

    for name in list(state.registered):
        try:
            registry.deregister(name)
        except Exception:
            logger.debug("utcp '%s': deregister %s failed", state.config.name, name, exc_info=True)
    state.registered = []


def _publish_config_problems(problems: Sequence[ConfigProblem]) -> None:
    global _config_problems
    with _lock:
        _config_problems = list(problems)


def _discover_one(provider: UtcpProviderConfig) -> _ProviderState:
    """Discover and register a single provider.  Never raises."""
    state = _ProviderState(config=provider)
    started = time.monotonic()
    try:
        client, specs, rejected = _run(
            _discover_provider(state),
            timeout=provider.connect_timeout,
        )
        state.client = client
        state.specs = specs
        state.semaphore = asyncio.Semaphore(max(1, int(provider.max_concurrent_calls)))
        state.connected = True
        state.error = None
        state.last_discovery = time.time()
        state.discovery_ms = (time.monotonic() - started) * 1000.0
        if not specs:
            note = "no tools discovered"
            if rejected:
                note += f" ({len(rejected)} rejected; see logs)"
            state.error = note
        logger.info(
            "utcp '%s': %d tool(s) discovered in %.0f ms",
            provider.name, len(specs), state.discovery_ms or 0.0,
        )
    except Exception as exc:
        state.connected = False
        state.error = _safe_error_text(exc, provider)
        state.discovery_ms = (time.monotonic() - started) * 1000.0
        logger.warning("utcp '%s': discovery failed: %s", provider.name, state.error)
        with _lock:
            _states[provider.name] = state
        return state

    with _lock:
        _states[provider.name] = state
    try:
        state.registered = _register_state_tools(state)
    except Exception as exc:  # never let one provider's registry issue escape
        state.error = _safe_error_text(exc, provider)
        logger.warning("utcp '%s': tool registration failed: %s", provider.name, state.error)
    if state.specs and not state.registered and not state.error:
        # Everything was filtered out or collided with an existing tool; say so
        # rather than looking healthy with zero tools.
        state.error = "all discovered tools were rejected (see logs)"
    return state


def discover_utcp_tools(
    *, force: bool = False, allowed_names: Optional[Sequence[str]] = None
) -> list[str]:
    """Discover tools for every enabled UTCP provider.

    Idempotent: providers already connected are skipped unless *force* is set.
    Provider failures are isolated — the remaining providers still register.

    ``allowed_names`` restricts discovery to the named providers (a
    ``-t/--toolsets`` allowlist, mirroring the MCP server filter). ``None``
    discovers every configured, enabled provider.
    """
    global _discovery_started
    with _discovery_lock:
        settings, providers, problems = load_utcp_config()
        _publish_config_problems(problems)
        for problem in problems:
            logger.warning("utcp: %s", problem.message)
        if not settings.enabled:
            logger.debug("utcp: disabled in config; skipping discovery")
            return []
        _set_settings(settings)

        with _lock:
            already = {
                name for name, state in _states.items()
                if state.connected and state.client is not None
            }
        allowed = {name.strip() for name in allowed_names or () if str(name).strip()}
        allowed = {name.removeprefix("utcp-") for name in allowed}

        def _selected(provider: UtcpProviderConfig) -> bool:
            return not allowed or provider.name in allowed

        enabled = [p for p in providers if p.enabled and _selected(p)]
        skipped = [p for p in providers if not p.enabled or not _selected(p)]
        for provider in skipped:
            logger.info(
                "utcp '%s': skipped (disabled or filtered by --toolsets)", provider.name
            )

        registered: list[str] = []
        for provider in enabled:
            if provider.name in already and not force:
                # Already discovered: report its tools without re-fetching the
                # manual (discovery belongs to lifecycle boundaries, not to
                # every tool-definition refresh).
                with _lock:
                    registered.extend(_states[provider.name].registered)
                continue
            if provider.name in already:
                with _lock:
                    previous = _states.pop(provider.name, None)
                if previous is not None:
                    _deregister_state_tools(previous)
                    if previous.client is not None:
                        try:
                            _run(previous.client.close(), timeout=10)
                        except Exception:
                            logger.debug(
                                "utcp '%s': closing client before re-discovery failed",
                                provider.name, exc_info=True,
                            )
            state = _discover_one(provider)
            registered.extend(state.registered)
        _discovery_started = True
        return sorted(set(registered))


def _set_settings(settings: UtcpSettings) -> None:
    global _settings
    with _lock:
        _settings = settings


def refresh_utcp_provider(name: Optional[str] = None) -> dict[str, list[str]]:
    """Re-discover one provider (or all) and re-register its tools.

    Returns ``{provider: [registry tool names]}``.
    """
    settings, providers, problems = load_utcp_config()
    _publish_config_problems(problems)
    if not settings.enabled:
        raise UtcpConfigError("UTCP support is disabled (utcp.enabled=false)")
    by_name = {provider.name: provider for provider in providers}
    targets = [name] if name else list(by_name)
    if name and name not in by_name:
        raise UtcpConfigError(f"unknown utcp provider '{name}'")
    _set_settings(settings)

    result: dict[str, list[str]] = {}
    for provider_name in targets:
        provider = by_name[provider_name]
        if not provider.enabled:
            result[provider_name] = []
            continue
        with _lock:
            previous = _states.get(provider_name)
        if previous is not None:
            _deregister_state_tools(previous)
            if previous.client is not None:
                try:
                    _run(previous.client.close(), timeout=10)
                except Exception:
                    logger.debug("utcp '%s': closing old client failed", provider_name, exc_info=True)
        state = _discover_one(provider)
        result[provider_name] = list(state.registered)
    return result


def shutdown_utcp_providers() -> None:
    """Close every provider client and stop the background loop."""
    global _loop, _loop_thread, _discovery_started

    with _lock:
        states = list(_states.values())
        _states.clear()
    for state in states:
        _deregister_state_tools(state)
        if state.client is not None:
            try:
                _run(state.client.close(), timeout=10)
            except Exception:
                logger.debug("utcp '%s': close failed during shutdown", state.config.name, exc_info=True)
            state.client = None
        state.connected = False

    with _loop_lock:
        loop = _loop
        thread = _loop_thread
        _loop = None
        _loop_thread = None
    if loop is not None:
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
    if thread is not None:
        thread.join(timeout=5)
    if loop is not None:
        try:
            loop.close()
        except Exception:
            pass
    _discovery_started = False


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _breaker_open(state: _ProviderState) -> Optional[str]:
    if state.consecutive_failures < FAILURE_THRESHOLD:
        return None
    age = time.monotonic() - state.breaker_opened_at
    if age >= FAILURE_COOLDOWN_SECONDS:
        return None  # half-open: allow one probe
    remaining = max(1, int(FAILURE_COOLDOWN_SECONDS - age))
    return (
        f"UTCP provider '{state.config.name}' is failing repeatedly "
        f"({state.consecutive_failures} consecutive errors). Retry in ~{remaining}s or "
        "use another tool; do not call this tool again immediately."
    )


def _note_success(state: _ProviderState) -> None:
    state.consecutive_failures = 0
    state.breaker_opened_at = 0.0


def _note_failure(state: _ProviderState) -> None:
    state.consecutive_failures += 1
    if state.consecutive_failures == FAILURE_THRESHOLD:
        state.breaker_opened_at = time.monotonic()


async def _invoke(state: _ProviderState, spec: UtcpToolSpec, args: dict[str, Any]) -> Any:
    semaphore = state.semaphore or asyncio.Semaphore(max(1, int(state.config.max_concurrent_calls)))
    async with semaphore:
        return await asyncio.wait_for(
            state.client.call_tool(spec.utcp_name, args),
            timeout=state.config.timeout,
        )


def _make_tool_handler(state: _ProviderState, spec: UtcpToolSpec) -> Callable[..., str]:
    """Build the sync registry handler for one UTCP tool."""

    def _handler(args: dict, **kwargs) -> str:
        provider = state.config
        if not isinstance(args, dict):
            return _tool_error("arguments must be a JSON object", "invalid_arguments", state, spec)
        if state.client is None or not state.connected:
            return _tool_error(
                f"UTCP provider '{provider.name}' is not connected"
                + (f": {state.error}" if state.error else ""),
                "provider_unavailable", state, spec,
            )
        breaker = _breaker_open(state)
        if breaker:
            return _tool_error(breaker, "provider_unavailable", state, spec)

        try:
            encoded = json.dumps(args, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return _tool_error("arguments are not JSON-serializable", "invalid_arguments", state, spec)
        if len(encoded.encode("utf-8")) > _settings.max_arg_bytes:
            return _tool_error(
                f"arguments exceed the {_settings.max_arg_bytes}-byte limit",
                "invalid_arguments", state, spec,
            )

        problems = validate_arguments(spec.parameters, args)
        if problems:
            return _tool_error(
                f"invalid arguments for '{spec.raw_name}': " + "; ".join(problems),
                "invalid_arguments", state, spec,
            )

        try:
            result = _run(_invoke(state, spec, args), timeout=provider.timeout + 5.0)
        except UtcpTimeoutError:
            _note_failure(state)
            return _tool_error(
                f"UTCP tool '{spec.raw_name}' timed out after {provider.timeout:.0f}s",
                "timeout", state, spec,
            )
        except Exception as exc:
            _note_failure(state)
            kind = classify_error(exc)
            detail = _safe_error_text(exc, provider)
            return _tool_error(
                f"UTCP tool '{spec.raw_name}' failed ({kind}): {detail}",
                kind, state, spec,
            )

        _note_success(state)
        limit = int(provider.max_result_chars)
        text = _dump_result({"result": serialize_result(result)})
        if len(text) > limit:
            # Keep the envelope valid JSON: truncate the payload, not the
            # document, and say so explicitly so the model knows it is partial.
            truncated = _dump_result(
                {
                    "result": text[: max(0, limit - 120)],
                    "truncated": True,
                    "result_chars": len(text),
                }
            )
            return truncated[:limit] if len(truncated) > limit else truncated
        return text

    _handler.__name__ = f"utcp_{spec.registry_name}"
    _handler.__doc__ = spec.description
    return _handler


def _tool_error(
    message: str,
    error_type: str,
    state: _ProviderState,
    spec: UtcpToolSpec,
) -> str:
    payload = {
        "error": redact_secrets(message, state.config),
        "error_type": error_type,
        "provider": state.config.name,
        "tool": spec.raw_name,
    }
    return json.dumps(payload, ensure_ascii=False)


def _make_check_fn(provider_name: str) -> Callable[[], bool]:
    def _check() -> bool:
        with _lock:
            state = _states.get(provider_name)
        return bool(state is not None and state.connected and state.client is not None)

    return _check


# ---------------------------------------------------------------------------
# Introspection (CLI / dashboard)
# ---------------------------------------------------------------------------


def get_utcp_status() -> list[dict[str, Any]]:
    """Return provider-level status; never includes credentials."""
    with _lock:
        states = {name: state for name, state in _states.items()}
        problems = list(_config_problems)
    rows: list[dict[str, Any]] = []
    for name, state in sorted(states.items()):
        config = state.config
        rows.append(
            {
                "name": name,
                "transport": config.call_template_type,
                "url": config.template.get("url"),
                "enabled": config.enabled,
                "connected": state.connected,
                "tools": len(state.specs),
                "registered": len(state.registered),
                "error": state.error,
                "last_discovery": state.last_discovery,
                "discovery_ms": state.discovery_ms,
                "variables": list(config.variable_names),
            }
        )
    for problem in problems:
        rows.append(
            {
                "name": problem.name,
                "transport": None,
                "url": None,
                "enabled": False,
                "connected": False,
                "tools": 0,
                "registered": 0,
                "error": redact_secrets(problem.message),
                "last_discovery": None,
                "discovery_ms": None,
                "variables": [],
            }
        )
    return rows


def get_utcp_tools(provider: Optional[str] = None) -> list[dict[str, Any]]:
    """Return the discovered tool inventory (registry names + identities)."""
    with _lock:
        states = list(_states.values())
    rows: list[dict[str, Any]] = []
    for state in states:
        if provider and state.config.name != provider:
            continue
        for spec in state.specs:
            rows.append(
                {
                    "provider": spec.provider,
                    "tool": spec.raw_name,
                    "utcp_name": spec.utcp_name,
                    "registry_name": spec.registry_name,
                    "description": spec.description,
                    "tags": list(spec.tags),
                }
            )
    return sorted(rows, key=lambda row: row["registry_name"])


def utcp_available() -> bool:
    """True when the optional SDK can be imported."""
    try:
        _import_utcp()
        return True
    except Exception:
        return False


def reset_utcp_state_for_tests() -> None:
    """Drop module state without touching the registry (unit-test helper)."""
    global _discovery_started, _config_problems
    with _lock:
        _states.clear()
        _config_problems = []
        _discovery_started = False
