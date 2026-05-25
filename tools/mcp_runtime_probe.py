"""Redacted runtime-depth probing for Hermes/Codex/Claude MCP servers."""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
import argparse
import asyncio
import importlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlparse

try:
    yaml: Any | None = importlib.import_module("yaml")
except ImportError:  # pragma: no cover - yaml is optional; _load_yaml handles absence
    yaml = None


STATUS_CLASSES = (
    "ok",
    "disabled",
    "auth_needed",
    "offline",
    "unsupported",
    "method_unsupported",
    "protocol_error",
    "error",
)

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _parse_boolish(value: Any, *, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _env_refs(value: Any, *, include_mapping_keys: bool = False) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        for match in _ENV_REF_RE.finditer(value):
            refs.add(match.group(1) or match.group(2))
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                include_mapping_keys
                and isinstance(key, str)
                and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key)
            ):
                refs.add(key)
            refs.update(_env_refs(nested, include_mapping_keys=include_mapping_keys))
    elif isinstance(value, list):
        for nested in value:
            refs.update(_env_refs(nested, include_mapping_keys=include_mapping_keys))
    return refs


def _bearer_env_from_headers(headers: Mapping[str, Any]) -> str | None:
    for key, value in headers.items():
        if str(key).lower() != "authorization" or not isinstance(value, str):
            continue
        if not value.lower().strip().startswith("bearer "):
            continue
        refs = sorted(_env_refs(value))
        return refs[0] if refs else None
    return None


def _url_summary(url: str | None) -> dict[str, str] | None:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.netloc:
        return None
    return {"host": parsed.netloc, "path": parsed.path or "/"}


def _command_summary(command: str | None, args: tuple[str, ...]) -> dict[str, Any] | None:
    if not command:
        return None
    return {"basename": os.path.basename(command), "args_count": len(args)}


@dataclass(frozen=True)
class NormalizedMCPServer:
    source: str
    name: str
    config: Mapping[str, Any]
    transport: str
    enabled: bool = True
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env_vars: tuple[str, ...] = ()
    bearer_env_var: str | None = None

    @property
    def id(self) -> str:
        return f"{self.source}:{self.name}"

    def redacted_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "name": self.name,
            "transport": self.transport,
            "enabled": self.enabled,
        }
        url = _url_summary(self.url)
        if url:
            summary["url"] = url
        command = _command_summary(self.command, self.args)
        if command:
            summary["command"] = command
        env_vars = sorted(set(self.env_vars) | ({self.bearer_env_var} if self.bearer_env_var else set()))
        if env_vars:
            summary["env_vars"] = env_vars
        return summary


@dataclass
class _ProbeSessionAdapter:
    task: Any
    _initialized: bool = False

    async def initialize(self) -> Any:
        self._initialized = True
        return getattr(self.task, "initialize_result", None)

    async def list_tools(self) -> Any:
        tools = getattr(self.task, "_tools", None)
        if tools is not None:
            return type("ListToolsResult", (), {"tools": tools})()
        session = getattr(self.task, "session", None)
        if session is None:
            return type("ListToolsResult", (), {"tools": []})()
        return await session.list_tools()

    async def list_resources(self) -> Any:
        session = getattr(self.task, "session", None)
        if session is None:
            raise RuntimeError("MCP session is not connected")
        return await session.list_resources()

    async def list_prompts(self) -> Any:
        session = getattr(self.task, "session", None)
        if session is None:
            raise RuntimeError("MCP session is not connected")
        return await session.list_prompts()

    async def shutdown(self) -> None:
        shutdown = getattr(self.task, "shutdown", None)
        if shutdown is None:
            return
        result = shutdown()
        if hasattr(result, "__await__"):
            await result


Connector = Callable[[NormalizedMCPServer], Awaitable[Any] | Any]


def normalize_mcp_servers(raw_configs: Mapping[str, Any]) -> list[NormalizedMCPServer]:
    """Normalize Hermes/Codex/Claude MCP config shapes into one redacted model."""

    normalized: list[NormalizedMCPServer] = []
    for source, raw_config in raw_configs.items():
        if not isinstance(raw_config, Mapping):
            continue
        servers = (
            raw_config.get("mcp_servers")
            or raw_config.get("mcpServers")
            or (raw_config.get("mcp") or {}).get("servers")
        )
        if not isinstance(servers, Mapping):
            continue
        for name, cfg in servers.items():
            if not isinstance(cfg, Mapping):
                continue
            args = tuple(str(arg) for arg in (cfg.get("args") or []))
            url = str(cfg.get("url") or "").strip() or None
            command = str(cfg.get("command") or "").strip() or None
            transport = str(cfg.get("transport") or "").strip().lower()
            if not transport:
                transport = "http" if url else "stdio" if command else "unsupported"
            if transport in {"streamable_http", "streamable-http", "sse"}:
                transport = "http"
            bearer_env_var = (
                cfg.get("bearer_token_env_var")
                or cfg.get("bearerTokenEnvVar")
                or cfg.get("token_env_var")
                or cfg.get("tokenEnvVar")
            )
            headers = cfg.get("headers") if isinstance(cfg.get("headers"), Mapping) else {}
            bearer_env_var = str(bearer_env_var or _bearer_env_from_headers(headers) or "").strip() or None
            env_vars = set()
            env_vars.update(_env_refs(cfg.get("env") or {}, include_mapping_keys=True))
            env_vars.update(_env_refs(headers))
            if bearer_env_var:
                env_vars.add(bearer_env_var)
            normalized.append(
                NormalizedMCPServer(
                    source=str(source),
                    name=str(name),
                    config=cfg,
                    transport=transport,
                    enabled=_parse_boolish(cfg.get("enabled", True), default=True),
                    url=url,
                    command=command,
                    args=args,
                    env_vars=tuple(sorted(env_vars)),
                    bearer_env_var=bearer_env_var,
                )
            )
    return normalized


def _is_google_drive(server: NormalizedMCPServer) -> bool:
    haystack = " ".join(
        part.lower()
        for part in (
            server.source,
            server.name,
            server.url or "",
            server.command or "",
            " ".join(server.args),
        )
    )
    return "google" in haystack and "drive" in haystack


def _stdio_command_available(server: NormalizedMCPServer) -> bool:
    if not server.command:
        return False
    command = os.path.expanduser(server.command)
    if os.sep in command:
        return os.path.isfile(command) and os.access(command, os.X_OK)
    return shutil.which(command) is not None


def _sdk_available() -> bool:
    try:
        from tools import mcp_tool

        return bool(getattr(mcp_tool, "_MCP_AVAILABLE", False))
    except Exception:
        return False


async def _default_connector(server: NormalizedMCPServer) -> _ProbeSessionAdapter:
    from tools.mcp_tool import _connect_server

    config = dict(server.config)
    if server.bearer_env_var and os.getenv(server.bearer_env_var):
        headers = dict(config.get("headers") or {})
        headers.setdefault("Authorization", f"Bearer {os.environ[server.bearer_env_var]}")
        config["headers"] = headers
    task = await _connect_server(server.name, config)
    return _ProbeSessionAdapter(task)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _shutdown_session(session: Any) -> None:
    for name in ("shutdown", "close", "aclose"):
        closer = getattr(session, name, None)
        if closer is None:
            continue
        await _maybe_await(closer())
        return


def _classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(term in text for term in ("401", "403", "unauthorized", "forbidden", "auth", "token")):
        return "auth_needed"
    if any(term in text for term in ("method not found", "-32601", "not implemented")):
        return "method_unsupported"
    if any(term in text for term in ("protocol", "json-rpc", "jsonrpc", "parse error", "invalid request")):
        return "protocol_error"
    if any(
        term in text
        for term in (
            "connection",
            "refused",
            "timed out",
            "timeout",
            "name resolution",
            "not found",
            "no such file",
            "network",
            "offline",
        )
    ):
        return "offline"
    return "error"


def _count_items(result: Any, attr: str) -> int:
    items = getattr(result, attr, None)
    if items is None:
        return 0
    try:
        return len(items)
    except TypeError:
        return 0


def _capability_unsupported(initialize_result: Any, name: str) -> bool:
    capabilities = getattr(initialize_result, "capabilities", None)
    return capabilities is not None and getattr(capabilities, name, None) is None


async def _probe_one(server: NormalizedMCPServer, connector: Connector) -> dict[str, Any]:
    entry = server.redacted_summary()
    entry["checks"] = {}
    entry["tools_count"] = 0
    entry["resources_count"] = 0
    entry["prompts_count"] = 0

    if not server.enabled:
        entry["status"] = "disabled"
        return entry
    if server.transport not in {"http", "stdio"}:
        entry["status"] = "unsupported"
        return entry
    if server.transport == "stdio" and not _stdio_command_available(server):
        entry["status"] = "offline"
        entry["checks"]["executable"] = "offline"
        return entry
    if server.bearer_env_var and not os.getenv(server.bearer_env_var):
        entry["status"] = "auth_needed"
        entry["checks"]["auth"] = "auth_needed"
        return entry

    session = None
    try:
        session = await _maybe_await(connector(server))
        initialize_result = await session.initialize()
        entry["checks"]["initialize"] = "ok"

        try:
            tools_result = await session.list_tools()
        except Exception as exc:
            status = _classify_exception(exc)
            entry["checks"]["tools"] = status
            entry["status"] = status
            return entry
        entry["checks"]["tools"] = "ok"
        entry["tools_count"] = _count_items(tools_result, "tools")

        if _capability_unsupported(initialize_result, "resources"):
            entry["checks"]["resources"] = "unsupported"
        else:
            try:
                resources_result = await session.list_resources()
                entry["checks"]["resources"] = "ok"
                entry["resources_count"] = _count_items(resources_result, "resources")
            except Exception as exc:
                status = _classify_exception(exc)
                entry["checks"]["resources"] = status
                if status != "method_unsupported":
                    entry["status"] = status
                    return entry
        if _capability_unsupported(initialize_result, "prompts"):
            entry["checks"]["prompts"] = "unsupported"
        else:
            try:
                prompts_result = await session.list_prompts()
                entry["checks"]["prompts"] = "ok"
                entry["prompts_count"] = _count_items(prompts_result, "prompts")
            except Exception as exc:
                status = _classify_exception(exc)
                entry["checks"]["prompts"] = status
                if status != "method_unsupported":
                    entry["status"] = status
                    return entry
        entry["status"] = "ok"
        return entry
    except Exception as exc:
        status = _classify_exception(exc)
        if not entry["checks"]:
            entry["checks"]["initialize"] = status
        entry["status"] = status
        return entry
    finally:
        if session is not None:
            await _shutdown_session(session)


def _add_status_samples(report: dict[str, Any]) -> None:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in report["servers"]:
        status = entry["status"]
        if len(samples[status]) >= 3:
            continue
        sample = {
            key: entry[key]
            for key in ("id", "source", "name", "transport", "url", "command", "checks")
            if key in entry
        }
        samples[status].append(sample)
    report["status_samples"] = dict(samples)


def _build_report(servers: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(entry["status"] for entry in servers)
    check_counts = Counter(
        status
        for entry in servers
        for status in (entry.get("checks") or {}).values()
    )
    report = {
        "status_classes": list(STATUS_CLASSES),
        "status_counts": dict(sorted(status_counts.items())),
        "check_status_counts": dict(sorted(check_counts.items())),
        "servers": servers,
    }
    _add_status_samples(report)
    return report


async def probe_servers(
    servers: list[NormalizedMCPServer],
    *,
    connector: Connector | None = None,
    sdk_available: bool | None = None,
    skip_google_drive_auth_needed: bool = False,
) -> dict[str, Any]:
    if connector is None:
        connector = _default_connector
    if sdk_available is None:
        sdk_available = True if connector is not _default_connector else _sdk_available()

    entries: list[dict[str, Any]] = []
    for server in servers:
        missing_auth = bool(server.bearer_env_var and not os.getenv(server.bearer_env_var))
        if skip_google_drive_auth_needed and missing_auth and _is_google_drive(server):
            continue
        if not sdk_available and server.enabled:
            entry = server.redacted_summary()
            entry["status"] = "unsupported"
            entry["checks"] = {"sdk": "unsupported"}
            entry["tools_count"] = 0
            entry["resources_count"] = 0
            entry["prompts_count"] = 0
            entries.append(entry)
            continue
        entries.append(await _probe_one(server, connector))
    return _build_report(entries)


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return {}


def _load_toml(path: Path) -> Mapping[str, Any] | None:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return None
    except Exception:
        return {}


def _load_yaml(path: Path) -> Mapping[str, Any] | None:
    if yaml is None:
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, Mapping) else {}
    except FileNotFoundError:
        return None
    except Exception:
        return {}


def load_default_config_sources(
    home: Path | None = None,
    *,
    runtime: str = "all",
) -> dict[str, Mapping[str, Any]]:
    home = home or Path.home()
    sources: dict[str, Mapping[str, Any]] = {}

    if runtime in {"all", "hermes"}:
        hermes_home = Path(os.getenv("HERMES_HOME", str(home / ".hermes")))
        hermes_config = _load_yaml(hermes_home / "config.yaml")
        if hermes_config is not None:
            sources["hermes"] = hermes_config

    if runtime in {"all", "codex"}:
        codex_home = Path(os.getenv("CODEX_HOME", str(home / ".codex")))
        codex_config = _load_toml(codex_home / "config.toml")
        if codex_config is not None:
            sources["codex"] = codex_config

    if runtime in {"all", "claude"}:
        claude_config = _load_json(home / ".claude" / "settings.json")
        if claude_config is not None:
            sources["claude"] = claude_config

    return sources


async def probe_default_sources(
    *,
    runtime: str = "all",
    server_name: str | None = None,
    skip_google_drive_auth_needed: bool = True,
) -> dict[str, Any]:
    sources = load_default_config_sources(runtime=runtime)
    if not sources:
        return _build_report([])
    servers = normalize_mcp_servers(sources)
    if server_name:
        servers = [server for server in servers if server.name == server_name or server.id == server_name]
    return await probe_servers(
        servers,
        skip_google_drive_auth_needed=skip_google_drive_auth_needed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["codex", "claude", "hermes", "all"], default="all")
    parser.add_argument("--server", help="probe only one server name or source:name id")
    parser.add_argument("--json", action="store_true", help="emit JSON; retained for command symmetry")
    parser.add_argument("--redact", action="store_true", help="all output is always redacted")
    parser.add_argument("--timeout", type=int, default=120, help="overall probe timeout in seconds")
    parser.add_argument(
        "--skip-auth-needed",
        action="store_true",
        help="skip entries known to need interactive auth, including Google Drive",
    )
    parser.add_argument(
        "--include-google-drive-auth-needed",
        action="store_true",
        help="include Google Drive MCP entries that are only missing auth",
    )
    args = parser.parse_args(argv)

    async def _run() -> dict[str, Any]:
        return await asyncio.wait_for(
            probe_default_sources(
                runtime=args.runtime,
                server_name=args.server,
                skip_google_drive_auth_needed=(
                    args.skip_auth_needed or not args.include_google_drive_auth_needed
                ),
            ),
            timeout=args.timeout if args.timeout > 0 else None,
        )

    try:
        report = asyncio.run(_run())
    except asyncio.TimeoutError:
        report = {
            "status": "timeout",
            "status_counts": {"timeout": 1},
            "check_status_counts": {"timeout": 1},
            "servers": [],
            "status_classes": ["timeout"],
            "status_samples": {},
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 124
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
