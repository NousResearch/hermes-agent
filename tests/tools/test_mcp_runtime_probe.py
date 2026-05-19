import asyncio
import json
from types import SimpleNamespace


class FakeSession:
    def __init__(
        self,
        *,
        tools=None,
        resources=None,
        prompts=None,
        resource_error=None,
        tool_error=None,
        prompt_error=None,
        capabilities=None,
    ):
        self._tools = tools or []
        self._resources = resources or []
        self._prompts = prompts or []
        self._resource_error = resource_error
        self._tool_error = tool_error
        self._prompt_error = prompt_error
        self._capabilities = capabilities

    async def initialize(self):
        return SimpleNamespace(capabilities=self._capabilities)

    async def list_tools(self):
        if self._tool_error is not None:
            raise self._tool_error
        return SimpleNamespace(tools=self._tools)

    async def list_resources(self):
        if self._resource_error is not None:
            raise self._resource_error
        return SimpleNamespace(resources=self._resources)

    async def list_prompts(self):
        if self._prompt_error is not None:
            raise self._prompt_error
        return SimpleNamespace(prompts=self._prompts)


def _run(coro):
    return asyncio.run(coro)


def test_normalizes_hermes_codex_and_claude_configs_without_secret_values():
    from tools.mcp_runtime_probe import normalize_mcp_servers

    servers = normalize_mcp_servers(
        {
            "hermes": {
                "mcp_servers": {
                    "github": {
                        "url": "https://api.githubcopilot.com/mcp/readonly?token=secret",
                        "headers": {
                            "Authorization": "Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}",
                        },
                    }
                }
            },
            "codex": {
                "mcp_servers": {
                    "cloudflare-api": {
                        "url": "https://mcp.cloudflare.com/mcp",
                        "bearer_token_env_var": "CLOUDFLARE_API_TOKEN",
                    }
                }
            },
            "claude": {
                "mcpServers": {
                    "filesystem": {
                        "command": "/usr/local/bin/npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "env": {"FS_TOKEN": "${FILESYSTEM_TOKEN}"},
                    }
                }
            },
        }
    )

    summaries = {server.id: server.redacted_summary() for server in servers}
    assert summaries["hermes:github"]["url"] == {
        "host": "api.githubcopilot.com",
        "path": "/mcp/readonly",
    }
    assert summaries["codex:cloudflare-api"]["env_vars"] == ["CLOUDFLARE_API_TOKEN"]
    assert summaries["claude:filesystem"]["command"] == {
        "basename": "npx",
        "args_count": 3,
    }
    assert summaries["claude:filesystem"]["env_vars"] == ["FILESYSTEM_TOKEN", "FS_TOKEN"]

    encoded = json.dumps([server.redacted_summary() for server in servers], sort_keys=True)
    assert "secret" not in encoded
    assert "Bearer" not in encoded
    assert "/usr/local/bin/npx" not in encoded


def test_missing_stdio_executable_reports_offline_and_does_not_connect(monkeypatch):
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    calls = []

    async def connector(server):
        calls.append(server.name)
        return FakeSession()

    server = NormalizedMCPServer(
        source="hermes",
        name="missing",
        config={"command": "definitely-not-a-real-mcp-command"},
        transport="stdio",
        command="definitely-not-a-real-mcp-command",
    )
    monkeypatch.setattr("tools.mcp_runtime_probe.shutil.which", lambda command: None)

    report = _run(probe_servers([server], connector=connector))

    assert report["servers"][0]["status"] == "offline"
    assert report["status_counts"]["offline"] == 1
    assert calls == []


def test_missing_bearer_env_var_reports_auth_needed_and_redacts_name_only(monkeypatch):
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    server = NormalizedMCPServer(
        source="codex",
        name="github-readonly",
        config={},
        transport="http",
        url="https://api.githubcopilot.com/mcp/readonly",
        bearer_env_var="GITHUB_PERSONAL_ACCESS_TOKEN",
    )
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)

    report = _run(probe_servers([server], connector=lambda _server: None))

    entry = report["servers"][0]
    assert entry["status"] == "auth_needed"
    assert entry["env_vars"] == ["GITHUB_PERSONAL_ACCESS_TOKEN"]
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in json.dumps(report)


def test_http_401_reports_auth_needed(monkeypatch):
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    async def connector(server):
        raise RuntimeError("401 Unauthorized")

    server = NormalizedMCPServer(
        source="codex",
        name="cloudflare-api",
        config={},
        transport="http",
        url="https://mcp.cloudflare.com/mcp",
        bearer_env_var="CLOUDFLARE_API_TOKEN",
    )
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "not-printed")

    report = _run(probe_servers([server], connector=connector))

    assert report["servers"][0]["status"] == "auth_needed"
    assert "not-printed" not in json.dumps(report)


def test_resources_method_not_found_is_method_unsupported_but_tools_are_counted():
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    async def connector(server):
        return FakeSession(
            tools=[SimpleNamespace(name="search")],
            resource_error=RuntimeError("-32601 Method not found"),
            capabilities=SimpleNamespace(resources=SimpleNamespace()),
        )

    server = NormalizedMCPServer(
        source="hermes",
        name="resources-missing",
        config={},
        transport="http",
        url="https://example.test/mcp",
    )

    report = _run(probe_servers([server], connector=connector))

    entry = report["servers"][0]
    assert entry["status"] == "ok"
    assert entry["checks"]["tools"] == "ok"
    assert entry["checks"]["resources"] == "method_unsupported"
    assert entry["tools_count"] == 1
    assert report["check_status_counts"]["method_unsupported"] == 1


def test_tools_failure_is_attributed_to_tools_check():
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    async def connector(server):
        return FakeSession(tool_error=RuntimeError("connection timed out"))

    server = NormalizedMCPServer(
        source="hermes",
        name="wedged",
        config={},
        transport="http",
        url="https://example.test/mcp",
    )

    report = _run(probe_servers([server], connector=connector))

    entry = report["servers"][0]
    assert entry["status"] == "offline"
    assert entry["checks"]["initialize"] == "ok"
    assert entry["checks"]["tools"] == "offline"


def test_prompts_are_probed_and_counted_when_advertised():
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    async def connector(server):
        return FakeSession(
            tools=[SimpleNamespace(name="search")],
            resources=[],
            prompts=[SimpleNamespace(name="summarize")],
            capabilities=SimpleNamespace(
                resources=SimpleNamespace(),
                prompts=SimpleNamespace(),
            ),
        )

    server = NormalizedMCPServer(
        source="hermes",
        name="prompts",
        config={},
        transport="http",
        url="https://example.test/mcp",
    )

    report = _run(probe_servers([server], connector=connector))

    entry = report["servers"][0]
    assert entry["status"] == "ok"
    assert entry["checks"]["prompts"] == "ok"
    assert entry["prompts_count"] == 1


def test_advertised_prompts_failure_degrades_status():
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    async def connector(server):
        return FakeSession(
            tools=[SimpleNamespace(name="search")],
            resources=[],
            prompt_error=RuntimeError("401 Unauthorized"),
            capabilities=SimpleNamespace(
                resources=SimpleNamespace(),
                prompts=SimpleNamespace(),
            ),
        )

    server = NormalizedMCPServer(
        source="hermes",
        name="prompts-auth",
        config={},
        transport="http",
        url="https://example.test/mcp",
    )

    report = _run(probe_servers([server], connector=connector))

    entry = report["servers"][0]
    assert entry["status"] == "auth_needed"
    assert entry["checks"]["prompts"] == "auth_needed"


def test_tools_only_server_skips_resources_probe_and_reports_ok():
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    async def connector(server):
        return FakeSession(
            tools=[SimpleNamespace(name="resolve-library-id")],
            capabilities=SimpleNamespace(resources=None),
        )

    server = NormalizedMCPServer(
        source="hermes",
        name="context7",
        config={},
        transport="http",
        url="https://context7.example/mcp",
    )

    report = _run(probe_servers([server], connector=connector))

    entry = report["servers"][0]
    assert entry["status"] == "ok"
    assert entry["checks"]["resources"] == "unsupported"
    assert entry["checks"]["prompts"] == "unsupported"
    assert entry["tools_count"] == 1


def test_sdk_unavailable_reports_unsupported_without_connecting():
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    calls = []

    async def connector(server):
        calls.append(server.name)
        return FakeSession()

    server = NormalizedMCPServer(
        source="hermes",
        name="github",
        config={},
        transport="http",
        url="https://api.githubcopilot.com/mcp/readonly",
    )

    report = _run(probe_servers([server], connector=connector, sdk_available=False))

    assert report["servers"][0]["status"] == "unsupported"
    assert calls == []


def test_cli_default_skips_google_drive_auth_needed_entries(monkeypatch):
    from tools.mcp_runtime_probe import NormalizedMCPServer, probe_servers

    server = NormalizedMCPServer(
        source="claude",
        name="Google Drive",
        config={},
        transport="http",
        url="https://mcp.google.com/drive",
        bearer_env_var="GOOGLE_DRIVE_TOKEN",
    )
    monkeypatch.delenv("GOOGLE_DRIVE_TOKEN", raising=False)

    report = _run(probe_servers([server], skip_google_drive_auth_needed=True))

    assert report["servers"] == []
    assert report["status_counts"] == {}


def test_default_source_loader_filters_runtime(monkeypatch, tmp_path):
    from tools.mcp_runtime_probe import load_default_config_sources

    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    hermes = tmp_path / ".hermes"
    codex = tmp_path / ".codex"
    claude = tmp_path / ".claude"
    hermes.mkdir()
    codex.mkdir()
    claude.mkdir()
    (hermes / "config.yaml").write_text("mcp_servers:\n  arxiv:\n    command: uvx\n", encoding="utf-8")
    (codex / "config.toml").write_text("[mcp_servers.context7]\ncommand = \"npx\"\n", encoding="utf-8")
    (claude / "settings.json").write_text("{\"mcpServers\":{\"filesystem\":{\"command\":\"npx\"}}}", encoding="utf-8")

    assert set(load_default_config_sources(tmp_path, runtime="all")) == {"hermes", "codex", "claude"}
    assert set(load_default_config_sources(tmp_path, runtime="codex")) == {"codex"}


def test_probe_default_sources_filters_server(monkeypatch, tmp_path):
    from tools import mcp_runtime_probe

    async def fake_probe_servers(servers, **kwargs):
        return {"servers": [server.id for server in servers], "kwargs": kwargs}

    monkeypatch.setattr(
        mcp_runtime_probe,
        "load_default_config_sources",
        lambda runtime="all": {
            "codex": {
                "mcp_servers": {
                    "context7": {"command": "npx"},
                    "arxiv": {"command": "uvx"},
                }
            }
        },
    )
    monkeypatch.setattr(mcp_runtime_probe, "probe_servers", fake_probe_servers)

    report = _run(mcp_runtime_probe.probe_default_sources(runtime="codex", server_name="context7"))

    assert report["servers"] == ["codex:context7"]
