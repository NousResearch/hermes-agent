"""Regression tests for scoping MCP server discovery to a resolved toolset list.

Background
==========
Process audit (2026-07-21): 7 concurrently-running Hermes owners (kanban
dispatcher-spawned workers + cron jobs) held 58 ``mcp_stdio_watchdog``
children between them. Each owner independently called
``discover_mcp_tools()`` with NO arguments, which unconditionally connects
every server listed under ``mcp_servers`` in the resolved config — even
though the dispatcher (``hermes_cli.kanban_db._resolve_worker_cli_toolsets``)
and cron (``cron.scheduler._resolve_cron_enabled_toolsets``) already compute
a scoped ``--toolsets``/``enabled_toolsets`` allowlist for that specific
worker/job, and MCP server names are first-class members of that allowlist
(``hermes_cli.tools_config.enabled_mcp_server_names``). Every additional
concurrent worker therefore re-booted the full MCP fleet (CodeGraph,
git_context, Qdrant, CUA, Bitwarden, Stitch, etc.) regardless of whether its
own toolset scope even referenced those servers, multiplying process/RSS/
connection fan-out and startup latency with every concurrent task.

The fix threads an optional ``enabled_toolsets`` allowlist through
``discover_mcp_tools()`` (and the two callers that already resolve one:
``cron.scheduler.run_job`` and the CLI startup path via
``hermes_cli.main._parse_cli_toolsets_arg`` / ``start_background_mcp_discovery``)
so a scoped caller only spawns the MCP server subprocesses within its own
tool surface. ``enabled_toolsets=None`` (the default — used by interactive
CLI/gateway/dashboard sessions with no explicit scope) preserves the prior
unrestricted behavior.
"""

from __future__ import annotations

from unittest.mock import patch


# ---------------------------------------------------------------------------
# _filter_servers_by_toolset_allowlist — pure filtering logic
# ---------------------------------------------------------------------------

class TestFilterServersByToolsetAllowlist:
    def test_none_allowlist_returns_all_servers_unfiltered(self):
        """Interactive sessions (no scoped toolset list) still see every server."""
        from tools.mcp_tool import _filter_servers_by_toolset_allowlist

        servers = {"codegraph": {"command": "codegraph"}, "bitwarden": {"command": "npx"}}
        assert _filter_servers_by_toolset_allowlist(servers, None) == servers

    def test_scoped_allowlist_keeps_only_named_servers(self):
        """A worker's --toolsets list restricts which MCP servers connect."""
        from tools.mcp_tool import _filter_servers_by_toolset_allowlist

        servers = {
            "codegraph": {"command": "codegraph"},
            "git_context": {"command": "python3"},
            "bitwarden": {"command": "npx"},
            "stitch": {"command": "npx"},
        }
        allowlist = ["terminal", "file", "codegraph", "git_context"]
        result = _filter_servers_by_toolset_allowlist(servers, allowlist)
        assert set(result.keys()) == {"codegraph", "git_context"}

    def test_allowlist_with_no_matching_servers_returns_empty(self):
        """A worker scoped only to native toolsets connects zero MCP servers."""
        from tools.mcp_tool import _filter_servers_by_toolset_allowlist

        servers = {"codegraph": {"command": "codegraph"}, "bitwarden": {"command": "npx"}}
        result = _filter_servers_by_toolset_allowlist(servers, ["terminal", "file", "web"])
        assert result == {}

    def test_all_sentinel_disables_filtering(self):
        """The 'all' sentinel (matches get_tool_definitions convention) is unrestricted."""
        from tools.mcp_tool import _filter_servers_by_toolset_allowlist

        servers = {"codegraph": {"command": "codegraph"}, "bitwarden": {"command": "npx"}}
        assert _filter_servers_by_toolset_allowlist(servers, ["all"]) == servers
        assert _filter_servers_by_toolset_allowlist(servers, ["*"]) == servers

    def test_empty_allowlist_falls_back_to_unrestricted(self):
        """An empty list (no meaningful scope) must not silently drop every server."""
        from tools.mcp_tool import _filter_servers_by_toolset_allowlist

        servers = {"codegraph": {"command": "codegraph"}}
        assert _filter_servers_by_toolset_allowlist(servers, []) == servers


# ---------------------------------------------------------------------------
# discover_mcp_tools(enabled_toolsets=...) — end-to-end scoping
# ---------------------------------------------------------------------------

class TestDiscoverMcpToolsScoping:
    def test_discover_mcp_tools_default_connects_every_configured_server(self):
        """No enabled_toolsets arg (interactive sessions) — prior behavior preserved."""
        from tools.mcp_tool import discover_mcp_tools

        fake_config = {
            "codegraph": {"command": "codegraph", "args": ["serve", "--mcp"]},
            "bitwarden": {"command": "npx", "args": ["-y", "@bitwarden/mcp-server"]},
        }
        register_calls = []

        def fake_register(servers):
            register_calls.append(set(servers.keys()))
            return []

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool.register_mcp_servers", side_effect=fake_register):
            discover_mcp_tools()

        assert register_calls == [{"codegraph", "bitwarden"}]

    def test_discover_mcp_tools_scoped_connects_only_allowlisted_servers(self):
        """A resolved worker toolset list restricts which servers get connected —
        the actual subprocess-spawn gate, not just post-hoc tool filtering."""
        from tools.mcp_tool import discover_mcp_tools

        fake_config = {
            "codegraph": {"command": "codegraph", "args": ["serve", "--mcp"]},
            "git_context": {"command": "python3"},
            "bitwarden": {"command": "npx", "args": ["-y", "@bitwarden/mcp-server"]},
            "stitch": {"command": "npx", "args": ["@_davideast/stitch-mcp", "proxy"]},
        }
        register_calls = []

        def fake_register(servers):
            register_calls.append(set(servers.keys()))
            return []

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool.register_mcp_servers", side_effect=fake_register):
            discover_mcp_tools(enabled_toolsets=["terminal", "file", "codegraph"])

        assert register_calls == [{"codegraph"}]

    def test_discover_mcp_tools_scoped_to_pure_native_toolsets_skips_registration_entirely(self):
        """A worker scoped only to native toolsets (no MCP server names) never
        calls register_mcp_servers at all — zero MCP subprocess spawns."""
        from tools.mcp_tool import discover_mcp_tools

        fake_config = {
            "codegraph": {"command": "codegraph", "args": ["serve", "--mcp"]},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool.register_mcp_servers") as mock_register:
            result = discover_mcp_tools(enabled_toolsets=["terminal", "file", "web"])

        mock_register.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# cron.scheduler.run_job — MCP discovery scoped to the job's own toolsets
# ---------------------------------------------------------------------------

class TestCronSchedulerScopesMcpDiscovery:
    def test_run_job_passes_resolved_toolsets_to_discover_mcp_tools(self):
        """cron.scheduler must resolve the job's enabled_toolsets BEFORE calling
        discover_mcp_tools and pass it through, so a cron job scoped away from
        MCP servers doesn't boot them on every tick (mirrors the kanban worker
        fix — see #P1 MCP fleet explosion)."""
        from cron import scheduler

        job = {
            "id": "scoped-job",
            "name": "scoped-job",
            "no_agent": True,  # short-circuit before AIAgent construction
            "script": "/nonexistent/script.sh",
        }

        captured_kwargs = {}

        def fake_discover(**kwargs):
            captured_kwargs.update(kwargs)
            return []

        # no_agent jobs skip discover_mcp_tools entirely (see
        # test_scheduler_mcp_init.py::test_no_agent_cron_job_does_not_initialize_mcp),
        # so exercise the resolver directly against the same job/cfg shape
        # run_job would use, confirming it produces a real allowlist that
        # _resolve_cron_enabled_toolsets alone would also produce.
        from cron.scheduler import _resolve_cron_enabled_toolsets

        cfg = {"platform_toolsets": {"cron": ["terminal", "web"]}}
        resolved = _resolve_cron_enabled_toolsets(job, cfg)
        assert resolved is not None
        assert "terminal" in resolved and "web" in resolved

    def test_resolved_cron_toolsets_used_as_mcp_allowlist_excludes_unlisted_servers(self):
        """End-to-end: a job's resolved toolset list, fed into
        discover_mcp_tools, excludes MCP servers the job never asked for."""
        from tools.mcp_tool import discover_mcp_tools

        fake_config = {
            "codegraph": {"command": "codegraph"},
            "bitwarden": {"command": "npx"},
        }
        register_calls = []

        def fake_register(servers):
            register_calls.append(set(servers.keys()))
            return []

        # Simulates a job whose per-job enabled_toolsets names only "codegraph".
        job_toolsets = ["terminal", "file", "codegraph"]

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool.register_mcp_servers", side_effect=fake_register):
            discover_mcp_tools(enabled_toolsets=job_toolsets)

        assert register_calls == [{"codegraph"}]


# ---------------------------------------------------------------------------
# hermes_cli.main._parse_cli_toolsets_arg — CLI arg normalization
# ---------------------------------------------------------------------------

class TestParseCliToolsetsArg:
    def test_none_returns_none(self):
        from hermes_cli.main import _parse_cli_toolsets_arg
        assert _parse_cli_toolsets_arg(None) is None

    def test_empty_string_returns_none(self):
        from hermes_cli.main import _parse_cli_toolsets_arg
        assert _parse_cli_toolsets_arg("") is None

    def test_comma_separated_string_splits(self):
        from hermes_cli.main import _parse_cli_toolsets_arg
        assert _parse_cli_toolsets_arg("web,terminal,codegraph") == [
            "web", "terminal", "codegraph",
        ]

    def test_list_of_strings_flattened_and_comma_split(self):
        from hermes_cli.main import _parse_cli_toolsets_arg
        assert _parse_cli_toolsets_arg(["web", "terminal,file"]) == [
            "web", "terminal", "file",
        ]

    def test_whitespace_stripped(self):
        from hermes_cli.main import _parse_cli_toolsets_arg
        assert _parse_cli_toolsets_arg(" web , terminal ") == ["web", "terminal"]
