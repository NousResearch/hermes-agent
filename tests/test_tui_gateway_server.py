import importlib
import json
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.continuation_engine import should_use_continuation_engine
from agent.intent_preclassifier import preclassify_intent
from tui_gateway import server


class _ChunkyStdout:
    def __init__(self):
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        for ch in text:
            self.parts.append(ch)
            time.sleep(0.0001)
        return len(text)

    def flush(self) -> None:
        return None


class _BrokenStdout:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def test_write_json_serializes_concurrent_writes(monkeypatch):
    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)

    threads = [
        threading.Thread(target=server.write_json, args=({"seq": i, "text": "x" * 24},))
        for i in range(8)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    lines = "".join(out.parts).splitlines()

    assert len(lines) == 8
    assert {json.loads(line)["seq"] for line in lines} == set(range(8))


def test_write_json_returns_false_on_broken_pipe(monkeypatch):
    monkeypatch.setattr(server, "_real_stdout", _BrokenStdout())

    assert server.write_json({"ok": True}) is False


def test_status_callback_emits_kind_and_text():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("context_pressure", "85% to compaction")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "context_pressure", "text": "85% to compaction"},
    )


def test_status_callback_accepts_single_message_argument():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("thinking...")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "status", "text": "thinking..."},
    )


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def test_config_set_yolo_toggles_session_scope():
    from tools.approval import clear_session, is_session_yolo_enabled

    server._sessions["sid"] = _session()
    try:
        resp_on = server.handle_request({"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "yolo"}})
        assert resp_on["result"]["value"] == "1"
        assert is_session_yolo_enabled("session-key") is True

        resp_off = server.handle_request({"id": "2", "method": "config.set", "params": {"session_id": "sid", "key": "yolo"}})
        assert resp_off["result"]["value"] == "0"
        assert is_session_yolo_enabled("session-key") is False
    finally:
        clear_session("session-key")
        server._sessions.clear()


def test_enable_gateway_prompts_sets_gateway_env(monkeypatch):
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    server._enable_gateway_prompts()

    assert server.os.environ["HERMES_GATEWAY_SESSION"] == "1"
    assert server.os.environ["HERMES_EXEC_ASK"] == "1"
    assert server.os.environ["HERMES_INTERACTIVE"] == "1"


def test_setup_status_reports_provider_config(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "claude", "provider": "anthropic"}})

    resp = server.handle_request({"id": "1", "method": "setup.status", "params": {}})

    assert resp["result"] == {
        "provider": "anthropic",
        "model": "claude",
        "provider_configured": False,
    }


def test_setup_status_delegates_to_setup_bridge(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server,
        "_build_setup_status_payload_impl",
        lambda **kwargs: seen.update(kwargs) or {"provider": "anthropic", "model": "claude", "provider_configured": True},
    )

    resp = server.handle_request({"id": "1", "method": "setup.status", "params": {}})

    assert resp["result"] == {"provider": "anthropic", "model": "claude", "provider_configured": True}
    assert seen["load_cfg"] is server._load_cfg
    assert seen["resolve_model"] is server._resolve_model


def test_setup_catalog_lists_native_bootstrap_providers(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "claude", "provider": "anthropic"}})
    monkeypatch.setattr(
        server,
        "_native_setup_catalog",
        lambda: [
            {
                "slug": "anthropic",
                "name": "Anthropic",
                "description": "Anthropic desc",
                "auth_type": "api_key",
                "configured": False,
                "key_env_var": "ANTHROPIC_API_KEY",
                "supports_native": True,
                "models": ["claude-sonnet-4-5"],
                "total_models": 1,
            }
        ],
    )

    resp = server.handle_request({"id": "1", "method": "setup.catalog", "params": {}})

    assert resp["result"] == {
        "provider_configured": False,
        "provider": "anthropic",
        "model": "claude",
        "providers": [
            {
                "slug": "anthropic",
                "name": "Anthropic",
                "description": "Anthropic desc",
                "auth_type": "api_key",
                "configured": False,
                "key_env_var": "ANTHROPIC_API_KEY",
                "supports_native": True,
                "models": ["claude-sonnet-4-5"],
                "total_models": 1,
            }
        ],
    }


def test_setup_catalog_delegates_to_setup_bridge(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server,
        "_build_setup_catalog_payload_impl",
        lambda **kwargs: seen.update(kwargs) or {"provider_configured": False, "provider": "", "model": "m", "providers": []},
    )

    resp = server.handle_request({"id": "1", "method": "setup.catalog", "params": {}})

    assert resp["result"] == {"provider_configured": False, "provider": "", "model": "m", "providers": []}
    assert seen["catalog_builder"] is server._native_setup_catalog


def test_setup_bootstrap_persists_provider_and_model(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        server,
        "_bootstrap_provider_config",
        lambda provider, *, api_key="", model="", base_url="": captured.update(
            {"provider": provider, "api_key": api_key, "model": model, "base_url": base_url}
        )
        or {"provider": provider, "model": model},
    )

    resp = server.handle_request(
        {
            "id": "1",
            "method": "setup.bootstrap",
            "params": {"provider": "anthropic", "api_key": "***", "model": "claude-sonnet-4-5"},
        }
    )

    assert resp["result"] == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "provider_configured": True,
    }
    assert captured == {"provider": "anthropic", "api_key": "***", "model": "claude-sonnet-4-5", "base_url": ""}


def test_setup_bootstrap_delegates_to_setup_bridge(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server,
        "_run_setup_bootstrap_impl",
        lambda **kwargs: seen.update(kwargs) or {"provider": "anthropic", "model": "claude", "provider_configured": True},
    )

    params = {"provider": "anthropic", "api_key": "sekret", "model": "claude"}
    resp = server.handle_request({"id": "1", "method": "setup.bootstrap", "params": params})

    assert resp["result"] == {"provider": "anthropic", "model": "claude", "provider_configured": True}
    assert seen["provider"] == "anthropic"
    assert seen["bootstrapper"] is server._bootstrap_provider_config


def test_native_setup_catalog_keeps_external_process_explicit_and_excludes_oauth_device_code(monkeypatch):
    fake_models = types.SimpleNamespace(
        CANONICAL_PROVIDERS=[
            types.SimpleNamespace(slug="anthropic", tui_desc="Anthropic desc", label="Anthropic"),
            types.SimpleNamespace(slug="copilot-acp", tui_desc="Copilot desc", label="Copilot ACP"),
            types.SimpleNamespace(slug="nous", tui_desc="Nous desc", label="Nous")
        ]
    )

    monkeypatch.setitem(sys.modules, "hermes_cli.models", fake_models)
    monkeypatch.setattr(
        server,
        "_setup_provider_status",
        lambda provider: {
            "anthropic": {"auth_type": "api_key", "configured": False, "key_env_var": "ANTHROPIC_API_KEY", "name": "Anthropic"},
            "copilot-acp": {"auth_type": "external_process", "configured": False, "key_env_var": "", "name": "Copilot ACP"},
            "nous": {"auth_type": "oauth_device_code", "configured": False, "key_env_var": "", "name": "Nous"},
        }[provider],
    )
    monkeypatch.setattr(server, "_setup_provider_models", lambda provider: ([f"{provider}/model"], ""))

    providers = server._native_setup_catalog()

    assert [p["slug"] for p in providers] == ["anthropic", "copilot-acp"]
    assert providers[0]["supports_native"] is True
    assert providers[1]["supports_native"] is False
    assert providers[1]["auth_type"] == "external_process"


def test_config_set_reasoning_updates_live_session_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)

    resp_effort = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "reasoning", "value": "low"}}
    )
    assert resp_effort["result"]["value"] == "low"
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}

    resp_show = server.handle_request(
        {"id": "2", "method": "config.set", "params": {"session_id": "sid", "key": "reasoning", "value": "show"}}
    )
    assert resp_show["result"]["value"] == "show"
    assert server._sessions["sid"]["show_reasoning"] is True


def test_config_set_verbose_updates_session_mode_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(verbose_logging=False)
    server._sessions["sid"] = _session(agent=agent)

    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "verbose", "value": "cycle"}}
    )

    assert resp["result"]["value"] == "verbose"
    assert server._sessions["sid"]["tool_progress_mode"] == "verbose"
    assert agent.verbose_logging is True


def test_config_set_model_uses_live_switch_path(monkeypatch):
    server._sessions["sid"] = _session()
    seen = {}

    def _fake_apply(sid, session, raw):
        seen["args"] = (sid, session["session_key"], raw)
        return {"value": "new/model", "warning": "catalog unreachable"}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)
    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "model", "value": "new/model"}}
    )

    assert resp["result"]["value"] == "new/model"
    assert resp["result"]["warning"] == "catalog unreachable"
    assert seen["args"] == ("sid", "session-key", "new/model")


def test_routing_owner_promotes_swarm_to_native_surface():
    assert server._routing_owner("swarm") == "native"
    assert server._routing_owner("handoff") == "slash-live"
    assert server._routing_owner("tools") == "native"


def test_config_set_model_global_persists(monkeypatch):
    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-old"

        def switch_model(self, **kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="anthropic/claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-new",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )
    seen = {}
    saved = {}

    def _switch_model(**kwargs):
        seen.update(kwargs)
        return result

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch_model)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.update(cfg))

    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "model", "value": "anthropic/claude-sonnet-4.6 --global"}}
    )

    assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
    assert seen["is_global"] is True
    assert saved["model"]["default"] == "anthropic/claude-sonnet-4.6"
    assert saved["model"]["provider"] == "anthropic"
    assert saved["model"]["base_url"] == "https://api.anthropic.com"


def test_tools_catalog_reports_provider_and_mcp_state(monkeypatch):
    import hermes_cli.config as config_mod
    import hermes_cli.tools_config as tools_config_mod

    monkeypatch.setattr(tools_config_mod, "CONFIGURABLE_TOOLSETS", [("web", "Web", "search"), ("plugin_x", "Plugin", "plugin")])
    monkeypatch.setattr(
        tools_config_mod,
        "TOOL_CATEGORIES",
        {"web": {"providers": [{"name": "Firecrawl", "web_backend": "firecrawl", "env_vars": [{"key": "FIRECRAWL_API_KEY"}]}]}},
    )
    monkeypatch.setattr(tools_config_mod, "_get_effective_configurable_toolsets", lambda: [("web", "Web", "search"), ("plugin_x", "Plugin", "plugin")])
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda cfg, platform, include_default_mcp_servers=False: {"web", "plugin_x"})
    monkeypatch.setattr(config_mod, "load_config", lambda: {"web": {"backend": "firecrawl"}, "mcp_servers": {"github": {"enabled": True, "tools": {"exclude": ["create_issue"]}}}})
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: "sekret" if key == "FIRECRAWL_API_KEY" else "")

    resp = server.handle_request({"id": "1", "method": "tools.catalog", "params": {}})

    assert resp["result"] == {
        "toolsets": [
            {
                "active_provider": "",
                "configurable": False,
                "description": "plugin",
                "enabled": True,
                "kind": "builtin",
                "name": "plugin_x",
                "providers": [],
                "title": "Plugin",
            },
            {
                "active_provider": "firecrawl",
                "configurable": True,
                "description": "search",
                "enabled": True,
                "kind": "builtin",
                "name": "web",
                "providers": [
                    {
                        "active": True,
                        "badge": "",
                        "env_vars": [
                            {"configured": True, "key": "FIRECRAWL_API_KEY", "prompt": "FIRECRAWL_API_KEY", "url": ""}
                        ],
                        "name": "Firecrawl",
                        "slug": "firecrawl",
                        "tag": "",
                    }
                ],
                "title": "Web",
            },
        ],
        "mcp_servers": [{"enabled": True, "exclude": ["create_issue"], "include": [], "name": "github"}],
    }


def test_tools_catalog_delegates_to_tools_bridge(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server,
        "_build_tools_catalog_payload_impl",
        lambda **kwargs: seen.update(kwargs) or {"toolsets": [], "mcp_servers": []},
    )

    resp = server.handle_request({"id": "1", "method": "tools.catalog", "params": {}})

    assert resp["result"] == {"toolsets": [], "mcp_servers": []}
    assert seen["load_config"]
    assert seen["effective_toolsets"]


def test_toolsets_list_derives_canonical_repo_surface_enabled_from_visible_members(monkeypatch):
    import model_tools
    import toolsets

    visible_defs = [
        {"function": {"name": "read_file", "description": "Read file contents."}},
        {"function": {"name": "search_files", "description": "Search files."}},
        {"function": {"name": "ast_list_defs", "description": "List AST defs."}},
        {"function": {"name": "ast_find_nodes", "description": "Find AST nodes."}},
        {"function": {"name": "lsp_document_symbols", "description": "List document symbols."}},
        {"function": {"name": "lsp_definition", "description": "Find definitions."}},
        {"function": {"name": "lsp_diagnostics", "description": "Show diagnostics."}},
    ]
    info_map = {
        "file": {
            "description": "File tools",
            "resolved_tools": ["read_file", "write_file", "patch", "search_files"],
        },
        "code_intel": {
            "description": "Code intel",
            "resolved_tools": [
                "ast_list_defs",
                "ast_find_nodes",
                "lsp_document_symbols",
                "lsp_definition",
                "lsp_diagnostics",
            ],
        },
        "repo-code-knowledge": {
            "description": "Canonical repo knowledge",
            "resolved_tools": [
                "read_file",
                "search_files",
                "ast_list_defs",
                "ast_find_nodes",
                "lsp_document_symbols",
                "lsp_definition",
                "lsp_diagnostics",
            ],
        },
    }
    contract_map = {
        "file": {
            "canonical_name": "file",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": False,
            "boundary_note": "Built-in toolset",
        },
        "code_intel": {
            "canonical_name": "code_intel",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": False,
            "boundary_note": "Built-in toolset",
        },
        "repo-code-knowledge": {
            "canonical_name": "repo-code-knowledge",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": True,
            "boundary_note": "Canonical built-in control-plane surface",
        },
    }

    monkeypatch.setattr(
        model_tools,
        "get_available_toolsets",
        lambda: {
            name: {
                "description": info_map[name]["description"],
                "tools": list(info_map[name]["resolved_tools"]),
                **contract_map[name],
            }
            for name in ("file", "code_intel", "repo-code-knowledge")
        },
    )
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: visible_defs)
    monkeypatch.setattr(toolsets, "get_toolset_info", lambda name: {**info_map[name], "tool_count": len(info_map[name]["resolved_tools"])})
    monkeypatch.setattr(toolsets, "get_toolset_contract", lambda name: contract_map[name])

    server._sessions["sid"] = _session(agent=types.SimpleNamespace(enabled_toolsets=["file", "code_intel"]))
    try:
        resp = server.handle_request({"id": "1", "method": "toolsets.list", "params": {"session_id": "sid"}})
    finally:
        server._sessions.clear()

    rows = {row["name"]: row for row in resp["result"]["toolsets"]}
    assert rows["repo-code-knowledge"]["enabled"] is True
    assert rows["repo-code-knowledge"]["canonical_name"] == "repo-code-knowledge"
    assert rows["repo-code-knowledge"]["is_canonical_knowledge_surface"] is True
    assert rows["file"]["enabled"] is True
    assert rows["code_intel"]["enabled"] is True


def test_tools_show_exposes_canonical_knowledge_surfaces_as_first_class_sections(monkeypatch):
    import model_tools
    import toolsets

    visible_defs = [
        {"function": {"name": "read_file", "description": "Read file contents. Extra detail."}},
        {"function": {"name": "search_files", "description": "Search files. Extra detail."}},
        {"function": {"name": "web_search", "description": "Search the web. Extra detail."}},
        {"function": {"name": "web_extract", "description": "Extract web pages. Extra detail."}},
        {"function": {"name": "browser_navigate", "description": "Navigate browser. Extra detail."}},
        {"function": {"name": "browser_snapshot", "description": "Snapshot browser. Extra detail."}},
        {"function": {"name": "vision_analyze", "description": "Analyze images. Extra detail."}},
    ]
    info_map = {
        "repo-code-knowledge": {
            "description": "Canonical repo knowledge",
            "resolved_tools": ["read_file", "search_files"],
        },
        "web-research-knowledge": {
            "description": "Canonical web knowledge",
            "resolved_tools": ["web_search", "web_extract"],
        },
        "document-pdf-diagram-intelligence": {
            "description": "Canonical document intelligence",
            "resolved_tools": ["browser_navigate", "browser_snapshot", "vision_analyze"],
        },
    }
    contract_map = {
        name: {
            "canonical_name": name,
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": True,
            "boundary_note": "Canonical built-in control-plane surface",
        }
        for name in info_map
    }
    owner_map = {
        "read_file": "file",
        "search_files": "file",
        "web_search": "web",
        "web_extract": "web",
        "browser_navigate": "browser",
        "browser_snapshot": "browser",
        "vision_analyze": "vision",
    }

    monkeypatch.setattr(
        model_tools,
        "get_available_toolsets",
        lambda: {
            name: {
                "description": info_map[name]["description"],
                "tools": list(info_map[name]["resolved_tools"]),
                **contract_map[name],
            }
            for name in info_map
        },
    )
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: visible_defs)
    monkeypatch.setattr(model_tools, "get_toolset_for_tool", lambda name: owner_map[name])
    monkeypatch.setattr(toolsets, "get_toolset_info", lambda name: {**info_map[name], "tool_count": len(info_map[name]["resolved_tools"])})
    monkeypatch.setattr(toolsets, "get_toolset_contract", lambda name: contract_map[name])

    resp = server.handle_request({"id": "1", "method": "tools.show", "params": {}})

    section_names = [section["name"] for section in resp["result"]["sections"]]
    assert section_names == [
        "document-pdf-diagram-intelligence",
        "repo-code-knowledge",
        "web-research-knowledge",
    ]
    sections = {section["name"]: section for section in resp["result"]["sections"]}
    assert [tool["name"] for tool in sections["repo-code-knowledge"]["tools"]] == ["read_file", "search_files"]
    assert [tool["name"] for tool in sections["web-research-knowledge"]["tools"]] == ["web_search", "web_extract"]
    assert [tool["name"] for tool in sections["document-pdf-diagram-intelligence"]["tools"]] == [
        "browser_navigate",
        "browser_snapshot",
        "vision_analyze",
    ]
    legacy_names = [section["name"] for section in resp["result"]["legacy_sections"]]
    assert legacy_names == ["browser", "file", "vision", "web"]
    assert resp["result"]["tools"][0]["owner_toolset"] in {"file", "web", "browser", "vision"}


def test_tools_catalog_remains_provider_config_oriented_and_skips_visibility_derivation(monkeypatch):
    monkeypatch.setattr(server, "_derive_visible_toolsets", lambda enabled_toolsets: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(server, "_tools_catalog_payload", lambda: {"toolsets": [{"name": "web", "enabled": True}], "mcp_servers": []})

    resp = server.handle_request({"id": "1", "method": "tools.catalog", "params": {}})

    assert resp["result"] == {"toolsets": [{"name": "web", "enabled": True}], "mcp_servers": []}


def test_tools_provider_configure_updates_config_and_resets_session(monkeypatch):
    server._sessions["sid"] = _session()
    saved_env = {}
    saved_cfg = {}
    reset_calls = []
    fake_tools = types.SimpleNamespace(
        TOOL_CATEGORIES={"web": {"providers": [{"name": "Firecrawl", "web_backend": "firecrawl", "env_vars": [{"key": "FIRECRAWL_API_KEY"}]}]}},
        _visible_providers=lambda cat, cfg: cat["providers"],
    )
    fake_config = types.SimpleNamespace(
        load_config=lambda: {"web": {}, "mcp_servers": {}},
        save_config=lambda cfg: saved_cfg.update(cfg),
        save_env_value=lambda key, value: saved_env.update({key: value}),
        get_env_value=lambda key: saved_env.get(key, ""),
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.tools_config", fake_tools)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_config)
    monkeypatch.setitem(sys.modules, "hermes_cli.nous_subscription", types.SimpleNamespace(get_nous_subscription_features=lambda cfg: {}))
    monkeypatch.setitem(sys.modules, "tools.tool_backend_helpers", types.SimpleNamespace(managed_nous_tools_enabled=lambda: False))
    monkeypatch.setattr(server, "_tools_catalog_payload", lambda: {"mcp_servers": [], "toolsets": [{"name": "web"}]})
    monkeypatch.setattr(server, "_reset_session_agent", lambda sid, session: reset_calls.append((sid, session["session_key"])) or {"model": "claude", "skills": {}, "tools": {}})

    resp = server.handle_request(
        {
            "id": "1",
            "method": "tools.provider.configure",
            "params": {
                "env": {"FIRECRAWL_API_KEY": "sekret"},
                "provider": "firecrawl",
                "session_id": "sid",
                "toolset": "web",
            },
        }
    )

    assert saved_env == {"FIRECRAWL_API_KEY": "sekret"}
    assert saved_cfg["web"]["backend"] == "firecrawl"
    assert saved_cfg["web"]["use_gateway"] is False
    assert reset_calls == [("sid", "session-key")]
    assert resp["result"] == {
        "mcp_servers": [],
        "toolsets": [{"name": "web"}],
        "info": {"model": "claude", "skills": {}, "tools": {}},
        "provider": "firecrawl",
        "reset": True,
        "toolset": "web",
    }


def test_tools_provider_configure_delegates_to_tools_bridge(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server,
        "_configure_tool_provider_impl",
        lambda **kwargs: seen.update(kwargs) or {"missing_env": ["FIRECRAWL_API_KEY"], "provider": "firecrawl", "toolset": "web"},
    )

    resp = server.handle_request(
        {
            "id": "1",
            "method": "tools.provider.configure",
            "params": {"toolset": "web", "provider": "firecrawl", "env": {"FIRECRAWL_API_KEY": "sekret"}},
        }
    )

    assert resp["result"] == {"missing_env": ["FIRECRAWL_API_KEY"], "provider": "firecrawl", "toolset": "web"}
    assert seen["catalog_payload_builder"] is server._tools_catalog_payload


def test_tools_mcp_configure_delegates_to_tools_bridge(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server,
        "_configure_mcp_servers_impl",
        lambda **kwargs: seen.update(kwargs) or {"changed": ["github"], "info": None, "reset": False, "unknown": [], "mcp_servers": [], "toolsets": []},
    )

    resp = server.handle_request(
        {"id": "1", "method": "tools.mcp.configure", "params": {"action": "disable", "names": ["github"]}}
    )

    assert resp["result"] == {
        "changed": ["github"],
        "info": None,
        "reset": False,
        "unknown": [],
        "mcp_servers": [],
        "toolsets": [],
    }
    assert seen["catalog_payload_builder"] is server._tools_catalog_payload


def test_tools_mcp_probe_delegates_to_tools_bridge(monkeypatch):
    monkeypatch.setattr(server, "_probe_mcp_servers_impl", lambda **kwargs: {"servers": [{"name": "github", "reachable": True, "tools": []}]})

    resp = server.handle_request({"id": "1", "method": "tools.mcp.probe", "params": {}})

    assert resp["result"] == {"servers": [{"name": "github", "reachable": True, "tools": []}]}


def test_tools_configure_delegates_to_tools_bridge(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server,
        "_configure_tools_impl",
        lambda **kwargs: seen.update(kwargs) or {
            "changed": ["web"],
            "enabled_toolsets": ["terminal"],
            "info": None,
            "missing_servers": [],
            "reset": False,
            "unknown": [],
        },
    )

    resp = server.handle_request(
        {"id": "1", "method": "tools.configure", "params": {"action": "disable", "names": ["web"]}}
    )

    assert resp["result"] == {
        "changed": ["web"],
        "enabled_toolsets": ["terminal"],
        "info": None,
        "missing_servers": [],
        "reset": False,
        "unknown": [],
    }
    assert seen["session_lookup"]("missing") is None


def test_skills_manage_delegates_to_bridge(monkeypatch):
    seen = {}
    fake_banner = types.SimpleNamespace(get_available_skills=lambda: ["alpha", "beta"])
    fake_source_hub = types.SimpleNamespace(
        GitHubAuth=lambda: "auth",
        create_source_router=lambda auth: {"auth": auth},
        unified_search=lambda *args, **kwargs: [],
    )
    fake_skills = types.SimpleNamespace(
        browse_skills=lambda **kwargs: {"items": []},
        do_audit=lambda *args, **kwargs: None,
        do_check=lambda *args, **kwargs: None,
        do_install=lambda *args, **kwargs: None,
        do_uninstall=lambda *args, **kwargs: None,
        do_update=lambda *args, **kwargs: None,
        inspect_skill=lambda query: {"identifier": query},
    )

    monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
    monkeypatch.setitem(sys.modules, "hermes_cli.skills_hub", fake_skills)
    monkeypatch.setitem(sys.modules, "tools.skills_hub", fake_source_hub)
    monkeypatch.setattr(
        server,
        "_dispatch_skills_manage_impl",
        lambda params, **kwargs: seen.update({"params": params, **kwargs}) or {"skills": ["alpha", "beta"]},
    )

    resp = server.handle_request({"id": "1", "method": "skills.manage", "params": {"action": "LIST"}})

    assert resp["result"] == {"skills": ["alpha", "beta"]}
    assert seen["params"] == {"action": "list", "query": ""}
    assert seen["get_available_skills"] is fake_banner.get_available_skills
    assert seen["capture_console_output"] is server._capture_console_output
    assert seen["browse_skills"] is fake_skills.browse_skills
    assert seen["inspect_skill"] is fake_skills.inspect_skill
    assert seen["do_install"] is fake_skills.do_install
    assert seen["do_update"] is fake_skills.do_update
    assert seen["do_audit"] is fake_skills.do_audit
    assert seen["do_uninstall"] is fake_skills.do_uninstall
    assert seen["source_router_factory"]("auth") == {"auth": "auth"}
    assert seen["auth_factory"]() == "auth"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"action": "search"}, "skills.search requires query"),
        ({"action": "install", "query": ""}, "skills.install requires query"),
        ({"action": "list", "query": "alpha"}, "unexpected skills params for list: query"),
        ({"action": "browse", "page": 0}, "skills.browse requires positive page"),
        ({"action": "audit", "query": "all", "source": "github"}, "unexpected skills params for audit: source"),
        ({"action": "wat"}, "unknown skills action: wat"),
    ],
)
def test_skills_manage_contract_rejects_invalid_action_shapes(params, message):
    resp = server.handle_request({"id": "1", "method": "skills.manage", "params": params})

    assert resp["error"]["code"] == 4017
    assert resp["error"]["message"] == message


def test_config_set_personality_rejects_unknown_name(monkeypatch):
    monkeypatch.setattr(server, "_available_personalities", lambda cfg=None: {"helpful": "You are helpful."})
    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"key": "personality", "value": "bogus"}}
    )

    assert "error" in resp
    assert "Unknown personality" in resp["error"]["message"]


def test_config_set_personality_resets_history_and_returns_info(monkeypatch):
    session = _session(agent=types.SimpleNamespace(), history=[{"role": "user", "text": "hi"}], history_version=4)
    new_agent = types.SimpleNamespace(model="x")
    emits = []

    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_available_personalities", lambda cfg=None: {"helpful": "You are helpful."})
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None: new_agent)
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": getattr(agent, "model", "?")})
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    monkeypatch.setattr(server, "_write_config_key", lambda path, value: None)

    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "personality", "value": "helpful"}}
    )

    assert resp["result"]["history_reset"] is True
    assert resp["result"]["info"] == {"model": "x"}
    assert session["history"] == []
    assert session["history_version"] == 5
    assert ("session.info", "sid", {"model": "x"}) in emits


def test_session_compress_uses_compress_helper(monkeypatch):
    agent = types.SimpleNamespace()
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(server, "_compress_session_history", lambda session, focus_topic=None: (2, {"total": 42}))
    monkeypatch.setattr(server, "_session_info", lambda _agent: {"model": "x"})

    with patch("tui_gateway.server._emit") as emit:
        resp = server.handle_request({"id": "1", "method": "session.compress", "params": {"session_id": "sid"}})

    assert resp["result"]["removed"] == 2
    assert resp["result"]["usage"]["total"] == 42
    emit.assert_called_once_with("session.info", "sid", {"model": "x"})


def test_prompt_submit_sets_approval_session_key(monkeypatch):
    from tools.approval import get_current_session_key

    captured = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            captured["session_key"] = get_current_session_key(default="")
            return {"final_response": "ok", "messages": [{"role": "assistant", "content": "ok"}]}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    resp = server.handle_request({"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "ping"}})

    assert resp["result"]["status"] == "streaming"
    assert captured["session_key"] == "session-key"


def test_prompt_submit_expands_context_refs(monkeypatch):
    captured = {}

    class _Agent:
        model = "test/model"
        base_url = ""
        api_key = ""

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            captured["prompt"] = prompt
            return {"final_response": "ok", "messages": [{"role": "assistant", "content": "ok"}]}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    fake_ctx = types.ModuleType("agent.context_references")
    fake_ctx.preprocess_context_references = lambda message, **kwargs: types.SimpleNamespace(
        blocked=False, message="expanded prompt", warnings=[], references=[], injected_tokens=0
    )
    fake_meta = types.ModuleType("agent.model_metadata")
    fake_meta.get_model_context_length = lambda *args, **kwargs: 100000

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setitem(sys.modules, "agent.context_references", fake_ctx)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", fake_meta)

    server.handle_request({"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "@diff"}})

    assert captured["prompt"] == "expanded prompt"


def test_session_resume_reopens_persisted_history_into_a_fresh_runtime_session(monkeypatch):
    reopened = []
    captured = {}

    class _DB:
        def get_session(self, _sid):
            return {"id": "persisted-session"}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, sid):
            reopened.append(sid)

        def get_messages_as_conversation(self, _sid):
            return [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "world",
                    "metadata": {
                        "swarm": {
                            "subagents": [
                                {
                                    "goal": "Audit slash routes",
                                    "id": "sa:0:Audit slash routes",
                                    "index": 0,
                                    "notes": ["confirmed"],
                                    "status": "completed",
                                    "summary": "done",
                                    "taskCount": 1,
                                    "thinking": [],
                                    "tools": []
                                }
                            ],
                            "turnStatus": "persisted"
                        }
                    }
                },
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda _target: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None: types.SimpleNamespace(session_id=session_id, runtime_sid=sid))
    monkeypatch.setattr(
        server,
        "_init_session",
        lambda sid, key, agent, history, cols=80: captured.update(
            {"sid": sid, "session_key": key, "history": history, "agent": agent, "cols": cols}
        ),
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent: {"model": "test/model"})

    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": {"session_id": "persisted-session", "cols": 120}}
    )

    assert resp["result"]["resumed"] == "persisted-session"
    assert resp["result"]["message_count"] == 2
    assert resp["result"]["messages"][1]["swarm"]["turnStatus"] == "persisted"
    assert resp["result"]["messages"][1]["swarm"]["subagents"][0]["goal"] == "Audit slash routes"
    assert reopened == ["persisted-session"]
    assert captured["session_key"] == "persisted-session"
    assert captured["sid"] != "persisted-session"
    assert captured["agent"].session_id == "persisted-session"
    assert captured["cols"] == 120


def test_image_attach_appends_local_image(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png"}
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: Path("/tmp/cat.png")

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request({"id": "1", "method": "image.attach", "params": {"session_id": "sid", "path": "/tmp/cat.png"}})

    assert resp["result"]["attached"] is True
    assert resp["result"]["name"] == "cat.png"
    assert len(server._sessions["sid"]["attached_images"]) == 1


def test_command_dispatch_exec_nonzero_surfaces_error(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": {"boom": {"type": "exec", "command": "boom"}}})
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="failed"),
    )

    resp = server.handle_request({"id": "1", "method": "command.dispatch", "params": {"name": "boom"}})

    assert "error" in resp
    assert "failed" in resp["error"]["message"]


def test_routing_owner_lane_locks_authoritative_route_sets():
    assert server._NATIVE_PRODUCT_COMMANDS == frozenset({"setup", "skills", "swarm", "tools"})
    assert server._LIVE_SLASH_COMMANDS == frozenset({"handoff", "init-deep", "model", "provider", "ralph-loop", "start-work", "ulw-loop"})
    assert server._LEGACY_SLASH_WORKER_COMMANDS == frozenset({
        "agents",
        "browser",
        "config",
        "cron",
        "debug",
        "fast",
        "gquota",
        "history",
        "insights",
        "platforms",
        "plugins",
        "profile",
        "reload",
        "reload-mcp",
        "rollback",
        "save",
        "snapshot",
        "status",
        "stop",
        "title",
        "toolsets",
    })
    assert server._SLASH_EXEC_COMMANDS == server._LIVE_SLASH_COMMANDS | server._LEGACY_SLASH_WORKER_COMMANDS


def test_command_dispatch_rejects_route_owned_commands():
    native = server.handle_request({"id": "1", "method": "command.dispatch", "params": {"name": "tools"}})
    live = server.handle_request({"id": "2", "method": "command.dispatch", "params": {"name": "start-work"}})
    legacy = server.handle_request({"id": "3", "method": "command.dispatch", "params": {"name": "config"}})

    assert native["error"]["message"] == "/tools is handled by native product routing; command.dispatch is fallback-only"
    assert live["error"]["message"] == "/start-work is handled by slash.exec; command.dispatch is fallback-only"
    assert legacy["error"]["message"] == "/config is handled by slash.exec; command.dispatch is fallback-only"


def test_slash_worker_locks_legacy_only_execution(monkeypatch):
    fake_cli = types.ModuleType("cli")

    class FakeHermesCLI:
        def process_command(self, command):
            print(f"ran:{command}")

    fake_cli.HermesCLI = FakeHermesCLI
    fake_cli._cprint = lambda text: None
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    import tui_gateway.slash_worker as slash_worker

    slash_worker = importlib.reload(slash_worker)

    assert slash_worker._NATIVE_PRODUCT_COMMANDS == frozenset({"setup", "skills", "tools"})
    assert slash_worker._LIVE_SLASH_COMMANDS == frozenset({"handoff", "init-deep", "model", "provider", "ralph-loop", "start-work", "ulw-loop"})
    assert slash_worker._LEGACY_SLASH_WORKER_COMMANDS == server._LEGACY_SLASH_WORKER_COMMANDS

    cli = FakeHermesCLI()
    with patch.object(slash_worker.cli_mod, "_cprint", lambda text: None):
        assert slash_worker._run(cli, "/config") == "ran:/config"

    with patch.object(slash_worker.cli_mod, "_cprint", lambda text: None):
        try:
            slash_worker._run(cli, "/tools")
        except RuntimeError as exc:
            assert str(exc) == "/tools is handled by native product routing; slash worker is legacy-only"
        else:
            raise AssertionError("expected native route rejection")

    with patch.object(slash_worker.cli_mod, "_cprint", lambda text: None):
        try:
            slash_worker._run(cli, "/provider claude")
        except RuntimeError as exc:
            assert str(exc) == "/provider is handled by slash.exec live routing; slash worker is legacy-only"
        else:
            raise AssertionError("expected live route rejection")

    with patch.object(slash_worker.cli_mod, "_cprint", lambda text: None):
        try:
            slash_worker._run(cli, "/resume")
        except RuntimeError as exc:
            assert str(exc) == "/resume is not eligible for slash worker execution"
        else:
            raise AssertionError("expected non-legacy rejection")


@pytest.mark.parametrize(
    ("command_text", "canonical", "raw_args"),
    [
        ("handoff request", "handoff", "request"),
        ("init-deep investigate", "init-deep", "investigate"),
        ("start-work ship it", "start-work", "ship it"),
        ("ralph-loop close it", "ralph-loop", "close it"),
        ("ulw-loop finish it", "ulw-loop", "finish it"),
    ],
)
def test_slash_exec_work_command_submits_live_prompt(monkeypatch, command_text, canonical, raw_args):
    captured = {}
    session = _session()
    server._sessions["sid"] = session

    monkeypatch.setattr(
        server,
        "_start_prompt_submit",
        lambda rid, sid, session, text: captured.update({"args": (rid, sid, session["session_key"], text)}) or {"result": {"status": "streaming"}},
    )
    monkeypatch.setattr(
        "hermes_cli.work_command_adapter.prepare_work_command",
        lambda canonical, raw_args, session_id, cwd: f"prepared::{canonical}::{raw_args}::{session_id}",
    )

    resp = server.handle_request({"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": command_text}})

    assert resp["result"]["output"] == f"/{canonical} started"
    assert captured["args"] == ("1", "sid", "session-key", f"prepared::{canonical}::{raw_args}::session-key")


@pytest.mark.parametrize(
    ("command_text", "expected_runtime_mode", "should_continue"),
    [
        ("handoff request", "default", False),
        ("init-deep investigate", "default", False),
        ("start-work ship it", "default", False),
        ("ralph-loop close it", "ralph", True),
        ("ulw-loop finish it", "ultrawork", True),
    ],
)
def test_slash_exec_work_command_preserves_continuation_semantics(monkeypatch, command_text, expected_runtime_mode, should_continue):
    captured = {}
    session = _session()
    server._sessions["sid"] = session

    def _capture_submit(rid, sid, session, prepared):
        captured["prepared"] = prepared
        return {"result": {"status": "streaming"}}

    monkeypatch.setattr(server, "_start_prompt_submit", _capture_submit)

    resp = server.handle_request({"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": command_text}})

    prepared = captured["prepared"]
    classification = preclassify_intent({"message": prepared.task_contract["task"], "task_contract": prepared.task_contract})

    assert resp["result"]["output"] == f"/{prepared.command_name} started"
    assert classification.inferred_runtime_mode == expected_runtime_mode
    assert should_use_continuation_engine(classification.inferred_runtime_mode, {
        "outcomeStatus": "interrupted",
        "activeTodos": [{"id": "todo-1", "content": "Finish the delegated task", "status": "in_progress"}],
    }) is should_continue
    if prepared.command_name == "start-work":
        assert "NAMED_WORKFLOW_JSON:" in prepared.agent_message
    elif should_continue:
        assert f'"runtime_mode": "{expected_runtime_mode}"' in prepared.agent_message


@pytest.mark.parametrize(
    "command_text",
    [
        "handoff request",
        "init-deep investigate",
        "start-work ship it",
        "ralph-loop close it",
        "ulw-loop finish it",
    ],
)
def test_slash_exec_work_command_busy_is_deterministic(monkeypatch, command_text):
    server._sessions["sid"] = _session(running=True)
    monkeypatch.setattr(
        "hermes_cli.work_command_adapter.prepare_work_command",
        lambda canonical, raw_args, session_id, cwd: "prepared prompt",
    )

    resp = server.handle_request({"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": command_text}})

    assert "error" in resp
    assert resp["error"]["code"] == 4009
    assert resp["error"]["message"] == "session busy"



def test_slash_exec_model_uses_live_switch_path(monkeypatch):
    server._sessions["sid"] = _session()
    seen = {}

    def _fake_apply(sid, session, raw):
        seen["args"] = (sid, session["session_key"], raw)
        return {"value": "new/model", "warning": "catalog unreachable"}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)
    resp = server.handle_request(
        {"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": "model new/model"}}
    )

    assert resp["result"]["output"] == "model → new/model"
    assert resp["result"]["warning"] == "catalog unreachable"
    assert seen["args"] == ("sid", "session-key", "new/model")



def test_slash_exec_provider_alias_uses_live_model_path(monkeypatch):
    server._sessions["sid"] = _session()
    seen = {}

    def _fake_apply(sid, session, raw):
        seen["args"] = (sid, session["session_key"], raw)
        return {"value": "provider/model", "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)
    resp = server.handle_request(
        {"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": "provider provider/model"}}
    )

    assert resp["result"]["output"] == "model → provider/model"
    assert seen["args"] == ("sid", "session-key", "provider/model")



def test_slash_exec_native_product_command_rejects_worker(monkeypatch):
    session = _session()
    server._sessions["sid"] = session
    worker_ctor = []

    monkeypatch.setattr(server, "_SlashWorker", lambda *args, **kwargs: worker_ctor.append((args, kwargs)))
    resp = server.handle_request({"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": "tools"}})

    assert "error" in resp
    assert resp["error"]["message"] == "/tools is handled by native product routing; slash.exec is legacy-only"
    assert worker_ctor == []
    assert session["slash_worker"] is None



def test_slash_exec_legacy_command_still_uses_worker(monkeypatch):
    session = _session()
    server._sessions["sid"] = session
    seen = {}

    class _Worker:
        def run(self, command):
            seen["command"] = command
            return "legacy output"

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(server, "_SlashWorker", lambda session_key, model: _Worker())
    resp = server.handle_request({"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": "config"}})

    assert resp["result"]["output"] == "legacy output"
    assert seen["command"] == "config"



def test_plugins_list_surfaces_loader_error(monkeypatch):
    with patch("hermes_cli.plugins.get_plugin_manager", side_effect=Exception("boom")):
        resp = server.handle_request({"id": "1", "method": "plugins.list", "params": {}})

    assert "error" in resp
    assert "boom" in resp["error"]["message"]


def test_complete_slash_surfaces_completer_error(monkeypatch):
    with patch("hermes_cli.commands.SlashCommandCompleter", side_effect=Exception("no completer")):
        resp = server.handle_request({"id": "1", "method": "complete.slash", "params": {"text": "/mo"}})

    assert "error" in resp
    assert "no completer" in resp["error"]["message"]


def test_input_detect_drop_attaches_image(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: {
        "path": Path("/tmp/cat.png"),
        "is_image": True,
        "remainder": "",
    }

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request(
        {"id": "1", "method": "input.detect_drop", "params": {"session_id": "sid", "text": "/tmp/cat.png"}}
    )

    assert resp["result"]["matched"] is True
    assert resp["result"]["is_image"] is True
    assert resp["result"]["text"] == "[User attached image: cat.png]"


def test_rollback_restore_resolves_number_and_file_path():
    calls = {}

    class _Mgr:
        enabled = True

        def list_checkpoints(self, cwd):
            return [{"hash": "aaa111"}, {"hash": "bbb222"}]

        def restore(self, cwd, target, file_path=None):
            calls["args"] = (cwd, target, file_path)
            return {"success": True, "message": "done"}

    server._sessions["sid"] = _session(agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()), history=[])
    resp = server.handle_request(
        {
            "id": "1",
            "method": "rollback.restore",
            "params": {"session_id": "sid", "hash": "2", "file_path": "src/app.tsx"},
        }
    )

    assert resp["result"]["success"] is True
    assert calls["args"][1] == "bbb222"
    assert calls["args"][2] == "src/app.tsx"
