from __future__ import annotations

import pytest

from tui_gateway import entry


def test_prepare_tui_gateway_hooks_registers_shell_hooks(monkeypatch):
    calls = []

    def fake_discover_plugins():
        calls.append(("discover_plugins",))

    def fake_load_config():
        calls.append(("load_config",))
        return {"hooks": {"post_llm_call": [{"command": "echo {}", "timeout": 5}]}}

    def fake_register_from_config(cfg, *, accept_hooks=False):
        calls.append(("register_from_config", cfg, accept_hooks))

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", fake_discover_plugins)
    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        fake_register_from_config,
    )

    entry._prepare_tui_gateway_hooks()

    assert calls == [
        ("discover_plugins",),
        ("load_config",),
        (
            "register_from_config",
            {"hooks": {"post_llm_call": [{"command": "echo {}", "timeout": 5}]}},
            False,
        ),
    ]


def test_main_prepares_hooks_before_gateway_ready(monkeypatch):
    events = []

    monkeypatch.setattr(
        entry,
        "_install_sidecar_publisher",
        lambda: events.append("sidecar"),
    )
    monkeypatch.setattr(
        entry,
        "_prepare_tui_gateway_hooks",
        lambda: events.append("hooks"),
    )
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"mcp_servers": {}},
    )
    monkeypatch.setattr(
        entry,
        "_log_exit",
        lambda reason: events.append(("exit", reason)),
    )

    def fake_write_json(payload):
        events.append(("write_json", payload["params"]["type"]))
        return False

    monkeypatch.setattr(entry, "write_json", fake_write_json)

    with pytest.raises(SystemExit):
        entry.main()

    assert events[:3] == [
        "sidecar",
        "hooks",
        ("write_json", "gateway.ready"),
    ]
