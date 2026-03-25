"""Tests for tools/service_expose_tool.py."""

import json
from subprocess import CompletedProcess
from unittest.mock import patch

from tools.service_expose_tool import service_expose_tool


def test_describe_lists_supported_strategies():
    result = json.loads(service_expose_tool({"action": "describe"}))

    names = {item["name"] for item in result["strategies"]}
    assert {"localhost", "reverse-proxy", "tailscale-serve", "tailscale-funnel", "command"} <= names
    assert "{local_port}" in result["template_placeholders"]


def test_localhost_exposure_returns_direct_url():
    result = json.loads(
        service_expose_tool(
            {
                "action": "expose",
                "strategy": "localhost",
                "local_port": 19432,
                "path": "/review",
                "service_name": "plannotator",
            }
        )
    )

    assert result["success"] is True
    assert result["url"] == "http://127.0.0.1:19432/review"
    assert result["public"] is False


def test_command_strategy_runs_template_and_parses_url_pid_log():
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout="URL=https://demo.example/review\nPID=123\nLOG=/tmp/demo.log\n",
        stderr="",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(
            service_expose_tool(
                {
                    "action": "expose",
                    "strategy": "command",
                    "local_port": 19432,
                    "service_name": "plannotator",
                    "command_template": "launcher --port {local_port} --name {service_name}",
                }
            )
        )

    assert result["success"] is True
    assert result["url"] == "https://demo.example/review"
    assert result["pid"] == "123"
    assert result["log"] == "/tmp/demo.log"
    invoked = run_mock.call_args.args[0]
    assert invoked[:2] == ["bash", "-lc"]
    assert "launcher --port 19432 --name plannotator" in invoked[2]


def test_named_strategy_uses_env_template(monkeypatch):
    monkeypatch.setenv(
        "HERMES_SERVICE_EXPOSE_REVERSE_PROXY_TEMPLATE",
        "router-expose --listen {local_url} --host {requested_host}",
    )
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout="URL=https://review.example/\n",
        stderr="",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(
            service_expose_tool(
                {
                    "action": "expose",
                    "strategy": "reverse-proxy",
                    "local_port": 9999,
                    "requested_host": "review.example",
                }
            )
        )

    assert result["success"] is True
    assert result["url"] == "https://review.example/"
    assert "router-expose --listen http://127.0.0.1:9999 --host review.example" in run_mock.call_args.args[0][2]


def test_reverse_proxy_uses_legacy_env_template_for_compat(monkeypatch):
    monkeypatch.setenv(
        "HERMES_SERVICE_EXPOSE_CLOUD77_TEMPLATE",
        "legacy-expose --listen {local_url} --host {requested_host}",
    )
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout="URL=https://legacy.example/\n",
        stderr="",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(
            service_expose_tool(
                {
                    "action": "expose",
                    "strategy": "reverse-proxy",
                    "local_port": 7777,
                    "requested_host": "legacy.example",
                }
            )
        )

    assert result["success"] is True
    assert result["url"] == "https://legacy.example/"
    assert "legacy-expose --listen http://127.0.0.1:7777 --host legacy.example" in run_mock.call_args.args[0][2]


def test_missing_template_returns_clear_error():
    result = json.loads(
        service_expose_tool(
            {
                "action": "expose",
                "strategy": "tailscale-funnel",
                "local_port": 8080,
            }
        )
    )

    assert "No command template configured" in result["error"]
