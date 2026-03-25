"""Tests for tools/service_expose_tool.py."""

import json
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

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


@pytest.mark.parametrize(
    "strategy,env_name,expected_url,is_public",
    [
        ("reverse-proxy", "HERMES_SERVICE_EXPOSE_REVERSE_PROXY_TEMPLATE", "https://review.example/", True),
        ("tailscale-serve", "HERMES_SERVICE_EXPOSE_TAILSCALE_SERVE_TEMPLATE", "https://tailnet.example/", False),
        ("tailscale-funnel", "HERMES_SERVICE_EXPOSE_TAILSCALE_FUNNEL_TEMPLATE", "https://public.example/", True),
    ],
)
def test_named_strategies_use_env_templates(monkeypatch, strategy, env_name, expected_url, is_public):
    monkeypatch.setenv(
        env_name,
        "launcher --listen {local_url} --host {requested_host} --mode {strategy}",
    )
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout=f"URL={expected_url}\n",
        stderr="",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(
            service_expose_tool(
                {
                    "action": "expose",
                    "strategy": strategy,
                    "local_port": 9999,
                    "requested_host": "review.example",
                }
            )
        )

    assert result["success"] is True
    assert result["url"] == expected_url
    assert result["public"] is is_public
    command = run_mock.call_args.args[0][2]
    assert "launcher --listen http://127.0.0.1:9999 --host review.example --mode" in command


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
