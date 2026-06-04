from __future__ import annotations


def test_connectivity_check_uses_url_hostname_not_config_key():
    from tools.mcp_tool import _run_mcp_connectivity_check

    probed_hosts: list[str] = []

    _run_mcp_connectivity_check(
        "github",
        {"url": "https://api.githubcopilot.com/mcp/"},
        probed_hosts.append,
    )

    assert probed_hosts == ["api.githubcopilot.com"]
    assert "github" not in probed_hosts


def test_connectivity_check_skips_dns_for_stdio_transport():
    from tools.mcp_tool import _run_mcp_connectivity_check

    probed_hosts: list[str] = []

    _run_mcp_connectivity_check(
        "local-tools",
        {"command": "python", "args": ["-m", "example_mcp_server"]},
        probed_hosts.append,
    )

    assert probed_hosts == []


def test_connectivity_check_handles_missing_url_gracefully():
    from tools.mcp_tool import _run_mcp_connectivity_check

    probed_hosts: list[str] = []

    _run_mcp_connectivity_check("broken-server", {}, probed_hosts.append)

    assert probed_hosts == []
