"""Tests for unset ``${VAR}`` placeholders in MCP server configs.

`_interpolate_env_vars` leaves the literal `${VAR}` in place when VAR is unset, and that placeholder
used to be handed to the server as if it were a value. Observed: a turn ran in a process without
JIRA_URL, `${JIRA_URL}` reached mcp-atlassian verbatim, and the failure surfaced as
`requests.exceptions.MissingSchema: Invalid URL '${JIRA_URL}/rest/api/2/search...'` from inside a
subprocess -- a message that names `requests` rather than the missing variable.
"""

import logging

import pytest

from tools.mcp_tool import _strip_unresolved_placeholders, _unresolved_env_refs


@pytest.fixture(autouse=True)
def _no_placeholder_vars(monkeypatch):
    for name in ("MCP_TEST_URL", "MCP_TEST_TOKEN", "MCP_TEST_OPTIONAL"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- _unresolved_env_refs

def test_no_refs_in_a_resolved_config():
    assert _unresolved_env_refs({"url": "https://example.com", "env": {"A": "1"}}) == []


def test_refs_found_in_nested_structures():
    cfg = {
        "url": "${MCP_TEST_URL}/mcp",
        "args": ["--token", "${MCP_TEST_TOKEN}"],
        "env": {"NESTED": {"deep": "${MCP_TEST_TOKEN}"}},
    }
    # Deduped, in encounter order.
    assert _unresolved_env_refs(cfg) == ["MCP_TEST_URL", "MCP_TEST_TOKEN"]


def test_cursor_style_env_prefix_is_normalised():
    assert _unresolved_env_refs({"url": "${env:MCP_TEST_URL}"}) == ["MCP_TEST_URL"]


def test_context_vars_are_not_env_refs():
    # ${userHome} and friends always resolve, so they must not be reported as missing.
    assert _unresolved_env_refs({"args": ["${userHome}/bin/x"]}) == []


# ------------------------------------------------------------- _strip_unresolved_placeholders

def test_resolved_config_passes_through_unchanged(caplog):
    cfg = {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer abc"}}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        out, dropped = _strip_unresolved_placeholders("srv", cfg)
    assert out == cfg
    assert dropped == []
    assert caplog.records == []


def test_one_unset_entry_is_dropped_and_the_server_survives(caplog):
    # The case that must NOT disable a server: an optional credential the downstream server ignores.
    cfg = {
        "command": "mcp-atlassian",
        "env": {"URL": "https://example.atlassian.net", "PERSONAL_TOKEN": "${MCP_TEST_OPTIONAL}"},
    }
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        out, dropped = _strip_unresolved_placeholders("jira", cfg)
    assert out is not None
    assert out["env"] == {"URL": "https://example.atlassian.net"}
    assert dropped == ["MCP_TEST_OPTIONAL"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("jira" in m and "MCP_TEST_OPTIONAL" in m for m in messages)
    # No placeholder survives into the config handed to the server.
    assert "${" not in repr(out)


def test_every_entry_unset_refuses_the_server(caplog):
    # A mapping that was entirely placeholders leaves the server with nothing to talk to.
    cfg = {"command": "mcp-atlassian", "env": {"URL": "${MCP_TEST_URL}", "TOKEN": "${MCP_TEST_TOKEN}"}}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        out, offending = _strip_unresolved_placeholders("jira", cfg)
    assert out is None
    assert offending == ["MCP_TEST_URL", "MCP_TEST_TOKEN"]
    assert any("NOT registered" in r.getMessage() for r in caplog.records)


def test_unset_url_refuses_the_server(caplog):
    # The reported failure: a `url` that is half a placeholder is not a URL, and registering it moves
    # the error into a subprocess that cannot name the variable.
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        out, offending = _strip_unresolved_placeholders("remote", {"url": "${MCP_TEST_URL}/mcp"})
    assert out is None
    assert offending == ["MCP_TEST_URL"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("MCP_TEST_URL" in m and "NOT registered" in m for m in messages)


def test_unset_arg_refuses_the_server():
    out, offending = _strip_unresolved_placeholders(
        "srv", {"command": "npx", "args": ["-y", "pkg", "--token=${MCP_TEST_TOKEN}"]}
    )
    assert out is None
    assert offending == ["MCP_TEST_TOKEN"]


def test_a_set_variable_is_never_reported(monkeypatch):
    monkeypatch.setenv("MCP_TEST_URL", "https://example.com")
    from tools.mcp_tool import _interpolate_env_vars

    cfg = _interpolate_env_vars({"url": "${MCP_TEST_URL}/mcp"})
    out, dropped = _strip_unresolved_placeholders("srv", cfg)
    assert out == {"url": "https://example.com/mcp"}
    assert dropped == []


def test_the_value_never_reaches_the_log(caplog, monkeypatch):
    # The whole point of logging names: a warning that printed the secret would be a worse defect than
    # the one this guards against.
    monkeypatch.setenv("MCP_TEST_TOKEN", "s3cret-value")
    from tools.mcp_tool import _interpolate_env_vars

    cfg = _interpolate_env_vars(
        {"command": "x", "env": {"TOKEN": "${MCP_TEST_TOKEN}", "OTHER": "${MCP_TEST_OPTIONAL}"}}
    )
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        out, dropped = _strip_unresolved_placeholders("srv", cfg)
    assert out is not None
    assert dropped == ["MCP_TEST_OPTIONAL"]
    assert not any("s3cret-value" in r.getMessage() for r in caplog.records)
