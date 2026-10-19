"""Stdio MCP launch env: an optional ``"VAR": "${VAR}"`` forward never reaches the child literally."""

from unittest.mock import patch


def test_unset_whole_placeholder_env_entry_is_dropped():
    """An optional ``"VAR": "${VAR}"`` forward never reaches the child as the literal
    placeholder: unset drops the key, set passes the value, partial refs keep today's text."""
    from tools.mcp_tool_config import _build_safe_env, _interpolate_env_vars

    user_env = {"OPT_DIR": "${OPT_DIR}", "OPT_ENV": "${env:OPT_ENV}", "SET_DIR": "${SET_DIR}",
                "PARTIAL": "${OPT_DIR}/bin", "PLAIN": "x", "XDG_CONFIG_HOME": "${OPT_CONFIG_HOME}"}
    parent = {"PATH": "/usr/bin", "SET_DIR": "/opt/set", "XDG_CONFIG_HOME": "/parent/config"}
    with patch.dict("os.environ", parent, clear=True):
        result = _build_safe_env(_interpolate_env_vars(user_env))

    assert "OPT_DIR" not in result and "OPT_ENV" not in result
    assert "XDG_CONFIG_HOME" not in result  # unset override does not fall back to the parent's value
    assert result["SET_DIR"] == "/opt/set"
    assert result["PARTIAL"] == "${OPT_DIR}/bin"
    assert result["PLAIN"] == "x"
