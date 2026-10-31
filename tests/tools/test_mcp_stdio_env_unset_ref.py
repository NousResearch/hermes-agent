"""Stdio MCP launch env: an env entry rendered from one unset ``${VAR}`` counts as absent."""

import json
import subprocess
import sys
from unittest.mock import patch

USER_ENV = {"OPT_DIR": "${OPT_DIR}", "OPT_ENV": "${env:OPT_ENV}", "EMPTY": "${EMPTY}", "SET_DIR": "${SET_DIR}",
            "PARTIAL": "${OPT_DIR}/bin", "PLAIN": "x", "XDG_CONFIG_HOME": "${OPT_CONFIG_HOME}",
            "TERM": "${OPT_TERM}", "RESOLVED_LITERAL": "${ACTUAL_LITERAL}"}
PARENT = {"PATH": "/usr/bin:/bin", "TERM": "parent-term", "XDG_CONFIG_HOME": "/parent/config", "SET_DIR": "/opt/set",
          "EMPTY": "", "ACTUAL_LITERAL": "${STILL_UNSET}"}


def _rendered():
    from tools.mcp_tool_config import _interpolate_env_vars
    # Load renders once; connect renders again under the owner's scope.
    return _interpolate_env_vars(_interpolate_env_vars({"command": sys.executable, "env": USER_ENV}))


def test_unset_whole_ref_counts_as_absent_and_set_values_pass_verbatim():
    from tools.mcp_tool_config import _build_safe_env

    with patch.dict("os.environ", PARENT, clear=True):
        result = _build_safe_env(_rendered()["env"])

    assert not {"OPT_DIR", "OPT_ENV", "EMPTY"} & set(result)
    assert result["XDG_CONFIG_HOME"] == "/parent/config"  # as if the line weren't there
    assert result["SET_DIR"] == "/opt/set" and result["PLAIN"] == "x"
    assert result["PARTIAL"] == "${OPT_DIR}/bin"
    assert result["RESOLVED_LITERAL"] == "${STILL_UNSET}"  # a set value is data, whatever it looks like

    # Unset at load, set (even to its own placeholder text) by connect: the re-render passes it.
    from tools.mcp_tool_config import _interpolate_env_vars
    loaded = _interpolate_env_vars({"FORWARD": "${LATE}"})
    with patch.dict("os.environ", {**PARENT, "LATE": "${LATE}"}, clear=True):
        assert _build_safe_env(_interpolate_env_vars(loaded))["FORWARD"] == "${LATE}"


def test_child_spawned_through_the_sdk_merge_sees_the_documented_env():
    from mcp.client.stdio import get_default_environment
    from tools.mcp_tool_transport import _stdio_launch

    with patch.dict("os.environ", PARENT, clear=True):
        command, env, _ = _stdio_launch(_rendered())
        child_env = get_default_environment() | env  # what mcp.client.stdio.stdio_client spawns with
    probe = "import json, os; print(json.dumps(dict(os.environ)))"
    seen = json.loads(subprocess.run([command, "-c", probe], env=child_env, capture_output=True, text=True,
                                     check=True).stdout)

    assert "OPT_DIR" not in seen and "EMPTY" not in seen
    assert seen["TERM"] == "parent-term"
    assert seen["RESOLVED_LITERAL"] == "${STILL_UNSET}"
