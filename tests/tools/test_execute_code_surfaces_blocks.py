"""``execute_code`` must SURFACE a guard block, never return empty success.

The gateway-lifecycle guard refuses a ``terminal()`` call by RETURNING a
result dict. Inside the ``execute_code`` sandbox that is a value, not an
error — so the overwhelmingly common script shape ::

    r = terminal("sudo launchctl bootstrap system /path/to.plist")

runs to completion without inspecting ``r``, and ``execute_code`` reports
``status=success``, ``exit_code=0``, ``output=""``: a silent block, strictly
worse than the direct terminal tool, which at least prints why it refused.

The generated ``hermes_tools`` stub therefore raises ``ToolCallBlocked`` on
any result carrying the guard's marker key, so the refusal lands in the
cell's traceback and ``execute_code`` reports ``status=error``.
"""

from __future__ import annotations

import inspect
import json

from cron.lifecycle_guard import GATEWAY_LIFECYCLE_BLOCK_MARKER
from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    _TOOL_STUBS,
    generate_hermes_tools_module,
)


def _stub_namespace(tool_names, call_result, transport="uds"):
    """Exec the generated module with ``_call`` replaced by a canned result."""
    source = generate_hermes_tools_module(list(tool_names), transport=transport)
    namespace: dict = {}
    exec(compile(source, "<hermes_tools>", "exec"), namespace)
    namespace["_call"] = lambda tool, args: call_result
    return namespace


def _blocked_result():
    """The exact shape tools/terminal_tool_guards._blocked_json returns."""
    return {
        "output": "",
        "exit_code": 1,
        "error": (
            "Blocked: launchctl submit/bootstrap is restricted inside a "
            "supervised gateway regardless of the job label."
        ),
        "status": "error",
        "blocked_by": GATEWAY_LIFECYCLE_BLOCK_MARKER,
    }


class TestStubRaisesOnBlock:
    def test_terminal_stub_raises_on_a_blocked_result(self):
        namespace = _stub_namespace(["terminal"], _blocked_result())
        blocked = namespace["ToolCallBlocked"]
        try:
            namespace["terminal"]("sudo launchctl bootstrap system /tmp/x.plist")
        except blocked as exc:
            assert "was blocked and did not run" in str(exc)
            assert "launchctl submit" in str(exc)
        else:
            raise AssertionError(
                "a blocked terminal() call returned instead of raising"
            )

    def test_blocked_marker_key_matches_the_producer(self):
        """Contract test: the stub's key is the guard's key.

        The stub is generated source and cannot import the guard, so the key
        is duplicated. Derive BOTH sides here rather than retyping either — a
        rename on the producer side must fail this test, not silently disarm
        the stub.
        """
        source = generate_hermes_tools_module(["terminal"])
        assert '_BLOCKED_MARKER_KEY = "blocked_by"' in source
        result = _blocked_result()
        assert result["blocked_by"] == GATEWAY_LIFECYCLE_BLOCK_MARKER
        namespace = _stub_namespace(["terminal"], result)
        assert namespace["_BLOCKED_MARKER_KEY"] in result

    def test_every_generated_stub_routes_through_the_guard(self):
        """Not just terminal(): every tool the sandbox exposes.

        Placeholder arguments are derived from each stub's own signature, so
        a newly-added tool is covered without editing this test.
        """
        names = sorted(SANDBOX_ALLOWED_TOOLS & set(_TOOL_STUBS))
        assert names, "no sandbox tool stubs to check"
        namespace = _stub_namespace(names, _blocked_result())
        blocked = namespace["ToolCallBlocked"]
        checked = []
        for name in names:
            func = namespace[name]
            args = [
                ["x"] if parameter.annotation is list else "x"
                for parameter in inspect.signature(func).parameters.values()
                if parameter.default is inspect.Parameter.empty
            ]
            try:
                func(*args)
            except blocked:
                checked.append(name)
                continue
            raise AssertionError(f"{name}() did not raise on a blocked result")
        assert checked == names

    def test_file_transport_stub_raises_too(self):
        """Remote backends generate a different header, same contract."""
        namespace = _stub_namespace(
            ["terminal"], _blocked_result(), transport="file"
        )
        blocked = namespace["ToolCallBlocked"]
        try:
            namespace["terminal"]("launchctl submit -l x -- /bin/true")
        except blocked:
            return
        raise AssertionError("file-transport stub returned a blocked result")


class TestNormalResultsUnaffected:
    def test_successful_result_passes_through_unchanged(self):
        ok = {"output": "hi\n", "exit_code": 0, "error": None}
        namespace = _stub_namespace(["terminal"], ok)
        assert namespace["terminal"]("echo hi") == ok

    def test_ordinary_nonzero_exit_is_not_a_block(self):
        """A command that RAN and failed must still return, not raise.

        The discriminating half: if the stub raised on any non-zero exit
        code, every legitimate `grep` miss or `exit 7` would explode.
        """
        failed = {"output": "", "exit_code": 7, "error": None}
        namespace = _stub_namespace(["terminal"], failed)
        assert namespace["terminal"]("exit 7") == failed

    def test_error_status_without_the_marker_is_not_a_block(self):
        """`status: error` alone is not a refusal — only the marker is."""
        errored = {
            "output": "",
            "exit_code": -1,
            "error": "Failed to execute command: boom",
            "status": "error",
        }
        namespace = _stub_namespace(["terminal"], errored)
        assert namespace["terminal"]("boom") == errored

    def test_non_dict_result_passes_through(self):
        namespace = _stub_namespace(["terminal"], "raw string result")
        assert namespace["terminal"]("echo hi") == "raw string result"

    def test_falsy_marker_value_is_not_a_block(self):
        result = {"output": "", "exit_code": 0, "blocked_by": ""}
        namespace = _stub_namespace(["terminal"], result)
        assert namespace["terminal"]("echo hi") == result


class TestGuardEnvelopeStampsTheMarker:
    def test_blocked_json_stamps_every_refusal(self):
        """The producer side: the single choke point stamps the marker."""
        from tools.terminal_tool_guards import _blocked_json

        payload = json.loads(_blocked_json("nope", "error"))
        assert payload["blocked_by"] == GATEWAY_LIFECYCLE_BLOCK_MARKER
        assert payload["exit_code"] == 1
        assert payload["status"] == "error"

    def test_blocked_json_output_drives_the_stub(self):
        """End-to-end across the seam: real envelope in, raise out."""
        from tools.terminal_tool_guards import _blocked_json

        namespace = _stub_namespace(
            ["terminal"], json.loads(_blocked_json("refused because X", "error"))
        )
        blocked = namespace["ToolCallBlocked"]
        try:
            namespace["terminal"]("launchctl submit -l x -- /bin/true")
        except blocked as exc:
            assert "refused because X" in str(exc)
            return
        raise AssertionError("real guard envelope did not raise in the stub")
