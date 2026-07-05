"""Tests for the ``bearer_refresh_cmd`` config option in tools/mcp_tool.

Covers:
  * Happy path: command exits 0 with a clean token → header swapped in place.
  * Feature disabled: no command configured → returns False, no side effects.
  * Cooldown: second call within 60s window → skipped without spawning.
  * Command exit != 0 → returns False, header unchanged, stderr logged.
  * Output validation: empty / too-short / whitespace-in-token / error-string
    outputs are all rejected (script bug must not silently install garbage).
  * Headers field is not a dict → defensive bail-out.

The reconnect-loop integration (auth error → refresh fires → loop continues
without burning a retry) is covered by the existing reconnect tests +
manual VM smoke; here we test the helper in isolation so failures point
straight at the helper logic.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _BEARER_REFRESH_COOLDOWN_S


def _make_wrapper(headers: dict | None = None, cmd: str | None = None) -> MCPServerTask:
    """Build a wrapper with the bare minimum state needed to drive the
    helper. Real wrappers are spawned from the registry; for unit-isolation
    we set the attributes the helper reads."""
    w = MCPServerTask.__new__(MCPServerTask)
    cfg: dict = {"url": "http://example.test/mcp"}
    if headers is not None:
        cfg["headers"] = headers
    if cmd is not None:
        cfg["bearer_refresh_cmd"] = cmd
    w._config = cfg
    w.name = "test-server"
    return w


class _FakeProc:
    """Stand-in for asyncio.subprocess.Process with adjustable behavior."""
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"",
                 hang: bool = False):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(30)  # forces caller's wait_for to time out
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True


def _patch_subprocess(proc: _FakeProc):
    return patch(
        "tools.mcp_tool.asyncio.create_subprocess_shell",
        AsyncMock(return_value=proc),
    )


@pytest.mark.asyncio
async def test_happy_path_installs_token_and_replaces_header():
    w = _make_wrapper(
        headers={"Authorization": "Bearer old-stale-token"},
        cmd="/etc/hermes/refresh-bearer.sh super_admin",
    )
    proc = _FakeProc(returncode=0, stdout=b"new-fresh-token-abc123\n")
    with _patch_subprocess(proc):
        ok = await w._refresh_bearer_via_command()
    assert ok is True
    assert w._config["headers"]["Authorization"] == "Bearer new-fresh-token-abc123"


@pytest.mark.asyncio
async def test_feature_disabled_when_no_cmd_configured():
    w = _make_wrapper(headers={"Authorization": "Bearer original"}, cmd=None)
    with patch(
        "tools.mcp_tool.asyncio.create_subprocess_shell",
        AsyncMock(),
    ) as spawn:
        ok = await w._refresh_bearer_via_command()
    assert ok is False
    spawn.assert_not_called()  # never spawned
    assert w._config["headers"]["Authorization"] == "Bearer original"


@pytest.mark.asyncio
async def test_cooldown_skips_repeated_attempts():
    w = _make_wrapper(
        headers={"Authorization": "Bearer original"},
        cmd="/etc/hermes/mint.sh",
    )
    # First call: spawn fakeproc, succeed.
    proc = _FakeProc(returncode=0, stdout=b"new-token-1234567890\n")
    with _patch_subprocess(proc):
        assert await w._refresh_bearer_via_command() is True

    # Second call BEFORE cooldown elapses: must NOT spawn again.
    with patch(
        "tools.mcp_tool.asyncio.create_subprocess_shell",
        AsyncMock(),
    ) as spawn:
        ok = await w._refresh_bearer_via_command()
    assert ok is False
    spawn.assert_not_called()

    # Skip the cooldown clock forward and confirm the next call spawns again.
    w._bearer_refresh_last_attempt -= (_BEARER_REFRESH_COOLDOWN_S + 1)
    proc2 = _FakeProc(returncode=0, stdout=b"newer-token-9876543210\n")
    with _patch_subprocess(proc2):
        assert await w._refresh_bearer_via_command() is True
    assert w._config["headers"]["Authorization"] == "Bearer newer-token-9876543210"


@pytest.mark.asyncio
async def test_command_nonzero_exit_returns_false_and_keeps_header():
    w = _make_wrapper(
        headers={"Authorization": "Bearer original"},
        cmd="/etc/hermes/mint.sh staff",
    )
    proc = _FakeProc(returncode=1, stderr=b"KV unreachable\n")
    with _patch_subprocess(proc):
        ok = await w._refresh_bearer_via_command()
    assert ok is False
    assert w._config["headers"]["Authorization"] == "Bearer original"


@pytest.mark.asyncio
async def test_command_timeout_kills_process_and_returns_false():
    w = _make_wrapper(
        headers={"Authorization": "Bearer original"},
        cmd="/etc/hermes/mint.sh",
    )
    proc = _FakeProc(returncode=0, hang=True)
    with patch(
        "tools.mcp_tool.asyncio.create_subprocess_shell",
        AsyncMock(return_value=proc),
    ), patch("tools.mcp_tool._BEARER_REFRESH_TIMEOUT_S", 0.05):
        ok = await w._refresh_bearer_via_command()
    assert ok is False
    assert proc.killed is True
    assert w._config["headers"]["Authorization"] == "Bearer original"


@pytest.mark.parametrize("bad_output", [
    b"\n",                    # empty
    b"short\n",               # < 16 chars
    b"token with space\n",    # whitespace inside (script smearing two values together)
    b"ERROR: vault unreachable\n",   # error string emitted to stdout instead of token
    b"FAILED to mint token please retry later\n",
    b"<html>oauth error</html>\n",   # HTML error page from a misconfigured endpoint
    b"{\"error\": \"invalid_client\"}\n",  # JSON error blob
])
@pytest.mark.asyncio
async def test_bad_outputs_are_rejected(bad_output):
    w = _make_wrapper(
        headers={"Authorization": "Bearer original"},
        cmd="/etc/hermes/mint.sh",
    )
    proc = _FakeProc(returncode=0, stdout=bad_output)
    with _patch_subprocess(proc):
        ok = await w._refresh_bearer_via_command()
    assert ok is False, f"should have rejected: {bad_output!r}"
    assert w._config["headers"]["Authorization"] == "Bearer original"


@pytest.mark.asyncio
async def test_headers_field_not_a_dict_is_handled():
    w = _make_wrapper(headers=None, cmd="/etc/hermes/mint.sh")
    # Force a non-dict headers value (e.g. operator misconfig).
    w._config["headers"] = "Bearer literal-string-not-a-dict"
    proc = _FakeProc(returncode=0, stdout=b"fresh-token-aaaaaa\n")
    with _patch_subprocess(proc):
        ok = await w._refresh_bearer_via_command()
    assert ok is False
    assert w._config["headers"] == "Bearer literal-string-not-a-dict"


def test_is_auth_error_unwraps_exception_group():
    """The whole point of the patch: a 401 wrapped inside anyio's
    TaskGroup ExceptionGroup must still be detected as an auth error.
    Real-world signature seen in production: anyio.TaskGroup raises
    BaseExceptionGroup containing an httpx.HTTPStatusError(401)."""
    import httpx
    from tools.mcp_tool import _is_auth_error

    # Construct a 401 HTTPStatusError the same way httpx does at runtime.
    req = httpx.Request("POST", "http://example.test/mcp")
    resp = httpx.Response(401, request=req)
    inner = httpx.HTTPStatusError(
        "Unauthorized", request=req, response=resp,
    )

    # Direct: still detected (regression guard for the fast path).
    assert _is_auth_error(inner) is True

    # Single-level wrap (anyio TaskGroup shape).
    one_level = BaseExceptionGroup("unhandled errors in a TaskGroup", [inner])
    assert _is_auth_error(one_level) is True

    # Nested wrap (TaskGroup-of-TaskGroup, depth 3).
    nested = BaseExceptionGroup(
        "outer", [BaseExceptionGroup("inner", [BaseExceptionGroup("deep", [inner])])],
    )
    assert _is_auth_error(nested) is True

    # Mixed siblings: non-auth error alongside the 401 still detects.
    mixed = BaseExceptionGroup(
        "siblings",
        [ConnectionResetError("peer closed"), inner],
    )
    assert _is_auth_error(mixed) is True


def test_is_auth_error_rejects_non_auth_exception_groups():
    """A TaskGroup of NON-auth errors (e.g. ConnectionResetError) must
    NOT fire the auth path — that would force every transient blip
    through bearer_refresh_cmd unnecessarily."""
    from tools.mcp_tool import _is_auth_error

    grp = BaseExceptionGroup(
        "transient network",
        [ConnectionResetError("peer closed"), TimeoutError("read timeout")],
    )
    assert _is_auth_error(grp) is False


def test_is_auth_error_handles_non_401_http_status_in_group():
    """A 500 wrapped in TaskGroup must NOT register as auth error."""
    import httpx
    from tools.mcp_tool import _is_auth_error

    req = httpx.Request("POST", "http://example.test/mcp")
    resp_500 = httpx.Response(500, request=req)
    inner_500 = httpx.HTTPStatusError(
        "Server error", request=req, response=resp_500,
    )
    grp = BaseExceptionGroup("transport task group", [inner_500])
    assert _is_auth_error(grp) is False


@pytest.mark.asyncio
async def test_preserves_operator_header_casing():
    """If operator wrote 'authorization' (lowercase), the swap keeps that
    casing — don't surprise them with a renamed key."""
    w = _make_wrapper(
        headers={"authorization": "Bearer old"},
        cmd="/etc/hermes/mint.sh",
    )
    proc = _FakeProc(returncode=0, stdout=b"fresh-token-1234567890\n")
    with _patch_subprocess(proc):
        ok = await w._refresh_bearer_via_command()
    assert ok is True
    assert "authorization" in w._config["headers"]
    assert "Authorization" not in w._config["headers"]
    assert w._config["headers"]["authorization"] == "Bearer fresh-token-1234567890"
