"""cua-driver subprocess environment scrubbing (#37878)."""

from __future__ import annotations

import os


class TestCuaEnvironmentScrubbing:
    """Verify that cua-driver subprocess environment is sanitized (issue #37878)."""

    def test_cua_session_sanitizes_provider_env_vars(self):
        """_CuaDriverSession lifecycle must sanitize sensitive env vars.

        The cua-driver MCP subprocess should not inherit Hermes-managed
        credentials or other sensitive environment variables — only
        runtime-required vars. Regression test for issue #37878.

        After the lifecycle-owner refactor, env scrubbing happens inside
        `_lifecycle_coro`; this test drives that coroutine directly with
        all the MCP/stdio plumbing mocked, captures the env arg passed
        to StdioServerParameters, and asserts the scrub contract.
        """
        from unittest.mock import MagicMock, patch, AsyncMock
        from tools.computer_use.cua_backend_session import _CuaDriverSession, _AsyncBridge
        import asyncio

        bridge = _AsyncBridge()
        session = _CuaDriverSession(bridge)

        captured_env: dict[str, str] = {}

        async def drive_lifecycle():
            test_env = {
                "OPENAI_API_KEY": "sk-secret",         # blocked
                "ANTHROPIC_API_KEY": "sk-ant-secret",  # blocked
                "PATH": "/usr/bin:/bin",               # safe
                "HOME": "/home/user",                  # safe
                "SAFE_VAR": "allowed",                 # safe
            }

            def capture_env(**kwargs):
                captured_env.update(kwargs.get("env", {}))
                # Return any sentinel — never actually used by the
                # patched stdio_client path below.
                return MagicMock()

            with patch.dict(os.environ, test_env, clear=True), \
                 patch("tools.computer_use.cua_backend_driver.resolve_cua_driver_cmd",
                       return_value="cua-driver"), \
                 patch("tools.computer_use.cua_backend_driver._resolve_mcp_invocation",
                       return_value=("cua-driver", ["mcp"])), \
                 patch("mcp.StdioServerParameters", side_effect=capture_env), \
                 patch("mcp.client.stdio.stdio_client") as mock_stdio, \
                 patch("mcp.ClientSession") as mock_session_class:

                # stdio_client(params) is used as `async with`.
                mock_stdio.return_value.__aenter__ = AsyncMock(
                    return_value=(MagicMock(), MagicMock()))
                mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

                # ClientSession(read, write) is used as `async with`.
                fake_session = MagicMock()
                fake_session.initialize = AsyncMock()
                # tools/list yields nothing — keeps _populate_capabilities
                # quiet without us needing to fully mock the response shape.
                fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=fake_session)
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Run the lifecycle with the shutdown event pre-set so it
                # tears down right after setup. We can't pre-set
                # session._shutdown_event because _lifecycle_coro creates
                # it inside the coroutine; instead, kick a background
                # task that signals as soon as the event exists.
                async def _signal_shutdown_when_ready():
                    for _ in range(200):  # ~1s budget
                        if session._shutdown_event is not None:
                            session._shutdown_event.set()
                            return
                        await asyncio.sleep(0.005)

                signal_task = asyncio.create_task(_signal_shutdown_when_ready())
                lifecycle_error: BaseException | None = None
                try:
                    await session._lifecycle_coro()
                except BaseException as exc:  # mocks may raise after the env capture
                    lifecycle_error = exc
                finally:
                    signal_task.cancel()
                    try:
                        await signal_task
                    except asyncio.CancelledError:
                        pass

                # The env capture is this test's contract. When it never landed the
                # lifecycle failed before it got to the scrub, and swallowing that
                # turns the real cause -- e.g. the home-I/O guard refusing PM's
                # `<checkout>/../manifest.json` probe on a default install -- into
                # the misleading PATH assert below. Fail on the cause instead.
                if not captured_env:
                    raise AssertionError(
                        "_lifecycle_coro() failed before the env capture landed"
                    ) from lifecycle_error

        asyncio.run(drive_lifecycle())

        # Blocked credentials must NOT have been passed to the subprocess.
        assert "OPENAI_API_KEY" not in captured_env, \
            "OPENAI_API_KEY should be stripped from cua-driver subprocess"
        assert "ANTHROPIC_API_KEY" not in captured_env, \
            "ANTHROPIC_API_KEY should be stripped from cua-driver subprocess"
        # At least one safe var must survive the scrub.
        assert "PATH" in captured_env or "SAFE_VAR" in captured_env, \
            "At least one safe environment variable should be preserved"
