"""Startup-latency regressions: probe-mode aux clients, lazy MCP SDK,
non-blocking banner update check.

These pin the CLI cold-start contract established in the sub-400ms pass:
- check_fn availability probes must not import the OpenAI SDK or build
  real HTTP clients (aux_probe_mode).
- tools/mcp_tool must not import the `mcp` SDK at module import time.
- build_welcome_banner must not block on the update-check prefetch.
"""

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class TestAuxProbeMode:
    def test_probe_mode_returns_stub_without_openai_import(self):
        import agent.auxiliary_client as aux

        with aux.aux_probe_mode():
            client = aux._create_openai_client(api_key="k", base_url="https://x.invalid/v1")
        assert isinstance(client, aux._AuxProbeClientStub)
        assert client.api_key == "k"

    def test_probe_stub_never_cached(self):
        import agent.auxiliary_client as aux

        stub = aux._AuxProbeClientStub()
        key = ("probe-test", False, "", "", "", (), False, "", None, "m")
        aux._store_cached_client(key, stub, "m")
        with aux._client_cache_lock:
            assert key not in aux._client_cache

    def test_probe_stub_raises_on_runtime_use(self):
        import agent.auxiliary_client as aux

        stub = aux._AuxProbeClientStub()
        with pytest.raises(RuntimeError, match="availability checks only"):
            _ = stub.chat

    def test_probe_mode_is_scoped_and_reentrant(self):
        import agent.auxiliary_client as aux

        assert not aux._aux_probe_active()
        with aux.aux_probe_mode():
            assert aux._aux_probe_active()
            with aux.aux_probe_mode():
                assert aux._aux_probe_active()
            # inner exit must not clear the outer scope
            assert aux._aux_probe_active()
        assert not aux._aux_probe_active()

    def test_probe_mode_is_thread_local(self):
        import agent.auxiliary_client as aux

        seen = {}

        def other_thread():
            seen["active"] = aux._aux_probe_active()

        with aux.aux_probe_mode():
            t = threading.Thread(target=other_thread)
            t.start()
            t.join()
        assert seen["active"] is False

    def test_maybe_wrap_anthropic_passes_stub_through(self):
        import agent.auxiliary_client as aux

        stub = aux._AuxProbeClientStub(base_url="https://api.anthropic.com")
        out = aux._maybe_wrap_anthropic(stub, "m", "key", "https://api.anthropic.com")
        assert out is stub

    def test_to_async_client_passes_stub_through(self):
        import agent.auxiliary_client as aux

        stub = aux._AuxProbeClientStub()
        client, model = aux._to_async_client(stub, "m")
        assert client is stub
        assert model == "m"


class TestVisionCheckUsesProbeMode:
    def test_check_vision_requirements_enters_probe_mode(self):
        from tools import vision_tools
        import agent.auxiliary_client as aux

        states = []

        def fake_resolver(*a, **k):
            states.append(aux._aux_probe_active())
            return ("nous", aux._AuxProbeClientStub(), "m")

        with patch.object(aux, "resolve_vision_provider_client", fake_resolver):
            assert vision_tools.check_vision_requirements() is True
        assert states and all(states)


class TestLazyMcpSdk:
    def test_module_import_does_not_import_mcp_sdk(self):
        """Importing tools.mcp_tool must not pull in the `mcp` package."""
        import subprocess

        code = (
            "import sys; sys.modules.pop('mcp', None); "
            "import tools.mcp_tool; "
            "assert 'mcp' not in sys.modules, 'mcp imported eagerly'; "
            "print('ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout

    def test_availability_flag_reflects_find_spec(self):
        import importlib.util
        from tools import mcp_tool

        expected = importlib.util.find_spec("mcp") is not None
        assert mcp_tool._MCP_AVAILABLE is expected

    def test_ensure_mcp_sdk_binds_symbols(self):
        import importlib.util
        from tools import mcp_tool

        if importlib.util.find_spec("mcp") is None:
            pytest.skip("mcp SDK not installed")
        assert mcp_tool._ensure_mcp_sdk() is True
        assert mcp_tool.ClientSession is not None
        assert mcp_tool.stdio_client is not None

    def test_ensure_respects_patched_unavailable(self):
        from tools import mcp_tool

        with patch.object(mcp_tool, "_MCP_AVAILABLE", False):
            assert mcp_tool._ensure_mcp_sdk() is False

    def test_lazy_symbol_getattr_resolves_via_ensure(self):
        import importlib.util
        from tools import mcp_tool

        if importlib.util.find_spec("mcp") is None:
            pytest.skip("mcp SDK not installed")
        # getattr through the module (what mock.patch does when saving the
        # original) must materialize the symbol instead of AttributeError.
        assert getattr(mcp_tool, "StdioServerParameters") is not None


class TestBannerUpdateCheckNonBlocking:
    def test_banner_does_not_block_on_pending_update_check(self):
        """When the prefetch hasn't finished, the banner path must return in
        well under the old 500ms blocking wait."""
        import hermes_cli.banner as banner

        class _NullConsole:
            def print(self, *a, **k):
                pass

        with patch.object(banner, "_update_check_done", threading.Event()), \
             patch.object(banner, "_deferred_update_notice_started", False):
            start = time.perf_counter()
            behind = banner.get_update_result(timeout=0.05)
            if behind is None and not banner._update_check_done.is_set():
                banner._defer_update_notice(_NullConsole())
            elapsed = time.perf_counter() - start
        assert elapsed < 0.3, f"banner update check blocked {elapsed:.3f}s"

    def test_deferred_notice_prints_when_result_lands(self):
        import hermes_cli.banner as banner

        printed = []
        console = MagicMock()

        done = threading.Event()
        with patch.object(banner, "_update_check_done", done), \
             patch.object(banner, "_update_result", None), \
             patch.object(banner, "_deferred_update_notice_started", False), \
             patch("builtins.print", side_effect=printed.append):
            banner._defer_update_notice(console, max_wait=5.0)
            banner._update_result = 3
            done.set()
            deadline = time.time() + 5
            while not printed and time.time() < deadline:
                time.sleep(0.02)
        assert printed, "deferred update notice never printed"
        assert "3 commits behind" in printed[0]

    def test_deferred_notice_uses_prompt_toolkit_safe_printer(self):
        """Late Rich output must not write ANSI into patch_stdout's proxy."""
        import hermes_cli.banner as banner

        printed = []

        unsafe_console = MagicMock()
        unsafe_console.print.side_effect = AssertionError(
            "raw Rich console must not handle late notice"
        )

        done = threading.Event()
        with patch.object(banner, "_update_check_done", done), \
             patch.object(banner, "_update_result", None), \
             patch.object(banner, "_deferred_update_notice_started", False), \
             patch("builtins.print", side_effect=printed.append):
            banner._defer_update_notice(
                unsafe_console,
                max_wait=5.0,
            )
            banner._update_result = 922
            done.set()
            deadline = time.time() + 5
            while not printed and time.time() < deadline:
                time.sleep(0.02)

        assert printed, "safe deferred printer was never called"
        assert "922 commits behind" in printed[0]
        assert "?[" not in printed[0]
        assert "\x1b[" not in printed[0]
        assert "[bold" not in printed[0]

    def test_deferred_notice_silent_when_up_to_date(self):
        import hermes_cli.banner as banner

        printed = []

        class _Console:
            def print(self, msg, *a, **k):
                printed.append(msg)

        done = threading.Event()
        with patch.object(banner, "_update_check_done", done), \
             patch.object(banner, "_update_result", 0), \
             patch.object(banner, "_deferred_update_notice_started", False):
            banner._defer_update_notice(_Console(), max_wait=2.0)
            done.set()
            time.sleep(0.3)
        assert not printed
