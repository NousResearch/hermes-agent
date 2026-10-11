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
from unittest.mock import patch

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

    @pytest.mark.parametrize("wrap", [False, True], ids=["bare-stub", "adapter-wrapped-stub"])
    def test_repeat_probes_stay_resolvable_and_never_cache(self, wrap):
        """Probes go through _get_cached_client's inline store, not _store_cached_client,
        so the stub (bare, or wrapped in a Codex adapter whose leaf is the stub) used to
        land under the runtime key; the next probe then hit _compat_model() on the stub
        and check_vision_requirements() flipped to False for the process (#87654).
        """
        import agent.auxiliary_client as aux

        def _resolve(*a, **k):
            stub = aux._AuxProbeClientStub(api_key="k", base_url="https://x.invalid/v1")
            return (aux.CodexAuxiliaryClient(stub, "v/m") if wrap else stub), "v/m"

        aux.shutdown_cached_clients()
        with patch.object(aux, "resolve_provider_client", _resolve), aux.aux_probe_mode():
            answers = [aux._get_cached_client("probe-guard-test", "vendor/model")[0] is not None for _ in range(3)]
        assert answers == [True, True, True]
        with aux._client_cache_lock:
            assert not [k for k in aux._client_cache if k[1:2] == (False,) and k[-1] == "vendor/model"]

    def test_probe_mode_does_not_cache_wrapped_client_or_evict_runtime_client(self, tmp_path, monkeypatch):
        """A probe wrapper must not replace a usable runtime client in the shared cache."""
        import agent.auxiliary_client as aux

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(aux, "_CLIENT_CACHE_MAX_SIZE", 1)
        runtime_key = aux._client_cache_key("runtime", async_mode=False, model="runtime-model")
        probe_key = aux._client_cache_key("probe", async_mode=False, model="probe-model")
        # Construct the real SDK client without credentials or a network request; the wrapper is
        # the shape returned by Codex/Responses routes during an availability probe.
        runtime_client = aux.CodexAuxiliaryClient(
            aux._load_openai_cls()(api_key="test", base_url="http://127.0.0.1:9/v1"),
            "runtime-model",
        )

        builds = []

        def resolve_probe_wrapper(*_args, **_kwargs):
            client = aux.CodexAuxiliaryClient(
                aux._create_openai_client(api_key="test", base_url="http://127.0.0.1:9/v1"),
                "probe-model",
            )
            builds.append(client)
            return client, "probe-model"

        aux.shutdown_cached_clients()
        try:
            aux._store_cached_client(runtime_key, runtime_client, "runtime-model")
            monkeypatch.setattr(aux, "resolve_provider_client", resolve_probe_wrapper)
            with aux.aux_probe_mode():
                probe_client, probe_model = aux._get_cached_client("probe", "probe-model")

            assert isinstance(probe_client, aux.CodexAuxiliaryClient)
            assert probe_model == "probe-model"
            with aux._client_cache_lock:
                assert aux._client_cache[runtime_key][0] is runtime_client
                assert probe_key not in aux._client_cache
            real_client, _ = aux._get_cached_client("probe", "probe-model")
            assert len(builds) == 2
            assert real_client is not None
            assert real_client is not probe_client
            assert isinstance(probe_client._real_client, aux._AuxProbeClientStub)
            assert not isinstance(real_client._real_client, aux._AuxProbeClientStub)
            assert real_client._real_client.responses is not None
            cached_client, _ = aux._get_cached_client("probe", "probe-model")
            assert cached_client is real_client
            assert len(builds) == 2
        finally:
            aux.shutdown_cached_clients()

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
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout

    def test_lazy_symbol_getattr_resolves_via_ensure(self):
        import importlib.util
        from tools import mcp_tool

        if importlib.util.find_spec("mcp") is None:
            pytest.skip("mcp SDK not installed")
        # getattr through the module (what mock.patch does when saving the
        # original) must materialize the symbol instead of AttributeError.
        assert mcp_tool.StdioServerParameters is not None

class TestBannerUpdateCheckNonBlocking:
    def test_banner_does_not_block_on_pending_update_check(self):
        """When the prefetch hasn't finished, the banner path must return in
        well under the old 500ms blocking wait."""
        from hermes_cli import banner

        with patch.object(banner, "_update_check_done", threading.Event()), \
             patch.object(banner, "_deferred_update_notice_started", False):
            start = time.perf_counter()
            behind = banner.get_update_result(timeout=0.05)
            if behind is None and not banner._update_check_done.is_set():
                banner._defer_update_notice()
            elapsed = time.perf_counter() - start
        assert elapsed < 0.3, f"banner update check blocked {elapsed:.3f}s"

    def test_deferred_notice_prints_through_prompt_toolkit_renderer(self):
        """The late notice lands after patch_stdout owns stdout, where raw ESC bytes are
        sanitized into visible ``?[1;33m`` text (#83969). It must reach prompt_toolkit as a
        parsed ANSI fragment — never as a bare ``Console.print`` to stdout."""
        from hermes_cli import banner
        from prompt_toolkit.formatted_text import ANSI, to_formatted_text

        printed = []
        done = threading.Event()
        with patch.object(banner, "_update_check_done", done), \
             patch.object(banner, "_update_result", None), \
             patch.object(banner, "_deferred_update_notice_started", False), \
             patch("prompt_toolkit.print_formatted_text", side_effect=lambda *a, **k: printed.append(a[0])):
            banner._defer_update_notice(max_wait=5.0)
            banner._update_result = 3
            done.set()
            deadline = time.time() + 5
            while not printed and time.time() < deadline:
                time.sleep(0.02)
        assert printed, "deferred update notice never reached prompt_toolkit's renderer"
        assert isinstance(printed[0], ANSI)
        visible = "".join(text for _style, text, *_ in to_formatted_text(printed[0]))
        assert "3 commits behind" in visible
        assert "\x1b" not in visible and "[bold" not in visible
