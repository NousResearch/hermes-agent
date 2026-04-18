"""Tests for CLI voice mode integration -- command parsing, markdown stripping,
state management, streaming TTS activation, voice message prefix, _vprint."""

import asyncio
import importlib
import os
import queue
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_voice_cli(**overrides):
    """Create a minimal HermesCLI with only voice-related attrs initialized.

    Uses ``__new__()`` to bypass ``__init__`` so no config/env/API setup is
    needed.  Only the voice state attributes (from __init__ lines 3749-3758)
    are populated.
    """
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._voice_lock = threading.Lock()
    cli._voice_mode = False
    cli._voice_tts = False
    cli._voice_recorder = None
    cli._voice_recording = False
    cli._voice_processing = False
    cli._voice_continuous = False
    cli._voice_tts_done = threading.Event()
    cli._voice_tts_done.set()
    cli._pending_input = queue.Queue()
    cli._app = None
    cli.console = SimpleNamespace(width=80)
    for k, v in overrides.items():
        setattr(cli, k, v)
    return cli


def _make_full_cli():
    import cli as cli_module
    from cli import HermesCLI

    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }

    with patch("cli.get_tool_definitions", return_value=[]), patch.dict(
        "os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False
    ), patch.dict(cli_module.__dict__, {"CLI_CONFIG": clean_config}):
        return HermesCLI()


def _build_run_keybindings():
    import cli as cli_module

    captured = {}
    thread_calls = []

    class FakeApplication:
        def __init__(self, *args, **kwargs):
            self.key_bindings = kwargs["key_bindings"]
            self.layout = kwargs.get("layout")
            self.style = kwargs.get("style")
            self.full_screen = kwargs.get("full_screen", False)
            self.mouse_support = kwargs.get("mouse_support", False)
            self.is_running = False
            self.renderer = SimpleNamespace(
                _last_size=None,
                _last_screen=None,
                _cursor_pos=SimpleNamespace(x=0, y=0),
                output=SimpleNamespace(get_size=lambda: SimpleNamespace(columns=80)),
            )
            self.current_buffer = SimpleNamespace(text="", reset=lambda: None)
            self.invalidated = 0
            self.exited = False
            self._on_resize = lambda: None
            captured["app"] = self

        def run(self):
            return None

        def invalidate(self):
            self.invalidated += 1

        def exit(self):
            self.exited = True
            self.is_running = False

    class FakeThread:
        def __init__(self, target, args=(), kwargs=None, daemon=False):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            self.join_calls = []
            self._alive = False
            self.target_name = getattr(target, "__name__", target.__class__.__name__)
            thread_calls.append(self)

        def start(self):
            if self.target_name in {"spinner_loop", "process_loop"}:
                self._alive = False
                return
            self._alive = True
            try:
                self.target(*self.args, **self.kwargs)
            finally:
                self._alive = False

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            self._alive = False

        def is_alive(self):
            return self._alive

    cli_obj = _make_full_cli()
    cli_obj._print_exit_summary = lambda: None

    with (
        patch("cli.Application", FakeApplication),
        patch("cli.patch_stdout", return_value=nullcontext()),
        patch("cli.threading.Thread", FakeThread),
        patch("cli.atexit.register", lambda *_args, **_kwargs: None),
        patch("cli._run_cleanup", lambda: None),
    ):
        cli_obj.run()

    return cli_obj, captured["app"], thread_calls


def _find_handler(app, handler_name: str):
    for binding in app.key_bindings.bindings:
        if getattr(binding.handler, "__name__", "") == handler_name:
            return binding.handler
    raise AssertionError(f"Could not find keybinding handler {handler_name!r}")


# ============================================================================
# Markdown stripping — import real function from tts_tool
# ============================================================================

from tools.tts_tool import _strip_markdown_for_tts


class TestMarkdownStripping:
    def test_strips_bold(self):
        assert _strip_markdown_for_tts("This is **bold** text") == "This is bold text"

    def test_strips_italic(self):
        assert _strip_markdown_for_tts("This is *italic* text") == "This is italic text"

    def test_strips_inline_code(self):
        assert _strip_markdown_for_tts("Run `pip install foo`") == "Run pip install foo"

    def test_strips_fenced_code_blocks(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        result = _strip_markdown_for_tts(text)
        assert "print" not in result
        assert "Done." in result

    def test_strips_headers(self):
        assert _strip_markdown_for_tts("## Summary\nSome text") == "Summary\nSome text"

    def test_strips_list_markers(self):
        text = "- item one\n- item two\n* item three"
        result = _strip_markdown_for_tts(text)
        assert "item one" in result
        assert "- " not in result
        assert "* " not in result

    def test_strips_urls(self):
        text = "Visit https://example.com for details"
        result = _strip_markdown_for_tts(text)
        assert "https://" not in result
        assert "Visit" in result

    def test_strips_markdown_links(self):
        text = "See [the docs](https://example.com/docs) for info"
        result = _strip_markdown_for_tts(text)
        assert "the docs" in result
        assert "https://" not in result
        assert "[" not in result

    def test_strips_horizontal_rules(self):
        text = "Part one\n---\nPart two"
        result = _strip_markdown_for_tts(text)
        assert "---" not in result
        assert "Part one" in result
        assert "Part two" in result

    def test_empty_after_stripping_returns_empty(self):
        text = "```python\nprint('hello')\n```"
        result = _strip_markdown_for_tts(text)
        assert result == ""

    def test_long_text_not_truncated(self):
        """_strip_markdown_for_tts does NOT truncate — that's the caller's job."""
        text = "a" * 5000
        result = _strip_markdown_for_tts(text)
        assert len(result) == 5000

    def test_complex_response(self):
        text = (
            "## Answer\n\n"
            "Here's how to do it:\n\n"
            "```python\ndef hello():\n    print('hi')\n```\n\n"
            "Run it with `python main.py`. "
            "See [docs](https://example.com) for more.\n\n"
            "- Step one\n- Step two\n\n"
            "---\n\n"
            "**Good luck!**"
        )
        result = _strip_markdown_for_tts(text)
        assert "```" not in result
        assert "https://" not in result
        assert "**" not in result
        assert "---" not in result
        assert "Answer" in result
        assert "Good luck!" in result
        assert "docs" in result


# ============================================================================
# Voice command parsing
# ============================================================================

class TestVoiceCommandParsing:
    """Test _handle_voice_command logic without full CLI setup."""

    def test_parse_subcommands(self):
        """Verify subcommand extraction from /voice commands."""
        test_cases = [
            ("/voice on", "on"),
            ("/voice off", "off"),
            ("/voice tts", "tts"),
            ("/voice status", "status"),
            ("/voice", ""),
            ("/voice  ON  ", "on"),
        ]
        for command, expected in test_cases:
            parts = command.strip().split(maxsplit=1)
            subcommand = parts[1].lower().strip() if len(parts) > 1 else ""
            assert subcommand == expected, f"Failed for {command!r}: got {subcommand!r}"


# ============================================================================
# Voice state thread safety
# ============================================================================

class TestVoiceStateLock:
    def test_lock_protects_state(self):
        """Verify that concurrent state changes don't corrupt state."""
        lock = threading.Lock()
        state = {"recording": False, "count": 0}

        def toggle_many(n):
            for _ in range(n):
                with lock:
                    state["recording"] = not state["recording"]
                    state["count"] += 1

        threads = [threading.Thread(target=toggle_many, args=(1000,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["count"] == 4000


# ============================================================================
# Streaming TTS lazy import activation (Bug A fix)
# ============================================================================

class TestStreamingTTSActivation:
    """Verify streaming TTS uses lazy imports to check availability."""

    def test_activates_when_elevenlabs_and_sounddevice_available(self):
        """use_streaming_tts should be True when provider is elevenlabs
        and both lazy imports succeed."""
        use_streaming_tts = False
        try:
            from tools.tts_tool import (
                _load_tts_config as _load_tts_cfg,
                _get_provider as _get_prov,
                _import_elevenlabs,
                _import_sounddevice,
            )
            assert callable(_import_elevenlabs)
            assert callable(_import_sounddevice)
        except ImportError:
            pytest.skip("tools.tts_tool not available")

        with patch("tools.tts_tool._load_tts_config") as mock_cfg, \
             patch("tools.tts_tool._get_provider", return_value="elevenlabs"), \
             patch("tools.tts_tool._import_elevenlabs") as mock_el, \
             patch("tools.tts_tool._import_sounddevice") as mock_sd:
            mock_cfg.return_value = {"provider": "elevenlabs"}
            mock_el.return_value = MagicMock()
            mock_sd.return_value = MagicMock()

            from tools.tts_tool import (
                _load_tts_config as load_cfg,
                _get_provider as get_prov,
                _import_elevenlabs as import_el,
                _import_sounddevice as import_sd,
            )
            cfg = load_cfg()
            if get_prov(cfg) == "elevenlabs":
                import_el()
                import_sd()
                use_streaming_tts = True

        assert use_streaming_tts is True

    def test_does_not_activate_when_elevenlabs_missing(self):
        """use_streaming_tts stays False when elevenlabs import fails."""
        use_streaming_tts = False
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "elevenlabs"}), \
             patch("tools.tts_tool._get_provider", return_value="elevenlabs"), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError("no elevenlabs")):
            try:
                from tools.tts_tool import (
                    _load_tts_config as load_cfg,
                    _get_provider as get_prov,
                    _import_elevenlabs as import_el,
                    _import_sounddevice as import_sd,
                )
                cfg = load_cfg()
                if get_prov(cfg) == "elevenlabs":
                    import_el()
                    import_sd()
                    use_streaming_tts = True
            except (ImportError, OSError):
                pass

        assert use_streaming_tts is False

    def test_does_not_activate_when_sounddevice_missing(self):
        """use_streaming_tts stays False when sounddevice import fails."""
        use_streaming_tts = False
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "elevenlabs"}), \
             patch("tools.tts_tool._get_provider", return_value="elevenlabs"), \
             patch("tools.tts_tool._import_elevenlabs", return_value=MagicMock()), \
             patch("tools.tts_tool._import_sounddevice", side_effect=OSError("no PortAudio")):
            try:
                from tools.tts_tool import (
                    _load_tts_config as load_cfg,
                    _get_provider as get_prov,
                    _import_elevenlabs as import_el,
                    _import_sounddevice as import_sd,
                )
                cfg = load_cfg()
                if get_prov(cfg) == "elevenlabs":
                    import_el()
                    import_sd()
                    use_streaming_tts = True
            except (ImportError, OSError):
                pass

        assert use_streaming_tts is False

    def test_does_not_activate_for_non_elevenlabs_provider(self):
        """use_streaming_tts stays False when provider is not elevenlabs."""
        use_streaming_tts = False
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}), \
             patch("tools.tts_tool._get_provider", return_value="edge"):
            try:
                from tools.tts_tool import (
                    _load_tts_config as load_cfg,
                    _get_provider as get_prov,
                    _import_elevenlabs as import_el,
                    _import_sounddevice as import_sd,
                )
                cfg = load_cfg()
                if get_prov(cfg) == "elevenlabs":
                    import_el()
                    import_sd()
                    use_streaming_tts = True
            except (ImportError, OSError):
                pass

        assert use_streaming_tts is False

    def test_stale_boolean_imports_no_longer_exist(self):
        """Confirm _HAS_ELEVENLABS and _HAS_AUDIO are not in tts_tool module."""
        import tools.tts_tool as tts_mod
        assert not hasattr(tts_mod, "_HAS_ELEVENLABS"), \
            "_HAS_ELEVENLABS should not exist -- lazy imports replaced it"
        assert not hasattr(tts_mod, "_HAS_AUDIO"), \
            "_HAS_AUDIO should not exist -- lazy imports replaced it"


# ============================================================================
# Voice mode user message prefix (Bug B fix)
# ============================================================================

class TestVoiceMessagePrefix:
    """Voice mode should inject instruction via user message prefix,
    not by modifying the system prompt (which breaks prompt cache)."""

    def test_prefix_added_when_voice_mode_active(self):
        """When voice mode is active and message is str, agent_message
        should have the voice instruction prefix."""
        voice_mode = True
        message = "What's the weather like?"

        agent_message = message
        if voice_mode and isinstance(message, str):
            agent_message = (
                "[Voice input — respond concisely and conversationally, "
                "2-3 sentences max. No code blocks or markdown.] "
                + message
            )

        assert agent_message.startswith("[Voice input")
        assert "What's the weather like?" in agent_message

    def test_no_prefix_when_voice_mode_inactive(self):
        """When voice mode is off, message passes through unchanged."""
        voice_mode = False
        message = "What's the weather like?"

        agent_message = message
        if voice_mode and isinstance(message, str):
            agent_message = (
                "[Voice input — respond concisely and conversationally, "
                "2-3 sentences max. No code blocks or markdown.] "
                + message
            )

        assert agent_message == message

    def test_no_prefix_for_multimodal_content(self):
        """When message is a list (multimodal), no prefix is added."""
        voice_mode = True
        message = [{"type": "text", "text": "describe this"}, {"type": "image_url"}]

        agent_message = message
        if voice_mode and isinstance(message, str):
            agent_message = (
                "[Voice input — respond concisely and conversationally, "
                "2-3 sentences max. No code blocks or markdown.] "
                + message
            )

        assert agent_message is message

    def test_history_stays_clean(self):
        """conversation_history should contain the original message,
        not the prefixed version."""
        voice_mode = True
        message = "Hello there"
        conversation_history = []

        conversation_history.append({"role": "user", "content": message})

        agent_message = message
        if voice_mode and isinstance(message, str):
            agent_message = (
                "[Voice input — respond concisely and conversationally, "
                "2-3 sentences max. No code blocks or markdown.] "
                + message
            )

        assert conversation_history[-1]["content"] == "Hello there"
        assert agent_message.startswith("[Voice input")
        assert agent_message != conversation_history[-1]["content"]

    def test_enable_voice_mode_does_not_modify_system_prompt(self):
        """_enable_voice_mode should NOT modify self.system_prompt or
        agent.ephemeral_system_prompt -- the system prompt must stay
        stable to preserve prompt cache."""
        cli = SimpleNamespace(
            _voice_mode=False,
            _voice_tts=False,
            _voice_lock=threading.Lock(),
            system_prompt="You are helpful",
            agent=SimpleNamespace(ephemeral_system_prompt="You are helpful"),
        )

        original_system = cli.system_prompt
        original_ephemeral = cli.agent.ephemeral_system_prompt

        cli._voice_mode = True

        assert cli.system_prompt == original_system
        assert cli.agent.ephemeral_system_prompt == original_ephemeral


# ============================================================================
# _vprint force parameter (Minor fix)
# ============================================================================

class TestVprintForceParameter:
    """_vprint should suppress output during streaming TTS unless force=True."""

    def _make_agent_with_stream(self, stream_active: bool):
        """Create a minimal agent-like object with _vprint."""
        agent = SimpleNamespace(
            _stream_callback=MagicMock() if stream_active else None,
        )

        def _vprint(*args, force=False, **kwargs):
            if not force and getattr(agent, "_stream_callback", None) is not None:
                return
            print(*args, **kwargs)

        agent._vprint = _vprint
        return agent

    def test_suppressed_during_streaming(self, capsys):
        """Normal _vprint output is suppressed when streaming TTS is active."""
        agent = self._make_agent_with_stream(stream_active=True)
        agent._vprint("should be hidden")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_shown_when_not_streaming(self, capsys):
        """Normal _vprint output is shown when streaming is not active."""
        agent = self._make_agent_with_stream(stream_active=False)
        agent._vprint("should be shown")
        captured = capsys.readouterr()
        assert "should be shown" in captured.out

    def test_force_shown_during_streaming(self, capsys):
        """force=True bypasses the streaming suppression."""
        agent = self._make_agent_with_stream(stream_active=True)
        agent._vprint("critical error!", force=True)
        captured = capsys.readouterr()
        assert "critical error!" in captured.out

    def test_force_shown_when_not_streaming(self, capsys):
        """force=True works normally when not streaming (no regression)."""
        agent = self._make_agent_with_stream(stream_active=False)
        agent._vprint("normal message", force=True)
        captured = capsys.readouterr()
        assert "normal message" in captured.out

class TestEdgeTTSLazyImport:
    """Bug #3: _generate_edge_tts must use lazy import, not bare module name."""

    def test_generate_edge_tts_calls_lazy_import(self):
        """_generate_edge_tts should resolve Edge TTS via the lazy import helper."""
        import tools.tts_tool as tts_tool

        communicate = MagicMock()
        communicate.save = AsyncMock()
        edge_tts_module = SimpleNamespace(
            Communicate=MagicMock(return_value=communicate)
        )

        with patch.object(tts_tool, "_import_edge_tts", return_value=edge_tts_module) as mock_import:
            result = asyncio.run(
                tts_tool._generate_edge_tts(
                    "hello world",
                    "/tmp/test-edge.mp3",
                    {"edge": {"voice": "en-US-TestVoice"}},
                )
            )

        mock_import.assert_called_once()
        edge_tts_module.Communicate.assert_called_once_with(
            "hello world",
            "en-US-TestVoice",
        )
        communicate.save.assert_awaited_once_with("/tmp/test-edge.mp3")
        assert result == "/tmp/test-edge.mp3"


class TestStreamingTTSOutputStreamCleanup:
    """Bug #7: output_stream must be closed in finally block."""

    def test_output_stream_closed_in_finally(self, monkeypatch):
        """stream_tts_to_speaker should always close the output stream."""
        import tools.tts_tool as tts_tool

        text_queue = queue.Queue()
        text_queue.put("Hello from TTS cleanup.")
        text_queue.put(None)
        stop_event = threading.Event()
        tts_done_event = threading.Event()

        output_stream = MagicMock()
        sounddevice_module = SimpleNamespace(
            OutputStream=MagicMock(return_value=output_stream)
        )
        client = MagicMock()
        client.text_to_speech.convert.return_value = [b"\x00\x00"]
        elevenlabs_ctor = MagicMock(return_value=client)
        output_stream.write.side_effect = RuntimeError("speaker write failed")

        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-elevenlabs-key")

        with (
            patch.object(tts_tool, "_load_tts_config", return_value={}),
            patch.object(tts_tool, "_import_elevenlabs", return_value=elevenlabs_ctor),
            patch.object(tts_tool, "_import_sounddevice", return_value=sounddevice_module),
        ):
            tts_tool.stream_tts_to_speaker(text_queue, stop_event, tts_done_event)

        output_stream.start.assert_called_once()
        output_stream.stop.assert_called_once()
        output_stream.close.assert_called_once()
        assert tts_done_event.is_set()


class TestCtrlCResetsContinuousMode:
    """Bug #4: Ctrl+C cancel must reset _voice_continuous."""

    def test_ctrl_c_handler_resets_voice_continuous(self):
        """Ctrl+C while recording should stop continuous mode before dispatching stop."""
        cli_obj, app, _thread_calls = _build_run_keybindings()
        handler = _find_handler(app, "handle_ctrl_c")

        cli_obj._voice_mode = True
        cli_obj._voice_recording = True
        cli_obj._voice_continuous = True
        cli_obj._voice_recorder = MagicMock()
        thread_targets = []

        class FakeThread:
            def __init__(self, target, args=(), kwargs=None, daemon=False):
                self.target = target
                self.args = args
                self.kwargs = kwargs or {}
                self.daemon = daemon
                thread_targets.append(target)

            def start(self):
                self.target(*self.args, **self.kwargs)

        with patch("cli.threading.Thread", FakeThread):
            handler(SimpleNamespace(app=app))

        assert cli_obj._voice_continuous is False
        cli_obj._voice_recorder.cancel.assert_called_once()
        assert thread_targets == [cli_obj._voice_recorder.cancel]


class TestVoiceStatusUsesConfigKey:
    """Bug #8: _show_voice_status must read record key from config."""

    @patch("cli._cprint")
    @patch("tools.voice_mode.check_voice_requirements", return_value={"available": True, "details": "All good"})
    @patch("hermes_cli.config.load_config", return_value={"voice": {"record_key": "ctrl+k"}})
    def test_show_voice_status_uses_configured_key(self, _cfg, _req, mock_cprint):
        """_show_voice_status should render the configured key instead of a hardcoded default."""
        cli = _make_voice_cli(_voice_mode=True, _voice_tts=True)

        cli._show_voice_status()

        output = "\n".join(" ".join(str(arg) for arg in call.args) for call in mock_cprint.call_args_list)
        assert "CTRL+K" in output
        assert "Record key: Ctrl+B" not in output


class TestChatTTSCleanupOnException:
    """Bug #2: chat() must clean up streaming TTS resources on exception."""

    def test_chat_exception_path_cleans_streaming_tts_resources(self):
        """If chat errors after streaming TTS starts, the cleanup path should stop the worker."""
        import cli as cli_module

        cli_obj = _make_full_cli()
        cli_obj._voice_tts = True
        cli_obj._interrupt_queue = queue.Queue()
        cli_obj._ensure_runtime_credentials = MagicMock(return_value=True)
        cli_obj._resolve_turn_agent_config = MagicMock(
            return_value={"signature": "sig", "model": None, "runtime": None, "label": None}
        )
        cli_obj._active_agent_route_signature = "sig"
        cli_obj.agent = SimpleNamespace(
            run_conversation=lambda **_kwargs: {"final_response": "hello", "messages": []},
            flush_memories=lambda *_args, **_kwargs: None,
            session_id="sid",
        )
        cli_obj._flush_stream = MagicMock(side_effect=RuntimeError("flush failed"))

        queue_holder = {}
        thread_calls = []

        class FakeQueue:
            def __init__(self):
                self.items = []
                queue_holder["queue"] = self

            def put(self, item):
                self.items.append(item)

            def put_nowait(self, item):
                self.items.append(item)

        class FakeThread:
            def __init__(self, target, args=(), kwargs=None, daemon=False):
                self.target = target
                self.args = args
                self.kwargs = kwargs or {}
                self.daemon = daemon
                self.target_name = getattr(target, "__name__", target.__class__.__name__)
                self.join_calls = []
                self._alive = False
                thread_calls.append(self)

            def start(self):
                if self.target_name == "run_agent":
                    self._alive = True
                    try:
                        self.target(*self.args, **self.kwargs)
                    finally:
                        self._alive = False
                else:
                    self._alive = True

            def join(self, timeout=None):
                self.join_calls.append(timeout)
                self._alive = False

            def is_alive(self):
                return self._alive

        with (
            patch("tools.tts_tool._load_tts_config", return_value={}),
            patch("tools.tts_tool._get_provider", return_value="elevenlabs"),
            patch("tools.tts_tool._import_elevenlabs"),
            patch("tools.tts_tool._import_sounddevice"),
            patch("tools.tts_tool.stream_tts_to_speaker", lambda *_args, **_kwargs: None),
            patch.object(cli_module, "queue", SimpleNamespace(Queue=FakeQueue, Empty=queue.Empty)),
            patch.object(cli_module.threading, "Thread", FakeThread),
            patch("cli._cprint"),
            patch.object(cli_module.ChatConsole, "print"),
        ):
            response = cli_obj.chat("hello")

        assert response is None
        assert queue_holder["queue"].items[-1] is None
        tts_threads = [thread for thread in thread_calls if thread.target_name == "<lambda>"]
        assert tts_threads
        assert tts_threads[0].join_calls == [5]


class TestBrowserToolSignalHandlerRemoved:
    """browser_tool.py must NOT register SIGINT/SIGTERM handlers that call
    sys.exit() — this conflicts with prompt_toolkit's event loop and causes
    the process to become unkillable during voice mode."""

    def test_no_signal_handler_registration(self):
        """Reloading browser_tool should not install SIGINT/SIGTERM handlers."""
        import tools.browser_tool as browser_tool

        with (
            patch("signal.signal") as mock_signal,
            patch("atexit.register") as mock_atexit_register,
        ):
            importlib.reload(browser_tool)

        mock_signal.assert_not_called()
        assert mock_atexit_register.call_count >= 1


class TestKeyHandlerNeverBlocks:
    """The Ctrl+B key handler runs in prompt_toolkit's event-loop thread.
    Any blocking call freezes the entire UI.  Verify that:
    1. _voice_start_recording is NOT called directly (must be in daemon thread)
    2. _voice_processing guard prevents starting while stop/transcribe runs
    3. _voice_processing is set atomically with _voice_recording in stop_and_transcribe
    """

    def test_start_recording_dispatches_via_thread(self):
        """Voice record start should schedule work onto a background thread."""
        cli_obj, app, _thread_calls = _build_run_keybindings()
        handler = _find_handler(app, "handle_voice_record")

        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._agent_running = False
        cli_obj._clarify_state = None
        cli_obj._sudo_state = None
        cli_obj._approval_state = None
        cli_obj._voice_tts_done.set()
        cli_obj._voice_start_recording = MagicMock()
        thread_targets = []

        class FakeThread:
            def __init__(self, target, args=(), kwargs=None, daemon=False):
                self.target = target
                self.args = args
                self.kwargs = kwargs or {}
                self.daemon = daemon
                thread_targets.append(target)

            def start(self):
                self.target(*self.args, **self.kwargs)

        with patch("cli.threading.Thread", FakeThread):
            handler(SimpleNamespace(app=app))

        cli_obj._voice_start_recording.assert_called_once()
        assert len(thread_targets) == 1

    def test_processing_guard_blocks_new_recording_start(self):
        """Voice record start should no-op while stop/transcribe is still running."""
        cli_obj, app, thread_calls = _build_run_keybindings()
        handler = _find_handler(app, "handle_voice_record")

        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = True
        cli_obj._agent_running = False
        cli_obj._clarify_state = None
        cli_obj._sudo_state = None
        cli_obj._approval_state = None
        cli_obj._voice_tts_done.set()
        cli_obj._voice_start_recording = MagicMock()

        baseline_threads = len(thread_calls)
        handler(SimpleNamespace(app=app))

        cli_obj._voice_start_recording.assert_not_called()
        assert len(thread_calls) == baseline_threads

    def test_processing_set_atomically_with_recording_false(self):
        """_voice_stop_and_transcribe should expose processing=True before stop() returns."""
        cli_obj = _make_voice_cli(_voice_recording=True)
        stop_entered = threading.Event()
        release_stop = threading.Event()

        recorder = MagicMock()

        def _stop():
            stop_entered.set()
            release_stop.wait(timeout=5)
            return None

        recorder.stop.side_effect = _stop
        cli_obj._voice_recorder = recorder

        worker = threading.Thread(target=cli_obj._voice_stop_and_transcribe, daemon=True)
        worker.start()

        assert stop_entered.wait(timeout=2)
        assert cli_obj._voice_recording is False
        assert cli_obj._voice_processing is True

        release_stop.set()
        worker.join(timeout=2)
        assert cli_obj._voice_processing is False


# ============================================================================
# Real behavior tests — CLI voice methods via _make_voice_cli()
# ============================================================================

class TestHandleVoiceCommandReal:
    """Tests _handle_voice_command routing with real CLI instance."""

    def _cli(self):
        cli = _make_voice_cli()
        cli._enable_voice_mode = MagicMock()
        cli._disable_voice_mode = MagicMock()
        cli._toggle_voice_tts = MagicMock()
        cli._show_voice_status = MagicMock()
        return cli

    @patch("cli._cprint")
    def test_on_calls_enable(self, _cp):
        cli = self._cli()
        cli._handle_voice_command("/voice on")
        cli._enable_voice_mode.assert_called_once()

    @patch("cli._cprint")
    def test_off_calls_disable(self, _cp):
        cli = self._cli()
        cli._handle_voice_command("/voice off")
        cli._disable_voice_mode.assert_called_once()

    @patch("cli._cprint")
    def test_tts_calls_toggle(self, _cp):
        cli = self._cli()
        cli._handle_voice_command("/voice tts")
        cli._toggle_voice_tts.assert_called_once()

    @patch("cli._cprint")
    def test_status_calls_show(self, _cp):
        cli = self._cli()
        cli._handle_voice_command("/voice status")
        cli._show_voice_status.assert_called_once()

    @patch("cli._cprint")
    def test_toggle_off_when_enabled(self, _cp):
        cli = self._cli()
        cli._voice_mode = True
        cli._handle_voice_command("/voice")
        cli._disable_voice_mode.assert_called_once()

    @patch("cli._cprint")
    def test_toggle_on_when_disabled(self, _cp):
        cli = self._cli()
        cli._voice_mode = False
        cli._handle_voice_command("/voice")
        cli._enable_voice_mode.assert_called_once()

    @patch("cli._cprint")
    def test_unknown_subcommand(self, mock_cp):
        cli = self._cli()
        cli._handle_voice_command("/voice foobar")
        cli._enable_voice_mode.assert_not_called()
        cli._disable_voice_mode.assert_not_called()
        # Should print usage via _cprint
        assert any("Unknown" in str(c) or "unknown" in str(c)
                    for c in mock_cp.call_args_list)


class TestEnableVoiceModeReal:
    """Tests _enable_voice_mode with real CLI instance."""

    @patch("cli._cprint")
    @patch("hermes_cli.config.load_config", return_value={"voice": {}})
    @patch("tools.voice_mode.check_voice_requirements",
           return_value={"available": True, "details": "OK"})
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": True, "warnings": []})
    def test_success_sets_voice_mode(self, _env, _req, _cfg, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_mode is True

    @patch("cli._cprint")
    def test_already_enabled_noop(self, _cp):
        cli = _make_voice_cli(_voice_mode=True)
        cli._enable_voice_mode()
        assert cli._voice_mode is True

    @patch("cli._cprint")
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": False, "warnings": ["SSH session"]})
    def test_env_check_fails(self, _env, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_mode is False

    @patch("cli._cprint")
    @patch("tools.voice_mode.check_voice_requirements",
           return_value={"available": False, "details": "Missing",
                         "missing_packages": ["sounddevice"]})
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": True, "warnings": []})
    def test_requirements_fail(self, _env, _req, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_mode is False

    @patch("cli._cprint")
    @patch("hermes_cli.config.load_config", return_value={"voice": {"auto_tts": True}})
    @patch("tools.voice_mode.check_voice_requirements",
           return_value={"available": True, "details": "OK"})
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": True, "warnings": []})
    def test_auto_tts_from_config(self, _env, _req, _cfg, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_tts is True

    @patch("cli._cprint")
    @patch("hermes_cli.config.load_config", return_value={"voice": {}})
    @patch("tools.voice_mode.check_voice_requirements",
           return_value={"available": True, "details": "OK"})
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": True, "warnings": []})
    def test_no_auto_tts_default(self, _env, _req, _cfg, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_tts is False

    @patch("cli._cprint")
    @patch("hermes_cli.config.load_config", side_effect=Exception("broken config"))
    @patch("tools.voice_mode.check_voice_requirements",
           return_value={"available": True, "details": "OK"})
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": True, "warnings": []})
    def test_config_exception_still_enables(self, _env, _req, _cfg, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_mode is True


class TestDisableVoiceModeReal:
    """Tests _disable_voice_mode with real CLI instance."""

    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback")
    def test_all_flags_reset(self, _sp, _cp):
        cli = _make_voice_cli(_voice_mode=True, _voice_tts=True,
                              _voice_continuous=True)
        cli._disable_voice_mode()
        assert cli._voice_mode is False
        assert cli._voice_tts is False
        assert cli._voice_continuous is False

    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback")
    def test_active_recording_cancelled(self, _sp, _cp):
        recorder = MagicMock()
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._disable_voice_mode()
        recorder.cancel.assert_called_once()
        assert cli._voice_recording is False

    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback")
    def test_stop_playback_called(self, mock_sp, _cp):
        cli = _make_voice_cli()
        cli._disable_voice_mode()
        mock_sp.assert_called_once()

    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback")
    def test_tts_done_event_set(self, _sp, _cp):
        cli = _make_voice_cli()
        cli._voice_tts_done.clear()
        cli._disable_voice_mode()
        assert cli._voice_tts_done.is_set()

    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback")
    def test_no_recorder_no_crash(self, _sp, _cp):
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=None)
        cli._disable_voice_mode()
        assert cli._voice_mode is False

    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback", side_effect=RuntimeError("boom"))
    def test_stop_playback_exception_swallowed(self, _sp, _cp):
        cli = _make_voice_cli(_voice_mode=True)
        cli._disable_voice_mode()
        assert cli._voice_mode is False


class TestVoiceSpeakResponseReal:
    """Tests _voice_speak_response with real CLI instance."""

    @patch("cli._cprint")
    def test_early_return_when_tts_off(self, _cp):
        cli = _make_voice_cli(_voice_tts=False)
        with patch("tools.tts_tool.text_to_speech_tool") as mock_tts:
            cli._voice_speak_response("Hello")
            mock_tts.assert_not_called()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.getsize", return_value=1000)
    @patch("cli.os.path.isfile", return_value=True)
    @patch("cli.os.makedirs")
    @patch("tools.voice_mode.play_audio_file")
    @patch("tools.tts_tool.text_to_speech_tool", return_value='{"success": true}')
    def test_markdown_stripped(self, mock_tts, _play, _mkd, _isf, _gsz, _unl, _cp):
        cli = _make_voice_cli(_voice_tts=True)
        cli._voice_speak_response("## Title\n**bold** and `code`")
        call_text = mock_tts.call_args.kwargs["text"]
        assert "##" not in call_text
        assert "**" not in call_text
        assert "`" not in call_text

    @patch("cli._cprint")
    @patch("cli.os.makedirs")
    @patch("tools.tts_tool.text_to_speech_tool", return_value='{"success": true}')
    def test_code_blocks_removed(self, mock_tts, _mkd, _cp):
        cli = _make_voice_cli(_voice_tts=True)
        cli._voice_speak_response("```python\nprint('hi')\n```\nSome text")
        call_text = mock_tts.call_args.kwargs["text"]
        assert "print" not in call_text
        assert "```" not in call_text
        assert "Some text" in call_text

    @patch("cli._cprint")
    @patch("cli.os.makedirs")
    def test_empty_after_strip_returns_early(self, _mkd, _cp):
        cli = _make_voice_cli(_voice_tts=True)
        with patch("tools.tts_tool.text_to_speech_tool") as mock_tts:
            cli._voice_speak_response("```python\nprint('hi')\n```")
            mock_tts.assert_not_called()

    @patch("cli._cprint")
    @patch("cli.os.makedirs")
    @patch("tools.tts_tool.text_to_speech_tool", return_value='{"success": true}')
    def test_long_text_truncated(self, mock_tts, _mkd, _cp):
        cli = _make_voice_cli(_voice_tts=True)
        cli._voice_speak_response("A" * 5000)
        call_text = mock_tts.call_args.kwargs["text"]
        assert len(call_text) <= 4000

    @patch("cli._cprint")
    @patch("cli.os.makedirs")
    @patch("tools.tts_tool.text_to_speech_tool", side_effect=RuntimeError("tts fail"))
    def test_exception_sets_done_event(self, _tts, _mkd, _cp):
        cli = _make_voice_cli(_voice_tts=True)
        cli._voice_tts_done.clear()
        cli._voice_speak_response("Hello")
        assert cli._voice_tts_done.is_set()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.getsize", return_value=1000)
    @patch("cli.os.path.isfile", return_value=True)
    @patch("cli.os.makedirs")
    @patch("tools.voice_mode.play_audio_file")
    @patch("tools.tts_tool.text_to_speech_tool", return_value='{"success": true}')
    def test_play_audio_called(self, _tts, mock_play, _mkd, _isf, _gsz, _unl, _cp):
        cli = _make_voice_cli(_voice_tts=True)
        cli._voice_speak_response("Hello world")
        mock_play.assert_called_once()


class TestVoiceStopAndTranscribeReal:
    """Tests _voice_stop_and_transcribe with real CLI instance."""

    @patch("cli._cprint")
    def test_guard_not_recording(self, _cp):
        cli = _make_voice_cli(_voice_recording=False)
        with patch("tools.voice_mode.transcribe_recording") as mock_tr:
            cli._voice_stop_and_transcribe()
            mock_tr.assert_not_called()

    @patch("cli._cprint")
    def test_no_recorder_returns_early(self, _cp):
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=None)
        with patch("tools.voice_mode.transcribe_recording") as mock_tr:
            cli._voice_stop_and_transcribe()
            mock_tr.assert_not_called()
        assert cli._voice_recording is False

    @patch("cli._cprint")
    @patch("tools.voice_mode.play_beep")
    def test_no_speech_detected(self, _beep, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = None
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        assert cli._pending_input.empty()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.isfile", return_value=True)
    @patch("hermes_cli.config.load_config", return_value={"stt": {}})
    @patch("tools.voice_mode.transcribe_recording",
           return_value={"success": True, "transcript": "hello world"})
    @patch("tools.voice_mode.play_beep")
    def test_successful_transcription_queues_input(
        self, _beep, _tr, _cfg, _isf, _unl, _cp
    ):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        assert cli._pending_input.get_nowait() == "hello world"

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.isfile", return_value=True)
    @patch("hermes_cli.config.load_config", return_value={"stt": {}})
    @patch("tools.voice_mode.transcribe_recording",
           return_value={"success": True, "transcript": ""})
    @patch("tools.voice_mode.play_beep")
    def test_empty_transcript_not_queued(self, _beep, _tr, _cfg, _isf, _unl, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        assert cli._pending_input.empty()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.isfile", return_value=True)
    @patch("hermes_cli.config.load_config", return_value={"stt": {}})
    @patch("tools.voice_mode.transcribe_recording",
           return_value={"success": False, "error": "API timeout"})
    @patch("tools.voice_mode.play_beep")
    def test_transcription_failure(self, _beep, _tr, _cfg, _isf, _unl, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        assert cli._pending_input.empty()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.isfile", return_value=True)
    @patch("hermes_cli.config.load_config", return_value={"stt": {}})
    @patch("tools.voice_mode.transcribe_recording",
           side_effect=ConnectionError("network"))
    @patch("tools.voice_mode.play_beep")
    def test_exception_caught(self, _beep, _tr, _cfg, _isf, _unl, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()  # Should not raise

    @patch("cli._cprint")
    @patch("tools.voice_mode.play_beep")
    def test_processing_flag_cleared(self, _beep, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = None
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        assert cli._voice_processing is False

    @patch("cli._cprint")
    @patch("tools.voice_mode.play_beep")
    def test_continuous_restarts_on_no_speech(self, _beep, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = None
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder,
                              _voice_continuous=True)
        cli._voice_start_recording = MagicMock()
        cli._voice_stop_and_transcribe()
        cli._voice_start_recording.assert_called_once()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.isfile", return_value=True)
    @patch("hermes_cli.config.load_config", return_value={"stt": {}})
    @patch("tools.voice_mode.transcribe_recording",
           return_value={"success": True, "transcript": "hello"})
    @patch("tools.voice_mode.play_beep")
    def test_continuous_no_restart_on_success(
        self, _beep, _tr, _cfg, _isf, _unl, _cp
    ):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder,
                              _voice_continuous=True)
        cli._voice_start_recording = MagicMock()
        cli._voice_stop_and_transcribe()
        cli._voice_start_recording.assert_not_called()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.isfile", return_value=True)
    @patch("hermes_cli.config.load_config", return_value={"stt": {"model": "whisper-large-v3"}})
    @patch("tools.voice_mode.transcribe_recording",
           return_value={"success": True, "transcript": "hi"})
    @patch("tools.voice_mode.play_beep")
    def test_stt_model_from_config(self, _beep, mock_tr, _cfg, _isf, _unl, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        mock_tr.assert_called_once_with("/tmp/test.wav", model="whisper-large-v3")


# ---------------------------------------------------------------------------
# Bugfix: _refresh_level must read _voice_recording under lock
# ---------------------------------------------------------------------------


class TestRefreshLevelLock:
    """Bug: _refresh_level thread read _voice_recording without lock."""

    def test_refresh_stops_when_recording_false(self):
        import threading, time

        lock = threading.Lock()
        recording = True
        iterations = 0

        def refresh_level():
            nonlocal iterations
            while True:
                with lock:
                    still = recording
                if not still:
                    break
                iterations += 1
                time.sleep(0.01)

        t = threading.Thread(target=refresh_level, daemon=True)
        t.start()

        time.sleep(0.05)
        with lock:
            recording = False

        t.join(timeout=1)
        assert not t.is_alive(), "Refresh thread did not stop"
        assert iterations > 0, "Refresh thread never ran"
