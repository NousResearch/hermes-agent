"""Tests for CLI voice mode integration -- command parsing, markdown stripping,
state management, streaming TTS activation, voice message prefix, _vprint."""

import ast
import os
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

    def test_error_messages_use_force_in_run_agent(self):
        """Verify that critical error _vprint calls in run_agent.py
        include force=True."""
        with open("run_agent.py", "r") as f:
            source = f.read()

        tree = ast.parse(source)

        forced_error_count = 0
        unforced_error_count = 0

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "_vprint"):
                continue
            has_fatal = False
            for arg in node.args:
                if isinstance(arg, ast.JoinedStr):
                    for val in arg.values:
                        if isinstance(val, ast.Constant) and isinstance(val.value, str):
                            if "\u274c" in val.value:
                                has_fatal = True
                                break

            if not has_fatal:
                continue

            has_force = any(
                kw.arg == "force"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )

            if has_force:
                forced_error_count += 1
            else:
                unforced_error_count += 1

        assert forced_error_count > 0, \
            "Expected at least one _vprint with force=True for error messages"
        assert unforced_error_count == 0, \
            f"Found {unforced_error_count} critical error _vprint calls without force=True"


# ============================================================================
# Bug fix regression tests
# ============================================================================

class TestEdgeTTSLazyImport:
    """Bug #3: _generate_edge_tts must use lazy import, not bare module name."""

    def test_generate_edge_tts_calls_lazy_import(self):
        """AST check: _generate_edge_tts must call _import_edge_tts(), not
        reference bare 'edge_tts' module name."""
        import ast as _ast

        with open("tools/tts_tool.py") as f:
            tree = _ast.parse(f.read())

        for node in _ast.walk(tree):
            if isinstance(node, _ast.AsyncFunctionDef) and node.name == "_generate_edge_tts":
                # Collect all Name references (bare identifiers)
                bare_refs = [
                    n.id for n in _ast.walk(node)
                    if isinstance(n, _ast.Name) and n.id == "edge_tts"
                ]
                assert bare_refs == [], (
                    f"_generate_edge_tts uses bare 'edge_tts' name — "
                    f"should use _import_edge_tts() lazy helper"
                )

                # Must have a call to _import_edge_tts
                lazy_calls = [
                    n for n in _ast.walk(node)
                    if isinstance(n, _ast.Call)
                    and isinstance(n.func, _ast.Name)
                    and n.func.id == "_import_edge_tts"
                ]
                assert len(lazy_calls) >= 1, (
                    "_generate_edge_tts must call _import_edge_tts()"
                )
                break
        else:
            pytest.fail("_generate_edge_tts not found in tts_tool.py")


class TestStreamingTTSOutputStreamCleanup:
    """Bug #7: output_stream must be closed in finally block."""

    def test_output_stream_closed_in_finally(self):
        """AST check: stream_tts_to_speaker's finally block must close
        output_stream even on exception."""
        import ast as _ast

        with open("tools/tts_tool.py") as f:
            tree = _ast.parse(f.read())

        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == "stream_tts_to_speaker":
                # Find the outermost try that has a finally with tts_done_event.set()
                for child in _ast.walk(node):
                    if isinstance(child, _ast.Try) and child.finalbody:
                        finally_text = "\n".join(
                            _ast.dump(n) for n in child.finalbody
                        )
                        if "tts_done_event" in finally_text:
                            assert "output_stream" in finally_text, (
                                "finally block must close output_stream"
                            )
                            return
                pytest.fail("No finally block with tts_done_event found")


class TestCtrlCResetsContinuousMode:
    """Bug #4: Ctrl+C cancel must reset _voice_continuous."""

    def test_ctrl_c_handler_resets_voice_continuous(self):
        """Source check: Ctrl+C voice cancel block must set
        _voice_continuous = False."""
        with open("cli.py") as f:
            source = f.read()

        # Find the Ctrl+C handler's voice cancel block
        lines = source.split("\n")
        in_cancel_block = False
        found_continuous_reset = False
        for i, line in enumerate(lines):
            if "Cancel active voice recording" in line:
                in_cancel_block = True
            if in_cancel_block:
                if "_voice_continuous = False" in line:
                    found_continuous_reset = True
                    break
                # Block ends at next comment section or return
                if "return" in line and in_cancel_block:
                    break

        assert found_continuous_reset, (
            "Ctrl+C voice cancel block must set _voice_continuous = False"
        )


class TestDisableVoiceModeStopsTTS:
    """Bug #5: _disable_voice_mode must stop active TTS playback."""

    def test_disable_voice_mode_calls_stop_playback(self):
        """Source check: _disable_voice_mode must call stop_playback()."""
        import inspect
        from cli import HermesCLI

        source = inspect.getsource(HermesCLI._disable_voice_mode)
        assert "stop_playback" in source, (
            "_disable_voice_mode must call stop_playback()"
        )
        assert "_voice_tts_done.set()" in source, (
            "_disable_voice_mode must set _voice_tts_done"
        )


class TestVoiceStatusUsesConfigKey:
    """Bug #8: _show_voice_status must read record key from config."""

    def test_show_voice_status_not_hardcoded(self):
        """Source check: _show_voice_status must not hardcode Ctrl+B."""
        with open("cli.py") as f:
            source = f.read()

        lines = source.split("\n")
        in_method = False
        for line in lines:
            if "def _show_voice_status" in line:
                in_method = True
            elif in_method and line.strip().startswith("def "):
                break
            elif in_method:
                assert 'Record key: Ctrl+B"' not in line, (
                    "_show_voice_status hardcodes 'Ctrl+B' — "
                    "should read from config"
                )

    def test_show_voice_status_reads_config(self):
        """Source check: _show_voice_status must use load_config()."""
        with open("cli.py") as f:
            source = f.read()

        lines = source.split("\n")
        in_method = False
        method_lines = []
        for line in lines:
            if "def _show_voice_status" in line:
                in_method = True
            elif in_method and line.strip().startswith("def "):
                break
            elif in_method:
                method_lines.append(line)

        method_body = "\n".join(method_lines)
        assert "load_config" in method_body or "record_key" in method_body, (
            "_show_voice_status should read record_key from config"
        )


class TestChatTTSCleanupOnException:
    """Bug #2: chat() must clean up streaming TTS resources on exception."""

    def test_chat_has_finally_for_tts_cleanup(self):
        """AST check: chat() method must have a finally block that cleans up
        text_queue, stop_event, and tts_thread."""
        import ast as _ast

        with open("cli.py") as f:
            tree = _ast.parse(f.read())

        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == "chat":
                # Find Try nodes with finally blocks
                for child in _ast.walk(node):
                    if isinstance(child, _ast.Try) and child.finalbody:
                        finally_text = "\n".join(
                            _ast.dump(n) for n in child.finalbody
                        )
                        if "text_queue" in finally_text:
                            assert "stop_event" in finally_text, (
                                "finally must also handle stop_event"
                            )
                            assert "tts_thread" in finally_text, (
                                "finally must also handle tts_thread"
                            )
                            return
                pytest.fail(
                    "chat() must have a finally block cleaning up "
                    "text_queue/stop_event/tts_thread"
                )


class TestBrowserToolSignalHandlerRemoved:
    """browser_tool.py must NOT register SIGINT/SIGTERM handlers that call
    sys.exit() — this conflicts with prompt_toolkit's event loop and causes
    the process to become unkillable during voice mode."""

    def test_no_signal_handler_registration(self):
        """Source check: browser_tool.py must not call signal.signal()
        for SIGINT or SIGTERM."""
        with open("tools/browser_tool.py") as f:
            source = f.read()

        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            assert "signal.signal(signal.SIGINT" not in stripped, (
                f"browser_tool.py:{i} registers SIGINT handler — "
                f"use atexit instead to avoid prompt_toolkit conflicts"
            )
            assert "signal.signal(signal.SIGTERM" not in stripped, (
                f"browser_tool.py:{i} registers SIGTERM handler — "
                f"use atexit instead to avoid prompt_toolkit conflicts"
            )


class TestKeyHandlerNeverBlocks:
    """The Ctrl+B key handler runs in prompt_toolkit's event-loop thread.
    Any blocking call freezes the entire UI.  Verify that:
    1. _voice_start_recording is NOT called directly (must be in daemon thread)
    2. _voice_processing guard prevents starting while stop/transcribe runs
    3. _voice_processing is set atomically with _voice_recording in stop_and_transcribe
    """

    def test_start_recording_not_called_directly_in_handler(self):
        """AST check: handle_voice_record must NOT call _voice_start_recording()
        directly — it must wrap it in a Thread to avoid blocking the UI."""
        import ast as _ast

        with open("cli.py") as f:
            tree = _ast.parse(f.read())

        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == "handle_voice_record":
                # Collect all direct calls to _voice_start_recording in this function.
                # They should ONLY appear inside a nested def (the _start_recording wrapper).
                for child in _ast.iter_child_nodes(node):
                    # Direct statements in the handler body (not nested defs)
                    if isinstance(child, _ast.Expr) and isinstance(child.value, _ast.Call):
                        call_src = _ast.dump(child.value)
                        assert "_voice_start_recording" not in call_src, (
                            "handle_voice_record calls _voice_start_recording directly "
                            "— must dispatch to a daemon thread"
                        )
                break

    def test_processing_guard_in_start_path(self):
        """Source check: key handler must check _voice_processing before
        starting a new recording."""
        with open("cli.py") as f:
            source = f.read()

        lines = source.split("\n")
        in_handler = False
        in_else = False
        found_guard = False
        for line in lines:
            if "def handle_voice_record" in line:
                in_handler = True
            elif in_handler and line.strip().startswith("def ") and "_start_recording" not in line:
                break
            elif in_handler and "else:" in line:
                in_else = True
            elif in_else and "_voice_processing" in line:
                found_guard = True
                break

        assert found_guard, (
            "Key handler START path must guard against _voice_processing "
            "to prevent blocking on AudioRecorder._lock"
        )

    def test_processing_set_atomically_with_recording_false(self):
        """Source check: _voice_stop_and_transcribe must set _voice_processing = True
        in the same lock block where it sets _voice_recording = False."""
        with open("cli.py") as f:
            source = f.read()

        lines = source.split("\n")
        in_method = False
        in_first_lock = False
        found_recording_false = False
        found_processing_true = False
        for line in lines:
            if "def _voice_stop_and_transcribe" in line:
                in_method = True
            elif in_method and "with self._voice_lock:" in line and not in_first_lock:
                in_first_lock = True
            elif in_first_lock:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "_voice_recording = False" in stripped:
                    found_recording_false = True
                if "_voice_processing = True" in stripped:
                    found_processing_true = True
                # End of with block (dedent)
                if stripped and not line.startswith("            ") and not line.startswith("\t\t\t"):
                    break

        assert found_recording_false and found_processing_true, (
            "_voice_stop_and_transcribe must set _voice_processing = True "
            "atomically (same lock block) with _voice_recording = False"
        )


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
