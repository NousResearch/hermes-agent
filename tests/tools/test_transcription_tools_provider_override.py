"""CAF conversion, OpenAI retry, and explicit provider override contracts."""
import os
import sys
import struct
import subprocess
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


pytest_plugins = ["tests.tools.test_transcription_tools"]

pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")

class TestCafConversion:
    """Tests for _convert_caf_to_wav and CAF dispatch in transcribe_audio."""

    def test_convert_caf_with_ffmpeg(self, tmp_path, monkeypatch):
        """_convert_caf_to_wav uses ffmpeg when available."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)
        work_dir = tmp_path / "converted"
        work_dir.mkdir()
        wav_path = str(work_dir / "voice.wav")

        def fake_run(cmd, **kwargs):
            Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00")
            return MagicMock(returncode=0)

        monkeypatch.setattr(
            "tools.transcription_audio._find_ffmpeg_binary",
            lambda: "/usr/bin/ffmpeg",
        )
        monkeypatch.setattr(subprocess, "run", fake_run)

        from tools.transcription_tools import _convert_caf_to_wav
        result = _convert_caf_to_wav(str(caf_path), str(work_dir))
        assert result == wav_path
        assert Path(result).exists()


    def test_transcribe_caf_not_converted_for_local(self, tmp_path, monkeypatch):
        """CAF conversion is skipped for local provider (native handling)."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider",
                   return_value="local"), \
             patch("tools.transcription_tools._convert_caf_to_wav") as mock_convert, \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(caf_path))

        assert result["success"] is True
        mock_convert.assert_not_called()

    @pytest.mark.parametrize("outcome", ["success", "provider-error"])
    def test_caf_conversion_preserves_neighbors_and_removes_owned_output(
        self, tmp_path, monkeypatch, outcome
    ):
        """Cloud CAF conversion must not clobber a sibling ``<stem>.wav`` nor
        leave its converted output behind, whether the provider succeeds or
        raises."""
        from tools import transcription_audio as audio
        from tools import transcription_tools as stt

        source = tmp_path / "voice.caf"
        source.write_bytes(b"caff fixture")
        neighbor = source.with_suffix(".wav")
        neighbor.write_bytes(b"existing recording")
        outputs = []
        monkeypatch.setattr(stt, "_load_stt_config", lambda: {
            "provider": "groq", "cloud_trim_silence": False,
        })
        monkeypatch.setattr(audio, "_find_ffmpeg_binary", lambda: "ffmpeg")

        def encode(command, **_kwargs):
            output = Path(command[-1])
            outputs.append(output)
            output.write_bytes(b"converted recording")

        def transcribe(file_path, *_args):
            assert Path(file_path).read_bytes() == b"converted recording"
            if outcome == "provider-error":
                raise RuntimeError("transcription failed")
            return {"success": True, "transcript": "hello"}

        monkeypatch.setattr(audio, "_run_quiet", encode)
        monkeypatch.setattr(stt, "_dispatch_stt_provider", transcribe)
        if outcome == "provider-error":
            with pytest.raises(RuntimeError, match="transcription failed"):
                stt.transcribe_audio(str(source))
        else:
            assert stt.transcribe_audio(str(source))["success"] is True
        assert source.read_bytes() == b"caff fixture"
        assert neighbor.read_bytes() == b"existing recording"
        assert outputs and all(
            not path.exists() and not path.parent.exists() for path in outputs
        )


class TestTranscribeCredentialReadGuard:
    """transcribe_audio must refuse credential/secret stores before dispatch."""

    def test_transcribe_audio_blocks_credential_read(self, tmp_path):
        """A ``.env`` (secret-bearing) file is refused up front, so its
        plaintext is never shipped to an external STT provider — mirroring the
        read guard added to image-gen (587be5b5b) and xAI video-gen
        (104232979)."""
        from tools.transcription_tools import transcribe_audio
        from agent.file_safety import get_read_block_error

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-secret\n")

        expected = get_read_block_error(str(env_file))
        assert expected, "test setup: a .env file should be read-blocked"

        result = transcribe_audio(str(env_file))

        assert result["success"] is False
        # The error is the shared read-guard message, not an audio-validation
        # or provider error — proving the guard fired before dispatch.
        assert result["error"] == expected


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("progress", [True, False])
def test_command_stt_idle_timeout_preserves_transcription_contract(tmp_path, progress):
    import shlex
    from tools.transcription_command import _transcribe_command_stt

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture audio")
    script = tmp_path / "transcribe.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "assert Path(sys.argv[1]).read_bytes() == b'fixture audio'\n"
        + ("for i in range(6):\n    print('progress', file=sys.stderr, flush=True)\n    time.sleep(.6)\n"
           if progress else "time.sleep(30)\n")
        + "Path(sys.argv[2]).write_text('actual transcript', encoding='utf-8')\n",
        encoding="utf-8",
    )
    args = [sys.executable, "-u", str(script)]
    command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
    result = _transcribe_command_stt(str(audio), "probe", {
        "command": command + " {input_path} {output_path}", "timeout": 2,
    }, {})
    assert result["success"] is progress
    assert result["provider"] == "probe"
    if progress:
        assert result["transcript"] == "actual transcript"
    else:
        assert "STT command provider 'probe' timed out after 2s" in result["error"]


# ============================================================================
# Explicit openai selection keeps its selection-specific error (#93045)
# ============================================================================

class TestExplicitOpenaiSelectionError:
    """A managed-route outage must not be reported as generic setup guidance.

    When ``_resolve_openai_audio_client_config()`` raises its
    selection-specific ValueError (managed openai-audio gateway unavailable,
    with the ``hermes tools`` remediation for managed-Nous users), the old
    boolean probe flattened it into False — the log said "no API key" and
    the transcription result returned the all-provider install hint,
    pointing operators at unrelated setup instead of their managed route.
    """

    def _no_openai_credentials(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "tools.tool_backend_helpers.resolve_openai_audio_api_key",
            lambda: None,
        )
        monkeypatch.setattr(
            "tools.managed_tool_gateway.resolve_managed_tool_gateway",
            lambda vendor: None,
        )


    def test_dispatch_returns_selection_specific_error(self, monkeypatch):
        """The final transcription result carries the managed-route error and
        its hermes tools remediation instead of the all-provider install
        hint."""
        self._no_openai_credentials(monkeypatch)
        monkeypatch.setattr(
            "tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True
        )
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config", dict
        )
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch(
                 "tools.tool_backend_helpers.nous_tool_gateway_unavailable_message",
                 lambda what: f"managed route down for {what}; run `hermes tools`",
             ):
            from tools.transcription_tools import _dispatch_stt_provider

            result = _dispatch_stt_provider(
                "/tmp/nonexistent.wav", "none", {"provider": "openai"}
            )

        assert result["success"] is False
        assert "managed route down" in result["error"]
        assert "hermes tools" in result["error"]
        assert "No STT provider available" not in result["error"]

    def test_auto_detect_none_keeps_generic_hint(self, monkeypatch):
        """Auto-detect with no credentials at all still returns the generic
        all-provider hint — the selection-specific branch must not fire
        without an explicit provider choice."""
        self._no_openai_credentials(monkeypatch)
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config", dict
        )
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch(
                 "tools.transcription_tools._has_local_command", return_value=False
             ), \
             patch(
                 "tools.transcription_tools._try_lazy_install_stt",
                 return_value=False,
             ):
            from tools.transcription_tools import _dispatch_stt_provider

            result = _dispatch_stt_provider("/tmp/x.wav", "none", {})

        assert result["success"] is False
        assert "No STT provider available" in result["error"]

# _transcribe_openai — 5xx transcode-and-retry (#81644)
# ============================================================================


class TestTranscribeOpenaiFiveXxRetry:
    """A 5xx rejection of the audio container must reach the
    transcode-and-retry path, not propagate as a plain API error."""

    def _status_error(self, status_code: int, message: str) -> Exception:
        import httpx
        from openai import APIStatusError

        request = httpx.Request(
            "POST", "https://api.example.com/v1/audio/transcriptions"
        )
        return APIStatusError(
            message,
            response=httpx.Response(status_code, request=request),
            body={"type": "system_error"},
        )

    def test_server_error_triggers_transcode_and_retry(self, sample_wav, tmp_path):
        """The provider rejects the container with a 5xx (the gapgpt case in
        #81644): the transcode-and-retry must run and succeed."""
        converted = tmp_path / "retry.m4a"
        converted.write_bytes(b"fake audio")
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = [
            self._status_error(503, "503 system_error"),  # first attempt: 5xx
            "retried transcript",                          # retry after transcode
        ]

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch(
                 "tools.transcription_cloud._transcode_audio_for_stt",
                 return_value=(str(converted), None),
             ):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(
                sample_wav, "gpt-4o-transcribe", api_key="sk-test"
            )

        assert result["success"] is True
        assert result["transcript"] == "retried transcript"
        assert mock_client.audio.transcriptions.create.call_count == 2

    def test_server_error_without_transcode_keeps_provider_error(self, sample_wav):
        """No ffmpeg: the 5xx surfaces as the provider's own error, not a transcode message,
        and the file is not re-sent."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = self._status_error(503, "503 system_error")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch("tools.transcription_cloud._transcode_audio_for_stt",
                   return_value=(None, "ffmpeg not found")):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(sample_wav, "whisper-1", api_key="sk-test")

        assert result["success"] is False
        assert "503" in result["error"] and "ffmpeg" not in result["error"]
        assert mock_client.audio.transcriptions.create.call_count == 1


class TestProviderOverride:
    def test_explicit_provider_exception_log_excludes_provider_text(self, sample_wav, caplog, monkeypatch):
        import logging
        from tools import transcription_tools as stt

        def fail(*_args, **_kwargs):
            raise RuntimeError("provider-private-override-payload")

        monkeypatch.setattr(stt, "_transcribe_prepared_audio", fail)
        with caplog.at_level(logging.WARNING, logger=stt.logger.name):
            result = stt._transcribe_audio_with_provider(sample_wav, provider="openai")
        assert result["success"] is False
        assert result["provider"] == "openai"
        assert "provider-private-override-payload" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    def test_public_transcribe_audio_signature_does_not_expose_provider_override(self):
        import inspect
        from tools.transcription_tools import transcribe_audio
        from tools.registry import registry

        parameters = inspect.signature(transcribe_audio).parameters
        assert list(parameters) == ["file_path", "model", "source"]
        assert parameters["model"].default is None
        assert parameters["source"].default is None
        # Transcription is an internal service, not a registered model tool.
        assert registry.get_entry("transcribe_audio") is None
        assert registry.get_entry("_transcribe_audio_with_provider") is None
        with pytest.raises(TypeError, match="provider"):
            transcribe_audio("unused.wav", **{"provider": "openai"})

    @pytest.mark.parametrize("failure", [False, True])
    def test_override_is_exact_with_configured_backups(self, sample_wav, failure):
        from copy import deepcopy
        from tools.transcription_tools import _transcribe_audio_with_provider

        config = {"provider": "local", "fallback_providers": ["groq", "local"],
                  "cloud_trim_silence": False, "openai": {"model": "whisper-1", "api_key": "test-key"}}
        original = deepcopy(config)
        client = MagicMock()
        if failure:
            client.audio.transcriptions.create.side_effect = RuntimeError("selected route failed")
        else:
            client.audio.transcriptions.create.return_value = "selected route"
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=client), \
             patch("tools.transcription_tools._transcribe_groq") as groq, \
             patch("tools.transcription_tools._transcribe_local") as local, \
             patch("tools.transcription_tools._get_provider") as configured:
            result = _transcribe_audio_with_provider(sample_wav, provider="openai")
        assert result["success"] is not failure
        client.audio.transcriptions.create.assert_called_once()
        groq.assert_not_called()
        local.assert_not_called()
        configured.assert_not_called()
        assert config == original

    @pytest.mark.parametrize("stage", ["trim", "dispatch"])
    @pytest.mark.parametrize("explicit", [True, False])
    def test_explicit_route_contains_local_errors(self, sample_wav, stage, explicit):
        from copy import deepcopy
        from tools import transcription_tools as stt

        config = {"provider": "openai", "fallback_providers": ["groq", "local"],
                  "cloud_trim_silence": True}
        original = deepcopy(config)
        failure = RuntimeError("selected route preparation failed")
        with patch.object(stt, "_load_stt_config", return_value=config), \
             patch.object(stt, "_get_provider", return_value="openai") as configured, \
             patch.object(stt, "_trim_silence_for_cloud_stt", return_value=None,
                          side_effect=failure if stage == "trim" else None), \
             patch.object(stt, "_transcribe_openai", side_effect=failure) as selected, \
             patch.object(stt, "_transcribe_groq") as groq, \
             patch.object(stt, "_transcribe_local") as local:
            if explicit:
                result = stt._transcribe_audio_with_provider(sample_wav, provider="openai")
                assert result["success"] is False
                assert result["provider"] == "openai"
                assert "selected route preparation failed" in result["error"]
                configured.assert_not_called()
            else:
                with pytest.raises(RuntimeError, match="selected route preparation failed"):
                    stt.transcribe_audio(sample_wav)
            assert selected.call_count == int(stage == "dispatch")
            groq.assert_not_called()
            local.assert_not_called()
        assert config == original

    def test_override_and_default_calls_remain_isolated_while_interleaved(self, sample_wav):
        from concurrent.futures import ThreadPoolExecutor
        from copy import deepcopy
        from threading import Event
        from tools import transcription_tools as stt

        config = {"provider": "local", "local": {"model": "small"},
                  "fallback_providers": ["groq"], "cloud_trim_silence": False,
                  "openai": {"api_key": "test-key"}}
        original = deepcopy(config)
        entered, release = Event(), Event()
        client = MagicMock()

        def transcribe(**kwargs):
            entered.set()
            assert release.wait(5), "override was not released"
            return "override"

        client.audio.transcriptions.create.side_effect = transcribe
        with patch.object(stt, "_load_stt_config", return_value=config), \
             patch("tools.tool_backend_helpers.read_selection", return_value="local"), \
             patch.object(stt, "_HAS_FASTER_WHISPER", True), \
             patch.object(stt, "_HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=client), \
             patch.object(stt, "_transcribe_local", return_value={"success": True, "transcript": "default"}) as local, \
             ThreadPoolExecutor(max_workers=1) as executor:
            assert stt.transcribe_audio(sample_wav)["transcript"] == "default"
            future = executor.submit(stt._transcribe_audio_with_provider, sample_wav, provider="openai")
            try:
                assert entered.wait(5), "override never reached the selected API"
                assert config == original
                assert stt.transcribe_audio(sample_wav)["transcript"] == "default"
            finally:
                release.set()
            assert future.result(timeout=5)["transcript"] == "override"
            assert stt.transcribe_audio(sample_wav)["transcript"] == "default"
        assert local.call_args_list == [call(sample_wav, "small", language=None, prompt=None)] * 3
        client.audio.transcriptions.create.assert_called_once()
        assert config == original

    @pytest.mark.parametrize("guard, message", [
        ("disabled", "STT is disabled"), ("secret", ""),
        ("missing", "not found"), ("directory", "not a file"),
        ("symlink", "symbolic link"), ("unsupported", "Unsupported format"),
        ("oversized", "File too large"), ("silk_oversized", "File too large"),
    ])
    def test_override_cannot_bypass_native_guards(
        self, guard, message, sample_wav, oversized_wav, tmp_path,
    ):
        from tools.transcription_tools import _transcribe_audio_with_provider
        from agent.file_safety import get_read_block_error

        secret = tmp_path / ".env"
        secret.write_text("synthetic test data")
        unsupported = tmp_path / "audio.txt"
        unsupported.write_text("not audio")
        directory = tmp_path / "directory.wav"
        directory.mkdir()
        link = tmp_path / "linked.silk"
        link.symlink_to(sample_wav)
        silk = tmp_path / "oversized.silk"
        with silk.open("wb") as file:
            file.truncate(Path(oversized_wav).stat().st_size)
        paths = {"secret": secret, "missing": tmp_path / "absent.wav",
                 "directory": directory, "symlink": link, "unsupported": unsupported,
                 "oversized": oversized_wav, "silk_oversized": silk}
        path = str(paths.get(guard, sample_wav))
        config = {"provider": "local", "enabled": guard != "disabled",
                  "fallback_providers": ["groq"]}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._dispatch_stt_provider") as dispatch, \
             patch("tools.transcription_tools._get_provider") as configured, \
             patch("tools.transcription_audio._lazy_ensure_quietly") as install:
            result = _transcribe_audio_with_provider(path, provider="openai")
        assert result["success"] is False
        if guard == "secret":
            assert result["error"] == get_read_block_error(path)
        else:
            assert message in result["error"]
        dispatch.assert_not_called()
        configured.assert_not_called()
        install.assert_not_called()

    @pytest.mark.parametrize("prepared_name", ["invalid.txt", ".env"])
    def test_override_revalidates_prepared_input_and_cleans_it(self, sample_wav, tmp_path, prepared_name):
        from tools.transcription_tools import _transcribe_audio_with_provider

        work = tmp_path / "prepared"
        work.mkdir()
        prepared = work / prepared_name
        prepared.write_text("synthetic test data")
        with patch("tools.transcription_tools._prepare_audio_for_transcription",
                   return_value=(str(prepared), str(work), None)), \
             patch("tools.transcription_tools._dispatch_stt_provider") as dispatch:
            result = _transcribe_audio_with_provider(sample_wav, provider="openai")
        assert result["success"] is False
        dispatch.assert_not_called()
        assert not work.exists()
        assert Path(sample_wav).exists()

    @pytest.mark.parametrize("provider", [None, "openai"])
    def test_override_composes_with_native_caf_trim_cleanup(
        self, tmp_path, monkeypatch, provider,
    ):
        from copy import deepcopy
        from tools import transcription_tools as stt

        caf = tmp_path / "voice.caf"
        caf.write_bytes(b"caff" * 20)
        neighbor = caf.with_suffix(".wav")
        neighbor.write_bytes(b"existing recording")
        config = {"provider": "openai", "fallback_providers": ["local", "local"],
                  "local": {"model": "small"}, "openai": {"api_key": "test-key"}}
        original = deepcopy(config)
        outputs = []

        def run_audio_command(command, **kwargs):
            if command[0] == "test-ffprobe":
                duration = "20" if command[-1] == str(outputs[0]) else "5"
                return subprocess.CompletedProcess(command, 0, stdout=duration)
            output = Path(command[-1])
            output.write_bytes(b"RIFF" * 20)
            outputs.append(output)
            return subprocess.CompletedProcess(command, 0, stdout="")

        monkeypatch.setattr("tools.transcription_audio._find_ffmpeg_binary", lambda: "test-ffmpeg")
        monkeypatch.setattr("tools.transcription_audio._find_ffprobe_binary", lambda: "test-ffprobe")
        monkeypatch.setattr("tools.transcription_audio._run_quiet", run_audio_command)
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = RuntimeError("selected route failed")
        with patch.object(stt, "_load_stt_config", return_value=config), \
             patch.object(stt, "_HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=client), \
             patch.object(stt, "_transcribe_local", return_value={"success": True, "transcript": "fallback"}) as local:
            result = stt._transcribe_audio_with_provider(str(caf), model="whisper-1", source="gateway", provider=provider)
        assert result["success"] is False
        assert len(outputs) == 2
        assert client.audio.transcriptions.create.call_args.kwargs["file"].name == str(outputs[1])
        assert all(not output.parent.exists() for output in outputs)
        assert caf.exists()
        assert neighbor.read_bytes() == b"existing recording"
        local.assert_not_called()
        assert config == original

    def test_native_default_does_not_activate_configured_fallbacks(self, oversized_wav):
        from tools.transcription_tools import transcribe_audio

        config = {"provider": "local", "fallback_providers": ["openai"]}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.tool_backend_helpers.read_selection", return_value="local"), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._transcribe_local", return_value={"success": False}) as local, \
             patch("tools.transcription_tools._transcribe_openai") as cloud:
            result = transcribe_audio(oversized_wav)
        assert result["success"] is False
        assert result == {"success": False}
        local.assert_called_once()
        cloud.assert_not_called()
