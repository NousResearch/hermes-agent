"""Regression tests for resilient Edge TTS outbound behavior."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def test_generate_edge_tts_retries_same_voice_before_succeeding(tmp_path):
    from tools.tts_tool import _generate_edge_tts

    output_path = tmp_path / "out.mp3"
    call_counter = {"n": 0}

    async def flaky_save(path):
        call_counter["n"] += 1
        if call_counter["n"] < 3:
            raise RuntimeError("NoAudioReceived")
        Path(path).write_bytes(b"ok")

    mock_comm = MagicMock()
    mock_comm.save = AsyncMock(side_effect=flaky_save)
    mock_edge = MagicMock()
    mock_edge.Communicate = MagicMock(return_value=mock_comm)

    with patch("tools.tts_tool._import_edge_tts", return_value=mock_edge):
        result = asyncio.run(
            _generate_edge_tts(
                "Hello",
                str(output_path),
                {"edge": {"retry_attempts": 3}},
            )
        )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"ok"
    assert mock_edge.Communicate.call_count == 3


def test_generate_edge_tts_tries_fallback_voice_after_primary_failure(tmp_path):
    from tools.tts_tool import _generate_edge_tts

    output_path = tmp_path / "out.mp3"
    primary_comm = MagicMock()
    primary_comm.save = AsyncMock(side_effect=RuntimeError("NoAudioReceived"))

    async def fallback_save(path):
        Path(path).write_bytes(b"fallback-ok")

    fallback_comm = MagicMock()
    fallback_comm.save = AsyncMock(side_effect=fallback_save)

    def communicate_factory(text, **kwargs):
        if kwargs["voice"] == "en-US-AriaNeural":
            return primary_comm
        if kwargs["voice"] == "en-US-JennyNeural":
            return fallback_comm
        raise AssertionError(f"unexpected voice: {kwargs['voice']}")

    mock_edge = MagicMock()
    mock_edge.Communicate = MagicMock(side_effect=communicate_factory)

    with patch("tools.tts_tool._import_edge_tts", return_value=mock_edge):
        result = asyncio.run(
            _generate_edge_tts(
                "Hello",
                str(output_path),
                {
                    "edge": {
                        "voice": "en-US-AriaNeural",
                        "retry_attempts": 1,
                        "fallback_voices": ["en-US-JennyNeural"],
                    }
                },
            )
        )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"fallback-ok"
    used_voices = [call.kwargs["voice"] for call in mock_edge.Communicate.call_args_list]
    assert used_voices == ["en-US-AriaNeural", "en-US-JennyNeural"]
