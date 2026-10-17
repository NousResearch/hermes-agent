"""A TTS stream that closes early must never be delivered as a successful voice reply.

The OpenAI SDK's ``stream_to_file`` writes whatever bytes arrived before the HTTP body
ended and raises nothing, so a cut connection used to produce a clipped Ogg (no
end-of-stream page, seconds of audio for a long reply) that the tool reported as success.
Each chunk is now checked after synthesis, retried once with a fresh request, and failed
when it is still incomplete.
"""

import json
import shutil
import struct
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gateway.session_context import _UNSET, _VAR_MAP
from tools import tts_tool
from tools.tts_tool_integrity import _probe_duration, _speech_speed

LONG_TEXT = "This sentence is part of a long spoken reply about the weekly plan. " * 42
SHORT_TEXT = "Sure, the meeting moved to Thursday at ten."

_OPUS_PRE_SKIP = 312


def _ogg_page(header_type: int, granule: int, payload: bytes, sequence: int) -> bytes:
    lacing, remaining = [], len(payload)
    while remaining >= 255:
        lacing.append(255)
        remaining -= 255
    lacing.append(remaining)
    header = struct.pack("<4sBBqIIIB", b"OggS", 0, header_type, granule, 1, sequence, 0, len(lacing))
    return header + bytes(lacing) + payload


def _opus_ogg(seconds: float, *, eos: bool = True, cut_bytes: int = 0) -> bytes:
    """A structurally valid Ogg/Opus stream lasting *seconds* (granules only; payload is filler)."""
    head = b"OpusHead" + bytes([1, 1]) + struct.pack("<HIhB", _OPUS_PRE_SKIP, 48000, 0, 0)
    pages = [_ogg_page(0x02, 0, head, 0), _ogg_page(0x00, 0, b"OpusTags" + b"\x00" * 8, 1)]
    final_granule = _OPUS_PRE_SKIP + int(seconds * 48000)
    for step in range(1, 5):
        last = step == 4
        header_type = 0x04 if (last and eos) else 0x00
        pages.append(_ogg_page(header_type, final_granule * step // 4, b"\x55" * 600, step + 1))
    data = b"".join(pages)
    return data[:-cut_bytes] if cut_bytes else data


MP3_BYTES = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb" + b"\x00" * 400


@pytest.fixture(autouse=True)
def _openai_without_session_platform(monkeypatch):
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "openai"})
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


@pytest.fixture
def audio_dir(tmp_path):
    path = tmp_path / "audio"  # tmp_path also holds the conftest's isolated HERMES_HOME
    path.mkdir()
    return path


def _streaming_response(payload: bytes) -> MagicMock:
    def stream_to_file(path):
        with open(path, "wb") as fh:
            fh.write(payload)
    response = MagicMock()
    response.stream_to_file.side_effect = stream_to_file
    return response


def _run_openai(payloads: list, text: str, output_path: str):
    """Run the tool against a fake OpenAI client whose Nth request streams ``payloads[N]`` (or raises it)."""
    client = MagicMock()
    client.audio.speech.create.side_effect = [
        payload if isinstance(payload, BaseException) else _streaming_response(payload) for payload in payloads]
    with patch("tools.tts_tool._import_openai_client", return_value=MagicMock(return_value=client)), \
         patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
               return_value=("test-key", None, False)):
        result = json.loads(tts_tool.text_to_speech_tool(text, output_path=output_path))
    return result, client.audio.speech.create


@pytest.mark.parametrize("text, broken, reason", [
    pytest.param(LONG_TEXT, _opus_ogg(7.0, eos=False), "no end-of-stream flag", id="observed-7s-no-eos"),
    pytest.param(LONG_TEXT, _opus_ogg(205.0, eos=False), "no end-of-stream flag", id="full-length-no-eos"),
    pytest.param(LONG_TEXT, _opus_ogg(205.0, cut_bytes=100), "stops partway through a page", id="full-length-cut"),
    pytest.param(SHORT_TEXT, _opus_ogg(2.0, eos=False), "no end-of-stream flag", id="short-reply-no-eos"),
    pytest.param(SHORT_TEXT, _opus_ogg(2.0, cut_bytes=100), "stops partway through a page", id="short-reply-cut"),
])
def test_structurally_truncated_ogg_is_retried_once_then_fails(audio_dir, text, broken, reason):
    """The page walk alone must catch these: the full-length and short cases pass the duration floor."""
    result, create = _run_openai([broken, broken], text, str(audio_dir / "reply.ogg"))

    assert result["success"] is False
    assert "ended early" in result["error"] and reason in result["error"]
    assert " s of audio" in result["error"] and " bytes" in result["error"]
    assert create.call_count == 2
    keys = {call.kwargs["extra_headers"]["x-idempotency-key"] for call in create.call_args_list}
    assert len(keys) == 2, "the retry must be a fresh request, not a replay of the cut one"
    assert list(audio_dir.iterdir()) == []


def test_complete_but_implausibly_short_ogg_is_retried_then_fails(audio_dir):
    """A well-formed Ogg (EOS present, e.g. a cut MP3 that ffmpeg re-encoded) is judged by its granule."""
    clipped = _opus_ogg(7.0)

    result, create = _run_openai([clipped, clipped], LONG_TEXT, str(audio_dir / "reply.ogg"))

    assert result["success"] is False
    assert "far shorter than its text" in result["error"] and "7.0 s of audio" in result["error"]
    assert create.call_count == 2


@pytest.mark.parametrize("tts_config, delivered", [
    pytest.param({"provider": "openai"}, False, id="normal-speed"),
    pytest.param({"provider": "openai", "speed": 3.0}, True, id="global-speed"),
    pytest.param({"provider": "openai", "speed": 1.0, "openai": {"speed": 3.0}}, True, id="provider-speed-wins"),
])
def test_duration_floor_scales_with_configured_speed(audio_dir, monkeypatch, tts_config, delivered):
    """15 s for ~2.9k chars is clipped at 1x but plausible at 3x (floor ~36 s vs ~12 s)."""
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: tts_config)
    audio = _opus_ogg(15.0)

    result, create = _run_openai([audio, audio], LONG_TEXT, str(audio_dir / "reply.ogg"))

    assert result["success"] is delivered
    assert create.call_count == (1 if delivered else 2)


@pytest.mark.parametrize("tts_config, expected", [
    ({"speed": "fast"}, 1.0),
    ({"speed": 0.5}, 1.0),
    ({"speed": 9}, 4.0),
    ({"speed": 2.5}, 2.5),
    ({"speed": 1.0, "openai": {"speed": 2.0}}, 2.0),
    (None, 1.0),
])
def test_speech_speed_only_ever_relaxes_the_floor(tts_config, expected):
    assert _speech_speed(tts_config, "openai") == expected


def test_double_truncation_keeps_a_pre_existing_file_at_output_path(audio_dir):
    existing = audio_dir / "reply.ogg"
    existing.write_bytes(b"user data")
    broken = _opus_ogg(7.0, eos=False)

    result, create = _run_openai([broken, broken], LONG_TEXT, str(existing))

    assert result["success"] is False
    assert create.call_count == 2
    assert existing.read_bytes() == b"user data"
    assert sorted(p.name for p in audio_dir.iterdir()) == ["reply.ogg"]


def test_truncated_later_chunk_fails_the_reply_and_sweeps_earlier_chunks(audio_dir, monkeypatch):
    monkeypatch.setattr(tts_tool, "_load_tts_config",
                        lambda: {"provider": "openai", "openai": {"max_text_length": 1500}})
    broken = _opus_ogg(7.0, eos=False)

    result, create = _run_openai([_opus_ogg(110.0), broken, broken], LONG_TEXT, str(audio_dir / "reply.ogg"))

    assert result["success"] is False
    assert result["error"].startswith("TTS chunk 2 failed (openai)")
    assert create.call_count == 3
    assert list(audio_dir.iterdir()) == []


def test_retry_that_errors_reports_the_provider_error_and_leaves_nothing(audio_dir):
    result, create = _run_openai(
        [_opus_ogg(7.0, eos=False), ConnectionError("connection reset by peer")],
        LONG_TEXT, str(audio_dir / "reply.ogg"))

    assert result["success"] is False
    assert "connection reset by peer" in result["error"]
    assert create.call_count == 2
    assert list(audio_dir.iterdir()) == []


def test_complete_ogg_is_delivered_with_a_single_request(audio_dir):
    complete = _opus_ogg(205.0)

    result, create = _run_openai([complete], LONG_TEXT, str(audio_dir / "reply.ogg"))

    assert result["success"] is True
    assert create.call_count == 1
    assert (audio_dir / "reply.ogg").read_bytes() == complete


def test_retry_that_completes_delivers_the_complete_audio(audio_dir):
    complete = _opus_ogg(205.0)

    result, create = _run_openai(
        [_opus_ogg(7.0, eos=False), complete], LONG_TEXT, str(audio_dir / "reply.ogg"))

    assert result["success"] is True
    assert create.call_count == 2
    assert (audio_dir / "reply.ogg").read_bytes() == complete
    assert sorted(p.name for p in audio_dir.iterdir()) == ["reply.ogg"]


def test_short_reply_with_short_audio_is_never_flagged(audio_dir):
    with patch("tools.tts_tool_integrity._probe_duration",
               side_effect=AssertionError("short text must not be probed")):
        ogg_result, ogg_create = _run_openai([_opus_ogg(0.4)], SHORT_TEXT, str(audio_dir / "a.ogg"))
        mp3_result, mp3_create = _run_openai([MP3_BYTES], SHORT_TEXT, str(audio_dir / "b.mp3"))

    assert ogg_result["success"] is True and ogg_create.call_count == 1
    assert mp3_result["success"] is True and mp3_create.call_count == 1


def test_mp3_duration_check_is_skipped_without_ffprobe(audio_dir):
    with patch("tools.transcription_audio._find_ffprobe_binary", return_value=None):
        result, create = _run_openai([MP3_BYTES], LONG_TEXT, str(audio_dir / "reply.mp3"))

    assert result["success"] is True
    assert create.call_count == 1


def test_implausibly_short_mp3_is_retried_then_fails(audio_dir):
    with patch("tools.tts_tool_integrity._probe_duration", return_value=7.0):
        result, create = _run_openai([MP3_BYTES, MP3_BYTES], LONG_TEXT, str(audio_dir / "reply.mp3"))

    assert result["success"] is False
    assert "ended early" in result["error"]
    assert create.call_count == 2
    assert list(audio_dir.iterdir()) == []


def test_cut_vbr_mp3_measures_its_real_length_despite_xing_header(tmp_path):
    """A LAME Xing/Info header keeps declaring the full length after the stream is cut, so the
    probe must measure the packets actually present (skipped without a working ffmpeg)."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")
    full = tmp_path / "full.mp3"
    synth = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
         "-ac", "1", "-c:a", "libmp3lame", "-q:a", "5", str(full), "-y"],
        capture_output=True, check=False)
    if synth.returncode != 0:
        pytest.skip(f"ffmpeg on PATH cannot synthesize mp3 (exit {synth.returncode})")
    cut = tmp_path / "cut.mp3"
    cut.write_bytes(full.read_bytes()[: full.stat().st_size // 10])

    assert 55 < _probe_duration(str(full)) < 65
    assert _probe_duration(str(cut)) < 10
