"""Tests for the audio post-processing cluster of ``tools/tts_tool.py``.

This covers the ffmpeg-based conversion helpers (``_convert_to_opus``,
``_ffmpeg_transcode_to_opus``), the shared-container sniffer
(``_sniff_audio_container``), the .ogg container repair (``_repair_ogg_container``),
long-form audio concatenation (``_concat_audio_files``), delivery-file packing
(``_build_audio_delivery_files``, ``_pack_audio_files_for_delivery``), and the
per-platform audio-delivery profile resolver (``_resolve_audio_delivery_profile``).

Real TinyAudio fixtures are built with the stdlib ``wave`` module. Real ffmpeg
invocations are guarded by ``skipif`` so the suite stays green on machines
without ffmpeg; mock-based tests cover the same branches unconditionally by
patching the ``shutil.which`` / ``subprocess.run`` layer on the module.
"""

import shutil
import subprocess
import wave
from pathlib import Path
from unittest import mock

import pytest

import tools.tts_tool as tts


FFMPEG_MISSING = shutil.which("ffmpeg") is None
requires_ffmpeg = pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")


def _mk_wav(path, *, frames=8000, rate=16000):
    """Write a minimal mono 16-bit PCM WAV fixture with the stdlib ``wave``."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)
    return str(path)


def _mk_fake_subprocess(*, returncode=0, stderr=b"", writes_output=True):
    """Return a ``subprocess.run`` fake that writes a non-empty file to its last arg."""

    def _fake_run(args, **kwargs):
        if writes_output:
            # The ffmpeg command ends with "-y", so the output path is args[-2].
            Path(args[-2]).write_bytes(b"fake-encoded-audio")
        return mock.Mock(returncode=returncode, stderr=stderr)

    return _fake_run


class TestHasFfmpeg:
    def test_true_when_which_resolves(self, monkeypatch):
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        assert tts._has_ffmpeg() is True

    def test_false_when_which_missing(self, monkeypatch):
        monkeypatch.setattr(tts.shutil, "which", lambda name: None)
        assert tts._has_ffmpeg() is False


class TestConvertToOpus:
    def test_returns_none_without_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: False)
        assert tts._convert_to_opus("in.mp3") is None

    def test_swaps_extension_and_transcodes(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)

        def _fake_transcode(inp, ogg):
            calls["input"] = inp
            calls["ogg"] = ogg
            return ogg

        monkeypatch.setattr(tts, "_ffmpeg_transcode_to_opus", _fake_transcode)
        result = tts._convert_to_opus("/some/dir/voice.mp3")
        assert result == "/some/dir/voice.ogg"
        assert calls["input"] == "/some/dir/voice.mp3"
        assert calls["ogg"] == "/some/dir/voice.ogg"


class TestFfmpegTranscodeToOpus:
    @requires_ffmpeg
    def test_real_conversion_produces_ogg(self, tmp_path):
        wav = _mk_wav(tmp_path / "in.wav")
        ogg_path = str(tmp_path / "out.ogg")
        assert tts._ffmpeg_transcode_to_opus(wav, ogg_path) == ogg_path
        assert Path(ogg_path).exists()
        assert tts._sniff_audio_container(ogg_path) == "ogg"

    @requires_ffmpeg
    def test_in_place_transcode_replaces_source(self, tmp_path):
        src = _mk_wav(tmp_path / "inplace.ogg")
        assert tts._ffmpeg_transcode_to_opus(src, src) == src
        assert list(tmp_path.glob("*.ogg.tmp.ogg")) == []
        assert tts._sniff_audio_container(src) == "ogg"

    def test_returns_none_without_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: False)
        assert tts._ffmpeg_transcode_to_opus("a.mp3", "a.ogg") is None

    def test_nonzero_returncode_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)
        monkeypatch.setattr(
            tts.subprocess, "run", _mk_fake_subprocess(returncode=1, stderr=b"boom")
        )
        assert tts._ffmpeg_transcode_to_opus("a.mp3", str(tmp_path / "a.ogg")) is None

    def test_missing_output_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)
        # subprocess "succeeds" (returncode 0) but writes nothing -> no output file.
        monkeypatch.setattr(
            tts.subprocess, "run", _mk_fake_subprocess(returncode=0, writes_output=False)
        )
        assert tts._ffmpeg_transcode_to_opus("a.mp3", str(tmp_path / "a.ogg")) is None

    def test_timeout_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)

        def _raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("ffmpeg", 30)

        monkeypatch.setattr(tts.subprocess, "run", _raise_timeout)
        assert tts._ffmpeg_transcode_to_opus("a.mp3", str(tmp_path / "a.ogg")) is None

    def test_file_not_found_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)

        def _raise_fnf(*args, **kwargs):
            raise FileNotFoundError("ffmpeg missing")

        monkeypatch.setattr(tts.subprocess, "run", _raise_fnf)
        assert tts._ffmpeg_transcode_to_opus("a.mp3", str(tmp_path / "a.ogg")) is None

    def test_generic_exception_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)

        def _raise(*args, **kwargs):
            raise RuntimeError("exploded")

        monkeypatch.setattr(tts.subprocess, "run", _raise)
        assert tts._ffmpeg_transcode_to_opus("a.mp3", str(tmp_path / "a.ogg")) is None

    def test_success_renames_in_place(self, tmp_path, monkeypatch):
        src = tmp_path / "in.ogg"
        src.write_bytes(b"old-audio")
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)
        # Fake writes output to the ".tmp.ogg" work path (the last CLI arg).
        monkeypatch.setattr(tts.subprocess, "run", _mk_fake_subprocess(returncode=0))
        assert tts._ffmpeg_transcode_to_opus(str(src), str(src)) == str(src)
        assert src.read_bytes() == b"fake-encoded-audio"
        assert list(tmp_path.glob("*.ogg.tmp.ogg")) == []

    def test_in_place_cleanup_when_transcode_fails(self, tmp_path, monkeypatch):
        # In-place transcode that writes the .tmp.ogg file but fails (returncode != 0)
        # must clean up the leftover temp file in the finally block (lines 1466-1470).
        src = tmp_path / "in.ogg"
        src.write_bytes(b"old-audio")
        monkeypatch.setattr(tts, "_has_ffmpeg", lambda: True)
        monkeypatch.setattr(
            tts.subprocess, "run", _mk_fake_subprocess(returncode=1, stderr=b"boom")
        )
        assert tts._ffmpeg_transcode_to_opus(str(src), str(src)) is None
        assert list(tmp_path.glob("*.ogg.tmp.ogg")) == []


class TestSniffAudioContainer:
    def test_recognizes_wav(self, tmp_path):
        wav = _mk_wav(tmp_path / "a.wav")
        assert tts._sniff_audio_container(wav) == "wav"

    @requires_ffmpeg
    def test_recognizes_ogg(self, tmp_path):
        wav = _mk_wav(tmp_path / "a.wav")
        ogg = tmp_path / "a.ogg"
        tts._ffmpeg_transcode_to_opus(wav, str(ogg))
        assert tts._sniff_audio_container(str(ogg)) == "ogg"

    def test_recognizes_mp3_magic(self, tmp_path):
        p = tmp_path / "a.mp3"
        p.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64)
        assert tts._sniff_audio_container(str(p)) == "mp3"

    def test_unknown_on_unreadable(self, tmp_path):
        assert tts._sniff_audio_container(str(tmp_path / "does-not-exist.ogg")) == "unknown"

    def test_unknown_on_garbage(self, tmp_path):
        p = tmp_path / "junk.ogg"
        p.write_bytes(b"not audio at all " + b"\x00" * 20)
        assert tts._sniff_audio_container(str(p)) == "unknown"


class TestRepairOggContainer:
    def test_passthrough_when_not_ogg(self, tmp_path):
        p = tmp_path / "voice.mp3"
        p.write_bytes(b"anything")
        assert tts._repair_ogg_container(str(p)) == str(p)

    @requires_ffmpeg
    def test_passthrough_when_already_ogg(self, tmp_path):
        wav = _mk_wav(tmp_path / "in.wav")
        ogg = tmp_path / "voice.ogg"
        tts._ffmpeg_transcode_to_opus(wav, str(ogg))
        assert tts._repair_ogg_container(str(ogg)) == str(ogg)

    def test_passthrough_when_sniff_unknown(self, tmp_path):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"not audio at all " + b"\x00" * 20)
        assert tts._repair_ogg_container(str(p)) == str(p)

    @requires_ffmpeg
    def test_repairs_mismatched_container_in_place(self, tmp_path):
        # Valid WAV bytes hiding behind a .ogg name -> transcoded to real Opus.
        p = tmp_path / "voice.ogg"
        _mk_wav(p)
        assert tts._sniff_audio_container(str(p)) == "wav"
        assert tts._repair_ogg_container(str(p)) == str(p)
        assert tts._sniff_audio_container(str(p)) == "ogg"

    def test_renames_when_transcode_unavailable(self, tmp_path, monkeypatch):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64)
        monkeypatch.setattr(tts, "_ffmpeg_transcode_to_opus", lambda *a, **k: None)
        result = tts._repair_ogg_container(str(p))
        assert result == str(tmp_path / "voice.mp3")
        assert Path(result).exists()

    @requires_ffmpeg
    def test_renames_real_mismatch_when_transcode_fails(self, tmp_path):
        # ffmpeg rejects these collision-ridden MP3-ish bytes (returncode != 0),
        # so the honest-extension rename fallback fires.
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64)
        result = tts._repair_ogg_container(str(p))
        assert result == str(tmp_path / "voice.mp3")
        assert Path(result).exists()

    def test_returns_original_on_replace_failure(self, tmp_path, monkeypatch):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64)
        monkeypatch.setattr(tts, "_ffmpeg_transcode_to_opus", lambda *a, **k: None)

        def _raise(*args, **kwargs):
            raise OSError("no replace")

        monkeypatch.setattr(tts.os, "replace", _raise)
        assert tts._repair_ogg_container(str(p)) == str(p)


class TestConcatAudioFiles:
    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tts._concat_audio_files([], str(tmp_path / "out"))

    def test_single_file_copies_when_paths_differ(self, tmp_path):
        src = tmp_path / "a.wav"
        _mk_wav(src)
        out = tmp_path / "out.wav"
        assert tts._concat_audio_files([str(src)], str(out)) == str(out)
        assert out.exists()
        assert out.read_bytes() == src.read_bytes()

    def test_single_file_same_path_returns_output(self, tmp_path):
        src = tmp_path / "a.wav"
        _mk_wav(src)
        assert tts._concat_audio_files([str(src)], str(src)) == str(src)

    def test_no_ffmpeg_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts.shutil, "which", lambda name: None)
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        _mk_wav(a)
        _mk_wav(b)
        assert (
            tts._concat_audio_files([str(a), str(b)], str(tmp_path / "out.wav"))
            is None
        )

    def test_failure_returncode_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            tts.subprocess, "run", _mk_fake_subprocess(returncode=1, stderr=b"boom", writes_output=False)
        )
        _mk_wav(tmp_path / "a.wav")
        _mk_wav(tmp_path / "b.wav")
        assert (
            tts._concat_audio_files(
                [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")],
                str(tmp_path / "out.wav"),
            )
            is None
        )

    def test_timeout_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")

        def _raise(*args, **kwargs):
            raise subprocess.TimeoutExpired("ffmpeg", 120)

        monkeypatch.setattr(tts.subprocess, "run", _raise)
        _mk_wav(tmp_path / "a.wav")
        _mk_wav(tmp_path / "b.wav")
        assert (
            tts._concat_audio_files(
                [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")],
                str(tmp_path / "out.wav"),
            )
            is None
        )

    def test_oserror_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")

        def _raise(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(tts.subprocess, "run", _raise)
        _mk_wav(tmp_path / "a.wav")
        _mk_wav(tmp_path / "b.wav")
        assert (
            tts._concat_audio_files(
                [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")],
                str(tmp_path / "out.wav"),
            )
            is None
        )

    def test_reencode_default_branch_command(self, tmp_path, monkeypatch):
        # WAV inputs -> WAV output, not voice-compatible: default re-encode, no codec override.
        captured = {}
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            tts.subprocess,
            "run",
            _mk_fake_runner(captured),
        )
        _mk_wav(tmp_path / "a.wav")
        _mk_wav(tmp_path / "b.wav")
        out = tmp_path / "out.wav"
        assert tts._concat_audio_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")], str(out)) == str(out)
        assert "-c:a" not in captured["command"]

    def test_opm_codec_command_for_ogg_output(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(tts.subprocess, "run", _mk_fake_runner(captured))
        _mk_wav(tmp_path / "a.wav")
        _mk_wav(tmp_path / "b.wav")
        out = tmp_path / "out.ogg"
        assert tts._concat_audio_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")], str(out)) == str(out)
        assert captured["command"][captured["command"].index("-c:a") + 1] == "libopus"

    def test_opus_codec_when_voice_compatible(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(tts.subprocess, "run", _mk_fake_runner(captured))
        _mk_wav(tmp_path / "a.wav")
        _mk_wav(tmp_path / "b.wav")
        out = tmp_path / "out.wav"
        assert (
            tts._concat_audio_files(
                [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")], str(out), voice_compatible=True
            )
            == str(out)
        )
        assert captured["command"][captured["command"].index("-c:a") + 1] == "libopus"

    def test_mp3_copy_branch_command(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(tts.subprocess, "run", _mk_fake_runner(captured))
        (tmp_path / "a.mp3").write_bytes(b"id3")
        (tmp_path / "b.mp3").write_bytes(b"id3")
        out = tmp_path / "out.mp3"
        assert (
            tts._concat_audio_files([str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")], str(out))
            == str(out)
        )
        assert captured["command"][captured["command"].index("-c:a") + 1] == "copy"

    @requires_ffmpeg
    def test_real_concat_two_wavs(self, tmp_path):
        a = _mk_wav(tmp_path / "a.wav")
        b = _mk_wav(tmp_path / "b.wav")
        out = tmp_path / "combined.wav"
        assert tts._concat_audio_files([a, b], str(out)) == str(out)
        assert tts._sniff_audio_container(str(out)) == "wav"

    @requires_ffmpeg
    def test_real_concat_to_ogg_voice_compatible(self, tmp_path):
        a = _mk_wav(tmp_path / "a.wav")
        b = _mk_wav(tmp_path / "b.wav")
        out = tmp_path / "combined.ogg"
        assert (
            tts._concat_audio_files([a, b], str(out), voice_compatible=True)
            == str(out)
        )
        assert tts._sniff_audio_container(str(out)) == "ogg"


def _mk_fake_runner(captured):
    def _run(args, **kwargs):
        captured["command"] = args
        Path(args[-1]).write_bytes(b"combi")
        return mock.Mock(returncode=0, stderr=b"")

    return _run


def _mock_raise_oserror(self, *args, **kwargs):
    raise OSError("cannot unlink")


class TestPackAudioFilesForDelivery:
    def _profile(self, max_bytes, safety=0.5):
        return tts.AudioDeliveryProfile("telegram", max_bytes, safety)

    def test_empty(self):
        assert tts._pack_audio_files_for_delivery([], self._profile(1000)) == []

    def test_single_returns_one_group(self, tmp_path):
        f = _mk_wav(tmp_path / "a.wav")
        assert tts._pack_audio_files_for_delivery([f], self._profile(1_000_000)) == [[f]]

    def test_groups_multiple_under_target(self, tmp_path):
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(3)]
        profile = self._profile(max_bytes=10_000_000, safety=0.9)
        groups = tts._pack_audio_files_for_delivery(files, profile)
        assert groups == [files]

    def test_splits_when_target_exceeded(self, tmp_path):
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(2)]
        # Each WAV is ~32KB; set a 1-byte target so each file exceeds it.
        profile = tts.AudioDeliveryProfile("default", max_file_bytes=200, safety_ratio=0.85)
        groups = tts._pack_audio_files_for_delivery(files, profile)
        assert groups == [[files[0]], [files[1]]]

    def test_splits_on_suffix_change(self, tmp_path):
        w = _mk_wav(tmp_path / "a.wav")
        m = tmp_path / "b.mp3"
        m.write_bytes(b"x")
        files = [w, str(m)]
        profile = self._profile(1_000_000)
        assert tts._pack_audio_files_for_delivery(files, profile) == [[w], [str(m)]]


class TestBuildAudioDeliveryFiles:
    def _profile(self, max_bytes):
        return tts.AudioDeliveryProfile("telegram", max_bytes, 0.85)

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tts._build_audio_delivery_files([], str(tmp_path / "out.wav"), self._profile(1000))

    def test_single_chunk_above_limit_raises(self, tmp_path):
        f = _mk_wav(tmp_path / "big.wav")
        with pytest.raises(ValueError):
            tts._build_audio_delivery_files([f], str(tmp_path / "out.wav"), self._profile(10))

    def test_single_file_passthrough(self, tmp_path):
        f = _mk_wav(tmp_path / "a.wav")
        paths, combined = tts._build_audio_delivery_files(
            [f], str(tmp_path / "out.wav"), self._profile(50_000_000)
        )
        assert combined is False
        # The single deliverable is moved to the requested output path.
        assert paths == [str(tmp_path / "out.wav")]
        assert Path(f).exists() is False

    def test_combines_files_under_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts, "_concat_audio_files", _fake_concat)
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(2)]
        paths, combined = tts._build_audio_delivery_files(
            files, str(tmp_path / "out.wav"), self._profile(50_000_000)
        )
        assert combined is True
        assert len(paths) == 1

    def test_over_limit_combined_splits_recursively(self, tmp_path, monkeypatch):
        # _concat always produces an over-limit artifact, so the group must split.
        monkeypatch.setattr(tts, "_concat_audio_files", _fake_concat_over_limit)
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(2)]
        paths, combined = tts._build_audio_delivery_files(
            files, str(tmp_path / "part.wav"), self._profile(1_000_000)
        )
        # Single-member groups are returned un-combined, so both originals come back.
        assert combined is False
        assert len(paths) == 2

    def test_over_limit_combined_unlink_failure_still_splits(self, tmp_path, monkeypatch):
        # Even when removing the over-limit combined artifact raises, the group must
        # still split back to its valid constituents (covers the unlink OSError guard).
        monkeypatch.setattr(tts, "_concat_audio_files", _fake_concat_over_limit)
        monkeypatch.setattr(Path, "unlink", _mock_raise_oserror)
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(2)]
        paths, combined = tts._build_audio_delivery_files(
            files, str(tmp_path / "part.wav"), self._profile(1_000_000)
        )
        assert combined is False
        assert len(paths) == 2

    def test_combine_failure_returns_originals(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts, "_concat_audio_files", lambda *a, **k: None)
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(2)]
        paths, combined = tts._build_audio_delivery_files(
            files, str(tmp_path / "out.wav"), self._profile(50_000_000)
        )
        assert combined is False
        assert len(paths) == 2

    @requires_ffmpeg
    def test_real_single_combined_deliverable(self, tmp_path):
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(2)]
        paths, combined = tts._build_audio_delivery_files(
            files, str(tmp_path / "out.wav"), self._profile(50_000_000)
        )
        assert combined is True
        assert len(paths) == 1
        assert tts._sniff_audio_container(paths[0]) == "wav"

    def test_multiple_deliverables_are_partitioned(self, tmp_path, monkeypatch):
        # Unlike the split case, force combine to fail so originals pass through
        # and then get re-named into .partNN destinations.
        monkeypatch.setattr(tts, "_concat_audio_files", lambda *a, **k: None)
        files = [_mk_wav(tmp_path / f"a{i}.wav") for i in range(2)]
        paths, combined = tts._build_audio_delivery_files(
            files, str(tmp_path / "out.wav"), self._profile(50_000_000)
        )
        assert combined is False
        assert all(".part0" in Path(p).name for p in paths)


def _fake_concat(paths, output, **kwargs):
    Path(output).write_bytes(b"combined-audio")
    return output


def _fake_concat_over_limit(paths, output, **kwargs):
    Path(output).write_bytes(b"x" * 2_000_000)
    return output


class TestResolveAudioDeliveryProfile:
    def test_default_platform(self):
        profile = tts._resolve_audio_delivery_profile(None)
        assert profile.platform == "default"
        assert profile.max_file_bytes == 10 * 1024 * 1024

    def test_known_platform_case_insensitive(self):
        profile = tts._resolve_audio_delivery_profile("  Telegram ")
        assert profile.platform == "telegram"
        assert profile.max_file_bytes == 50 * 1024 * 1024

    def test_per_platform_override(self):
        cfg = {"delivery_profiles": {"telegram": {"max_file_bytes": 999}}}
        profile = tts._resolve_audio_delivery_profile("telegram", cfg)
        assert profile.max_file_bytes == 999

    def test_override_rejects_none_values(self):
        cfg = {"delivery_profiles": {"telegram": {"max_file_bytes": None, "safety_ratio": None}}}
        profile = tts._resolve_audio_delivery_profile("telegram", cfg)
        assert profile.max_file_bytes == 50 * 1024 * 1024

    def test_invalid_max_file_bytes_false_backs_off(self):
        # Invalid values fall back to the DEFAULT profile's limit (10MB), not the
        # platform-specific one that was originally selected.
        for bad in (True, "ten", 0, -5):
            profile = tts._resolve_audio_delivery_profile("telegram", {"delivery_profiles": {"telegram": {"max_file_bytes": bad}}})
            assert profile.max_file_bytes == 10 * 1024 * 1024

    def test_invalid_safety_ratio_falls_back(self):
        for bad in (True, 0, 1.5, -0.1, "high"):
            profile = tts._resolve_audio_delivery_profile("telegram", {"delivery_profiles": {"telegram": {"safety_ratio": bad}}})
            assert profile.safety_ratio == 0.85

    def test_valid_safety_ratio_kept(self):
        profile = tts._resolve_audio_delivery_profile(None, {"delivery_profiles": {"default": {"safety_ratio": 0.5}}})
        assert profile.safety_ratio == 0.5

    def test_target_file_bytes_property(self):
        profile = tts.AudioDeliveryProfile("x", 100, 0.5)
        assert profile.target_file_bytes == 50

    def test_target_file_bytes_min_one(self):
        profile = tts.AudioDeliveryProfile("x", 1, 0.85)
        assert profile.target_file_bytes == 1

    def test_non_dict_delivery_profiles_ignored(self):
        profile = tts._resolve_audio_delivery_profile("telegram", {"delivery_profiles": "not-a-dict"})
        assert profile.max_file_bytes == 50 * 1024 * 1024
