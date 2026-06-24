"""Tests for tools/neutts_synth.py — standalone NeuTTS synthesis helper."""

import sys
import wave
from unittest.mock import patch, MagicMock

import pytest


class _FakeNdarray:
    """Stand-in class so isinstance(x, np.ndarray) works with mocked numpy."""
    pass


def _mock_numpy():
    """Create a mock numpy module that handles array/clip/astype operations."""
    np = MagicMock()
    np.ndarray = _FakeNdarray
    np.created_arrays = []

    def _make_array(data, *args, **kwargs):
        n = len(data) if hasattr(data, '__len__') else 1
        arr = MagicMock()
        arr.__len__ = lambda self: n
        arr.__mul__ = lambda self, other: arr
        arr.flatten.return_value = arr
        arr.clip.return_value = arr
        arr.astype.return_value = arr
        arr.tobytes.return_value = b"\x00\x00" * n
        np.created_arrays.append(arr)
        return arr

    np.array.side_effect = _make_array
    np.clip.side_effect = lambda x, lo, hi: x  # return input unchanged
    np.float32 = "float32"
    np.int16 = "int16"
    return np


# ── _write_wav ────────────────────────────────────────────────────────


class TestWriteWav:
    def test_writes_valid_wav_file(self, tmp_path):
        """_write_wav produces a valid WAV file with correct header."""
        from tools.neutts_synth import _write_wav

        mock_np = _mock_numpy()
        with patch.dict("sys.modules", {"numpy": mock_np}):
            _write_wav(str(tmp_path / "out.wav"), [0.0, 0.5, -0.5], sample_rate=24000)

        with wave.open(str(tmp_path / "out.wav"), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000

    def test_converts_list_to_ndarray(self, tmp_path):
        """Non-ndarray input is converted to float32 array."""
        from tools.neutts_synth import _write_wav

        mock_np = _mock_numpy()
        with patch.dict("sys.modules", {"numpy": mock_np}):
            _write_wav(str(tmp_path / "out.wav"), [0.0, 0.5, -0.5], sample_rate=16000)

        with wave.open(str(tmp_path / "out.wav"), "rb") as wf:
            assert wf.getframerate() == 16000

    def test_clamps_samples(self, tmp_path):
        """Samples outside [-1, 1] are clamped before conversion."""
        from tools.neutts_synth import _write_wav

        mock_np = _mock_numpy()
        with patch.dict("sys.modules", {"numpy": mock_np}):
            _write_wav(str(tmp_path / "out.wav"), [2.0, -2.0, 0.0], sample_rate=24000)

        samples = mock_np.created_arrays[0]
        samples.flatten.assert_called_once_with()
        mock_np.clip.assert_called_once_with(samples, -1.0, 1.0)

    def test_flattens_multidimensional(self, tmp_path):
        """Multi-dimensional arrays are flattened."""
        from tools.neutts_synth import _write_wav

        mock_np = _mock_numpy()
        with patch.dict("sys.modules", {"numpy": mock_np}):
            _write_wav(str(tmp_path / "out.wav"), [[0.1, 0.2], [0.3, 0.4]], sample_rate=24000)

        samples = mock_np.created_arrays[0]
        samples.flatten.assert_called_once_with()
        mock_np.clip.assert_called_once_with(samples, -1.0, 1.0)

    def test_empty_samples(self, tmp_path):
        """Empty samples produce a valid WAV with 0 data."""
        from tools.neutts_synth import _write_wav

        mock_np = _mock_numpy()

        with patch.dict("sys.modules", {"numpy": mock_np}):
            _write_wav(str(tmp_path / "out.wav"), [], sample_rate=24000)

        with wave.open(str(tmp_path / "out.wav"), "rb") as wf:
            assert wf.getnframes() == 0

    def test_custom_sample_rate(self, tmp_path):
        """Custom sample rate is written to the header."""
        from tools.neutts_synth import _write_wav

        mock_np = _mock_numpy()
        with patch.dict("sys.modules", {"numpy": mock_np}):
            _write_wav(str(tmp_path / "out.wav"), [0.5], sample_rate=48000)

        with wave.open(str(tmp_path / "out.wav"), "rb") as wf:
            assert wf.getframerate() == 48000


# ── main() CLI entry point ─────────────────────────────────────────────


class TestMain:
    def test_missing_ref_audio_exits_1(self, tmp_path, capsys):
        """Missing reference audio file exits with code 1."""
        from tools.neutts_synth import main

        ref_text = tmp_path / "ref.txt"
        ref_text.write_text("hello")

        with pytest.raises(SystemExit) as exc_info:
            sys.argv = [
                "neutts_synth", "--text", "hi", "--out", str(tmp_path / "out.wav"),
                "--ref-audio", str(tmp_path / "nonexistent.wav"),
                "--ref-text", str(ref_text),
            ]
            main()
        assert exc_info.value.code == 1
        assert "reference audio not found" in capsys.readouterr().err

    def test_missing_ref_text_exits_1(self, tmp_path, capsys):
        """Missing reference text file exits with code 1."""
        from tools.neutts_synth import main

        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"fake audio")

        with pytest.raises(SystemExit) as exc_info:
            sys.argv = [
                "neutts_synth", "--text", "hi", "--out", str(tmp_path / "out.wav"),
                "--ref-audio", str(ref_audio),
                "--ref-text", str(tmp_path / "nonexistent.txt"),
            ]
            main()
        assert exc_info.value.code == 1
        assert "reference text not found" in capsys.readouterr().err

    def test_neutts_not_installed_exits_1(self, tmp_path, capsys):
        """When neutts is not installed, exits with code 1."""
        from tools.neutts_synth import main

        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"fake audio")
        ref_text = tmp_path / "ref.txt"
        ref_text.write_text("hello")

        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "neutts":
                raise ImportError("no neutts")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            with pytest.raises(SystemExit) as exc_info:
                sys.argv = [
                    "neutts_synth", "--text", "hi", "--out", str(tmp_path / "out.wav"),
                    "--ref-audio", str(ref_audio),
                    "--ref-text", str(ref_text),
                ]
                main()
        assert exc_info.value.code == 1
        assert "neutts not installed" in capsys.readouterr().err

    def test_successful_synthesis_with_soundfile(self, tmp_path):
        """Full synthesis path with soundfile available."""
        from tools.neutts_synth import main

        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"fake audio")
        ref_text = tmp_path / "ref.txt"
        ref_text.write_text("hello world")
        out_path = tmp_path / "output" / "out.wav"

        mock_wav = MagicMock()

        mock_tts = MagicMock()
        mock_tts.encode_reference.return_value = "codes"
        mock_tts.infer.return_value = mock_wav

        mock_neutts = MagicMock()
        mock_neutts.NeuTTS.return_value = mock_tts

        mock_sf = MagicMock()

        with patch.dict("sys.modules", {
            "neutts": mock_neutts,
            "soundfile": mock_sf,
        }):
            sys.argv = [
                "neutts_synth", "--text", "hi", "--out", str(out_path),
                "--ref-audio", str(ref_audio),
                "--ref-text", str(ref_text),
            ]
            main()

        mock_neutts.NeuTTS.assert_called_once()
        mock_tts.encode_reference.assert_called_once_with(str(ref_audio))
        mock_tts.infer.assert_called_once()
        mock_sf.write.assert_called_once()
        # Output directory was created
        assert out_path.parent.exists()

    def test_successful_synthesis_fallback_wav(self, tmp_path):
        """Full synthesis path without soundfile — falls back to _write_wav."""
        from tools.neutts_synth import main

        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"fake audio")
        ref_text = tmp_path / "ref.txt"
        ref_text.write_text("hello world")
        out_path = tmp_path / "out.wav"

        mock_wav = MagicMock()

        mock_tts = MagicMock()
        mock_tts.encode_reference.return_value = "codes"
        mock_tts.infer.return_value = mock_wav

        mock_neutts = MagicMock()
        mock_neutts.NeuTTS.return_value = mock_tts

        mock_np = _mock_numpy()

        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "soundfile":
                raise ImportError("no soundfile")
            if name == "neutts":
                return mock_neutts
            if name == "numpy":
                return mock_np
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            sys.argv = [
                "neutts_synth", "--text", "hi", "--out", str(out_path),
                "--ref-audio", str(ref_audio),
                "--ref-text", str(ref_text),
            ]
            main()

        # Fallback _write_wav was used — verify the file exists
        assert out_path.exists()

    def test_custom_model_and_device(self, tmp_path):
        """Custom --model and --device args are passed to NeuTTS."""
        from tools.neutts_synth import main

        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"fake audio")
        ref_text = tmp_path / "ref.txt"
        ref_text.write_text("hello")

        mock_wav = MagicMock()

        mock_tts = MagicMock()
        mock_tts.encode_reference.return_value = "codes"
        mock_tts.infer.return_value = mock_wav

        mock_neutts = MagicMock()
        mock_neutts.NeuTTS.return_value = mock_tts

        mock_sf = MagicMock()

        with patch.dict("sys.modules", {
            "neutts": mock_neutts,
            "soundfile": mock_sf,
        }):
            sys.argv = [
                "neutts_synth", "--text", "hi", "--out", str(tmp_path / "out.wav"),
                "--ref-audio", str(ref_audio),
                "--ref-text", str(ref_text),
                "--model", "custom/model-repo",
                "--device", "cuda",
            ]
            main()

        call_kwargs = mock_neutts.NeuTTS.call_args
        assert call_kwargs.kwargs["backbone_repo"] == "custom/model-repo"
        assert call_kwargs.kwargs["backbone_device"] == "cuda"
        assert call_kwargs.kwargs["codec_device"] == "cuda"
