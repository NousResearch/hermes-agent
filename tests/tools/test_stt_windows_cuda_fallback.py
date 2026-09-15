"""Regression tests for #103793: Windows STT fails with
"Library cublas64_12.dll is not found or cannot be loaded".

Two gaps let a first-use CUDA dlopen failure kill local STT with no CPU retry:

1. ``_CUDA_LIB_ERROR_MARKERS`` only listed POSIX library names (``libcublas``…).
   Windows DLLs carry no ``lib`` prefix, so ``cublas64_12.dll`` matched only by
   luck via the generic "cannot be loaded" phrasing.
2. faster-whisper's ``model.transcribe()`` is LAZY: it returns a generator and
   the decode — where the dlopen-on-first-use actually fires — happens while
   iterating segments. The mid-transcribe CUDA retry only wrapped the
   ``transcribe()`` call, so the error escaped the guard during iteration.
"""

from unittest.mock import MagicMock, patch
from importlib.machinery import ModuleSpec
import sys
import types

import pytest

# Stub faster_whisper when absent (same pattern as tests/tools/test_transcription_tools.py):
# patch("faster_whisper.WhisperModel", ...) needs an importable module, and CI
# does not install the STT extra.
if "faster_whisper" not in sys.modules:
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = MagicMock(name="WhisperModel")
    # find_spec() raises when __spec__ is None on a sys.modules stub.
    faster_whisper_stub.__spec__ = ModuleSpec("faster_whisper", loader=None)
    sys.modules["faster_whisper"] = faster_whisper_stub


class _FakeInfo:
    language = "en"
    duration = 1.0


class _Segments:
    """Iterable whose iteration itself fails — mirrors faster-whisper's lazy decode."""

    def __init__(self, exc):
        self._exc = exc

    def __iter__(self):
        raise self._exc
        yield  # pragma: no cover


# ===========================================================================
# Marker coverage: Windows DLL spellings must count as missing-CUDA-lib errors
# ===========================================================================

class TestWindowsCudaLibMarkers:
    @pytest.mark.parametrize("message", [
        "Library cublas64_12.dll is not found or cannot be loaded",
        "Library cublasLt64_12.dll is not found or cannot be loaded",
        "Library cudnn64_9.dll is not found or cannot be loaded",
        "Library cudart64_12.dll is not found or cannot be loaded",
    ])
    def test_windows_dll_names_are_cuda_lib_errors(self, message):
        from tools.transcription_local import _looks_like_cuda_lib_error
        assert _looks_like_cuda_lib_error(RuntimeError(message)) is True

    def test_posix_names_still_match(self):
        from tools.transcription_local import _looks_like_cuda_lib_error
        assert _looks_like_cuda_lib_error(
            RuntimeError("libcublas.so.12: cannot open shared object file")) is True

    def test_oom_still_not_a_missing_lib(self):
        from tools.transcription_local import _looks_like_cuda_lib_error
        assert _looks_like_cuda_lib_error(RuntimeError("CUDA out of memory")) is False


# ===========================================================================
# Mid-transcribe (iteration-time) dlopen failure retries on CPU
# ===========================================================================

class TestMidTranscribeDlopenFallback:
    def test_iteration_time_cuda_dlopen_retries_on_cpu(self, tmp_path):
        """The decode runs during segment ITERATION, not the transcribe() call:
        a first-use cuBLAS load failure there must evict the cached model and
        retry on CPU, not surface as a hard failure (#103793)."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        # First (CUDA) model: transcribe() succeeds, iteration raises the Windows
        # dlopen error. CPU model: fully succeeds.
        cuda_model = MagicMock()
        cuda_model.transcribe.return_value = (
            _Segments(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")),
            _FakeInfo(),
        )
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = (iter([]), _FakeInfo())

        mock_cls = MagicMock(side_effect=[cuda_model, cpu_model])

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True, result.get("error")
        # First load (CUDA) + CPU retry.
        assert mock_cls.call_count == 2
        # The retry must pin CPU + int8.
        retry_kwargs = mock_cls.call_args_list[1].kwargs
        assert retry_kwargs.get("device") == "cpu"
        assert retry_kwargs.get("compute_type") == "int8"

    def test_call_time_dlopen_still_retries_on_cpu(self, tmp_path):
        """The pre-existing shape — transcribe() itself raising the dlopen error —
        keeps working after the guard moves to cover iteration."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        cuda_model = MagicMock()
        cuda_model.transcribe.side_effect = RuntimeError(
            "libcublas.so.12: cannot open shared object file")
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = (iter([]), _FakeInfo())

        mock_cls = MagicMock(side_effect=[cuda_model, cpu_model])

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True, result.get("error")
        assert mock_cls.call_count == 2
        assert mock_cls.call_args_list[1].kwargs.get("device") == "cpu"

    def test_iteration_time_oom_surfaces_as_error(self, tmp_path):
        """A real runtime failure during iteration must NOT trigger the CPU retry."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        cuda_model = MagicMock()
        cuda_model.transcribe.return_value = (_Segments(RuntimeError("CUDA out of memory")), _FakeInfo())

        mock_cls = MagicMock(return_value=cuda_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]
        # No CPU retry for a non-missing-lib failure.
        assert mock_cls.call_count == 1