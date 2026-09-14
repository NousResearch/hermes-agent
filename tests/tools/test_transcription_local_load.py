"""Cache-first loading for tools.transcription_local._load_local_whisper_model (#111072).

huggingface_hub's snapshot_download looks the Hub up *before* the local cache, so a fully
cached model silently stalls for minutes when huggingface.co is unreachable. The loader must
try local_files_only=True first, fall back to the online path only on a cache miss, and
surface the HF_ENDPOINT / HF_HUB_DISABLE_XET escape hatches when the Hub is unreachable.
"""

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

if "faster_whisper" not in sys.modules:
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = MagicMock(name="WhisperModel")
    # Set ``__spec__`` so ``importlib.util.find_spec("faster_whisper")`` doesn't raise during
    # collection (same trick as test_transcription_tools.py).
    from importlib.machinery import ModuleSpec

    faster_whisper_stub.__spec__ = ModuleSpec("faster_whisper", loader=None)
    sys.modules["faster_whisper"] = faster_whisper_stub


# str-lookalikes of the two LocalEntryNotFoundError variants huggingface_hub raises (the real
# class may not be importable here since faster-whisper is stubbed out).
_CACHE_MISS = Exception(
    "Cannot find an appropriate cached snapshot folder for the specified revision on the local "
    "disk and outgoing traffic has been disabled. To enable repo look-ups and downloads online, "
    "pass 'local_files_only=False' as input."
)
_HUB_TIMEOUT = Exception(
    "Got: ConnectTimeout: [Errno 110] Connection timed out\n"
    "An error happened while trying to locate the files on the Hub, and we cannot find the "
    "appropriate snapshot folder for the specified revision on the local disk. Please check "
    "your internet connection and try again."
)
_XET_401 = Exception(
    "Task error: File reconstruction error: CAS Client Error: Request error: HTTP status client "
    "error (401 Unauthorized), domain: https://cas-server.xethub.hf.co/v2/reconstructions/abc"
)


@pytest.fixture
def no_force_cpu(monkeypatch):
    monkeypatch.setattr(
        "tools.transcription_local._should_force_faster_whisper_cpu", lambda: False
    )


def test_cached_model_loads_without_network(no_force_cpu):
    """A fully cached model must be served by the local_files_only=True attempt alone."""
    model = MagicMock(name="model")
    whisper_cls = MagicMock(return_value=model)
    with patch("faster_whisper.WhisperModel", whisper_cls):
        from tools.transcription_local import _load_local_whisper_model

        assert _load_local_whisper_model("base") is model
    whisper_cls.assert_called_once_with(
        "base", device="auto", compute_type="auto", local_files_only=True
    )


def test_cache_miss_falls_back_to_online_download(no_force_cpu):
    """Cold cache: the cache-only failure is retried online exactly once."""
    model = MagicMock(name="model")
    whisper_cls = MagicMock(side_effect=[_CACHE_MISS, model])
    with patch("faster_whisper.WhisperModel", whisper_cls):
        from tools.transcription_local import _load_local_whisper_model

        assert (
            _load_local_whisper_model("base", device="cuda", compute_type="float16")
            is model
        )
    assert whisper_cls.call_args_list == [
        call("base", device="cuda", compute_type="float16", local_files_only=True),
        call("base", device="cuda", compute_type="float16", local_files_only=False),
    ]


def test_unreachable_hub_error_names_mirror_escape_hatch(no_force_cpu):
    """Cold cache + unreachable Hub: the bare ConnectTimeout becomes an actionable error."""
    whisper_cls = MagicMock(side_effect=[_CACHE_MISS, _HUB_TIMEOUT])
    with patch("faster_whisper.WhisperModel", whisper_cls):
        from tools.transcription_local import _load_local_whisper_model

        with pytest.raises(RuntimeError) as excinfo:
            _load_local_whisper_model("base")
    assert "HF_ENDPOINT=https://hf-mirror.com" in str(excinfo.value)
    assert "HF_HUB_DISABLE_XET=1" in str(excinfo.value)
    assert "base" in str(excinfo.value)


def test_xet_401_through_mirror_names_escape_hatch(no_force_cpu):
    """The mirror+Xet trap (CAS bridge ignores HF_ENDPOINT, 401) gets the same hint."""
    whisper_cls = MagicMock(side_effect=[_CACHE_MISS, _XET_401])
    with patch("faster_whisper.WhisperModel", whisper_cls):
        from tools.transcription_local import _load_local_whisper_model

        with pytest.raises(RuntimeError, match="HF_ENDPOINT"):
            _load_local_whisper_model("base")


def test_non_cache_miss_failure_propagates_untouched(no_force_cpu):
    """An invalid model size fails before any cache lookup and must not be retried online."""
    whisper_cls = MagicMock(side_effect=ValueError("Invalid model size 'gargantuan'"))
    with patch("faster_whisper.WhisperModel", whisper_cls):
        from tools.transcription_local import _load_local_whisper_model

        with pytest.raises(ValueError, match="Invalid model size"):
            _load_local_whisper_model("gargantuan")
    assert whisper_cls.call_count == 1


def test_cuda_lib_error_still_falls_back_to_cpu_cache_only(no_force_cpu):
    """The pre-existing CUDA→CPU fallback keeps working inside the cache-only attempt."""
    model = MagicMock(name="model")
    whisper_cls = MagicMock(
        side_effect=[
            RuntimeError("libcublas.so: cannot open shared object file"),
            model,
        ]
    )
    with patch("faster_whisper.WhisperModel", whisper_cls):
        from tools.transcription_local import _load_local_whisper_model

        assert _load_local_whisper_model("base") is model
    assert whisper_cls.call_args_list == [
        call("base", device="auto", compute_type="auto", local_files_only=True),
        call("base", device="cpu", compute_type="int8", local_files_only=True),
    ]


def test_apple_silicon_cpu_pin_stays_cache_only(monkeypatch):
    """The force-CPU path must also pass local_files_only (macOS hosts behind blocked networks)."""
    monkeypatch.setattr(
        "tools.transcription_local._should_force_faster_whisper_cpu", lambda: True
    )
    model = MagicMock(name="model")
    whisper_cls = MagicMock(return_value=model)
    with patch("faster_whisper.WhisperModel", whisper_cls):
        from tools.transcription_local import _load_local_whisper_model

        assert _load_local_whisper_model("base") is model
    whisper_cls.assert_called_once_with(
        "base", device="cpu", compute_type="int8", local_files_only=True
    )


def test_replace_cached_model_on_cpu_uses_cache_only():
    """The mid-transcribe CUDA evict-and-reload must not hit the network either (#111072)."""
    model = MagicMock(name="model")
    whisper_cls = MagicMock(return_value=model)
    with (
        patch("faster_whisper.WhisperModel", whisper_cls),
        patch("tools.transcription_tools._local_model", None),
        patch("tools.transcription_tools._local_model_name", None),
    ):
        from tools.transcription_tools import _replace_cached_model_on_cpu

        assert _replace_cached_model_on_cpu("base") is model
    whisper_cls.assert_called_once_with(
        "base", device="cpu", compute_type="int8", local_files_only=True
    )
