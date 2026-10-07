"""Speech's Hub loader uses package cache for both offline lookup and download."""

import sys
from types import SimpleNamespace

import pytest

from hermes_platform import windows_appdata
from tools.transcription_local import _create_whisper_model


@pytest.mark.parametrize("owner", ["package", "unpackaged", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"])
def test_speech_cache_owner_survives_offline_miss(tmp_path, monkeypatch, owner):
    import hermes_cache

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    cache = tmp_path / "package-cache"
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(key, raising=False)
    if owner.startswith("HF") or owner == "HUGGINGFACE_HUB_CACHE":
        monkeypatch.setenv(owner, str(tmp_path / "shared-hub"))
    monkeypatch.setattr(windows_appdata, "local_cache_folder", lambda: None if owner == "unpackaged" else cache)
    monkeypatch.setattr(hermes_cache, "local_cache_folder", windows_appdata.local_cache_folder)
    monkeypatch.setattr(hermes_cache, "_get_platform_default_hermes_home", lambda: home)
    calls = []
    loaded = object()

    def load_model(name, **kwargs):
        calls.append((name, kwargs))
        if kwargs["local_files_only"]:
            raise RuntimeError("Unable to open file model.bin")
        return loaded

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=load_model))
    assert _create_whisper_model("base", device="cpu", compute_type="int8") is loaded
    assert [kwargs["local_files_only"] for _, kwargs in calls] == [True, False]
    for name, kwargs in calls:
        assert name == "base"
        if owner == "package":
            assert kwargs["download_root"] == str(cache / "hermes-inference/cache/faster-whisper")
        else:
            assert "download_root" not in kwargs
