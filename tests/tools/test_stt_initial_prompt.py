"""Tests for stt.local.initial_prompt wiring in transcription_tools._transcribe_local.

The `initial_prompt` parameter biases faster-whisper's vocabulary toward
domain-specific terms (product names, technical jargon, etc.). This test
asserts the config-driven wiring:

  - stt.local.initial_prompt set in config.yaml  -> passed to transcribe()
  - stt.local.initial_prompt unset               -> omitted (not None)

We don't run a real model — we monkeypatch the WhisperModel on
``tools.transcription_tools`` so we can capture the kwargs dict and
assert on it. Faster-whisper is the heavy lift; the wiring is the
lightweight contract this test guards.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _reload_with_config(tmp_path, monkeypatch, stt_local_cfg):
    """Spin up a hermes_home with a given stt.local config, import the module."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.dump({"stt": {"enabled": True, "local": stt_local_cfg}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Drop any cached import so _load_stt_config re-reads from disk.
    for mod in list(sys.modules):
        if mod == "tools.transcription_tools" or mod.startswith("tools.transcription_tools."):
            del sys.modules[mod]

    import tools.transcription_tools as tt  # noqa: WPS433 - intentional import after sys.path setup
    return tt


def test_initial_prompt_passed_when_set(tmp_path, monkeypatch):
    tt = _reload_with_config(
        tmp_path,
        monkeypatch,
        {"model": "base", "language": "de", "initial_prompt": "DeepSeek, MiMo, MiniMax"},
    )

    # Stub the slower bits: model load + transcribe call.
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([MagicMock(text="ok")], MagicMock(language="de", language_probability=1.0))
    tt._local_model = fake_model
    tt._local_model_name = "base"

    result = tt._transcribe_local("/tmp/fake.ogg", model_name="base")
    assert result["success"] is True

    # The decisive assertion: transcribe() was called with initial_prompt kwarg.
    fake_model.transcribe.assert_called_once()
    _args, kwargs = fake_model.transcribe.call_args
    assert kwargs.get("initial_prompt") == "DeepSeek, MiMo, MiniMax"
    assert kwargs.get("language") == "de"


def test_initial_prompt_omitted_when_unset(tmp_path, monkeypatch):
    tt = _reload_with_config(
        tmp_path,
        monkeypatch,
        {"model": "base", "language": "de"},
    )

    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([MagicMock(text="ok")], MagicMock(language="de", language_probability=1.0))
    tt._local_model = fake_model
    tt._local_model_name = "base"

    result = tt._transcribe_local("/tmp/fake.ogg", model_name="base")
    assert result["success"] is True

    fake_model.transcribe.assert_called_once()
    _args, kwargs = fake_model.transcribe.call_args
    # No initial_prompt key when config doesn't set it.
    assert "initial_prompt" not in kwargs
    assert kwargs.get("language") == "de"


def test_initial_prompt_omitted_when_empty_string(tmp_path, monkeypatch):
    """Empty string is treated as 'unset' (whisper's own None == empty contract)."""
    tt = _reload_with_config(
        tmp_path,
        monkeypatch,
        {"model": "base", "initial_prompt": ""},
    )

    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([MagicMock(text="ok")], MagicMock(language="de", language_probability=1.0))
    tt._local_model = fake_model
    tt._local_model_name = "base"

    tt._transcribe_local("/tmp/fake.ogg", model_name="base")

    fake_model.transcribe.assert_called_once()
    _args, kwargs = fake_model.transcribe.call_args
    assert "initial_prompt" not in kwargs
