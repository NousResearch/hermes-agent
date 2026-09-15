"""Language choices survive the fallback to Hermes-generated Whisper CLI calls."""

import json
import os
from pathlib import Path
import shlex
import sys
import wave

import pytest
import yaml


@pytest.fixture
def cli_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
    from tools import transcription_tools as tt

    monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", False)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    audio = tmp_path / "voice clip;not-a-command.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)
    return tmp_path, audio


def dispatch(home, audio, stt):
    from tools import transcription_tools as tt

    (home / "config.yaml").write_text(yaml.safe_dump({"stt": stt}), encoding="utf-8")
    cfg = tt._load_stt_config()
    provider = tt._get_provider(cfg)
    assert provider == "local_command"
    result = tt._dispatch_stt_provider(str(audio), provider, cfg)
    assert result["success"], result
    return json.loads(result["transcript"])


@pytest.mark.parametrize("global_lang,local_lang,env_lang,hook_lang,provider,expected", [
    ("", "", None, None, "local", None),
    ("", "", None, None, "local_command", None),
    ("en", "", None, None, "local", "en"),
    ("en", "zh", None, None, "local", "zh"),
    ("", "", "zh", None, "local", "zh"),
    ("en", "", "zh", None, "local", "en"),
    ("en", "zh", None, "ja", "local", "ja"),
], ids=["auto-fallback", "auto-explicit-cli", "global-en", "local-zh", "env-zh",
        "config-over-env", "hook-over-config"])
def test_discovered_cli_language(
    cli_home, monkeypatch, global_lang, local_lang, env_lang, hook_lang, provider, expected
):
    home, audio = cli_home
    binary = str(home / "Whisper Tools" / "whisper")
    monkeypatch.setattr("tools.transcription_local._find_whisper_binary", lambda: binary)
    if env_lang:
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", env_lang)
    if hook_lang:
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **kw: [{"language": hook_lang}])

    def capture(argv, **kwargs):
        assert argv[0] == binary
        assert argv[1] == str(audio)
        language = argv[argv.index("--language") + 1] if "--language" in argv else None
        out = Path(argv[argv.index("--output_dir") + 1]) / "transcript.txt"
        out.write_text(json.dumps({"language": language}), encoding="utf-8")

    monkeypatch.setattr("tools.transcription_local._run_quiet", capture)
    output = dispatch(home, audio, {"provider": provider, "language": global_lang,
                                    "local": {"model": "base", "language": local_lang}})
    assert output["language"] == expected


@pytest.mark.parametrize("template_lang,expected", [("{language}", "en"), ("auto", "auto")])
def test_custom_template_keeps_its_existing_language_behavior(cli_home, monkeypatch, template_lang, expected):
    home, audio = cli_home
    template = f"custom-asr {{input_path}} --output_dir {{output_dir}} --language {template_lang}"
    monkeypatch.setenv("HERMES_LOCAL_STT_COMMAND", template)

    def capture(argv, **kwargs):
        assert argv[0] == "custom-asr"
        assert argv[1] == str(audio)
        out = Path(argv[argv.index("--output_dir") + 1]) / "transcript.txt"
        out.write_text(json.dumps({"language": argv[argv.index("--language") + 1]}), encoding="utf-8")

    monkeypatch.setattr("tools.transcription_local._run_quiet", capture)
    output = dispatch(home, audio, {"provider": "local", "language": "", "local": {"language": ""}})
    assert output["language"] == expected


@pytest.mark.skipif(os.name == "nt", reason="Executable shebang fixture requires POSIX")
@pytest.mark.parametrize("language", ["", "zh"])
def test_discovered_cli_executes_with_expected_language(cli_home, monkeypatch, language):
    home, audio = cli_home
    binary = home / "Whisper Tools" / "whisper"
    binary.parent.mkdir()
    binary.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c "
        + shlex.quote(
            "import json,sys; from pathlib import Path; a=sys.argv[1:]; "
            "lang=a[a.index('--language')+1] if '--language' in a else None; "
            "out=Path(a[a.index('--output_dir')+1])/'fixture.txt'; "
            "out.write_text(json.dumps({'language':lang,'input':a[0]}))"
        ) + ' "$@"\n', encoding="utf-8"
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary.parent) + os.pathsep + os.environ.get("PATH", ""))
    from tools.transcription_audio import _find_whisper_binary

    assert Path(_find_whisper_binary()).resolve() == binary
    output = dispatch(home, audio, {"provider": "local", "language": language, "local": {"language": ""}})
    assert output["language"] == (language or None)
    assert output["input"] == str(audio)
