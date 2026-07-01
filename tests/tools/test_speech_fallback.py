"""Tests for the SpeechRouter fallback chain (Speech Router RFC, PR3).

Nothing here talks to a real TTS engine -- command providers are portable
shell commands (python -c) so these run identically on Linux/macOS/Windows,
matching the existing convention in test_tts_command_providers.py.
"""

import json
import sys
from unittest.mock import patch

from tools.speech.router import SpeechRouter
from tools.tts_tool import text_to_speech_tool


def _copy_command() -> str:
    """Shell command that copies {input_path} bytes -> {output_path}."""
    interpreter = sys.executable
    return (
        f'"{interpreter}" -c "import shutil, sys; '
        f'shutil.copyfile(sys.argv[1], sys.argv[2])" '
        f'{{input_path}} {{output_path}}'
    )


def _failing_command() -> str:
    """Shell command that always exits non-zero, producing no output file."""
    interpreter = sys.executable
    return f'"{interpreter}" -c "import sys; sys.exit(1)"'


def _config_with_chain(*provider_names: str, providers: dict) -> dict:
    return {
        "provider": provider_names[0],
        "router": {"providers": list(provider_names)},
        "providers": providers,
    }


class TestResolveProviderChain:
    def test_no_router_config_returns_single_default_provider(self):
        cfg = {"provider": "edge"}
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            chain, tts_config = SpeechRouter().resolve_provider_chain()
        assert chain == ["edge"]
        assert tts_config == cfg

    def test_empty_router_providers_list_falls_back_to_default(self):
        cfg = {"provider": "openai", "router": {"providers": []}}
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            chain, _ = SpeechRouter().resolve_provider_chain()
        assert chain == ["openai"]

    def test_configured_chain_is_returned_in_order(self):
        cfg = {
            "provider": "edge",
            "router": {"providers": ["local-gpu", "cloud-fallback", "edge"]},
        }
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            chain, _ = SpeechRouter().resolve_provider_chain()
        assert chain == ["local-gpu", "cloud-fallback", "edge"]


class TestTextToSpeechFallback:
    def test_first_provider_failure_falls_through_to_second(self, tmp_path):
        cfg = _config_with_chain(
            "dead-provider", "backup-provider",
            providers={
                "dead-provider": {"type": "command", "command": _failing_command()},
                "backup-provider": {"type": "command", "command": _copy_command()},
            },
        )
        out = tmp_path / "reply.mp3"
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result_json = text_to_speech_tool(text="hi", output_path=str(out))
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["provider"] == "backup-provider"

    def test_default_single_provider_does_not_fallback(self, tmp_path):
        """No tts.router.providers configured -- a failure must surface
        immediately, exactly like every pre-Router release. This is the
        compatibility guardrail: fallback is opt-in, never automatic."""
        cfg = {
            "provider": "dead-provider",
            "providers": {
                "dead-provider": {"type": "command", "command": _failing_command()},
            },
        }
        out = tmp_path / "reply.mp3"
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result_json = text_to_speech_tool(text="hi", output_path=str(out))
        result = json.loads(result_json)

        assert result["success"] is False

    def test_all_providers_failing_returns_last_error(self, tmp_path):
        cfg = _config_with_chain(
            "dead-one", "dead-two",
            providers={
                "dead-one": {"type": "command", "command": _failing_command()},
                "dead-two": {"type": "command", "command": _failing_command()},
            },
        )
        out = tmp_path / "reply.mp3"
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result_json = text_to_speech_tool(text="hi", output_path=str(out))
        result = json.loads(result_json)

        assert result["success"] is False

    def test_first_provider_success_never_tries_second(self, tmp_path):
        """A successful first attempt must not touch the fallback provider
        at all -- fallback only fires on failure."""
        cfg = _config_with_chain(
            "primary", "should-never-run",
            providers={
                "primary": {"type": "command", "command": _copy_command()},
                "should-never-run": {"type": "command", "command": _failing_command()},
            },
        )
        out = tmp_path / "reply.mp3"
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result_json = text_to_speech_tool(text="hi", output_path=str(out))
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["provider"] == "primary"
