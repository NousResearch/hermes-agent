import importlib

from hermes_constants import display_hermes_home


def test_text_to_speech_schema_uses_profile_safe_default_path():
    """Tool schema should not hardcode ~/.hermes paths (breaks profiles)."""
    # pytest's autouse fixture sets HERMES_HOME at runtime (after test collection),
    # so reload the module to ensure any schema strings computed at import time
    # reflect the isolated HERMES_HOME.
    tts_tool = importlib.import_module("tools.tts_tool")
    tts_tool = importlib.reload(tts_tool)

    desc = tts_tool.TTS_SCHEMA["parameters"]["properties"]["output_path"]["description"]

    # Should mention the active HERMES_HOME (or its ~/ shorthand) via display_hermes_home().
    assert display_hermes_home() in desc
    # Default directory should be the consolidated cache path for new installs.
    assert ("cache/audio" in desc) or ("audio_cache" in desc)

