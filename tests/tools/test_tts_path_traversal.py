"""Regression: text_to_speech_tool output_path must reject '..' traversal.

The TTS surface accepts agent/user-supplied absolute paths (writing to a
chosen file is the whole point). What it must reject is paths that use
``..`` components to escape their declared base — those are almost
always either a bug or prompt-injection-controlled
(e.g. ``output_path="audio/../../etc/cron.d/x"``).
"""

import json

from tools.tts_tool import text_to_speech_tool


def test_output_path_rejects_traversal_escape():
    """A path with '..' components must be rejected before any provider work."""
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path="audio/../../etc/cron.d/malicious",
    ))
    assert result["success"] is False
    assert "traversal" in result["error"].lower()


def test_output_path_rejects_bare_dotdot():
    """Bare '..' prefix must be rejected."""
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path="../escape.mp3",
    ))
    assert result["success"] is False
    assert "traversal" in result["error"].lower()


def test_output_path_rejects_hermes_oauth_store(tmp_path, monkeypatch):
    """TTS output_path must not bypass the shared protected-file write guard."""
    import agent.file_safety as file_safety

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: hermes_home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: hermes_home)

    target = hermes_home / ".anthropic_oauth.json"
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path=str(target),
    ))

    assert result["success"] is False
    assert "protected system/credential" in result["error"]
    assert not target.exists()


def test_output_path_rejects_mcp_token_directory(tmp_path, monkeypatch):
    """TTS output_path must not write synthesized audio over MCP token files."""
    import agent.file_safety as file_safety

    hermes_home = tmp_path / "hermes-home"
    token_dir = hermes_home / "mcp-tokens"
    token_dir.mkdir(parents=True)
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: hermes_home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: hermes_home)

    target = token_dir / "server.mp3"
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path=str(target),
    ))

    assert result["success"] is False
    assert "protected system/credential" in result["error"]
    assert not target.exists()


def test_output_path_safe_root_denial_labels_safe_root(tmp_path, monkeypatch):
    """A caller-supplied output_path outside HERMES_WRITE_SAFE_ROOT is a
    safe-root denial and must be labeled as such, not conflated with a
    credential/system path (#97110)."""
    import agent.file_safety as file_safety
    from tools import tts_tool

    safe_root = tmp_path / "safe-root"
    safe_root.mkdir()
    monkeypatch.setattr(
        file_safety, "get_safe_write_roots", lambda: [str(safe_root)]
    )

    outside = tmp_path / "outside" / "tts.mp3"
    outside.parent.mkdir(exist_ok=True)

    # Drive through the same helper text_to_speech_tool uses.
    from agent.file_safety import get_write_denied_error

    denial = get_write_denied_error(str(outside), verb="output_path")
    assert denial is not None
    assert "HERMES_WRITE_SAFE_ROOT" in denial
    assert "protected" not in denial

    # And the tool itself reports the safe-root message.
    result = json.loads(tts_tool.text_to_speech_tool(
        text="hello",
        output_path=str(outside),
    ))
    assert result["success"] is False
    assert "HERMES_WRITE_SAFE_ROOT" in result["error"]
    assert "protected credential or system path" not in result["error"]
