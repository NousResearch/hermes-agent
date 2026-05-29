"""Discord tool token resolution tests."""

from hermes_cli.config import get_config_path, get_env_path, invalidate_env_cache


def _write_discord_env(token: str = "discord-file-token") -> None:
    get_env_path().write_text(f"DISCORD_BOT_TOKEN={token}\n", encoding="utf-8")
    invalidate_env_cache()


def test_discord_tool_check_reads_hermes_env(monkeypatch):
    """The Discord REST tool should honor tokens stored in Hermes .env."""
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    _write_discord_env()

    from tools.discord_tool import _get_bot_token, check_discord_tool_requirements

    assert _get_bot_token() == "discord-file-token"
    assert check_discord_tool_requirements() is True


def test_discord_session_context_detects_hermes_env_token(monkeypatch):
    """Discord gateway context should not hide tools when token lives in .env."""
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    _write_discord_env()
    get_config_path().write_text(
        "platform_toolsets:\n"
        "  discord:\n"
        "  - discord\n",
        encoding="utf-8",
    )

    from gateway.session import _discord_token_available, _discord_tools_loaded

    assert _discord_token_available() is True
    assert _discord_tools_loaded() is True
