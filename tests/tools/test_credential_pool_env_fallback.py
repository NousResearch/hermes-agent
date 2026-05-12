"""Tests for credential_pool .env fallback and auth credential_pool lookup.

Covers the fix from #15914 / PR #15920:
- _seed_from_env reads API keys from ~/.hermes/.env when not in os.environ
- _resolve_api_key_provider_secret falls back to credential_pool when env vars are empty
- env vars take priority over .env file (handled by get_env_value itself)
- env vars take priority over credential pool (fallback only kicks in when env is empty)
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_pconfig(provider_id="deepseek", env_vars=None):
    """Create a minimal ProviderConfig for testing.

    Default provider_id is 'deepseek' because it's a real api_key provider
    in PROVIDER_REGISTRY (needed for _seed_from_env's generic path).
    """
    from hermes_cli.auth import ProviderConfig
    return ProviderConfig(
        id=provider_id,
        name=provider_id.title(),
        auth_type="api_key",
        api_key_env_vars=tuple(env_vars or [f"{provider_id.upper()}_API_KEY"]),
    )


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and clear known API key env vars.

    Also invalidates any cached get_env_value state by patching Path.home().
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Clear all known API key env vars so get_env_value falls through to .env
    for key in [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "ZAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    return home


def _write_env_file(home: Path, **kwargs) -> None:
    """Write key=value pairs to ~/.hermes/.env."""
    lines = [f"{k}={v}" for k, v in kwargs.items()]
    (home / ".env").write_text("\n".join(lines) + "\n")


class TestCredentialPoolSeedsFromDotEnv:
    """_seed_from_env must read keys from ~/.hermes/.env, not just os.environ.

    This is the load-bearing behaviour for the fix: when a user adds a key to
    .env mid-session or via a non-CLI entry point that doesn't run
    load_hermes_dotenv, the credential pool must still discover it.
    """

    def test_deepseek_key_from_dotenv_only(self, isolated_hermes_home):
        """Key in .env but not os.environ → _seed_from_env adds a pool entry."""
        _write_env_file(isolated_hermes_home, DEEPSEEK_API_KEY="sk-dotenv-only-12345")
        assert "DEEPSEEK_API_KEY" not in os.environ

        from agent.credential_pool import _seed_from_env
        entries = []
        changed, active_sources = _seed_from_env("deepseek", entries)

        assert changed is True
        assert "env:DEEPSEEK_API_KEY" in active_sources
        assert any(
            e.access_token == "sk-dotenv-only-12345"
            and e.source == "env:DEEPSEEK_API_KEY"
            for e in entries
        ), f"Expected seeded entry with dotenv key, got: {[(e.source, e.access_token) for e in entries]}"

    def test_openrouter_key_from_dotenv_only(self, isolated_hermes_home):
        """OpenRouter path has its own branch — verify it also reads .env."""
        _write_env_file(isolated_hermes_home, OPENROUTER_API_KEY="sk-or-dotenv-67890")
        assert "OPENROUTER_API_KEY" not in os.environ

        from agent.credential_pool import _seed_from_env
        entries = []
        changed, active_sources = _seed_from_env("openrouter", entries)

        assert changed is True
        assert "env:OPENROUTER_API_KEY" in active_sources
        assert any(
            e.access_token == "sk-or-dotenv-67890" for e in entries
        )

    def test_empty_dotenv_no_entries(self, isolated_hermes_home):
        """No .env file, no env vars → no entries seeded (and no crash)."""
        from agent.credential_pool import _seed_from_env
        entries = []
        changed, active_sources = _seed_from_env("deepseek", entries)
        assert changed is False
        assert active_sources == set()
        assert entries == []



class TestAuthResolvesFromDotEnv:
    """_resolve_api_key_provider_secret must also read from ~/.hermes/.env."""

    def test_key_from_dotenv_only(self, isolated_hermes_home):
        """Key in .env but not os.environ → _resolve returns it with the env var source."""
        _write_env_file(isolated_hermes_home, DEEPSEEK_API_KEY="sk-dot...-789")
        assert "DEEPSEEK_API_KEY" not in os.environ

        from hermes_cli.auth import _resolve_api_key_provider_secret
        key, source = _resolve_api_key_provider_secret(
            provider_id="deepseek",
            pconfig=_make_pconfig(),
        )
        assert key == "sk-dot...-789"
        assert source == "DEEPSEEK_API_KEY"

    def test_key_from_os_environ_overrides_dotenv(self, isolated_hermes_home, monkeypatch):
        """os.environ takes priority over .env file for the same key."""
        _write_env_file(isolated_hermes_home, DEEPSEEK_API_KEY="sk-dot...-789")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env...-abc")

        from hermes_cli.auth import _resolve_api_key_provider_secret
        key, source = _resolve_api_key_provider_secret(
            provider_id="deepseek",
            pconfig=_make_pconfig(),
        )
        assert key == "sk-env...-abc"
        assert source == "DEEPSEEK_API_KEY"

    def test_no_key_anywhere_returns_none(self, isolated_hermes_home):
        """Neither .env nor os.environ has the key → _resolve returns (None, None)."""
        assert "DEEPSEEK_API_KEY" not in os.environ

        from hermes_cli.auth import _resolve_api_key_provider_secret
        key, source = _resolve_api_key_provider_secret(
            provider_id="deepseek",
            pconfig=_make_pconfig(),
        )
        assert key in (None, ""), f"Expected None or empty, got key={key!r}"
        assert source in (None, ""), f"Expected None or empty, got source={source!r}"
