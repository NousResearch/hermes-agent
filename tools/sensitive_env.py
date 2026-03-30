"""Shared registry of Hermes-managed sensitive environment variables.

These vars are owned by Hermes itself (provider keys, gateway tokens,
tool credentials, etc.) and must never be exposed to sandboxed child
processes via skill-driven or config-driven passthrough.
"""

from __future__ import annotations


def build_sensitive_env_blocklist() -> frozenset[str]:
    """Return env vars that Hermes manages internally and should not forward."""
    blocked: set[str] = set()

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        for pconfig in PROVIDER_REGISTRY.values():
            blocked.update(pconfig.api_key_env_vars)
            if pconfig.base_url_env_var:
                blocked.add(pconfig.base_url_env_var)
    except ImportError:
        pass

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS

        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                blocked.add(name)
            elif category == "setting" and metadata.get("password"):
                blocked.add(name)
    except ImportError:
        pass

    blocked.update(
        {
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "OPENAI_ORG_ID",
            "OPENAI_ORGANIZATION",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "LLM_MODEL",
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "PERPLEXITY_API_KEY",
            "COHERE_API_KEY",
            "FIREWORKS_API_KEY",
            "XAI_API_KEY",
            "HELICONE_API_KEY",
            "PARALLEL_API_KEY",
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "TELEGRAM_HOME_CHANNEL",
            "TELEGRAM_HOME_CHANNEL_NAME",
            "DISCORD_HOME_CHANNEL",
            "DISCORD_HOME_CHANNEL_NAME",
            "DISCORD_REQUIRE_MENTION",
            "DISCORD_FREE_RESPONSE_CHANNELS",
            "DISCORD_AUTO_THREAD",
            "SLACK_HOME_CHANNEL",
            "SLACK_HOME_CHANNEL_NAME",
            "SLACK_ALLOWED_USERS",
            "WHATSAPP_ENABLED",
            "WHATSAPP_MODE",
            "WHATSAPP_ALLOWED_USERS",
            "SIGNAL_HTTP_URL",
            "SIGNAL_ACCOUNT",
            "SIGNAL_ALLOWED_USERS",
            "SIGNAL_GROUP_ALLOWED_USERS",
            "SIGNAL_HOME_CHANNEL",
            "SIGNAL_HOME_CHANNEL_NAME",
            "SIGNAL_IGNORE_STORIES",
            "HASS_TOKEN",
            "HASS_URL",
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
            "EMAIL_HOME_ADDRESS",
            "EMAIL_HOME_ADDRESS_NAME",
            "GATEWAY_ALLOWED_USERS",
            "GH_TOKEN",
            "GITHUB_APP_ID",
            "GITHUB_APP_PRIVATE_KEY_PATH",
            "GITHUB_APP_INSTALLATION_ID",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "DAYTONA_API_KEY",
        }
    )
    return frozenset(blocked)


HERMES_SENSITIVE_ENV_BLOCKLIST = build_sensitive_env_blocklist()
