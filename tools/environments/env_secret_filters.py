"""Hermes-managed secret filters used by subprocess env sanitizers.

This module is deliberately small and has no imports from ``tools.env_passthrough``
or from the rest of the environment toolchain. Keeping it free of circular
dependencies lets both ``tools.environments.local`` (which needs the passthrough
registry) and ``tools.env_passthrough`` (which needs these filters) import the
same single source of truth without risking a partially-initialised import.
"""

# Hermes-internal env vars that should NOT leak into terminal subprocesses.
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"

# Hermes-managed AWS *inference* credentials for ``auth_type="aws_sdk"``
# providers (Bedrock).  Scoped DELIBERATELY NARROW: this lists only the
# Bedrock-specific bearer token, which is a Hermes inference secret exactly
# analogous to ``OPENAI_API_KEY`` — nobody drives the ``aws``/``terraform``/
# ``boto3`` toolchain off it, so stripping it from terminal/execute_code
# subprocesses costs no user capability.
#
# The GENERAL AWS credential chain (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# AWS_SESSION_TOKEN, AWS_PROFILE, and the config/role pointers) is INTENTIONALLY
# left inheritable.  Per SECURITY.md §3.2 the local terminal is the user's
# trusted operator shell; the agent having the same general AWS access the
# user's own shell has is the intended posture, not a leak.  Hard-blocklisting
# those vars would (a) regress every user who runs aws/terraform/cdk/boto3 in
# the agent terminal — not just Bedrock users, since the registry is iterated
# unconditionally — and (b) be unrecoverable, because env_passthrough.py
# refuses to re-allow anything in this blocklist (GHSA-rhgp-j443-p4rf).  See
# issue #32314 discussion.
_AWS_SDK_CREDENTIAL_ENV_VARS = frozenset({
    "AWS_BEARER_TOKEN_BEDROCK",
})


def _build_provider_env_blocklist() -> frozenset:
    """Derive the blocklist from provider, tool, and gateway config."""
    blocked: set[str] = set()

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pconfig in PROVIDER_REGISTRY.values():
            blocked.update(pconfig.api_key_env_vars)
            if pconfig.auth_type == "aws_sdk":
                blocked.update(_AWS_SDK_CREDENTIAL_ENV_VARS)
            if pconfig.base_url_env_var:
                blocked.add(pconfig.base_url_env_var)
    except ImportError:
        pass

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS
        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"} or category == "setting" and metadata.get("password"):
                blocked.add(name)
    except ImportError:
        pass

    blocked.update({
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
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
        "HERMES_DASHBOARD_SESSION_TOKEN",
        "GATEWAY_ALLOWED_USERS",
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_DELIVERY_KEY",
    })
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()


def _is_hermes_internal_secret(key: str) -> bool:
    """Return True for Hermes-internal secrets injected under *dynamic* names.

    ``_HERMES_PROVIDER_ENV_BLOCKLIST`` is name-based and derived from the
    provider/tool registries, but the gateway and CLI also inject secrets into
    ``os.environ`` at runtime under names no static registry knows about:

    - ``AUXILIARY_<TASK>_API_KEY`` / ``AUXILIARY_<TASK>_BASE_URL`` — per-task
      side-LLM credentials bridged from ``config.yaml[auxiliary]`` by
      ``gateway/run.py`` and ``cli.py`` (vision, web_extract, approval,
      compression, and any plugin-registered auxiliary task). These are
      separate, often higher-spend API keys plus base URLs that may point at
      private endpoints; a model-authored shell command must never see them.
    - ``GATEWAY_RELAY_*_SECRET`` / ``GATEWAY_RELAY_*_KEY`` /
      ``GATEWAY_RELAY_*_TOKEN`` — relay-auth material provisioned by the
      gateway (``GATEWAY_RELAY_SECRET``, ``GATEWAY_RELAY_DELIVERY_KEY``).
      These are Tier-1 gateway secrets, like the messaging bot tokens in
      ``_ALWAYS_STRIP_KEYS``. Non-secret ``GATEWAY_RELAY_*`` routing hints
      (``GATEWAY_RELAY_URL``, ``GATEWAY_RELAY_PLATFORMS``, …) are NOT matched
      and remain visible.

    ``code_execution_tool.py`` already catches these via substring matching on
    ``KEY`` / ``SECRET`` / ``TOKEN``; the terminal backend's narrower name-based
    blocklist did not, which is the leak this predicate closes.

    This is the single source of truth for "Hermes-internal dynamic secret"
    across every spawn path — the terminal ``_make_run_env`` /
    ``_sanitize_subprocess_env`` filters, the Docker passthrough filter, and the
    non-terminal :func:`hermes_subprocess_env` helper all call it, so the
    dynamic patterns are stripped **unconditionally** regardless of
    ``env_passthrough`` skill registration or ``inherit_credentials``. Nothing
    a model-driving CLI legitimately needs matches these patterns.
    """
    upper = key.upper()
    if upper.startswith("AUXILIARY_") and (
        upper.endswith("_API_KEY") or upper.endswith("_BASE_URL")
    ):
        return True
    if upper.startswith("GATEWAY_RELAY_") and (
        upper.endswith("_SECRET") or upper.endswith("_KEY") or upper.endswith("_TOKEN")
    ):
        return True
    return False
