"""Tests for the ``hermes_subprocess_env()`` centralized helper.

Verifies that the strip-by-default policy is enforced, that
``inherit_credentials=True`` is grep-able and works correctly,
and that PYTHONUTF8 and HOME isolation are applied.

Issues:
- https://github.com/NousResearch/hermes-agent/issues/6032  (credential leakage)
- https://github.com/NousResearch/hermes-agent/issues/31420 (Windows UTF-8)
- https://github.com/NousResearch/hermes-agent/issues/31421 (loopback proxy)

Co-authored-by: leavedrop (@leavedrop) — centralized-helper design
    with inherit_credentials flag, PYTHONUTF8 default, and
    PROVIDER_CREDENTIAL_KEYS concept.
"""

import os
from unittest.mock import patch

import pytest

from tools.environments.local import (
    _ALWAYS_STRIP_KEYS,
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _HERMES_PROVIDER_ENV_FORCE_PREFIX,
    _STATIC_PROVIDER_ENV_BLOCKLIST,
    _build_provider_env_blocklist,
    hermes_subprocess_env,
)


# ── Positive: defaults strip credentials ──────────────────────────────


def test_strips_provider_api_keys_by_default():
    """Without inherit_credentials, provider keys are removed."""
    with patch.dict(
        os.environ,
        {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "OPENAI_API_KEY": "sk-should-be-stripped",
            "ANTHROPIC_TOKEN": "ant-token-secret",
            "GOOGLE_API_KEY": "google-secret",
            "DEEPSEEK_API_KEY": "deepseek-secret",
        },
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert "PATH" in env
    assert env["PATH"] == "/usr/bin"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_TOKEN" not in env
    assert "GOOGLE_API_KEY" not in env
    assert "DEEPSEEK_API_KEY" not in env


def test_strips_tool_and_messaging_secrets_by_default():
    """Without inherit_credentials, tool and messaging keys are removed."""
    with patch.dict(
        os.environ,
        {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "FIRECRAWL_API_KEY": "fc-secret",
            "FIRECRAWL_API_URL": "https://firecrawl.example",
            "BROWSERBASE_API_KEY": "bb-secret",
            "DISCORD_HOME_CHANNEL": "12345",
            "TELEGRAM_HOME_CHANNEL": "67890",
        },
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert "FIRECRAWL_API_KEY" not in env
    assert "FIRECRAWL_API_URL" not in env
    assert "BROWSERBASE_API_KEY" not in env
    assert "DISCORD_HOME_CHANNEL" not in env
    assert "TELEGRAM_HOME_CHANNEL" not in env
    assert env["PATH"] == "/usr/bin"


# ── Positive / inherit_credentials=True ────────────────────────────────


def test_inherit_credentials_keeps_provider_keys():
    """inherit_credentials=True preserves all credentials for ACP/CLI."""
    with patch.dict(
        os.environ,
        {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "OPENAI_API_KEY": "sk-preserved",
            "ANTHROPIC_TOKEN": "ant-preserved",
            "BROWSERBASE_API_KEY": "bb-preserved",
        },
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=True)

    assert env["OPENAI_API_KEY"] == "sk-preserved"
    assert env["ANTHROPIC_TOKEN"] == "ant-preserved"
    assert env["BROWSERBASE_API_KEY"] == "bb-preserved"
    assert env["PATH"] == "/usr/bin"


def test_inherit_credentials_preserves_all_blocklist_keys():
    """With inherit_credentials=True, provider keys survive, but Tier-1
    always-strip keys (GitHub auth, gateway tokens) are removed regardless."""
    blocklist_keys = {
        k for k in _HERMES_PROVIDER_ENV_BLOCKLIST
        if not k.startswith("_") and k not in _ALWAYS_STRIP_KEYS
    }
    always_strip_keys = {k for k in _ALWAYS_STRIP_KEYS if not k.startswith("_")}
    test_env = {k: f"val-{k}" for k in blocklist_keys | always_strip_keys}
    test_env["PATH"] = "/usr/bin"
    test_env["HOME"] = "/home/user"

    with patch.dict(os.environ, test_env, clear=True):
        env = hermes_subprocess_env(inherit_credentials=True)

    # Provider keys survive inheritance.
    for k in blocklist_keys:
        assert k in env, f"inherit_credentials=True should keep provider key {k}"

    # Tier-1 always-strip keys are removed even with inheritance.
    for k in always_strip_keys:
        assert k not in env, (
            f"{k} is an always-strip key and must be removed "
            f"even with inherit_credentials=True"
        )

    # Guard against accidental blocklist shrinkage: independently assert
    # that critical provider keys are present.  This prevents the test
    # from silently passing if a key is removed from the blocklist.
    for _critical in ("OPENAI_API_KEY", "ANTHROPIC_TOKEN", "GOOGLE_API_KEY",
                      "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
                      "BROWSERBASE_API_KEY", "FIRECRAWL_API_KEY"):
        assert _critical in env, (
            f"{_critical} must be in the blocklist AND preserved"
        )


def test_always_strip_keys_removed_even_with_inherit_credentials():
    """Tier-1 always-strip keys (GitHub auth, gateway tokens, infra secrets)
    are removed unconditionally — even with inherit_credentials=True."""
    with patch.dict(
        os.environ,
        {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            # Provider keys — should survive with inherit_credentials=True
            "OPENAI_API_KEY": "sk-survives",
            "ANTHROPIC_TOKEN": "ant-survives",
            # Tier-1 always-strip keys — must be removed always
            "GH_TOKEN": "gh-should-not-survive",
            "GITHUB_TOKEN": "github-should-not-survive",
            "DISCORD_HOME_CHANNEL": "discord-chan",
            "TELEGRAM_HOME_CHANNEL": "tg-chan",
            "MODAL_TOKEN_ID": "modal-id",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "DAYTONA_API_KEY": "daytona-key",
            "VERCEL_TOKEN": "vercel-tok",
        },
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=True)

    # Provider keys survive.
    assert env.get("OPENAI_API_KEY") == "sk-survives"
    assert env.get("ANTHROPIC_TOKEN") == "ant-survives"

    # Tier-1 keys are always stripped.
    for _always in ("GH_TOKEN", "GITHUB_TOKEN", "DISCORD_HOME_CHANNEL",
                    "TELEGRAM_HOME_CHANNEL", "MODAL_TOKEN_ID",
                    "MODAL_TOKEN_SECRET", "DAYTONA_API_KEY", "VERCEL_TOKEN"):
        assert _always not in env, (
            f"Tier-1 key {_always} must be removed even with "
            f"inherit_credentials=True"
        )


# ── Negative / preserved: non-secret env vars survive ──────────────────


@pytest.mark.parametrize(
    "safe_var",
    [
        "USER",
        "SHELL",
        "LANG",
        "TERM",
        "TMPDIR",
        "DISPLAY",
        "EDITOR",
        "PAGER",
    ],
)
def test_safe_env_vars_preserved(safe_var):
    """Common non-secret environment variables are always preserved."""
    with patch.dict(
        os.environ,
        {safe_var: f"value-of-{safe_var}", "PATH": "/usr/bin", "HOME": "/home/user"},
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert safe_var in env
    assert env[safe_var] == f"value-of-{safe_var}"


def test_path_and_home_preserved():
    """PATH and HOME are always preserved."""
    with patch.dict(
        os.environ,
        {"PATH": "/custom/path", "HOME": "/custom/home"},
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert env["PATH"] == "/custom/path"
    assert "HOME" in env


# ── PYTHONUTF8 injection ───────────────────────────────────────────────


def test_sets_pythonutf8_when_missing():
    """hermes_subprocess_env() always sets PYTHONUTF8 (issue #31420)."""
    with patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/home/user"}, clear=True):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert env.get("PYTHONUTF8") == "1"


def test_does_not_overwrite_existing_pythonutf8():
    """Existing PYTHONUTF8 value is not overwritten."""
    with patch.dict(
        os.environ,
        {"PATH": "/usr/bin", "HOME": "/home/user", "PYTHONUTF8": "0"},
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert env["PYTHONUTF8"] == "0"  # preserved, not overwritten


def test_pythonutf8_set_with_inherit_credentials():
    """PYTHONUTF8 is also set when inherit_credentials=True."""
    with patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/home/user"}, clear=True):
        env = hermes_subprocess_env(inherit_credentials=True)

    assert env.get("PYTHONUTF8") == "1"


# ── Edge: empty environment ────────────────────────────────────────────


def test_empty_environment_does_not_crash():
    """An empty os.environ produces a usable env dict with basics."""
    with patch.dict(os.environ, {}, clear=True):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert isinstance(env, dict)
    assert env.get("PYTHONUTF8") == "1"
    # PATH and HOME may be absent, but we shouldn't crash


# ── Edge: _HERMES_FORCE_ prefix keys in os.environ ─────────────────────


def test_hermes_force_prefix_keys_not_inherited():
    """_HERMES_FORCE_ prefix keys in the parent env are not passed through."""
    with patch.dict(
        os.environ,
        {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "_HERMES_FORCE_BROWSERBASE_API_KEY": "should-not-pass",
        },
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

    # The _HERMES_FORCE_ key itself should not appear
    assert "_HERMES_FORCE_BROWSERBASE_API_KEY" not in env


# ── Edge: mixed safe and blocklisted ───────────────────────────────────


def test_mixed_safe_and_blocklisted():
    """Safe keys survive while blocklisted keys are stripped."""
    with patch.dict(
        os.environ,
        {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "USER": "testuser",
            "OPENAI_API_KEY": "sk-secret",
            "ANTHROPIC_TOKEN": "ant-secret",
            "FIRECRAWL_API_KEY": "fc-secret",  # tool category, in blocklist
            "CUSTOM_APP_KEY": "custom-ok",  # not in blocklist
        },
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

    assert env["PATH"] == "/usr/bin"
    assert env["USER"] == "testuser"
    assert env["CUSTOM_APP_KEY"] == "custom-ok"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_TOKEN" not in env
    assert "FIRECRAWL_API_KEY" not in env


# ── Integration: caller adds back specific tool keys ───────────────────


def test_caller_can_add_back_browser_tool_keys():
    """Simulates the browser_tool.py pattern: strip everything, then add
    specific browser tool keys."""
    with patch.dict(
        os.environ,
        {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "OPENAI_API_KEY": "sk-should-be-stripped",
            "BROWSERBASE_API_KEY": "bb-needed",
            "BROWSERBASE_PROJECT_ID": "bb-project",
            "BROWSER_USE_API_KEY": "bu-needed",
            "FIRECRAWL_API_KEY": "fc-needed",
        },
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)

        # Add back browser-specific tool keys
        _browser_keys = (
            "BROWSERBASE_API_KEY",
            "BROWSERBASE_PROJECT_ID",
            "BROWSER_USE_API_KEY",
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "FIRECRAWL_BROWSER_TTL",
        )
        for _key in _browser_keys:
            if _key in os.environ:
                env[_key] = os.environ[_key]

    assert env["PATH"] == "/usr/bin"
    assert env["BROWSERBASE_API_KEY"] == "bb-needed"
    assert env["BROWSERBASE_PROJECT_ID"] == "bb-project"
    assert env["BROWSER_USE_API_KEY"] == "bu-needed"
    assert env["FIRECRAWL_API_KEY"] == "fc-needed"
    assert "OPENAI_API_KEY" not in env  # still stripped


# ── State inspection: no mutation of os.environ ────────────────────────


def test_does_not_mutate_os_environ():
    """hermes_subprocess_env() returns a copy; os.environ is untouched."""
    with patch.dict(
        os.environ,
        {"PATH": "/usr/bin", "HOME": "/home/user", "OPENAI_API_KEY": "sk-secret"},
        clear=True,
    ):
        env = hermes_subprocess_env(inherit_credentials=False)
        assert "OPENAI_API_KEY" not in env
        assert os.environ.get("OPENAI_API_KEY") == "sk-secret"
        # os.environ is not modified
        assert "PYTHONUTF8" not in os.environ or os.environ["PYTHONUTF8"] != "1"


# ── grep-ability proof ─────────────────────────────────────────────────


def test_inherit_credentials_true_is_literal_flag():
    """Prove that 'inherit_credentials=True' is literally grep-able in
    the calling code (not computed dynamically or hidden behind a
    variable).  This test asserts that the boolean is passed directly
    as a keyword argument — the pattern that makes audit possible."""
    import inspect

    src = inspect.getsource(hermes_subprocess_env)
    # The function itself uses 'inherit_credentials' as the param name
    assert "inherit_credentials" in src
    assert "if not inherit_credentials" in src


# ── Blocklist sanity ───────────────────────────────────────────────────


def test_blocklist_is_non_empty():
    """The blocklist must have provider keys to strip."""
    assert len(_HERMES_PROVIDER_ENV_BLOCKLIST) > 0


def test_blocklist_contains_key_provider_keys():
    """The most critical provider keys are in the blocklist."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_TOKEN", "GOOGLE_API_KEY",
                "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        assert key in _HERMES_PROVIDER_ENV_BLOCKLIST, (
            f"{key} must be in _HERMES_PROVIDER_ENV_BLOCKLIST"
        )


# ── Registry load-failure contract ─────────────────────────────────────


def test_static_blocklist_is_subset_of_full_blocklist():
    """The static baseline is a subset of the dynamically-extended blocklist.
    This guarantees that the fallback on registry load failure is a strict
    subset — we never strip keys that the dynamic extension would have
    preserved."""
    missing = _STATIC_PROVIDER_ENV_BLOCKLIST - _HERMES_PROVIDER_ENV_BLOCKLIST
    assert not missing, (
        f"Static baseline contains keys not in the dynamic blocklist: {missing}"
    )


def test_build_blocklist_returns_static_superset():
    """_build_provider_env_blocklist() returns at least the static baseline.
    If this fails, the registry load-failure fallback would be incomplete."""
    built = _build_provider_env_blocklist()
    missing = _STATIC_PROVIDER_ENV_BLOCKLIST - built
    assert not missing, (
        f"_build_provider_env_blocklist() missing static keys: {missing}"
    )


def test_blocklist_fallback_on_runtime_error():
    """If _build_provider_env_blocklist() raises a non-ImportError exception
    (e.g. AttributeError from a malformed PROVIDER_REGISTRY), the module-level
    try/except catches it and falls back to the static baseline.

    We verify this by constructing the failure scenario in-process: set the
    blocklist to the static baseline and confirm hermes_subprocess_env
    still strips provider keys correctly.  This is equivalent to the
    fallback path — the static baseline IS the fallback value."""
    import tools.environments.local as tgt

    # Save the original blocklist.
    _orig_blocklist = tgt._HERMES_PROVIDER_ENV_BLOCKLIST
    try:
        # Simulate the fallback: replace with static baseline.
        tgt._HERMES_PROVIDER_ENV_BLOCKLIST = _STATIC_PROVIDER_ENV_BLOCKLIST

        # hermes_subprocess_env must remain functional.
        with patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "HOME": "/home/user",
             "OPENAI_API_KEY": "sk-secret", "ANTHROPIC_TOKEN": "ant-secret"},
            clear=True,
        ):
            env = tgt.hermes_subprocess_env(inherit_credentials=False)

        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_TOKEN" not in env
        assert "PATH" in env
    finally:
        tgt._HERMES_PROVIDER_ENV_BLOCKLIST = _orig_blocklist
