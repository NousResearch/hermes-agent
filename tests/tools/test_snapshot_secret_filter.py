"""Regression tests for terminal snapshot secret filtering (issue #48441).

The snapshot mechanism uses ``export -p`` to dump all env vars to a file.
Without filtering, API keys, tokens, and passwords from .env are leaked to
disk in plaintext.  The ``_SECRET_ENV_GREP`` pattern filters them out.
"""

import subprocess

import pytest

from tools.environments.base import _SECRET_ENV_GREP


def _run_filter(input_text: str) -> str:
    """Run the grep filter on the given text and return stdout."""
    result = subprocess.run(
        ["bash", "-c", _SECRET_ENV_GREP],
        input=input_text,
        capture_output=True,
        text=True,
    )
    return result.stdout


FAKE_KEY = "sk-" + "x" * 20
FAKE_TOKEN = "ghp_" + "y" * 20


class TestSecretEnvGrep:
    """Verify _SECRET_ENV_GREP filters secret-bearing env vars."""

    def test_filters_api_key(self):
        output = _run_filter(f"declare -x OPENAI_API_KEY={FAKE_KEY}")
        assert "OPENAI_API_KEY" not in output

    def test_filters_secret(self):
        output = _run_filter("declare -x FEISHU_APP_SECRET=s3cret")
        assert "FEISHU_APP_SECRET" not in output

    def test_filters_token(self):
        output = _run_filter(f"declare -x GITHUB_TOKEN={FAKE_TOKEN}")
        assert "GITHUB_TOKEN" not in output

    def test_filters_password(self):
        output = _run_filter("declare -x DB_PASSWORD=hunter2")
        assert "DB_PASSWORD" not in output

    def test_filters_credential(self):
        output = _run_filter("declare -x AWS_CREDENTIAL=abc123")
        assert "AWS_CREDENTIAL" not in output

    def test_preserves_non_secret_vars(self):
        output = _run_filter('declare -x HOME="/home/user"')
        assert "HOME" in output

    def test_preserves_path(self):
        output = _run_filter('declare -x PATH="/usr/bin:/usr/local/bin"')
        assert "PATH" in output

    def test_preserves_custom_non_secret(self):
        output = _run_filter('declare -x MY_APP_VERSION="1.0.0"')
        assert "MY_APP_VERSION" in output

    def test_filters_multiple_secrets_among_normal_vars(self):
        input_text = "\n".join([
            'declare -x HOME="/home/user"',
            f"declare -x API_KEY={FAKE_KEY}",
            'declare -x PATH="/usr/bin"',
            f"declare -x DEEPSEEK_TOKEN={FAKE_TOKEN}",
            'declare -x EDITOR="vim"',
        ])
        output = _run_filter(input_text)
        assert "HOME" in output
        assert "PATH" in output
        assert "EDITOR" in output
        assert "API_KEY" not in output
        assert "DEEPSEEK_TOKEN" not in output

    def test_filters_verification_token(self):
        output = _run_filter("declare -x FEISHU_VERIFICATION_TOKEN=tok123")
        assert "FEISHU_VERIFICATION_TOKEN" not in output
