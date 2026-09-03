"""Tests for config-driven git identity injection (config.yaml ``git.identity``).

Inspired by Amp's orb git-identity model ("No Mailmap Required", Aug 2026):
the harness — not the model — decides the authorship on agent-made commits.
See issues #72556 (agent silently runs ``git config user.email``) and
#78374 (shared placeholder identity misattributed on GitHub).
"""

from unittest.mock import patch

from tools.environments.local import (
    _make_run_env,
    _read_git_identity_config,
    _sanitize_subprocess_env,
    apply_git_identity_env,
)

_ALL_KEYS = (
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
)


def _with_identity(name="Jane Doe", email="jane@users.noreply.github.com"):
    return patch(
        "tools.environments.local._read_git_identity_config",
        return_value=(name, email),
    )


class TestApplyGitIdentityEnv:
    def test_noop_when_unconfigured(self):
        env = {"PATH": "/usr/bin"}
        with _with_identity("", ""):
            apply_git_identity_env(env)
        assert env == {"PATH": "/usr/bin"}

    def test_sets_all_four_keys(self):
        env = {}
        with _with_identity():
            apply_git_identity_env(env)
        assert env["GIT_AUTHOR_NAME"] == "Jane Doe"
        assert env["GIT_COMMITTER_NAME"] == "Jane Doe"
        assert env["GIT_AUTHOR_EMAIL"] == "jane@users.noreply.github.com"
        assert env["GIT_COMMITTER_EMAIL"] == "jane@users.noreply.github.com"

    def test_existing_env_values_win(self):
        env = {"GIT_AUTHOR_NAME": "Explicit Export", "GIT_AUTHOR_EMAIL": "x@y.z"}
        with _with_identity():
            apply_git_identity_env(env)
        # user-exported values are never clobbered
        assert env["GIT_AUTHOR_NAME"] == "Explicit Export"
        assert env["GIT_AUTHOR_EMAIL"] == "x@y.z"
        # unset siblings still filled from config
        assert env["GIT_COMMITTER_NAME"] == "Jane Doe"
        assert env["GIT_COMMITTER_EMAIL"] == "jane@users.noreply.github.com"

    def test_email_only_configuration(self):
        env = {}
        with _with_identity(name="", email="jane@example.com"):
            apply_git_identity_env(env)
        assert "GIT_AUTHOR_NAME" not in env
        assert env["GIT_AUTHOR_EMAIL"] == "jane@example.com"
        assert env["GIT_COMMITTER_EMAIL"] == "jane@example.com"

    def test_name_only_configuration(self):
        env = {}
        with _with_identity(name="Jane Doe", email=""):
            apply_git_identity_env(env)
        assert env["GIT_AUTHOR_NAME"] == "Jane Doe"
        assert "GIT_AUTHOR_EMAIL" not in env


class TestReadGitIdentityConfig:
    def test_reads_from_config(self):
        cfg = {"git": {"identity": {"name": "Jane", "email": "j@e.com"}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _read_git_identity_config() == ("Jane", "j@e.com")

    def test_missing_section_returns_empty(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _read_git_identity_config() == ("", "")

    def test_malformed_identity_returns_empty(self):
        cfg = {"git": {"identity": "Jane <j@e.com>"}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _read_git_identity_config() == ("", "")

    def test_newlines_clamped_to_first_line(self):
        # embedded newlines would corrupt the commit object header
        cfg = {"git": {"identity": {"name": "Jane\nEvil: header", "email": "j@e.com\nX: y"}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _read_git_identity_config() == ("Jane", "j@e.com")

    def test_config_failure_returns_empty(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert _read_git_identity_config() == ("", "")


class TestSpawnPathIntegration:
    def test_make_run_env_injects_identity(self):
        with _with_identity():
            env = _make_run_env({"PATH": "/usr/bin"})
        for key in _ALL_KEYS:
            assert env.get(key), f"{key} missing from _make_run_env output"

    def test_sanitize_subprocess_env_injects_identity(self):
        with _with_identity():
            env = _sanitize_subprocess_env({"PATH": "/usr/bin"})
        for key in _ALL_KEYS:
            assert env.get(key), f"{key} missing from _sanitize_subprocess_env output"

    def test_unconfigured_leaves_spawn_env_untouched(self):
        with _with_identity("", ""):
            env = _make_run_env({"PATH": "/usr/bin"})
        for key in _ALL_KEYS:
            assert key not in env
