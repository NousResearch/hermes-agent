"""Tests for tools.env_passthrough — skill and config env var passthrough."""

import os
import pytest
import yaml

from agent import secret_scope as ss
import tools.env_passthrough as _ep_mod
from tools.env_passthrough import (
    clear_env_passthrough,
    get_all_passthrough,
    is_env_passthrough,
    register_env_passthrough,
    resolve_passthrough_value,
)


@pytest.fixture(autouse=True)
def _clean_passthrough():
    """Ensure a clean passthrough state for every test."""
    clear_env_passthrough()
    _ep_mod._config_passthrough = None
    ss.set_multiplex_active(False)
    yield
    clear_env_passthrough()
    _ep_mod._config_passthrough = None
    ss.set_multiplex_active(False)


class TestSkillScopedPassthrough:
    def test_register_and_check(self):
        assert not is_env_passthrough("TENOR_API_KEY")
        register_env_passthrough(["TENOR_API_KEY"])
        assert is_env_passthrough("TENOR_API_KEY")


    def test_skips_empty(self):
        register_env_passthrough(["", "  ", "VALID_KEY"])
        assert is_env_passthrough("VALID_KEY")
        assert not is_env_passthrough("")


class TestConfigPassthrough:
    def test_reads_from_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["MY_CUSTOM_KEY", "ANOTHER_TOKEN"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough = None

        assert is_env_passthrough("MY_CUSTOM_KEY")
        assert is_env_passthrough("ANOTHER_TOKEN")
        assert not is_env_passthrough("UNRELATED_VAR")


    def test_union_of_skill_and_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["CONFIG_KEY"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough = None

        register_env_passthrough(["SKILL_KEY"])
        all_pt = get_all_passthrough()
        assert "CONFIG_KEY" in all_pt
        assert "SKILL_KEY" in all_pt


class TestProfileScopedResolution:
    def test_active_scope_overrides_process_fallback(self):
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "profile-b"})
        try:
            assert resolve_passthrough_value("SERVICE_TOKEN", "profile-a") == "profile-b"
        finally:
            ss.reset_secret_scope(token)

    def test_active_scope_does_not_fall_back_to_another_profile(self):
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert resolve_passthrough_value("SERVICE_TOKEN", "profile-a") is None
        finally:
            ss.reset_secret_scope(token)

    def test_unscoped_multiplex_read_fails_closed(self):
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            resolve_passthrough_value("SERVICE_TOKEN", "profile-a")

    def test_single_profile_keeps_callers_fallback(self):
        assert resolve_passthrough_value("SERVICE_TOKEN", "profile-a") == "profile-a"

    def test_active_scope_keeps_explicit_global_override(self, monkeypatch):
        """Global terminal settings still honor a caller-provided override."""
        monkeypatch.setenv("TERMINAL_CWD", "/default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert resolve_passthrough_value("TERMINAL_CWD", "/explicit") == "/explicit"
        finally:
            ss.reset_secret_scope(token)


class TestExecuteCodeIntegration:
    """Verify that the passthrough is checked in execute_code's env filtering."""

    def test_secret_substring_blocked_by_default(self):
        """TENOR_API_KEY should be blocked without passthrough."""
        _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                              "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                              "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
        _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                              "PASSWD", "AUTH")

        test_env = {"PATH": "/usr/bin", "TENOR_API_KEY": "test123", "HOME": "/home/user"}
        child_env = {}
        for k, v in test_env.items():
            if is_env_passthrough(k):
                child_env[k] = v
                continue
            if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
                continue
            if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
                child_env[k] = v

        assert "PATH" in child_env
        assert "HOME" in child_env
        assert "TENOR_API_KEY" not in child_env

    def test_passthrough_allows_secret_through(self):
        """TENOR_API_KEY should pass through when registered."""
        _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                              "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                              "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
        _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                              "PASSWD", "AUTH")

        register_env_passthrough(["TENOR_API_KEY"])

        test_env = {"PATH": "/usr/bin", "TENOR_API_KEY": "test123", "HOME": "/home/user"}
        child_env = {}
        for k, v in test_env.items():
            if is_env_passthrough(k):
                child_env[k] = v
                continue
            if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
                continue
            if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
                child_env[k] = v

        assert "PATH" in child_env
        assert "HOME" in child_env
        assert "TENOR_API_KEY" in child_env
        assert child_env["TENOR_API_KEY"] == "test123"

    def test_execute_code_uses_active_profile_for_passthrough(self, monkeypatch):
        """The execute_code child must receive the routed profile's value."""
        from tools.code_execution_tool import _scrub_child_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile"})
        try:
            child_env = _scrub_child_env({"SERVICE_TOKEN": "token-for-default"})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert child_env["SERVICE_TOKEN"] == "token-for-routed-profile"

    def test_execute_code_omits_missing_scoped_passthrough(self, monkeypatch):
        """A missing routed secret must not leak into the execute_code child."""
        from tools.code_execution_tool import _scrub_child_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            child_env = _scrub_child_env({"SERVICE_TOKEN": "token-for-default"})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert "SERVICE_TOKEN" not in child_env


class TestTerminalIntegration:
    """Verify that the passthrough is checked in terminal's env sanitizers."""

    def test_background_terminal_uses_active_profile_for_passthrough(self, monkeypatch):
        """Background/PTY terminal children must use the routed profile value."""
        from tools.environments.local import _sanitize_subprocess_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile"})
        try:
            child_env = _sanitize_subprocess_env(
                {"SERVICE_TOKEN": "token-for-default"},
                {"SERVICE_TOKEN": "token-for-default"},
            )
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert child_env["SERVICE_TOKEN"] == "token-for-routed-profile"

    def test_background_terminal_omits_missing_scoped_passthrough(self, monkeypatch):
        """A missing routed secret must not leak into background terminal work."""
        from tools.environments.local import _sanitize_subprocess_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            child_env = _sanitize_subprocess_env({"SERVICE_TOKEN": "token-for-default"})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert "SERVICE_TOKEN" not in child_env

    def test_shared_local_snapshot_re_resolves_current_profile(self, monkeypatch, tmp_path):
        """A persistent shell snapshot must not retain the previous profile's value."""
        from tools.environments.local import LocalEnvironment

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        env = None
        token_b = None
        token_c = None
        try:
            token_a = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-profile-a"})
            try:
                env = LocalEnvironment(cwd=str(tmp_path))
                assert env.execute("printf '%s' \"$SERVICE_TOKEN\"")["output"] == "token-for-profile-a"
            finally:
                ss.reset_secret_scope(token_a)

            token_b = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-profile-b"})
            result = env.execute("printf '%s' \"$SERVICE_TOKEN\"")
            ss.reset_secret_scope(token_b)
            token_b = None

            token_c = ss.set_secret_scope({})
            missing = env.execute("printf '%s' \"${SERVICE_TOKEN-unset}\"")
        finally:
            if token_b is not None:
                ss.reset_secret_scope(token_b)
            if token_c is not None:
                ss.reset_secret_scope(token_c)
            ss.set_multiplex_active(False)
            if env is not None:
                env.cleanup()

        assert result["output"] == "token-for-profile-b"
        assert missing["output"] == "unset"

    def test_blocklisted_var_blocked_by_default(self):
        from tools.environments.local import _sanitize_subprocess_env, _HERMES_PROVIDER_ENV_BLOCKLIST

        # Pick a var we know is in the blocklist
        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        env = {blocked_var: "secret_value", "PATH": "/usr/bin"}
        result = _sanitize_subprocess_env(env)
        assert blocked_var not in result
        assert "PATH" in result

    def test_passthrough_cannot_override_provider_blocklist(self):
        """GHSA-rhgp-j443-p4rf: register_env_passthrough must NOT accept
        Hermes provider credentials — that was the bypass where a skill
        could declare ANTHROPIC_TOKEN / OPENAI_API_KEY as passthrough and
        defeat the execute_code sandbox scrubbing."""
        from tools.environments.local import (
            _sanitize_subprocess_env,
            _HERMES_PROVIDER_ENV_BLOCKLIST,
        )

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        # Attempt to register — must be silently refused (logged warning).
        register_env_passthrough([blocked_var])

        # is_env_passthrough must NOT report it as allowed
        assert not is_env_passthrough(blocked_var)

        # Sanitizer still strips the var from subprocess env
        env = {blocked_var: "secret_value", "PATH": "/usr/bin"}
        result = _sanitize_subprocess_env(env)
        assert blocked_var not in result
        assert "PATH" in result

    def test_passthrough_cannot_override_internal_dynamic_secret(self):
        """A skill must NOT be able to register dynamically-named Hermes
        secrets (AUXILIARY_*_API_KEY / _BASE_URL, GATEWAY_RELAY_* auth) as
        passthrough — they aren't in the static blocklist, so this is the
        defense-in-depth layer that keeps env_passthrough consistent with the
        unconditional strip in the sanitizers."""
        from tools.environments.local import _sanitize_subprocess_env

        for var in (
            "AUXILIARY_VISION_API_KEY",
            "AUXILIARY_VISION_BASE_URL",
            "GATEWAY_RELAY_SECRET",
            "GATEWAY_RELAY_DELIVERY_KEY",
        ):
            register_env_passthrough([var])
            assert not is_env_passthrough(var), (
                f"{var} should be refused passthrough registration"
            )
            result = _sanitize_subprocess_env({var: "secret", "PATH": "/usr/bin"})
            assert var not in result
            assert "PATH" in result

    def test_passthrough_allows_auxiliary_non_secret_routing(self):
        """AUXILIARY_*_PROVIDER / _MODEL and GATEWAY_RELAY routing hints are not
        secrets, so a skill may still register them (they're not protected)."""
        register_env_passthrough([
            "AUXILIARY_VISION_PROVIDER",
            "AUXILIARY_VISION_MODEL",
            "GATEWAY_RELAY_URL",
        ])
        assert is_env_passthrough("AUXILIARY_VISION_PROVIDER")
        assert is_env_passthrough("AUXILIARY_VISION_MODEL")
        assert is_env_passthrough("GATEWAY_RELAY_URL")

    def test_make_run_env_blocklist_override_rejected(self):
        """_make_run_env must NOT expose a blocklisted var to subprocess env
        even after a skill attempts to register it via passthrough."""
        from tools.environments.local import (
            _make_run_env,
            _HERMES_PROVIDER_ENV_BLOCKLIST,
        )

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        os.environ[blocked_var] = "secret_value"
        try:
            # Without passthrough — blocked
            result_before = _make_run_env({})
            assert blocked_var not in result_before

            # Skill tries to register it — must be refused, so still blocked
            register_env_passthrough([blocked_var])
            result_after = _make_run_env({})
            assert blocked_var not in result_after
        finally:
            os.environ.pop(blocked_var, None)

    def test_non_hermes_api_key_still_registerable(self):
        """Third-party API keys (TENOR_API_KEY, NOTION_TOKEN, etc.) are NOT
        Hermes provider credentials and must still pass through — skills
        that legitimately wrap third-party APIs must keep working."""
        # TENOR_API_KEY is a real example — used by the gif-search skill
        register_env_passthrough(["TENOR_API_KEY"])
        assert is_env_passthrough("TENOR_API_KEY")

        # Arbitrary skill-specific var
        register_env_passthrough(["MY_SKILL_CUSTOM_CONFIG"])
        assert is_env_passthrough("MY_SKILL_CUSTOM_CONFIG")

    def test_provider_blocklist_import_failure_fails_closed(self, monkeypatch):
        """If the dynamic provider blocklist can't be imported, provider
        credentials must be treated as protected and refused passthrough —
        otherwise a skill could tunnel a Hermes credential into the
        execute_code child (regression for #37950 / GHSA-rhgp-j443-p4rf).

        Verifies the full path: _is_hermes_provider_credential returns True,
        register_env_passthrough refuses the var, and _scrub_child_env keeps
        it out of the child env. A non-Hermes key is also rejected here (the
        fallback is conservative: when we can't tell, we fail closed), which
        is the safe direction.
        """
        import builtins

        from tools.code_execution_tool import _scrub_child_env

        real_import = builtins.__import__

        def fail_local_import(name, *args, **kwargs):
            if name == "tools.environments.local":
                raise ImportError("synthetic blocklist import failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_local_import)

        # Every name is now treated as a protected provider credential.
        assert _ep_mod._is_hermes_provider_credential("OPENAI_API_KEY")
        assert _ep_mod._is_hermes_provider_credential("ANTHROPIC_API_KEY")
        assert _ep_mod._is_hermes_provider_credential("GH_TOKEN")

        # Registration is refused while the blocklist is unavailable.
        register_env_passthrough(["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
        assert not is_env_passthrough("OPENAI_API_KEY")
        assert not is_env_passthrough("ANTHROPIC_API_KEY")

        # And the credential never reaches the execute_code child.
        child_env = _scrub_child_env(
            {
                "OPENAI_API_KEY": "synthetic-secret",
                "ANTHROPIC_API_KEY": "synthetic-secret",
                "PATH": "/usr/bin",
            },
            is_passthrough=is_env_passthrough,
            is_windows=False,
        )
        assert "OPENAI_API_KEY" not in child_env
        assert "ANTHROPIC_API_KEY" not in child_env
        assert child_env["PATH"] == "/usr/bin"


class TestBundledPlatformTerminalPassthrough:
    """Bundled platform manifests may mark env vars ``terminal_passthrough``.

    Those exact vars (e.g. the Buzz CLI credentials) must pass through the
    TERMINAL sanitizers so platform CLIs invoked by the agent inherit their
    auth (#76243), while remaining stripped from execute_code and generic
    subprocess spawns.
    """

    def test_buzz_cli_vars_are_terminal_passthrough(self):
        from tools.env_passthrough import is_terminal_env_passthrough

        for var in ("BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY", "BUZZ_AUTH_TAG"):
            assert is_terminal_env_passthrough(var), (
                f"{var} should be terminal-passthrough per bundled manifest"
            )

    def test_buzz_cli_vars_stay_stripped_from_generic_spawns(self):
        # The generic (non-terminal) check must NOT report them as allowed —
        # execute_code and generic subprocesses keep scrubbing them.
        for var in ("BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY", "BUZZ_AUTH_TAG"):
            assert not is_env_passthrough(var), (
                f"{var} must not leak into execute_code / generic spawns"
            )

    def test_terminal_sanitizer_keeps_buzz_vars(self):
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        env = {
            "BUZZ_PRIVATE_KEY": "nsec-secret",
            "BUZZ_RELAY_URL": "https://relay.example",
            "BUZZ_AUTH_TAG": '["x","y","z","sig"]',
            "PATH": "/usr/bin",
            "HOME": "/home/user",
        }
        # Foreground terminal run env
        result = _make_run_env(env)
        assert result["BUZZ_PRIVATE_KEY"] == "nsec-secret"
        assert result["BUZZ_RELAY_URL"] == "https://relay.example"
        assert result["BUZZ_AUTH_TAG"] == '["x","y","z","sig"]'

        # Background/PTY terminal spawn (terminal_scope=True)
        result_bg = _sanitize_subprocess_env(env, terminal_scope=True)
        assert result_bg["BUZZ_PRIVATE_KEY"] == "nsec-secret"
        assert result_bg["BUZZ_RELAY_URL"] == "https://relay.example"

        # execute_code / generic spawn (terminal_scope=False default) strips them
        result_gen = _sanitize_subprocess_env(env)
        assert "BUZZ_PRIVATE_KEY" not in result_gen
        assert "BUZZ_RELAY_URL" not in result_gen
        assert "BUZZ_AUTH_TAG" not in result_gen

    def test_generic_sanitizer_still_strips_buzz_without_scope(self, monkeypatch):
        from tools.environments.local import _sanitize_subprocess_env

        # Even if a skill/config registered one of the vars as generic
        # passthrough, the blocklist refusal in register_env_passthrough
        # keeps it out of execute_code (GHSA-rhgp-j443-p4rf).
        register_env_passthrough(["BUZZ_PRIVATE_KEY"])
        assert not is_env_passthrough("BUZZ_PRIVATE_KEY")

        env = {"BUZZ_PRIVATE_KEY": "nsec-secret", "PATH": "/usr/bin"}
        result = _sanitize_subprocess_env(env)
        assert "BUZZ_PRIVATE_KEY" not in result

    def test_execute_code_child_env_still_scrubs_buzz(self):
        from tools.code_execution_tool import _scrub_child_env

        child_env = _scrub_child_env(
            {
                "BUZZ_PRIVATE_KEY": "nsec-secret",
                "BUZZ_RELAY_URL": "https://relay.example",
                "PATH": "/usr/bin",
            },
            is_passthrough=is_env_passthrough,
            is_windows=False,
        )
        assert "BUZZ_PRIVATE_KEY" not in child_env
        assert "BUZZ_RELAY_URL" not in child_env
        assert child_env["PATH"] == "/usr/bin"


class TestConfigPassthroughTypeValidation:
    def test_scalar_config_warns_and_is_ignored(self, tmp_path, monkeypatch, caplog):
        """A JSON-looking quoted scalar must warn instead of silently producing
        an empty allowlist (#76243)."""
        config = {
            "terminal": {
                "env_passthrough": '["BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"]'
            }
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough = None

        import logging
        with caplog.at_level(logging.WARNING, logger="tools.env_passthrough"):
            assert not is_env_passthrough("BUZZ_RELAY_URL")
            assert not is_env_passthrough("BUZZ_PRIVATE_KEY")
        assert any("env_passthrough" in r.message and "must be a YAML list" in r.message
                   for r in caplog.records), (
            "scalar terminal.env_passthrough should emit a warning"
        )
