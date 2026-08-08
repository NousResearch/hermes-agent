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


class TestProvenanceScrub:
    """#77164: the child-env scrub must be provenance-aware — any env value
    present in the current home's applied-secrets snapshot
    (hermes_cli.env_loader._SECRET_SOURCE_VALUES_BY_HOME) is stripped from
    spawned children even under non-credential-shaped names (DATABASE_URL,
    FOO, 1Password item keys), while explicitly registered env_passthrough
    vars still receive their value.
    """

    def _seed_applied_secrets(self, home, values):
        from hermes_cli import env_loader

        home_key = str(home.resolve())
        env_loader._SECRET_SOURCE_VALUES_BY_HOME[home_key] = dict(values)
        return home_key

    def test_hermes_subprocess_env_strips_arbitrarily_named_applied_secret(self, tmp_path, monkeypatch):
        """The non-terminal factory (hermes_subprocess_env) must strip an
        externally-applied secret value from a real spawned child even when
        the env var name has no credential shape."""
        import json
        import subprocess
        import sys

        from hermes_cli import env_loader
        from tools.environments.local import hermes_subprocess_env

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        applied = {
            "DATABASE_URL": "postgres://svc:«redacted:db-pass»@db.internal:5432/app",
            "FOO": "«redacted:op-item-value»",
        }
        home_key = self._seed_applied_secrets(hermes_home, applied)
        try:
            for name, value in applied.items():
                monkeypatch.setenv(name, value)

            env = hermes_subprocess_env()  # no passthrough concept here

            code = (
                "import os, json; print(json.dumps({"
                "'k1': 'DATABASE_URL' in os.environ, "
                "'k2': 'FOO' in os.environ}))"
            )
            out = subprocess.run(
                [sys.executable, "-c", code],
                env=env, capture_output=True, text=True, timeout=60, check=True,
            )
            result = json.loads(out.stdout)
            assert result["k1"] is False, "DATABASE_URL leaked via hermes_subprocess_env"
            assert result["k2"] is False, "FOO leaked via hermes_subprocess_env"
        finally:
            env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)

    def test_hermes_subprocess_env_still_strips_on_inherit_credentials(self, tmp_path, monkeypatch):
        """The provenance scrub is Tier-1-like: it must apply even when the
        caller opts into inherit_credentials=True (model-driving CLIs)."""
        from hermes_cli import env_loader
        from tools.environments.local import hermes_subprocess_env

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        applied = {"FOO": "«redacted:op-item-value»"}
        home_key = self._seed_applied_secrets(hermes_home, applied)
        try:
            monkeypatch.setenv("FOO", "«redacted:op-item-value»")
            env = hermes_subprocess_env(inherit_credentials=True)
            assert "FOO" not in env
        finally:
            env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)

    def test_scrub_never_uses_another_profiles_snapshot(self, tmp_path, monkeypatch):
        """Cross-profile isolation: values snapshotted for home A must never
        be used to scrub a child built for home B."""
        from hermes_cli import env_loader
        from tools.environments.local import _sanitize_subprocess_env

        home_a = tmp_path / "profile-a"
        home_b = tmp_path / "profile-b"
        home_a.mkdir()
        home_b.mkdir()

        key_a = self._seed_applied_secrets(home_a, {"FOO": "«redacted:profile-a-secret»"})
        try:
            monkeypatch.setenv("HERMES_HOME", str(home_b))
            # Same value as profile-a's secret, but we're building for B.
            result = _sanitize_subprocess_env(
                {"FOO": "«redacted:profile-a-secret»", "PATH": "/usr/bin"}
            )
            assert result["FOO"] == "«redacted:profile-a-secret»"
            assert result["PATH"] == "/usr/bin"
        finally:
            env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(key_a, None)

    def test_empty_snapshot_degrades_gracefully(self, tmp_path, monkeypatch):
        """No snapshot for the current home → no value-based scrubbing, and
        ordinary vars keep flowing to the child."""
        from hermes_cli import env_loader
        from tools.environments.local import _sanitize_subprocess_env

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Deliberately NOT seeded.
        env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(str(hermes_home.resolve()), None)
        result = _sanitize_subprocess_env(
            {"DATABASE_URL": "postgres://user:pass@db:5432/app", "PATH": "/usr/bin"}
        )
        assert result["DATABASE_URL"] == "postgres://user:pass@db:5432/app"
        assert result["PATH"] == "/usr/bin"

    def test_passthrough_wins_over_provenance_scrub_e2e(self, tmp_path, monkeypatch):
        """A var registered in env_passthrough still reaches a real spawned
        child even though its value also appears in the applied-secrets
        snapshot — the explicit passthrough contract outranks the scrub."""
        import subprocess
        import sys

        from hermes_cli import env_loader
        from tools.environments.local import build_subprocess_env

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        db_url = "postgres://svc:«redacted:db-pass»@db.internal:5432/app"
        home_key = self._seed_applied_secrets(hermes_home, {"DATABASE_URL": db_url})
        try:
            monkeypatch.setenv("DATABASE_URL", db_url)
            register_env_passthrough(["DATABASE_URL"])
            env = build_subprocess_env()  # scrub on (default)
            out = subprocess.run(
                [sys.executable, "-c",
                 "import os; print(os.environ.get('DATABASE_URL', 'MISSING'))"],
                env=env, capture_output=True, text=True, timeout=60, check=True,
            )
            assert out.stdout.strip() == db_url
        finally:
            env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)

    def test_make_run_env_scrubs_applied_value_and_passthrough_wins(self, tmp_path, monkeypatch):
        """The terminal run-env path (_make_run_env) applies the same
        provenance scrub — applied secret values are stripped, and a
        passthrough-registered var keeps its value."""
        from hermes_cli import env_loader
        from tools.environments.local import _make_run_env

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        secret = "«redacted:make-run-env-secret»"
        home_key = self._seed_applied_secrets(hermes_home, {"DATABASE_URL": secret})
        try:
            monkeypatch.setenv("DATABASE_URL", secret)
            monkeypatch.setenv("PATH", "/usr/bin")

            run_env = _make_run_env({})
            assert "DATABASE_URL" not in run_env

            register_env_passthrough(["DATABASE_URL"])
            run_env2 = _make_run_env({})
            assert run_env2.get("DATABASE_URL") == secret
        finally:
            env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)
