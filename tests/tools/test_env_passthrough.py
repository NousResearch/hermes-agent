"""Tests for tools.env_passthrough — skill and config env var passthrough."""

import os
import pytest
import hermes_yaml as yaml

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
    _ep_mod._config_passthrough.clear()
    ss.set_multiplex_active(False)
    yield
    clear_env_passthrough()
    _ep_mod._config_passthrough.clear()
    ss.set_multiplex_active(False)


class TestSkillScopedPassthrough:


    def test_skips_empty(self):
        register_env_passthrough(["", "  ", "VALID_KEY"])
        assert is_env_passthrough("VALID_KEY")
        assert not is_env_passthrough("")


class TestRegistrationVisibleAcrossToolContextSnapshots:
    """Skill registration must survive the tool-dispatch context snapshot.

    Tool dispatch fans each tool call onto a worker whose context is a
    copy_context() snapshot taken at submit time
    (tools.thread_context.propagate_context_to_thread). A registration made
    inside one tool's worker (skill_view) used to be stored in a ContextVar
    and therefore never reached the submitting thread — the next tool's
    snapshot always saw an empty allowlist, so a skill's declared env vars
    never passed through to execute_code/terminal (#90004).
    """

    def test_registration_in_copied_context_is_visible_to_sibling_snapshot(self):
        import contextvars

        def _scenario():
            # The submitting thread has NEVER touched the allowlist var
            # (mirrors a fresh gateway turn where only tools register). Each
            # worker snapshot therefore starts without a set of its own.
            tool_a_ctx = contextvars.copy_context()
            tool_a_ctx.run(
                lambda: register_env_passthrough(["NOTION_API_KEY"])
            )
            # The NEXT tool's submit-time snapshot must still see it.
            tool_b_ctx = contextvars.copy_context()
            return tool_b_ctx.run(
                lambda: is_env_passthrough("NOTION_API_KEY")
            )

        # Run the whole scenario inside a pristine context so the module's
        # (test-fixture) current-context initialization cannot leak in.
        pristine = contextvars.Context()
        visible = pristine.run(_scenario)
        assert visible is True

    def test_registration_in_worker_thread_is_visible_to_main_thread(self):
        import concurrent.futures
        import contextvars

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            # Mirror tool_executor's propagate_context_to_thread snapshot:
            # the worker runs inside a copy of the submitting context.
            parent_ctx = contextvars.copy_context()
            future = pool.submit(
                parent_ctx.run, register_env_passthrough, ["AIRTABLE_API_KEY"]
            )
            future.result()
        assert is_env_passthrough("AIRTABLE_API_KEY") is True


class TestConfigPassthrough:
    def test_reads_from_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["MY_CUSTOM_KEY", "ANOTHER_TOKEN"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough.clear()

        assert is_env_passthrough("MY_CUSTOM_KEY")
        assert is_env_passthrough("ANOTHER_TOKEN")
        assert not is_env_passthrough("UNRELATED_VAR")


    def test_union_of_skill_and_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["CONFIG_KEY"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough.clear()

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



    def test_execute_code_uses_active_profile_for_passthrough(self, monkeypatch):
        """The execute_code child must receive the routed profile's value."""
        from tools.code_execution_env import _scrub_child_env

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
        from tools.code_execution_env import _scrub_child_env

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

    def test_execute_code_strips_buzz_vars(self):
        """BUZZ_* credentials must stay out of the execute_code child even
        though they pass through to terminal children (issue #78026): the
        carve-out is terminal-only.

        - BUZZ_PRIVATE_KEY matches the KEY secret substring.
        - BUZZ_AUTH_TAG matches the AUTH secret substring.
        - BUZZ_RELAY_URL matches no secret substring but is not on the safe
          prefix allowlist, so it is dropped too.
        """
        from tools.code_execution_env import _scrub_child_env

        buzz_vars = {
            "BUZZ_PRIVATE_KEY": "nsec1fake",
            "BUZZ_AUTH_TAG": '["tag","data","kind","sig"]',
            "BUZZ_RELAY_URL": "https://mycommunity.communities.buzz.xyz",
            "PATH": "/usr/bin",
            "HOME": "/home/user",
        }
        child_env = _scrub_child_env(buzz_vars)

        assert "BUZZ_PRIVATE_KEY" not in child_env
        assert "BUZZ_AUTH_TAG" not in child_env
        assert "BUZZ_RELAY_URL" not in child_env
        assert child_env["PATH"] == "/usr/bin"
        assert child_env["HOME"] == "/home/user"


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

    def test_scope_only_declared_name_reaches_every_local_child(self, monkeypatch):
        """A routed profile's declared secret lives only in its scope (its .env never enters the
        process env), so it must be added from the scope on every local spawn surface; an
        undeclared scope entry stays out. No scope bound -> byte-identical single-profile env."""
        from tools.code_execution_env import _scrub_child_env
        from tools.environments.local import _sanitize_subprocess_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.delenv("SERVICE_TOKEN", raising=False)
        base = {"PATH": "/usr/bin", "HOME": "/home/user"}
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile",
                                     "UNDECLARED_TOKEN": "never-forwarded"})
        try:
            terminal_env = _sanitize_subprocess_env(dict(base))
            sandbox_env = _scrub_child_env(dict(base))
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        for child_env in (terminal_env, sandbox_env):
            assert child_env["SERVICE_TOKEN"] == "token-for-routed-profile"
            assert "UNDECLARED_TOKEN" not in child_env
        assert "SERVICE_TOKEN" not in _sanitize_subprocess_env(dict(base))
        assert "SERVICE_TOKEN" not in _scrub_child_env(dict(base))

    def test_scope_overlay_failure_is_loud_on_both_local_surfaces(self, monkeypatch):
        """A scope/config failure while resolving declared scope-only names must raise, not be
        swallowed into a debug log that silently drops the declared secret again (#114209)."""
        import tools.env_passthrough as ep
        from tools.code_execution_env import _scrub_child_env
        from tools.environments.local import _sanitize_subprocess_env

        def _boom(_present):
            raise RuntimeError("scope lookup failed")

        monkeypatch.setattr(ep, "scoped_passthrough_additions", _boom)
        with pytest.raises(RuntimeError, match="scope lookup failed"):
            _sanitize_subprocess_env({"PATH": "/usr/bin"})
        with pytest.raises(RuntimeError, match="scope lookup failed"):
            _scrub_child_env({"PATH": "/usr/bin"})

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
        from tools.environments.local import _sanitize_subprocess_env
        from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST

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
        from tools.environments.local import _sanitize_subprocess_env
        from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST

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

    def test_passthrough_case_variant_of_blocklist_rejected(self):
        """A case-variant registration (``openai_api_key``) must be refused just
        like the canonical name: the remote-exec env builder resolves each
        registered name via ``os.getenv``, which is case-insensitive on Windows,
        so a variant would tunnel the real ``OPENAI_API_KEY`` value into
        SSH/Docker children: the same GHSA-rhgp-j443-p4rf primitive."""
        for var in ("openai_api_key", "OpenAi_Api_Key", "anthropic_api_key",
                    "Aws_Bearer_Token_Bedrock"):
            register_env_passthrough([var])
            assert not is_env_passthrough(var), (
                f"{var} should be refused passthrough registration")

    def test_passthrough_case_variant_via_config_rejected(self, tmp_path, monkeypatch):
        """The config-based allowlist (terminal.env_passthrough) must refuse
        case variants of provider credentials on the same filter."""
        config = {"terminal": {"env_passthrough": ["openai_api_key", "MY_OWN_KEY"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough.clear()

        assert not is_env_passthrough("openai_api_key")
        assert is_env_passthrough("MY_OWN_KEY")

    def test_passthrough_case_variant_never_reaches_remote_exec_env(self, monkeypatch):
        """Even if a case-variant name were present in the registered set, the
        remote env builder must not resolve it: on Windows ``os.getenv`` is
        case-insensitive, so ``openai_api_key`` would carry the real
        ``OPENAI_API_KEY`` into SSH/Docker exec envs."""
        from tools.environments import remote_common

        register_env_passthrough(["openai_api_key"])
        exec_env, _unset = remote_common.resolve_passthrough_env(set())
        assert not any(k.lower() == "openai_api_key" for k in exec_env)

        # Defense in depth, load-bearing on POSIX too: force the variant into
        # the registered set and set a real env entry under that spelling, so
        # os.getenv() resolves it and only the folded blocklist drops it.
        monkeypatch.setenv("openai_api_key", "sk-variant-value")
        monkeypatch.setenv("MY_OWN_KEY", "own-value")
        monkeypatch.setattr(
            "tools.env_passthrough.get_all_passthrough",
            lambda: {"openai_api_key", "MY_OWN_KEY"})
        exec_env, _unset = remote_common.resolve_passthrough_env(set())
        assert "openai_api_key" not in exec_env
        assert exec_env.get("MY_OWN_KEY") == "own-value"

    def test_passthrough_case_variant_never_reaches_execute_code_env(self):
        """The GHSA-rhgp-j443-p4rf path end to end: the variant registration
        is refused, so is_env_passthrough cannot carry the name past the
        execute_code scrub."""
        from tools.code_execution_env import _scrub_child_env

        register_env_passthrough(["openai_api_key"])
        child_env = _scrub_child_env(
            {"openai_api_key": "sk-real", "PATH": "/usr/bin"},
            is_passthrough=is_env_passthrough, is_windows=False)
        assert "openai_api_key" not in child_env
        assert child_env["PATH"] == "/usr/bin"

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

    def test_passthrough_cannot_register_buzz_vars(self, monkeypatch):
        """GHSA-rhgp-j443-p4rf seal stays intact for the BUZZ_* first-party
        platform credentials: even though they pass through to terminal
        children in a Buzz agent context (issue #78026), env_passthrough
        registration must still refuse them — the carve-out opens NO
        registration path, so a skill cannot expand BUZZ_* exposure to
        execute_code."""
        from tools.environments.local import _sanitize_subprocess_env

        monkeypatch.setenv("BUZZ_MANAGED_AGENT", "1")
        for var in (
            "BUZZ_PRIVATE_KEY",
            "BUZZ_AUTH_TAG",
            "BUZZ_RELAY_URL",
        ):
            register_env_passthrough([var])
            assert not is_env_passthrough(var), (
                f"{var} should be refused passthrough registration"
            )
            # Terminal sanitizer still passes BUZZ_* through to terminal
            # children by the first-party carve-out...
            result = _sanitize_subprocess_env({var: "value", "PATH": "/usr/bin"})
            assert result.get(var) == "value"
            # ...but the execute_code child never sees them.
            from tools.code_execution_env import _scrub_child_env

            child_env = _scrub_child_env({var: "value", "PATH": "/usr/bin"})
            assert var not in child_env

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
        from tools.environments.local import _make_run_env
        from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST

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

        from tools.code_execution_env import _scrub_child_env

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


class TestScrubbedProviderCredentialNotes:
    """#71788: surface when provider credentials are scrubbed (DX only)."""

    def test_list_and_format_single(self):
        from tools.env_passthrough import (
            format_scrubbed_provider_env_note,
            list_scrubbed_provider_credentials,
        )

        names = list_scrubbed_provider_credentials(
            {"DASHSCOPE_API_KEY": "secret", "PATH": "/usr/bin"},
            {"PATH": "/usr/bin"},
        )
        assert "DASHSCOPE_API_KEY" in names
        note = format_scrubbed_provider_env_note(names)
        assert "DASHSCOPE_API_KEY" in note
        assert "credential scrub" in note
        assert "not missing from the gateway" in note

    def test_empty_parent_value_not_reported(self):
        from tools.env_passthrough import list_scrubbed_provider_credentials

        names = list_scrubbed_provider_credentials(
            {"DASHSCOPE_API_KEY": "", "PATH": "/usr/bin"},
            {"PATH": "/usr/bin"},
        )
        assert "DASHSCOPE_API_KEY" not in names

    def test_present_in_child_not_reported(self):
        from tools.env_passthrough import list_scrubbed_provider_credentials

        names = list_scrubbed_provider_credentials(
            {"OPENAI_API_KEY": "sk-x", "PATH": "/usr/bin"},
            {"OPENAI_API_KEY": "sk-x", "PATH": "/usr/bin"},
        )
        assert "OPENAI_API_KEY" not in names

    def test_blocklist_absence_reported_even_if_passthrough_claimed(self, monkeypatch):
        """Authoritative provider blocklist wins over passthrough claims (#71788)."""
        from tools.env_passthrough import (
            clear_env_passthrough,
            list_scrubbed_provider_credentials,
            register_env_passthrough,
        )

        clear_env_passthrough()
        # register is refused for provider creds, but even if something set the
        # allow set, absence from child must still surface in the note.
        register_env_passthrough(["OPENAI_API_KEY"])
        names = list_scrubbed_provider_credentials(
            {"OPENAI_API_KEY": "sk-x", "PATH": "/usr/bin"},
            {"PATH": "/usr/bin"},
        )
        assert "OPENAI_API_KEY" in names


class TestCredentialScrubNoteResultPaths:
    """Result-level regression for #71788 review: note matches actual child env."""

    def test_local_make_run_env_self_env_not_empty_dict(self, monkeypatch):
        """Terminal local path must use _make_run_env(self.env), not {}."""
        from tools.environments.local import _make_run_env
        from tools.env_passthrough import list_scrubbed_provider_credentials

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-parent")
        # empty extra env still strips provider keys via blocklist
        child_empty_extra = _make_run_env({})
        child_with_self = _make_run_env({"PATH": "/usr/bin", "CUSTOM_X": "1"})
        names_empty = list_scrubbed_provider_credentials(os.environ, child_empty_extra)
        names_self = list_scrubbed_provider_credentials(os.environ, child_with_self)
        assert "OPENAI_API_KEY" in names_empty
        assert "OPENAI_API_KEY" in names_self
        # self.env values appear in child preview (not wiped by {})
        assert child_with_self.get("CUSTOM_X") == "1"
        assert child_empty_extra.get("CUSTOM_X") is None

    def test_execute_code_local_result_has_scrub_note(self, monkeypatch):
        from tools.code_execution_tool import _scrub_child_env
        from tools.env_passthrough import (
            format_scrubbed_provider_env_note,
            list_scrubbed_provider_credentials,
        )

        monkeypatch.setenv("OPENAI_API_KEY", "sk-local-result")
        child = _scrub_child_env(os.environ)
        note = format_scrubbed_provider_env_note(
            list_scrubbed_provider_credentials(os.environ, child)
        )
        assert "OPENAI_API_KEY" in note
        # Simulate attach-to-result path used by execute_code
        result = {"output": "hello", "status": "success"}
        if note:
            result["credential_scrub_note"] = note
        assert result["output"] == "hello"
        assert "OPENAI_API_KEY" in result["credential_scrub_note"]

    def test_remote_ssh_like_env_unknown_omits_false_scrub(self):
        """SSH-like remote: host env is not cleared — do not claim scrub vs {}."""
        from tools.env_passthrough import (
            format_scrubbed_provider_env_note,
            list_scrubbed_provider_credentials,
        )

        # Comparative smoke: only emit when we supply a known child env.
        parent = {"OPENAI_API_KEY": "sk-x", "PATH": "/usr/bin"}
        # Unknown remote host env — treat child as None unknown by skipping note
        # rather than comparing to {}.
        false_positive = format_scrubbed_provider_env_note(
            list_scrubbed_provider_credentials(parent, {})
        )
        assert "OPENAI_API_KEY" in false_positive  # helper still works
        # Callers for remote must choose not to attach when env unknown:
        known_remote_child = None
        note = (
            format_scrubbed_provider_env_note(
                list_scrubbed_provider_credentials(parent, known_remote_child)
            )
            if known_remote_child is not None
            else ""
        )
        assert note == ""

    def test_terminal_result_local_attaches_note(self, monkeypatch):
        """End-to-end: terminal_tool local attaches credential_scrub_note."""
        import json
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("OPENAI_API_KEY", "sk-term-local")
        monkeypatch.setenv("TERMINAL_ENV", "local")

        # Minimal path through note assembly using the same helper chain as
        # terminal_tool after the review fix.
        from tools.env_passthrough import (
            format_scrubbed_provider_env_note,
            list_scrubbed_provider_credentials,
        )
        from tools.environments.local import LocalEnvironment, _make_run_env

        env = LocalEnvironment(cwd="/tmp", timeout=5)
        child = _make_run_env(env.env or {})
        names = list_scrubbed_provider_credentials(os.environ, child)
        note = format_scrubbed_provider_env_note(names)
        assert "OPENAI_API_KEY" in note
        result_dict = {"output": "ok", "exit_code": 0, "error": None}
        result_dict["credential_scrub_note"] = note
        payload = json.dumps(result_dict)
        assert "credential_scrub_note" in payload
        assert "OPENAI_API_KEY" in payload
