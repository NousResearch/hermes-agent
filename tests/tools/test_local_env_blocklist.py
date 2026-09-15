"""Child-observed terminal policy, profile lifetime and PYTHONPATH provenance."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.tools._child_env_fixtures import child_env, observe_child, observe_terminal  # noqa: F401
from tools.environments import local
from tools.environments import local_pythonpath as pp


# Expectations come from independent provider/config declarations and literal
# policy examples, never the finished blocklist or a test-owned scrubber.
STATIC_BLOCKED = """
OPENAI_BASE_URL OPENAI_API_KEY OPENAI_API_BASE OPENAI_ORG_ID OPENAI_ORGANIZATION
OPENROUTER_API_KEY ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_TOKEN LLM_MODEL
VERTEX_CREDENTIALS_PATH GOOGLE_APPLICATION_CREDENTIALS AWS_BEARER_TOKEN_BEDROCK
GOOGLE_API_KEY DEEPSEEK_API_KEY MISTRAL_API_KEY GROQ_API_KEY TOGETHER_API_KEY
PERPLEXITY_API_KEY COHERE_API_KEY FIREWORKS_API_KEY XAI_API_KEY HELICONE_API_KEY
TELEGRAM_HOME_CHANNEL TELEGRAM_HOME_CHANNEL_NAME DISCORD_HOME_CHANNEL
DISCORD_HOME_CHANNEL_NAME DISCORD_REQUIRE_MENTION DISCORD_FREE_RESPONSE_CHANNELS
DISCORD_AUTO_THREAD SLACK_HOME_CHANNEL SLACK_HOME_CHANNEL_NAME SLACK_ALLOWED_USERS
WHATSAPP_ENABLED WHATSAPP_MODE WHATSAPP_ALLOWED_USERS SIGNAL_HTTP_URL SIGNAL_ACCOUNT
SIGNAL_ALLOWED_USERS SIGNAL_GROUP_ALLOWED_USERS SIGNAL_HOME_CHANNEL SIGNAL_HOME_CHANNEL_NAME
SIGNAL_IGNORE_STORIES HASS_TOKEN HASS_URL EMAIL_ADDRESS EMAIL_PASSWORD EMAIL_IMAP_HOST
EMAIL_SMTP_HOST EMAIL_HOME_ADDRESS EMAIL_HOME_ADDRESS_NAME HERMES_DASHBOARD_SESSION_TOKEN
GATEWAY_ALLOWED_USERS GATEWAY_ALLOW_ALL_USERS GH_TOKEN GITHUB_APP_ID
GITHUB_APP_PRIVATE_KEY_PATH GITHUB_APP_INSTALLATION_ID MODAL_TOKEN_ID MODAL_TOKEN_SECRET
DAYTONA_API_KEY VERCEL_OIDC_TOKEN VERCEL_TOKEN VERCEL_PROJECT_ID VERCEL_TEAM_ID GATEWAY_RELAY_ID
AUXILIARY_VISION_API_KEY AUXILIARY_WEB_EXTRACT_API_KEY AUXILIARY_APPROVAL_API_KEY
AUXILIARY_MY_PLUGIN_TASK_API_KEY AUXILIARY_VISION_BASE_URL AUXILIARY_COMPRESSION_BASE_URL
GATEWAY_RELAY_SECRET GATEWAY_RELAY_DELIVERY_KEY GATEWAY_RELAY_SESSION_TOKEN
""".split()
OPERATOR_ALLOWED = """
AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE AWS_DEFAULT_REGION
AWS_REGION AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN
CLAUDE_CODE_OAUTH_TOKEN AUXILIARY_VISION_PROVIDER AUXILIARY_VISION_MODEL GATEWAY_RELAY_URL
GATEWAY_RELAY_PLATFORMS MY_APP_KEY MY_CUSTOM_VAR
""".split()


def _running_site():
    return Path(sys.prefix) / ("Lib/site-packages" if os.name == "nt" else
                              f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")


def test_terminal_child_observes_declared_policy(child_env, monkeypatch):
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import OPTIONAL_ENV_VARS
    blocked = set(STATIC_BLOCKED)
    for config in PROVIDER_REGISTRY.values():
        blocked.update(config.api_key_env_vars)
        if config.base_url_env_var:
            blocked.add(config.base_url_env_var)
    blocked.update(name for name, meta in OPTIONAL_ENV_VARS.items()
                   if meta.get("category") in {"tool", "messaging"}
                   or (meta.get("category") == "setting" and meta.get("password")))
    blocked.discard("CLAUDE_CODE_OAUTH_TOKEN")  # operator's subscription, not Hermes inference
    for name in blocked | set(OPERATOR_ALLOWED):
        monkeypatch.setenv(name, "fake-" + name)
    before = dict(os.environ)
    env = local.LocalEnvironment(cwd=str(child_env), timeout=30,
                                 env={"OPENAI_BASE_URL": "fake-override", "MY_CUSTOM_VAR": "caller-value"})
    try:
        observed = observe_terminal(env, sorted(blocked | set(OPERATOR_ALLOWED)))
    finally:
        env.cleanup()
    assert observed == {**dict.fromkeys(blocked), **{k: "fake-" + k for k in OPERATOR_ALLOWED},
                        "MY_CUSTOM_VAR": "caller-value"}
    assert dict(os.environ) == before


@pytest.mark.parametrize("builder", ["foreground", "background", "factory", "nonterminal"])
def test_builders_strip_runtime_markers_and_owned_paths(child_env, monkeypatch, builder):
    repo, site = Path(__file__).resolve().parents[2], _running_site()
    user_path = str(child_env / "user-lib")
    for k, v in {"VIRTUAL_ENV": "/unrelated/venv", "CONDA_PREFIX": "/unrelated/conda",
                 "PYTHONHOME": "/nonexistent/python-home",
                 "PYTHONPATH": os.pathsep.join([str(repo), str(site), user_path])}.items():
        monkeypatch.setenv(k, v)
    factories = {
        "foreground": lambda: local._make_run_env({}),
        "background": lambda: local._sanitize_subprocess_env(dict(os.environ), {"VIRTUAL_ENV": "/extra/venv"}),
        "factory": local.build_subprocess_env,
        "nonterminal": local.hermes_subprocess_env,
    }
    before = dict(os.environ)
    actual = observe_child(factories[builder](), ["VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH", "HOME"])
    assert actual == {"VIRTUAL_ENV": None, "CONDA_PREFIX": None, "PYTHONHOME": None,
                      "PYTHONPATH": user_path, "HOME": str(child_env)}
    assert dict(os.environ) == before


@pytest.mark.parametrize("builder,base_force,extra_force", [
    ("foreground", "base-forced", "extra-forced"),
    ("background", None, "extra-forced"),
    ("factory", None, "extra-forced"),
    ("nonterminal", None, None),
])
def test_force_prefix_is_not_plugin_passthrough(child_env, monkeypatch, builder, base_force, extra_force):
    from tools.env_passthrough import register_env_passthrough, is_env_passthrough
    register_env_passthrough(["OPENAI_API_KEY", "AUXILIARY_VISION_API_KEY", "SERVICE_TOKEN"])
    assert not is_env_passthrough("OPENAI_API_KEY")
    assert not is_env_passthrough("AUXILIARY_VISION_API_KEY")
    assert is_env_passthrough("SERVICE_TOKEN")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-parent")
    monkeypatch.setenv("_HERMES_FORCE_OPENAI_API_KEY", "base-forced")
    extra = {"_HERMES_FORCE_OPENAI_BASE_URL": "extra-forced",
             "_HERMES_FORCE_AUXILIARY_VISION_API_KEY": "never-forward",
             "AUXILIARY_VISION_API_KEY": "never-forward", "MY_CUSTOM_VAR": "caller-value"}
    factories = {
        "foreground": lambda: local._make_run_env(extra),
        "background": lambda: local._sanitize_subprocess_env(dict(os.environ), extra),
        "factory": lambda: local.build_subprocess_env(extra=extra),
        "nonterminal": lambda: local.hermes_subprocess_env(base_env={**os.environ, **extra}),
    }
    result = factories[builder]()
    assert result.get("OPENAI_API_KEY") == base_force
    assert result.get("OPENAI_BASE_URL") == extra_force
    assert result["MY_CUSTOM_VAR"] == "caller-value"
    assert "AUXILIARY_VISION_API_KEY" not in result
    assert not any(k.startswith("_HERMES_FORCE_") for k in result)
    # Even a buggy plugin hook cannot bypass dynamic-secret exclusion.
    with patch("tools.env_passthrough.is_env_passthrough", return_value=True):
        assert "AUXILIARY_VISION_API_KEY" not in factories[builder]()


@pytest.mark.parametrize("managed,platform,allowed", [(True, "telegram", True), (False, "buzz", True), (False, "telegram", False)])
@pytest.mark.parametrize("scope", [None, {"BUZZ_PRIVATE_KEY": "fake-scoped"}])
def test_buzz_context_and_plain_process_value(child_env, monkeypatch, managed, platform, allowed, scope):
    from agent import secret_scope as ss
    from gateway.session_context import _SESSION_PLATFORM
    from tools.code_execution_env import _scrub_child_env
    from tools.env_passthrough import register_env_passthrough, is_env_passthrough
    buzz = {"BUZZ_PRIVATE_KEY": "fake-process", "BUZZ_AUTH_TAG": "fake-tag", "BUZZ_RELAY_URL": "fake-relay"}
    for k, v in buzz.items():
        monkeypatch.setenv(k, v)
    if managed:
        monkeypatch.setenv("BUZZ_MANAGED_AGENT", "1")
    platform_token = _SESSION_PLATFORM.set(platform)
    ss.set_multiplex_active(True)
    scope_token = ss.set_secret_scope(scope) if scope is not None else None
    try:
        register_env_passthrough(buzz)
        assert not any(is_env_passthrough(k) for k in buzz)
        for result in (local._make_run_env({}), local._sanitize_subprocess_env(dict(os.environ))):
            assert {k: result.get(k) for k in buzz} == (buzz if allowed else dict.fromkeys(buzz))
        for result in (local.hermes_subprocess_env(), _scrub_child_env(os.environ)):
            assert not set(buzz) & result.keys()
    finally:
        if scope_token is not None:
            ss.reset_secret_scope(scope_token)
        ss.set_multiplex_active(False)
        _SESSION_PLATFORM.reset(platform_token)


@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("first_platform,first_value", [("buzz", "fake-profile-a"), ("telegram", None)], ids=["buzz", "other"])
def test_buzz_secret_never_reaches_second_profile_via_snapshot(child_env, monkeypatch, first_platform, first_value):
    from agent import secret_scope as ss
    from gateway.session_context import _SESSION_PLATFORM
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "fake-profile-a")
    ss.set_multiplex_active(True)
    platform = _SESSION_PLATFORM.set(first_platform)
    env = None
    try:
        env = local.LocalEnvironment(cwd=str(child_env), timeout=30)
        snap = Path(env._snapshot_path)
        assert snap.exists()
        assert "BUZZ_PRIVATE_KEY" not in snap.read_text(encoding="utf-8")
        assert observe_terminal(env, ["BUZZ_PRIVATE_KEY"]) == {"BUZZ_PRIVATE_KEY": first_value}
        assert "fake-profile-a" not in snap.read_text(encoding="utf-8")
        monkeypatch.delenv("BUZZ_PRIVATE_KEY")
        _SESSION_PLATFORM.reset(platform)
        platform = _SESSION_PLATFORM.set("telegram")
        assert observe_terminal(env, ["BUZZ_PRIVATE_KEY"]) == {"BUZZ_PRIVATE_KEY": None}
        assert "BUZZ_PRIVATE_KEY" in env._snapshot_passthrough_names
        assert "BUZZ_PRIVATE_KEY" not in snap.read_text(encoding="utf-8")
    finally:
        if env is not None:
            env.cleanup()
        _SESSION_PLATFORM.reset(platform)
        ss.set_multiplex_active(False)


@pytest.mark.parametrize("scoped,expected", [({"SERVICE_TOKEN": "fake-routed"}, "fake-routed"), ({}, None)])
def test_profile_passthrough_in_terminal_child(child_env, monkeypatch, scoped, expected):
    from agent import secret_scope as ss
    from tools.env_passthrough import register_env_passthrough
    register_env_passthrough(["SERVICE_TOKEN"])
    monkeypatch.setenv("SERVICE_TOKEN", "fake-default")
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(scoped)
    env = None
    try:
        env = local.LocalEnvironment(cwd=str(child_env), timeout=30)
        assert observe_terminal(env, ["SERVICE_TOKEN"]) == {"SERVICE_TOKEN": expected}
        result = local._sanitize_subprocess_env(dict(os.environ))
        assert result.get("SERVICE_TOKEN") == expected
    finally:
        if env is not None:
            env.cleanup()
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)


@pytest.mark.parametrize("entries,expected", [
    (["SITE", "/user/lib"], ["/user/lib"]),
    (["REPO", "/user/lib"], ["/user/lib"]),
    (["SITE", "/user/lib", "SITE", "/user/lib"], ["/user/lib", "/user/lib"]),
    (["SITE", "REPO"], None),
    (["/first", "REPO", "", "SITE", "/last"], ["/first", "", "/last"]),
    ([" /opt/user-lib ", "relative/../lib", "", "/opt/user-lib", "/opt/user-lib"],
     [" /opt/user-lib ", "relative/../lib", "", "/opt/user-lib", "/opt/user-lib"]),
    (["/nix/store/user-plugin/lib/python3.12/site-packages", "/old/lib/python2.7/site-packages"],
     ["/nix/store/user-plugin/lib/python3.12/site-packages", "/old/lib/python2.7/site-packages"]),
    (["/opt/tools/python3.13/bin", "/opt/downloads/python3.13", "/custom/python3.13"],
     ["/opt/tools/python3.13/bin", "/opt/downloads/python3.13", "/custom/python3.13"]),
    ([""], [""]),
    (None, None),
])
def test_pythonpath_literal_policy(entries, expected):
    locations = {"REPO": str(Path(__file__).resolve().parents[2]), "SITE": str(_running_site())}
    env = {} if entries is None else {"PYTHONPATH": os.pathsep.join(locations.get(p, p) for p in entries)}
    pp._strip_hermes_owned_pythonpath(env)
    assert env.get("PYTHONPATH") == (os.pathsep.join(expected) if expected is not None else None)


def test_pythonpath_descendants_are_not_owned():
    repo, site = Path(__file__).resolve().parents[2], _running_site()
    entries = [str(site / "user-path"), str(repo / "tools"), str(repo / "tools/environments"),
               "/opt/other-venv/lib/python3.99/site-packages"]
    env = {"PYTHONPATH": os.pathsep.join(entries)}
    pp._strip_hermes_owned_pythonpath(env)
    assert env["PYTHONPATH"].split(os.pathsep) == entries


@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("link_at", ["home", "repo", "unrelated"])
@pytest.mark.parametrize("profile", [False, True])
def test_launcher_alias_provenance(child_env, monkeypatch, link_at, profile):
    from hermes_cli.gateway_windows import _preserve_hermes_home_path
    from hermes_cli.profiles import resolve_profile_env
    physical_home = child_env / "physical-home"
    physical_root = physical_home / "hermes-agent"
    physical_root.mkdir(parents=True)
    configured = child_env / "configured-home"
    if link_at == "home":
        _make_directory_link(configured, physical_home)
    else:
        configured.mkdir()
        if link_at == "repo":
            _make_directory_link(configured / "hermes-agent", physical_root)
        else:
            (configured / "hermes-agent").mkdir()
    (configured / "profiles/coder").mkdir(parents=True)
    unrelated = child_env / "user-tools/hermes-agent"
    unrelated.mkdir(parents=True)
    lexical_root = configured / "hermes-agent"
    monkeypatch.setenv("HERMES_HOME", str(configured))
    assert Path(resolve_profile_env("default")) == configured
    assert Path(resolve_profile_env("coder")) == configured / "profiles/coder"
    if link_at == "home":
        assert Path(_preserve_hermes_home_path(physical_root)) == lexical_root
    active_home = configured / "profiles/coder" if profile else configured
    aliases = pp._build_hermes_repo_root_aliases(physical_root.resolve(), physical_root, active_home)
    monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
    nested = lexical_root / "user-data"
    entries = [str(lexical_root), str(nested), str(unrelated), str(active_home / "not-the-repo")]
    env = {"PYTHONPATH": os.pathsep.join(entries)}
    pp._strip_hermes_owned_pythonpath(env)
    assert env["PYTHONPATH"].split(os.pathsep) == (entries if link_at == "unrelated" else entries[1:])
    if profile:
        assert active_home / "hermes-agent" not in aliases


@pytest.mark.parametrize("has_facts", [True, False])
def test_runtime_provenance_is_independent_of_aliases_and_virtual_env(child_env, monkeypatch, has_facts):
    from hermes_cli.runtime_paths import runtime_facts_path
    payload = child_env / "payload"
    runtime = payload / "state/environments/candidate/venv"
    site = runtime / ("Lib/site-packages" if os.name == "nt" else
                      f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    site.mkdir(parents=True)
    (runtime / "pyvenv.cfg").write_text("version = 3.14\n", encoding="utf-8")
    (payload / "tools").mkdir()
    (payload / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.runtime_paths.install_state_dir", lambda repo: payload / "state")
    if has_facts:
        facts = runtime_facts_path(Path(__file__).resolve().parents[2])
        facts.parent.mkdir(parents=True, exist_ok=True)
        facts.write_text(json.dumps({"packages": {"venv": {"environment": str(runtime)}}}), encoding="utf-8")
    monkeypatch.setattr(local, "_in_venv", False)
    monkeypatch.setattr(local, "_hermes_site_packages", None)
    alias = child_env / "unrelated-repo-alias"
    monkeypatch.setattr(local, "_hermes_repo_root_aliases", (alias,))
    user_venv = child_env / "user-venv"
    user_site = user_venv / "Lib/site-packages"
    user_site.mkdir(parents=True)
    (user_venv / "pyvenv.cfg").write_text("version = 3.13\n", encoding="utf-8")
    base = {"VIRTUAL_ENV": str(user_venv), "PYTHONPATH": os.pathsep.join(map(str, [site, alias, user_site]))}
    result = local._sanitize_subprocess_env(base)
    expected = [str(user_site)] if has_facts else [str(site), str(user_site)]
    assert result["PYTHONPATH"].split(os.pathsep) == expected
    assert "VIRTUAL_ENV" not in result
    assert base["VIRTUAL_ENV"] == str(user_venv)


@pytest.mark.parametrize("existing,expected", [
    (["/usr/bin", "/bin"], ["/opt/hermes/bin", "/usr/bin", "/bin"]),
    (["/usr/bin", "/opt/hermes/bin"], ["/usr/bin", "/opt/hermes/bin"]),
])
def test_background_hermes_path_repair_is_idempotent(child_env, monkeypatch, existing, expected):
    monkeypatch.setattr(local, "_HERMES_BIN_DIR", "/opt/hermes/bin")
    result = local._sanitize_subprocess_env({"PATH": os.pathsep.join(existing)})
    assert result["PATH"].split(os.pathsep) == expected
    assert local._sanitize_subprocess_env(result)["PATH"] == result["PATH"]


def test_hermes_bin_resolution_and_unresolved_noop(child_env, monkeypatch):
    bin_dir = child_env / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(local, "_HERMES_BIN_DIR", local._SENTINEL)
    monkeypatch.setattr(local.shutil, "which", lambda name: str(bin_dir / "hermes") if name == "hermes" else None)
    assert local._resolve_hermes_bin_dir() == str(bin_dir)
    monkeypatch.setattr(local, "_HERMES_BIN_DIR", None)
    assert local._prepend_hermes_bin_dir("/usr/bin") == "/usr/bin"


@pytest.mark.platforms("posix")
def test_foreground_minimal_path_preserves_operator_precedence(child_env, monkeypatch):
    monkeypatch.setattr(local, "_HERMES_BIN_DIR", "/opt/hermes/bin")
    monkeypatch.setenv("PATH", "/custom/bin:/custom/bin::/usr/bin")
    result = local._make_run_env({})["PATH"].split(":")
    assert result[:3] == ["/opt/hermes/bin", "/custom/bin", "/usr/bin"]
    assert "/opt/homebrew/bin" in result and "/opt/homebrew/sbin" in result
    assert "" not in result
    assert result.count("/custom/bin") == 1


def _make_directory_link(link: Path, target: Path) -> None:
    """Create a directory link without requiring symlink privileges.

    POSIX: Path.symlink_to.  Windows: try symlink_to first (works with
    Developer Mode enabled), then fall back to an unprivileged directory
    junction via `cmd /c mklink /J` -- junctions do not require the
    SeCreateSymbolicLinkPrivilege.  Raises the original error when no
    mechanism is available so callers can skip with a clear reason.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if sys.platform != "win32":
            raise
    # Binary capture: on a localized Windows the junction message is in the
    # console code page (e.g. GBK), which would raise UnicodeDecodeError in
    # the reader thread under UTF-8 mode.  Only the exit code matters.
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(detail or f"mklink /J failed: {result.returncode}")


class TestNativeEnvironmentContracts:
    @pytest.fixture(autouse=True)
    def _no_bin_injection(self, monkeypatch):
        monkeypatch.setattr(local, "_HERMES_BIN_DIR", None)

    @pytest.mark.platforms("windows")
    def test_windows_hermes_owned_paths_stripped(self):
        """On Windows, a Hermes venv site-packages entry written with
        backslashes is stripped by the same Hermes-owned check, while a
        user Windows path is preserved.  Windows-only: POSIX ``Path`` does
        not split on backslashes, so this cannot be meaningfully simulated
        on a POSIX host."""
        from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath

        venv_sp = str(_running_site())
        # Windows form: C:\...\venv\Lib\site-packages (backslashes)
        hermes_win = venv_sp
        user_win = "D:\\\\user\\\\lib"
        env = {
            "PYTHONPATH": ";".join([hermes_win, user_win]),
        }
        _strip_hermes_owned_pythonpath(env)
        entries = env["PYTHONPATH"].split(";")
        assert hermes_win not in entries
        assert user_win in entries

    @pytest.mark.platforms("macos")
    def test_make_run_env_real_launchd_path_gains_homebrew(self):
        """The literal macOS launchd PATH is the production trigger for #35613.

        macOS-only: the regression is the launchd environment on macOS, and
        the sane-path merge is a documented passthrough on Windows.
        """
        from tools.environments.local import _make_run_env
        launchd_env = {"PATH": os.pathsep.join(["/usr/bin", "/bin", "/usr/sbin", "/sbin"])}
        with patch.dict(os.environ, launchd_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(os.pathsep)
        assert "/opt/homebrew/bin" in path_entries
        assert "/opt/homebrew/sbin" in path_entries
        # Original entries keep their leading precedence.
        assert path_entries[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]


    @pytest.mark.platforms("windows")
    def test_make_run_env_preserves_windows_mixed_case_path_key(self, monkeypatch):
        """Windows-only: ``_path_env_key`` looks for a case-insensitive PATH
        key only on Windows, so the mixed-case ``Path`` preservation this
        asserts is a genuinely Windows-native behaviour.

        The Git Bash dir prepend is neutralised so the assertion is about the
        key casing alone (a real Windows box has those dirs).
        """
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        windows_env = {"Path": r"C:\Windows\System32;C:\Program Files\Git\bin"}
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
        with patch.object(local_mod.os, "environ", windows_env):
            result = _make_run_env({})
        assert result["Path"] == windows_env["Path"]
        assert "PATH" not in result
