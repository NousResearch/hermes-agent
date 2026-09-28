"""Child-observed terminal policy, profile lifetime and PYTHONPATH provenance."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.tools._child_env_fixtures import child_env, observe_child, observe_terminal  # noqa: F401
from tools.environments import local
from tools.environments import local_pythonpath as pp
from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST


def _running_venv_site_packages() -> Path:
    """Independently construct the host-native venv site-packages path."""
    if sys.platform == "win32":
        return Path(sys.prefix) / "Lib" / "site-packages"
    return Path(sys.prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"


def _physical_repo_root(tmp_path: Path) -> Path:
    """Create the physical repo checkout directory for junction tests."""
    physical_root = tmp_path / "physical-home" / "hermes-agent"
    physical_root.mkdir(parents=True)
    return physical_root


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
    (configured / "profiles/coder/config.yaml").write_text("{}\n", encoding="utf-8")
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
    from pm.environments import runtime_facts_path
    payload = child_env / "payload"
    runtime = payload / "state/environments/candidate/venv"
    site = runtime / ("Lib/site-packages" if os.name == "nt" else
                      f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    site.mkdir(parents=True)
    (runtime / "pyvenv.cfg").write_text("version = 3.14\n", encoding="utf-8")
    (payload / "tools").mkdir()
    (payload / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("pm.environments.install_state_dir", lambda repo: payload / "state")
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

    def test_empty_pythonpath_unchanged(self):
        """An empty PYTHONPATH is a no-op (falsy -> early return)."""
        from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath
        env = {"PYTHONPATH": ""}
        _strip_hermes_owned_pythonpath(env)
        # Empty string is falsy, so the function returns early without
        # modifying the dict.  The key stays as-is (empty string).
        assert env.get("PYTHONPATH") == ""

    def test_empty_component_preserved(self):
        """An empty component means cwd and must survive unchanged."""
        from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath

        user_pp = os.pathsep.join(["/foo", "", "/bar"])
        env = {"PYTHONPATH": user_pp}

        _strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"] == user_pp

    def test_raw_user_spelling_preserved(self):
        """The sanitizer does not trim, normalize, or deduplicate user entries."""
        from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath

        user_pp = os.pathsep.join([
            " /opt/user-lib ",
            "relative/../lib",
            "",
            "/opt/user-lib",
            "/opt/user-lib",
        ])
        env = {"PYTHONPATH": user_pp}

        _strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"] == user_pp


    @pytest.mark.platforms("windows")
    def test_base_python_sanitizer_uses_validated_separate_runtime_venv(self, tmp_path, monkeypatch):
        """A base interpreter strips the exact Windows runtime site-packages.

        This deliberately uses a synthetic Hermes venv separate from the test
        runner: sys.prefix represents base Python, while validated VIRTUAL_ENV
        identifies ``<repo>/venv`` as the Hermes runtime producer contract.
        """
        import tools.environments.local as local
        from tools.environments import local_pythonpath

        repo_root = tmp_path / "hermes-agent"
        runtime_venv = repo_root / "venv"
        runtime_sp = runtime_venv / "Lib" / "site-packages"
        runtime_sp.mkdir(parents=True)
        (runtime_venv / "pyvenv.cfg").write_text("version = 3.11\n", encoding="utf-8")
        base_prefix = tmp_path / "base-python"
        unrelated = "/custom/lib/python3.13/site-packages"

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", (repo_root,))
        monkeypatch.setattr(local, "_in_venv", False)
        monkeypatch.setattr(local, "_hermes_site_packages", None)
        monkeypatch.setattr(local.sys, "prefix", str(base_prefix))
        monkeypatch.setattr(local.sys, "base_prefix", str(base_prefix))

        env = {
            "VIRTUAL_ENV": str(runtime_venv),
            "PYTHONPATH": os.pathsep.join([str(runtime_sp), unrelated]),
        }
        result = local._sanitize_subprocess_env(env)

        assert Path(local.sys.prefix) == base_prefix
        assert runtime_venv != Path(local.sys.prefix)
        assert result["PYTHONPATH"] == unrelated
        assert "VIRTUAL_ENV" not in result

    def test_unrelated_virtual_env_is_not_runtime_provenance(self, tmp_path, monkeypatch):
        """An arbitrary inherited VIRTUAL_ENV cannot claim PYTHONPATH ownership."""
        import tools.environments.local as local
        from tools.environments import local_pythonpath

        repo_root = tmp_path / "hermes-agent"
        repo_root.mkdir()
        unrelated_venv = tmp_path / "user-venv"
        unrelated_sp = unrelated_venv / "Lib" / "site-packages"
        unrelated_sp.mkdir(parents=True)
        (unrelated_venv / "pyvenv.cfg").write_text("version = 3.13\n", encoding="utf-8")

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", (repo_root,))
        monkeypatch.setattr(local, "_in_venv", False)
        monkeypatch.setattr(local, "_hermes_site_packages", None)

        env = {
            "VIRTUAL_ENV": str(unrelated_venv),
            "PYTHONPATH": str(unrelated_sp),
        }
        local_pythonpath._strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"] == str(unrelated_sp)


    def test_no_pythonpath_key(self):
        """Missing PYTHONPATH key is a no-op."""
        from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath
        env = {"PATH": "/usr/bin"}
        _strip_hermes_owned_pythonpath(env)
        assert "PYTHONPATH" not in env


    @pytest.mark.parametrize("builder", [
        "_make_run_env",
        "_sanitize_subprocess_env",
        "hermes_subprocess_env",
    ])
    def test_builders_strip_hermes_venv_pythonpath(self, builder):
        """Every subprocess env builder applies the same sanitation contract:
        Hermes venv site-packages is stripped, user entries survive.
        """
        from tools.environments import local as local_mod

        venv_sp = str(_running_venv_site_packages())
        seed = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHONPATH": os.pathsep.join([venv_sp, "/home/user/my-lib"]),
        }
        with patch.dict(os.environ, seed, clear=True):
            if builder == "_make_run_env":
                result = local_mod._make_run_env({})
            elif builder == "_sanitize_subprocess_env":
                result = local_mod._sanitize_subprocess_env(dict(os.environ))
            else:
                result = local_mod.hermes_subprocess_env()
        pp = result.get("PYTHONPATH", "")
        entries = pp.split(os.pathsep) if pp else []
        assert venv_sp not in entries
        assert "/home/user/my-lib" in entries

    def test_scrub_child_env_strips_hermes_venv_pythonpath(self):
        """execute_code's _scrub_child_env path: after scrubbing, Hermes venv
        site-packages entries should be stripped when
        _strip_hermes_owned_pythonpath is applied (as the spawn path does),
        while user entries (even for another Python version) are preserved.
        """
        from tools.code_execution_env import _scrub_child_env
        from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath

        venv_sp = str(_running_venv_site_packages())
        other_sp = "/opt/other-venv/lib/python3.99/site-packages"
        source = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "PYTHONPATH": os.pathsep.join([venv_sp, other_sp, "/home/user/my-lib"]),
        }
        scrubbed = _scrub_child_env(source)
        # The scrubber passes PYTHONPATH through (it's in _SAFE_ENV_PREFIXES).
        assert "PYTHONPATH" in scrubbed
        # Now apply the selective strip (as the spawn path does).
        _strip_hermes_owned_pythonpath(scrubbed)
        pp = scrubbed.get("PYTHONPATH", "")
        entries = pp.split(os.pathsep) if pp else []
        assert venv_sp not in entries
        assert other_sp in entries
        assert "/home/user/my-lib" in entries

    @pytest.mark.parametrize("same_env", [True, False])
    def test_execute_code_composition_strips_inherited_hermes_entries(self, same_env):
        """Integration: execute_code's real spawn path composes a clean PYTHONPATH.

        Seeds a contaminated inherited PYTHONPATH (Hermes repo root + Hermes
        venv site-packages + user entries) through os.environ and drives
        execute_code all the way to Popen.  Proves the #84500 conditional
        composition and the #82581 selective strip compose correctly:

        * inherited Hermes venv site-packages never survive into the sandbox;
        * the staging tmpdir stays the first entry;
        * the repo root is deliberately re-added exactly once for a same-env
          child (the single occurrence proves the inherited copy was stripped
          first) and stays absent for an external-environment child;
        * user entries survive after the controlled entries.
        """
        import tools.code_execution_tool as cet
        from tools.code_execution_tool import execute_code

        def _mock_handle_function_call(function_name, function_args, task_id=None, user_task=None):
            return '{"output": "mock", "exit_code": 0}'

        hermes_root = str(Path(cet.__file__).resolve().parents[1])
        venv_sp = str(_running_venv_site_packages())
        user_a = "/home/user/my-lib"
        user_b = "/opt/project/lib"
        captured = {}

        def _fake_popen(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            captured["staging"] = os.path.dirname(cmd[1])
            proc = MagicMock()
            # The kernel's reader threads drain with read1(); a bare MagicMock never returns
            # EOF there, so the stderr thread spins forever appending mocks (a 1 GB/min leak
            # that outlived the test and OOM-killed the worker five times).
            proc.stdout.read.return_value = b""
            proc.stdout.read1.return_value = b""
            proc.stderr.read.return_value = b""
            proc.stderr.read1.return_value = b""
            proc.wait.return_value = 0
            proc.returncode = 0
            proc.poll.return_value = 0
            return proc

        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": "strict"}), \
             patch("model_tools.handle_function_call",
                   side_effect=_mock_handle_function_call), \
             patch("tools.code_execution_env._uses_hermes_python_environment",
                   return_value=same_env), \
             patch("subprocess.Popen", side_effect=_fake_popen), \
             patch.dict(os.environ, {
                 "PYTHONPATH": os.pathsep.join(
                     [hermes_root, venv_sp, user_a, user_b]),
             }):
            execute_code(code="pass", task_id="test-int", enabled_tools=[])

        assert "PYTHONPATH" in captured["env"], \
            "execute_code never reached Popen"
        parts = captured["env"]["PYTHONPATH"].split(os.pathsep)
        # Windows path comparison is case-insensitive: the inherited entries
        # and the re-added repo root can carry a different case than the
        # resolve()/abspath()-derived spellings used in this test (e.g. a
        # launcher-written lowercase PYTHONPATH).  Normalize with
        # os.path.normcase so a case-only difference never fails the
        # composition contract (identity on POSIX).
        norm_parts = [os.path.normcase(p) for p in parts]
        norm_staging = os.path.normcase(captured["staging"])
        norm_root = os.path.normcase(hermes_root)
        norm_venv = os.path.normcase(venv_sp)
        norm_user_a = os.path.normcase(user_a)
        norm_user_b = os.path.normcase(user_b)
        assert norm_parts[0] == norm_staging, \
            "staging tmpdir must be the first PYTHONPATH entry"
        assert norm_venv not in norm_parts, \
            "inherited Hermes venv site-packages must be stripped"
        assert norm_user_a in norm_parts and norm_user_b in norm_parts, \
            "user PYTHONPATH entries must survive"
        assert norm_parts.index(norm_user_a) > norm_parts.index(norm_staging), \
            "user entries must come after the staging tmpdir"
        if same_env:
            assert norm_parts.count(norm_root) == 1, \
                "repo root must be re-added exactly once for a same-env child"
            assert norm_parts.index(norm_user_a) > norm_parts.index(norm_root), \
                "user entries must come after the re-added repo root"
        else:
            assert norm_root not in norm_parts, \
                "repo root must stay absent for an external-env child"
        # The fake streams must hit EOF: the kernel's reader threads consume
        # ``read1()``, and an unconfigured MagicMock there is a truthy value
        # forever — the stderr reader spins after the test returns, growing
        # the pytest process by hundreds of MB per second (#115912).
        for thread in threading.enumerate():
            if "_reader" in thread.name:
                thread.join(timeout=2)
                assert not thread.is_alive(), \
                    f"{thread.name} is still spinning on the fake kernel stream"


    def test_repo_root_direct_child_preserved(self):
        """A direct child of the repo root (depth=1) is PRESERVED.

        Independent audit of every real launcher producer (Electron
        ``apps/desktop/electron/main.ts``,
        ``gateway/run.py::_ensure_windows_gateway_venv_imports``,
        ``cron/scheduler.py::_windows_cron_python_invocation``,
        ``tui_gateway/host_supervisor.py``) shows they all inject the exact
        repo root and/or the venv site-packages — none injects
        ``<repo>/tools`` or another direct child as an independent
        PYTHONPATH entry.  A user path that merely happens to live under
        the repo directory must therefore be preserved.
        """
        from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath

        local_file = Path(__import__("tools.environments.local", fromlist=["__file__"]).__file__).resolve()
        real_repo_root = local_file.parents[2]
        direct_child = str(real_repo_root / "tools")

        env = {
            "PYTHONPATH": os.pathsep.join([direct_child, "/home/user/my-lib"]),
        }
        _strip_hermes_owned_pythonpath(env)
        pp = env.get("PYTHONPATH", "")
        entries = pp.split(os.pathsep) if pp else []
        assert direct_child in entries
        assert "/home/user/my-lib" in entries

    def test_configured_home_alias_matches_launcher_output(self, tmp_path, monkeypatch):
        """The real producer spelling is derived and consumed end to end."""
        import tools.environments.local as local
        from tools.environments import local_pythonpath
        from hermes_cli.gateway_windows import _preserve_hermes_home_path

        physical_home = tmp_path / "physical-home"
        physical_root = _physical_repo_root(tmp_path)
        configured_home = tmp_path / "configured-home"
        try:
            _make_directory_link(configured_home, physical_home)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")
        monkeypatch.setenv("HERMES_HOME", str(configured_home))

        launcher_entry = Path(_preserve_hermes_home_path(physical_root))
        aliases = local_pythonpath._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )

        assert launcher_entry == configured_home / "hermes-agent"
        assert launcher_entry in aliases

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        nested_user_path = launcher_entry / "user-data"
        env = {
            "PYTHONPATH": os.pathsep.join([
                str(launcher_entry),
                str(nested_user_path),
                "/home/user/my-lib",
            ])
        }
        local_pythonpath._strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"].split(os.pathsep) == [
            str(nested_user_path),
            "/home/user/my-lib",
        ]

    def test_profile_rehome_keeps_junction_lexical_alias(self, tmp_path, monkeypatch):
        """Profile re-home must not lose the launcher's lexical repo-root spelling.

        The desktop/CLI spawn children with HERMES_HOME and PYTHONPATH in the
        configured (junction) spelling, but --profile / sticky active_profile
        re-home HERMES_HOME through resolve_profile_env() before the
        sanitizer loads.  Regression (junction + profile re-home): the alias
        builder must still recover the lexical root so the inherited lexical
        repo-root entry is stripped.
        """
        import tools.environments.local as local
        from tools.environments import local_pythonpath
        from hermes_cli.profiles import resolve_profile_env

        physical_home = tmp_path / "physical-home"
        physical_root = physical_home / "hermes-agent"
        physical_root.mkdir(parents=True)
        (physical_home / "profiles" / "coder").mkdir(parents=True)
        (physical_home / "profiles" / "coder" / "config.yaml").write_text("{}\n")  # identity marker
        configured_home = tmp_path / "configured-home"
        try:
            _make_directory_link(configured_home, physical_home)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        # Launcher contract: the configured spelling is the env and the root.
        monkeypatch.setenv("HERMES_HOME", str(configured_home))
        lexical_root = configured_home / "hermes-agent"

        # Profile re-home keeps the configured spelling (physically identical
        # through the link; lexically the launcher spelling is preserved).
        assert Path(resolve_profile_env("default")) == configured_home
        assert Path(resolve_profile_env("coder")) == configured_home / "profiles" / "coder"

        # The sanitizer now runs under the re-homed (profile) HERMES_HOME.
        aliases = local_pythonpath._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home / "profiles" / "coder",
        )
        assert any(local_pythonpath._same_path(a, lexical_root) for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        env = {"PYTHONPATH": os.pathsep.join([str(lexical_root), "/home/user/my-lib"])}
        local_pythonpath._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]


    def test_repo_level_junction_recovers_lexical_alias(self, tmp_path, monkeypatch):
        """The repo itself may be a junction under the configured root
        (e.g. D:\\hermes\\hermes-agent -> C:\\...\\hermes-agent) while the
        editable import spelling resolves to the physical location.  The
        alias builder must recover the lexical spelling via exact-identity
        proof (strict resolve), not a name-based guess.
        """
        import tools.environments.local as local
        from tools.environments import local_pythonpath

        physical_root = _physical_repo_root(tmp_path)
        configured_home = tmp_path / "configured-home"
        configured_home.mkdir()
        # repo-level link: <configured-home>/hermes-agent -> physical repo
        try:
            _make_directory_link(configured_home / "hermes-agent", physical_root)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        lexical_root = configured_home / "hermes-agent"
        aliases = local_pythonpath._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        assert any(local_pythonpath._same_path(a, lexical_root) for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        env = {"PYTHONPATH": os.pathsep.join([str(lexical_root), "/home/user/my-lib"])}
        local_pythonpath._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]

    def test_same_named_non_owned_directories_preserved(self, tmp_path, monkeypatch):
        """Negative controls: a directory that merely shares the repo's name
        -- whether under the configured root or in an unrelated location --
        is never aliased or stripped.  Exact filesystem identity decides,
        not the name; no ownership provenance means no strip.
        """
        import tools.environments.local as local
        from tools.environments import local_pythonpath

        physical_root = _physical_repo_root(tmp_path)
        configured_home = tmp_path / "configured-home"
        (configured_home / "hermes-agent").mkdir(parents=True)
        unrelated = tmp_path / "user-tools" / "hermes-agent"
        unrelated.mkdir(parents=True)

        aliases = local_pythonpath._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        for lookalike in (configured_home / "hermes-agent", unrelated):
            assert not any(local_pythonpath._same_path(a, lookalike) for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        for lookalike in (configured_home / "hermes-agent", unrelated):
            env = {"PYTHONPATH": os.pathsep.join([str(lookalike), "/home/user/my-lib"])}
            local_pythonpath._strip_hermes_owned_pythonpath(env)
            assert env["PYTHONPATH"].split(os.pathsep) == [str(lookalike), "/home/user/my-lib"]

    def test_profile_home_with_repo_level_junction(self, tmp_path, monkeypatch):
        """Profile re-home + repo-level junction together: the configured home
        is <root>/profiles/<name> while the repo is a link at <root>/hermes-agent.
        The root spelling must be derived (profiles -> grandparent) and then
        the lexical repo alias recovered from it.
        """
        import tools.environments.local as local
        from tools.environments import local_pythonpath

        physical_root = _physical_repo_root(tmp_path)
        configured_root = tmp_path / "configured-root"
        (configured_root / "profiles" / "coder").mkdir(parents=True)
        try:
            _make_directory_link(configured_root / "hermes-agent", physical_root)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        configured_home = configured_root / "profiles" / "coder"
        lexical_root = configured_root / "hermes-agent"
        aliases = local_pythonpath._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        assert any(local_pythonpath._same_path(a, lexical_root) for a in aliases)
        assert not any(local_pythonpath._same_path(a, configured_home / "hermes-agent") for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        env = {"PYTHONPATH": os.pathsep.join([str(lexical_root), "/home/user/my-lib"])}
        local_pythonpath._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]

    @pytest.mark.platforms("windows")
    def test_validated_runtime_venv_lexical_after_repo_recovery(self, tmp_path, monkeypatch):
        """uv-base gateway: once the lexical repo alias is recovered, a lexical
        VIRTUAL_ENV (<lexical repo>/venv) validates and its site-packages is
        stripped together with the repo root, while user entries survive.
        """
        import tools.environments.local as local
        from tools.environments import local_pythonpath

        physical_root = _physical_repo_root(tmp_path)
        venv_dir = physical_root / "venv"
        venv_dir.mkdir(parents=True)
        (venv_dir / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        configured_home = tmp_path / "configured-home"
        configured_home.mkdir()
        try:
            _make_directory_link(configured_home / "hermes-agent", physical_root)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        lexical_root = configured_home / "hermes-agent"
        aliases = local_pythonpath._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        assert any(local_pythonpath._same_path(a, lexical_root) for a in aliases)
        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)

        lexical_venv = lexical_root / "venv"
        validated = local_pythonpath._validated_runtime_venv({"VIRTUAL_ENV": str(lexical_venv)})
        assert validated is not None
        assert local_pythonpath._same_path(validated, lexical_venv)

        local._hermes_site_packages = None
        env = {"PYTHONPATH": os.pathsep.join([
            str(lexical_root),
            str(lexical_venv / "Lib" / "site-packages"),
            "/home/user/my-lib",
        ]), "VIRTUAL_ENV": str(lexical_venv)}
        local_pythonpath._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]





class TestPythonhomeSanitized:
    """PYTHONHOME must not leak from the Hermes runtime into subprocesses.

    The gateway inherits/sets PYTHONHOME in its process environment; a child
    interpreter (system Python, another venv, cron no_agent scripts) that
    inherits it redirects its stdlib search to the Hermes venv and crashes
    with version-mismatch errors before importing anything (#75018).
    """

    @pytest.mark.parametrize("builder", [
        "_make_run_env",
        "_sanitize_subprocess_env",
        "hermes_subprocess_env",
        "build_subprocess_env",
    ])
    def test_builders_strip_pythonhome(self, builder):
        """The gateway's inherited PYTHONHOME must not reach any subprocess
        builder -- terminal, background/PTY, cron no_agent scripts, and
        execute_code children (#75018).
        """
        from tools.environments import local as local_mod

        seed = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHONHOME": "/opt/hermes-venv",
        }
        with patch.dict(os.environ, seed, clear=True):
            if builder == "_make_run_env":
                result = local_mod._make_run_env({})
            elif builder == "_sanitize_subprocess_env":
                result = local_mod._sanitize_subprocess_env(dict(os.environ))
            elif builder == "hermes_subprocess_env":
                result = local_mod.hermes_subprocess_env()
            else:
                result = local_mod.build_subprocess_env()
        assert "PYTHONHOME" not in result


    def test_build_subprocess_env_no_scrub_preserves_pythonhome(self):
        """``build_subprocess_env(scrub_secrets=False)`` is the documented
        byte-for-byte escape hatch: no key is removed, so PYTHONHOME (and
        everything else) survives there by contract, not by omission.

        Callers that explicitly opt out of scrubbing (git credential flows,
        secret CLIs) must not have their environment silently altered — this
        test pins that exception as intentional.
        """
        from tools.environments.local import build_subprocess_env
        base = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHONHOME": "/opt/hermes-venv",
            "VIRTUAL_ENV": "/opt/hermes-venv",
            "SERVICE_TOKEN": "s3cr3t",
        }
        result = build_subprocess_env(base, scrub_secrets=False)
        assert result.get("PYTHONHOME") == "/opt/hermes-venv"
        assert result.get("VIRTUAL_ENV") == "/opt/hermes-venv"
        assert result.get("SERVICE_TOKEN") == "s3cr3t"


class TestProfileScopedPassthrough:
    def test_make_run_env_uses_active_profile_for_passthrough(self, monkeypatch):
        """Allowlisted values must come from the routed profile, not os.environ."""
        from agent import secret_scope as ss
        from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
        from tools.environments.local import _make_run_env

        clear_env_passthrough()
        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile"})
        try:
            result = _make_run_env({})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)
            clear_env_passthrough()

        assert result["SERVICE_TOKEN"] == "token-for-routed-profile"

    def test_make_run_env_omits_missing_scoped_passthrough(self, monkeypatch):
        """A missing routed secret must not fall back to the default profile."""
        from agent import secret_scope as ss
        from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
        from tools.environments.local import _make_run_env

        clear_env_passthrough()
        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            result = _make_run_env({})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)
            clear_env_passthrough()

        assert "SERVICE_TOKEN" not in result


class TestBlocklistCoverage:
    """Sanity checks that the blocklist covers all known providers."""


    def test_registry_vars_are_in_blocklist(self):
        """Every api_key_env_var and base_url_env_var from PROVIDER_REGISTRY
        must appear in the blocklist — ensures no drift.

        CLAUDE_CODE_OAUTH_TOKEN is the one deliberate exemption: it is owned
        by the user's Claude Code install, not Hermes (#55878).
        """
        from hermes_cli.auth import PROVIDER_REGISTRY

        exempt = {"CLAUDE_CODE_OAUTH_TOKEN"}
        for pconfig in PROVIDER_REGISTRY.values():
            for var in pconfig.api_key_env_vars:
                if var in exempt:
                    continue
                assert var in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Registry var {var} (provider={pconfig.id}) missing from blocklist"
                )
            if pconfig.base_url_env_var:
                assert pconfig.base_url_env_var in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Registry base_url_env_var {pconfig.base_url_env_var} "
                    f"(provider={pconfig.id}) missing from blocklist"
                )


    def test_general_aws_chain_not_in_blocklist(self):
        """The general AWS credential chain must NOT be in the blocklist —
        no-regression guard for #32314. These belong to the user's trusted
        operator shell (SECURITY.md §3.2), not to Hermes, and blocklisting
        them would be unrecoverable via env_passthrough (GHSA-rhgp-j443-p4rf).
        """
        general_chain = {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_REGION",
            "AWS_REGION",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
        }
        leaked_block = general_chain & _HERMES_PROVIDER_ENV_BLOCKLIST
        assert not leaked_block, (
            f"General AWS chain vars must stay inheritable, but these are "
            f"blocklisted: {sorted(leaked_block)} (capability regression, #32314)"
        )


    def test_claude_code_oauth_token_is_inheritable(self):
        """CLAUDE_CODE_OAUTH_TOKEN is owned by the user's Claude Code install
        (subscription OAuth), not a Hermes inference credential. Stripping it
        made agent-spawned ``claude`` fall through to the shared Keychain /
        ~/.claude credential store and clobber the user's interactive login
        on auth failure (#55878). It must stay inheritable."""
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in _HERMES_PROVIDER_ENV_BLOCKLIST

    def test_non_registry_provider_vars_are_in_blocklist(self):
        extras = {
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
        }
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_optional_tool_and_messaging_vars_are_in_blocklist(self):
        """Tool/messaging vars from OPTIONAL_ENV_VARS should stay covered."""
        from hermes_cli.config import OPTIONAL_ENV_VARS

        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                assert name in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Optional env var {name} (category={category}) missing from blocklist"
                )
            elif category == "setting" and metadata.get("password"):
                assert name in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Secret setting env var {name} missing from blocklist"
                )

    def test_gateway_runtime_vars_are_in_blocklist(self):
        extras = {
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
            "VERCEL_OIDC_TOKEN",
            "VERCEL_TOKEN",
            "VERCEL_PROJECT_ID",
            "VERCEL_TEAM_ID",
        }
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)


class TestSanePathIncludesHomebrew:
    """Verify _SANE_PATH includes macOS Homebrew directories."""

    @pytest.fixture(autouse=True)
    def _disable_hermes_bin_injection(self):
        """These tests assert the sane-path merge in isolation. Disable the
        hermes-install-dir prepend (a separate concern, covered by
        TestHermesBinDirOnPath) so a real ``hermes`` on the test runner's PATH
        doesn't shift the asserted PATH layout."""
        from tools.environments import local as local_mod
        saved = local_mod._HERMES_BIN_DIR
        local_mod._HERMES_BIN_DIR = None  # resolved -> no dir to inject
        yield
        local_mod._HERMES_BIN_DIR = saved



    def test_make_run_env_appends_homebrew_on_minimal_path(self, monkeypatch):
        """When PATH is minimal, _make_run_env appends missing sane entries.

        POSIX: the sane-path merge appends the Homebrew dirs.  Windows:
        _append_missing_sane_path_entries is a documented passthrough (the
        native PATH must not be touched), so the assertion is the unchanged
        input.  Git Bash dir prepending is neutralised so the merged PATH
        layout is deterministic on every host.
        """
        from tools.environments import local as local_mod
        from tools.environments.local import _SANE_PATH, _make_run_env
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
        minimal_env = {"PATH": "/some/custom/bin"}
        with patch.dict(os.environ, minimal_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(os.pathsep)
        assert path_entries[0] == "/some/custom/bin"
        if sys.platform == "win32":
            assert result["PATH"] == "/some/custom/bin"
        else:
            for entry in _SANE_PATH.split(os.pathsep):
                assert entry in path_entries


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
