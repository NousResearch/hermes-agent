"""
Regression tests for the shared-container task_id mapping.

The top-level agent and all delegate_task subagents share a single
terminal sandbox keyed by ``"default"``.  ``_resolve_container_task_id``
is the sole gatekeeper for which tool-call task_ids go to the shared
container vs. get their own isolated sandbox.  RL / benchmark
environments opt in to isolation by calling
``register_task_env_overrides(task_id, {...})`` before the agent loop;
every other task_id collapses back to ``"default"``.

If you change the collapse logic, update both the helper and these
tests -- see `hermes-agent-dev` skill, "Why do subagents get their own
containers?" section, and the Container lifecycle paragraph under
Docker Backend in ``website/docs/user-guide/configuration.md``.
"""

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Ensure no stray overrides from other tests leak in."""
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def test_none_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_cwd_only_override_collapses_to_default():
    """CWD-only overrides (ACP adapter workspace tracking) must NOT trigger
    container isolation — they should collapse to the shared 'default'
    container so all surfaces (TUI, gateway, dashboard) share one sandbox.
    Regression for #37361."""
    terminal_tool.register_task_env_overrides(
        "acp-session-abc", {"cwd": "/home/user/project"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("acp-session-abc")
            == "default"
        )
    finally:
        terminal_tool.clear_task_env_overrides("acp-session-abc")


def test_env_type_override_keeps_own_id():
    """env_type is an isolation key — must trigger per-task container."""
    terminal_tool.register_task_env_overrides(
        "bench-env", {"env_type": "sandbox", "cwd": "/work"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("bench-env")
            == "bench-env"
        )
    finally:
        terminal_tool.clear_task_env_overrides("bench-env")


# --- Cross-profile SSH-leak isolation (commit e00f940a9, re-applied) ---------
#
# When a session key is present (WebUI/gateway), each session must own its own
# slot in _active_environments so switching from profile A (ssh_host=10.0.0.1)
# to profile B (ssh_host=10.0.0.2) cannot reuse A's SSHEnvironment. Without this
# the shared "default" slot silently runs commands on the wrong remote host.


def test_distinct_session_keys_get_distinct_slots(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-A")
    a = terminal_tool._resolve_container_task_id(None)
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-B")
    b = terminal_tool._resolve_container_task_id(None)
    assert a == "session:sess-A"
    assert b == "session:sess-B"
    assert a != b


def test_subagent_collapses_onto_parent_session(monkeypatch):
    # Subagents inherit the parent's session key, so they share the parent's
    # container (the #16177 intent) rather than a global "default".
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-A")
    assert (
        terminal_tool._resolve_container_task_id("subagent-3-cafef00d")
        == "session:sess-A"
    )


def test_rl_override_wins_over_session_key(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-A")
    terminal_tool.register_task_env_overrides("tb2-z", {"docker_image": "z:1"})
    try:
        assert terminal_tool._resolve_container_task_id("tb2-z") == "tb2-z"
    finally:
        terminal_tool.clear_task_env_overrides("tb2-z")


def test_no_session_key_still_defaults(monkeypatch):
    # CLI mode: no session key -> unchanged "default" behaviour.
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    assert terminal_tool._resolve_container_task_id(None) == "default"


# --- Production gateway path: session key bound via ContextVars ---------------
#
# The tests above set HERMES_SESSION_KEY through os.environ, which only
# exercises the os.getenv() *fallback* branch of the scoping logic. Real
# gateway turns never write this process-global env var — they bind the
# identity through gateway.session_context.set_session_vars(), which stores it
# in a ContextVar, and _resolve_container_task_id reads it back via
# get_session_env(). These companion tests cover that production path with
# HERMES_SESSION_KEY absent from os.environ.


def test_session_key_from_contextvar_without_environ(monkeypatch):
    # Prove the fix works on the gateway path: HERMES_SESSION_KEY is NOT in
    # os.environ; the key lives only in the ContextVar bound by the gateway.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    tokens = set_session_vars(session_key="sess-ctx")
    try:
        assert (
            terminal_tool._resolve_container_task_id(None) == "session:sess-ctx"
        )
        # Subagents inherit the same ContextVar and collapse onto the parent.
        assert (
            terminal_tool._resolve_container_task_id("subagent-1-cafe")
            == "session:sess-ctx"
        )
    finally:
        clear_session_vars(tokens)


def test_contextvar_session_key_wins_over_environ(monkeypatch):
    # Two concurrent gateway sessions in one process must not cross-contaminate:
    # the ContextVar is authoritative even when a *different* value lingers in
    # os.environ (e.g. a CLI-set or previously-leaked global). The container
    # slot must follow the ContextVar-bound session, not the process global.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-ENV")
    tokens = set_session_vars(session_key="sess-CTX")
    try:
        assert (
            terminal_tool._resolve_container_task_id(None) == "session:sess-CTX"
        )
    finally:
        clear_session_vars(tokens)


# --- Persistent Docker is PROFILE-scoped, not session-scoped ------------------
#
# Product contract: TERMINAL_ENV=docker + container_persistent:true means ONE
# long-lived container per Hermes profile, shared by every session of that
# profile (CLI, gateway chats, WebUI). The a270c4ade session-key fallback must
# NOT fragment persistent Docker into per-session containers; it exists for
# backends where cross-session reuse is dangerous (SSH).


def _persistent_docker(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")


def test_persistent_docker_default_profile_shares_default_container(monkeypatch):
    # A gateway session of the default profile lands on the SAME container key
    # CLI mode uses — "default" — not a per-session key.
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    tokens = set_session_vars(session_key="agent:main:telegram:dm:123", profile="")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "default"
    finally:
        clear_session_vars(tokens)


def test_persistent_docker_two_sessions_same_profile_share_container(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    tokens = set_session_vars(session_key="agent:main:telegram:dm:123", profile="work")
    try:
        a = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_key="agent:main:discord:guild:456", profile="work")
    try:
        b = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    assert a == b == "profile:work"


def test_persistent_docker_distinct_profiles_get_distinct_containers(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    tokens = set_session_vars(session_key="sess-X", profile="work")
    try:
        a = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_key="sess-X", profile="research")
    try:
        b = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    assert a == "profile:work"
    assert b == "profile:research"
    assert a != b


def test_nonpersistent_docker_keeps_session_scoping(monkeypatch):
    # container_persistent:false is an explicit isolation statement (#82731) —
    # the profile-scope gate must not fire there.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    tokens = set_session_vars(session_key="sess-A", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "session:sess-A"
    finally:
        clear_session_vars(tokens)


def test_ssh_backend_keeps_session_scoping(monkeypatch):
    # The original a270c4ade leak: SSH environments must stay session-scoped
    # regardless of profile, or profile A's SSHEnvironment leaks into B.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    tokens = set_session_vars(session_key="sess-A", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "session:sess-A"
    finally:
        clear_session_vars(tokens)


# --- Trusted-profiles shared container opt-in (#84671) ------------------------


def test_shared_key_unifies_profiles(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team/workspace")
    tokens = set_session_vars(session_key="s1", profile="work")
    try:
        a = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_key="s2", profile="research")
    try:
        b = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    assert a == b == "shared:team/workspace"


def test_shared_key_applies_to_cli_no_session(monkeypatch):
    # CLI (no session key) must land in the same shared container as gateway
    # sessions, or the opt-in splits the container it exists to unify.
    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team/workspace")
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    assert terminal_tool._resolve_container_task_id(None) == "shared:team/workspace"


def test_empty_shared_key_keeps_profile_scoping(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "")
    tokens = set_session_vars(session_key="s1", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "profile:work"
    finally:
        clear_session_vars(tokens)


def test_shared_key_ignored_outside_persistent_docker(monkeypatch):
    # The opt-in is a persistent-Docker concept only: SSH keeps session
    # scoping even when the key is set.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team/workspace")
    tokens = set_session_vars(session_key="sess-A", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "session:sess-A"
    finally:
        clear_session_vars(tokens)


def _use_home(monkeypatch, tmp_path, name):
    """Run as a real home: ``"default"`` = the Hermes root, ``None`` = a home outside the root (which
    is then its own root), anything else = ``<root>/profiles/<name>``."""
    import hermes_constants

    root = hermes_constants._get_platform_default_hermes_home()  # conftest's per-test native root, any OS
    home = root if name == "default" else tmp_path / "elsewhere" if name is None else root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))


def test_sessionless_default_profile_stays_default(monkeypatch, tmp_path):
    """Default profile, no session: CLI and workers still share "default"."""
    from gateway.session_context import reset_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PROFILE", raising=False)
    reset_session_vars()
    _use_home(monkeypatch, tmp_path, "default")
    assert terminal_tool._resolve_container_task_id(None) == "default"
    assert terminal_tool._resolve_container_task_id("kanban-task") == "default"


def test_sessionless_named_profile_reuses_profile_container(monkeypatch, tmp_path):
    """Named-profile process, no session: reuse profile:<name>, not "default"."""
    from gateway.session_context import reset_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PROFILE", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)
    reset_session_vars()
    _use_home(monkeypatch, tmp_path, "tenant-a")
    assert terminal_tool._resolve_container_task_id(None) == "profile:tenant-a"
    assert terminal_tool._resolve_container_task_id("t_b865ceb0") == "profile:tenant-a"


def test_explicit_session_profile_wins_over_process_home(monkeypatch, tmp_path):
    """A bound session profile is authoritative, even if the process home differs."""
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    _use_home(monkeypatch, tmp_path, "tenant-a")
    tokens = set_session_vars(session_key="agent:main:desktop:1", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "profile:work"
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_key="agent:main:desktop:2", profile="tenant-a")
    try:
        assert terminal_tool._resolve_container_task_id("subagent-1") == "profile:tenant-a"
    finally:
        clear_session_vars(tokens)


def test_sessionless_named_profile_ignored_when_not_persistent_docker(monkeypatch, tmp_path):
    """The profile key is a persistent-Docker rule. Local stays "default"."""
    from gateway.session_context import reset_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    reset_session_vars()
    _use_home(monkeypatch, tmp_path, "tenant-a")
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_sessionless_home_outside_the_root_stays_default(monkeypatch, tmp_path):
    """A home outside the Hermes root is its own root: it keys "default", never profile:custom."""
    from gateway.session_context import reset_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    reset_session_vars()
    _use_home(monkeypatch, tmp_path, None)
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_named_launch_profile_chats_desktop_and_workers_share_one_container(monkeypatch, tmp_path):
    """A gateway launched as a named profile stamps no profile on its own chats; those chats, the
    profile's Desktop sessions and its session-less cron/kanban work must all key one container."""
    from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PROFILE", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)
    reset_session_vars()
    _use_home(monkeypatch, tmp_path, "tenant-a")
    keys = {"worker": terminal_tool._resolve_container_task_id("t_kanban")}
    for label, profile in (("chat", ""), ("desktop", "tenant-a")):
        tokens = set_session_vars(session_key=f"agent:main:{label}:1", profile=profile)
        try:
            keys[label] = terminal_tool._resolve_container_task_id("x")
        finally:
            clear_session_vars(tokens)
    assert keys == dict.fromkeys(keys, "profile:tenant-a")


def test_unnamed_session_in_default_or_outside_root_home_stays_default(monkeypatch, tmp_path):
    """The default root and a home outside the root keep the shared default key."""
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)
    for home_profile in ("default", None):
        _use_home(monkeypatch, tmp_path, home_profile)
        tokens = set_session_vars(session_key="agent:main:telegram:dm:1", profile="")
        try:
            assert terminal_tool._resolve_container_task_id("x") == "default"
        finally:
            clear_session_vars(tokens)


def test_file_panel_mirror_names_the_unnamed_chat_sandbox(monkeypatch, tmp_path):
    """The file panel's best candidate must be the sandbox an unnamed chat of a named launch
    profile actually uses, with and without a mounted workspace."""
    from gateway.platforms.docker_media_paths import _docker_sandbox_dir_candidates
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.environments.path_utils import sanitize_task_id_for_path

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    digest = "0123456789abcdef"
    terminal_tool._task_env_overrides["ws"] = {
        "cwd": "/host/proj", "cwd_source": "session", "workspace_key": digest}
    for home_profile in ("tenant-a", "custom", None):
        _use_home(monkeypatch, tmp_path, home_profile)
        tokens = set_session_vars(session_key="agent:main:telegram:dm:1", profile="")
        try:
            plain = terminal_tool._resolve_container_task_id("plain")
            mounted = terminal_tool._resolve_container_task_id("ws")
        finally:
            clear_session_vars(tokens)
        assert _docker_sandbox_dir_candidates("agent:main:telegram:dm:1")[0] == sanitize_task_id_for_path(plain)
        assert _docker_sandbox_dir_candidates(
            "agent:main:telegram:dm:1", digest)[0] == sanitize_task_id_for_path(mounted)


def test_a_profile_literally_named_custom_keys_its_own_container(monkeypatch, tmp_path):
    """``custom`` is a valid profile name: its chats and workers key ``profile:custom`` like its
    Desktop sessions, never ``default``."""
    from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PROFILE", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)
    reset_session_vars()
    _use_home(monkeypatch, tmp_path, "custom")
    assert terminal_tool._resolve_container_task_id(None) == "profile:custom"
    for profile in ("", "custom"):
        tokens = set_session_vars(session_key="agent:main:telegram:dm:1", profile=profile)
        try:
            assert terminal_tool._resolve_container_task_id("x") == "profile:custom"
        finally:
            clear_session_vars(tokens)


@pytest.mark.parametrize("home_parts", [("profiles", "work", "nested"), ("profiles", ".hidden"), ("loose",)])
def test_a_home_the_tui_stamps_custom_keys_one_container_for_every_path(monkeypatch, tmp_path, home_parts):
    """Inside the Hermes root but not a ``profiles/<name>`` home, the TUI stamps Desktop sessions
    ``custom``; that home's chats and session-less work key the same container, and it is the file
    panel's first candidate (the one a Save may write to), plain and mounted."""
    from gateway.platforms.docker_media_paths import _docker_sandbox_dir_candidates
    from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
    from tools.environments.path_utils import sanitize_task_id_for_path

    import hermes_constants

    home = hermes_constants._get_platform_default_hermes_home().joinpath(*home_parts)
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _persistent_docker(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PROFILE", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    from tui_gateway.server import _current_profile_name
    assert _current_profile_name() == "custom"
    reset_session_vars()
    keys = {"worker": terminal_tool._resolve_container_task_id(None)}
    digest = "0123456789abcdef"
    terminal_tool._task_env_overrides["ws"] = {
        "cwd": "/host/proj", "cwd_source": "session", "workspace_key": digest}
    for label, profile in (("chat", ""), ("desktop", "custom")):
        tokens = set_session_vars(session_key=f"agent:main:{label}:1", profile=profile)
        try:
            keys[label] = terminal_tool._resolve_container_task_id("x")
            keys[f"{label}-ws"] = terminal_tool._resolve_container_task_id("ws")
        finally:
            clear_session_vars(tokens)
    assert {k: v for k, v in keys.items() if not k.endswith("-ws")} == dict.fromkeys(
        ("worker", "chat", "desktop"), "profile:custom")
    assert keys["chat-ws"] == keys["desktop-ws"] == f"profile:custom:ws:{digest}"
    assert _docker_sandbox_dir_candidates("agent:main:chat:1")[0] == sanitize_task_id_for_path(keys["chat"])
    assert _docker_sandbox_dir_candidates(
        "agent:main:chat:1", digest)[0] == sanitize_task_id_for_path(keys["chat-ws"])


def test_home_profile_name_is_cached_per_home_never_across_homes(monkeypatch, tmp_path):
    from tools.terminal_tool_config import home_profile_name

    for name in ("tenant-a", "tenant-b", "default", "tenant-a"):
        _use_home(monkeypatch, tmp_path, name)
        assert home_profile_name() == name


def test_an_unresolvable_home_keeps_the_shared_default_key(monkeypatch):
    import hermes_constants
    from gateway.platforms.docker_media_paths import _docker_sandbox_dir_candidates
    from tools.terminal_tool_config import home_profile_name

    def _boom():
        raise RuntimeError("no home")

    monkeypatch.setattr(hermes_constants, "get_hermes_home", _boom)
    assert home_profile_name() == "default"
    assert _docker_sandbox_dir_candidates()[0] == "default"
