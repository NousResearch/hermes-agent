"""Gateway-session Bot Mode teammate routing (Yuki's Discord bug).

A Bot-Mode-managed profile's MESSAGING-GATEWAY session (Discord group chat,
Telegram DM, ...) must receive the ``message_agent`` tool and the teammate
protocol section, exactly like its canonical ``Bot Chat`` — otherwise a bot
living on Discord cannot route work to teammates at all and asks the user to
relay messages by hand.

The gate stays closed for:
- profiles on installs that are not participating in Bot Mode,
- paths that are not real members of a participating install's profile roster,
- self-owned and machine/API sessions (CLI, cron, A2A, Home Assistant, ...).

Fleet contract (rsi/JJ audit): EVERY local profile in a Bot-Mode-participating
install is a Bot Mode roster member, even when only some profiles have custom
``ui_meta['hermes-bots']`` presentation metadata. Unmanaged installs and
non-human session sources never receive the tool.
"""

import json
from pathlib import Path

import pytest

from gateway.config import Platform
from tools import bot_mode_dm, bot_mode_probe


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


# The exact audited install roster (13 profiles incl. default). Only five have
# custom Bot Mode presentation metadata in the installed topology; Bot Mode's
# roster itself is every profile returned by profiles.list.
AUDITED_PROFILES = (
    "default", "buggy", "coder", "jade", "jade-ops", "product", "qa",
    "research", "reviewer", "rsi", "x", "yuki", "yuki-ops",
)
AUDITED_PROFILES_WITH_BOT_META = ("jade", "jade-ops", "rsi", "yuki", "yuki-ops")

# Explicit security classification: only adapters carrying human-authored
# conversations qualify. Machine/API/event sources fail closed even though they
# are valid Platform values.
BUILTIN_MESSAGING_SOURCES = (
    "telegram", "discord", "whatsapp", "whatsapp_cloud", "slack", "signal",
    "mattermost", "matrix", "email", "sms", "dingtalk", "feishu", "wecom",
    "wecom_callback", "weixin", "bluebubbles", "qqbot", "yuanbao",
)
BUILTIN_DENIED_SOURCES = (
    "local", "homeassistant", "api_server", "webhook", "msgraph_webhook",
    "relay",
)
BUNDLED_MESSAGING_SOURCES = (
    "buzz", "dingtalk", "discord", "email", "feishu", "google_chat", "irc",
    "line", "matrix", "mattermost", "photon", "simplex", "slack", "sms",
    "teams", "telegram", "wecom", "whatsapp",
)
BUNDLED_DENIED_SOURCES = ("a2a", "homeassistant", "ntfy", "raft")
SELF_OWNED_SOURCES = (
    "", "cli", "tui", "desktop", "cron", "kanban", "subagent", "test",
    "webhook", "api_server", "msgraph_webhook", "local", "acp", "webui",
)
CANONICAL_BOT_CHAT_SOURCES = ("", "cli", "tui", "desktop")
DENIED_BOT_CHAT_SOURCES = tuple(
    sorted(
        (
            set(SELF_OWNED_SOURCES)
            | set(BUILTIN_DENIED_SOURCES)
            | set(BUNDLED_DENIED_SOURCES)
            | {"not-a-platform"}
        )
        - set(CANONICAL_BOT_CHAT_SOURCES)
    )
)


def _make_hermes_home(
    tmp_path: Path,
    managed_profiles=AUDITED_PROFILES_WITH_BOT_META,
    installed_profiles=AUDITED_PROFILES,
) -> Path:
    """Mirror a Bot Mode install: full roster, sparse presentation metadata."""
    home = tmp_path / ".hermes"
    home.mkdir()
    managed = set(managed_profiles)
    installed = set(installed_profiles)
    if "default" in managed:
        (home / "profile.yaml").write_text(
            "ui_meta:\n  hermes-bots:\n    shape: local\n", encoding="utf-8"
        )
    for name in sorted(installed - {"default"}):
        d = home / "profiles" / name
        d.mkdir(parents=True, exist_ok=True)
        metadata = (
            "\nui_meta:\n  hermes-bots:\n    shape: cloud\n"
            if name in managed
            else "\n"
        )
        (d / "profile.yaml").write_text(
            "description: teammate for tests\n" + metadata,
            encoding="utf-8",
        )
    return home


class _FakeDB:
    def __init__(self, home: Path, title: str):
        self.db_path = str(home / "state.db")
        self._title = title

    def get_session_title(self, _sid):
        return self._title


class _FakeAgent:
    def __init__(self, home: Path, *, title="", platform="", session_id="sess-1"):
        self._session_db = _FakeDB(home, title)
        self.session_id = session_id
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self.platform = platform
        self.tools: list = []
        self.valid_tool_names: set = set()
        # attributes the real system-prompt build reads
        self.model = "test-model"
        self.provider = "test"
        self.load_soul_identity = False
        self.skip_memory = True
        self.skip_context_files = True
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_store = None
        self._memory_manager = None
        self.session_start = None
        self.pass_session_id = False


# ── shared session-state helper ──────────────────────────────────────────────


def test_bot_managed_gateway_session_is_routed(tmp_path):
    """The headline repro: managed Yuki profile, Discord group session."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(
        yuki_home,
        title="Group: 1543883995182407728",  # Discord group room title shape
        platform="discord",
    )
    state = bot_mode_probe.bot_mode_session_state(agent)
    assert state["managed"] is True
    assert state["session_kind"] == "gateway"


@pytest.mark.parametrize("platform", CANONICAL_BOT_CHAT_SOURCES)
def test_canonical_bot_chat_kind_unchanged(tmp_path, platform):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(yuki_home, title="Bot Chat", platform=platform)
    state = bot_mode_probe.bot_mode_session_state(agent)
    assert state["managed"] is True
    assert state["session_kind"] == "bot_chat"


@pytest.mark.parametrize("platform", DENIED_BOT_CHAT_SOURCES)
def test_exact_bot_chat_title_does_not_bypass_denied_source(tmp_path, platform):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(yuki_home, title="Bot Chat", platform=platform)

    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False


def test_installed_profile_without_own_bot_metadata_is_roster_managed(tmp_path):
    """Installed topology: coder is a Bot Mode bot without presentation ui_meta."""
    home = _make_hermes_home(tmp_path)
    coder_home = home / "profiles" / "coder"
    assert "hermes-bots" not in (coder_home / "profile.yaml").read_text()

    gateway = _FakeAgent(coder_home, title="Group: 1", platform="discord")
    canonical = _FakeAgent(
        coder_home, title="Bot Chat", platform="cli", session_id="coder-bot-chat"
    )
    assert bot_mode_probe.bot_mode_session_state(gateway)["session_kind"] == "gateway"
    assert bot_mode_probe.bot_mode_session_state(canonical)["session_kind"] == "bot_chat"


def test_roster_profile_does_not_require_presentation_metadata_file(tmp_path):
    """profiles.list includes valid profile directories without profile.yaml."""
    home = _make_hermes_home(
        tmp_path, managed_profiles=("yuki",), installed_profiles=("yuki",)
    )
    bare = home / "profiles" / "bare"
    bare.mkdir()
    agent = _FakeAgent(bare, title="Group: 1", platform="discord", session_id="bare")
    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] == "gateway"


def test_deleted_and_invalid_profile_directories_are_not_roster_members(tmp_path):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    deleted = home / "profiles" / "coder"
    tombstone = home / "profiles" / ".deleted" / "coder"
    tombstone.parent.mkdir(exist_ok=True)
    tombstone.write_text("deleted\n")
    invalid = home / "profiles" / "INVALID!"
    invalid.mkdir()

    for profile_home, sid in ((deleted, "deleted"), (invalid, "invalid")):
        agent = _FakeAgent(
            profile_home, title="Group: 1", platform="discord", session_id=sid
        )
        assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] is None


def test_symlinked_profile_cannot_escape_install_containment(tmp_path):
    home = _make_hermes_home(
        tmp_path, managed_profiles=("yuki",), installed_profiles=("yuki",)
    )
    external = tmp_path / "external-managed-home"
    external.mkdir()
    (external / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    shape: external\n"
    )
    escaped = home / "profiles" / "escaped"
    escaped.symlink_to(external, target_is_directory=True)

    agent = _FakeAgent(
        escaped, title="Group: 1", platform="discord", session_id="escaped"
    )
    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] is None


def test_multiplex_context_override_beats_shared_launch_db(tmp_path):
    """The routed profile ContextVar, not shared gateway state.db, owns auth."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    # Managed launch/default profile must not bless a routed path that is not
    # a real member of the install's profile roster.
    home = _make_hermes_home(
        tmp_path, managed_profiles=("default",), installed_profiles=("default",)
    )
    yuki_home = home / "profiles" / "INVALID!"
    yuki_home.mkdir(parents=True)
    shared_agent = _FakeAgent(home, title="Group: 1", platform="discord")
    token = set_hermes_home_override(yuki_home)
    try:
        assert bot_mode_probe._agent_home(shared_agent) == str(yuki_home)
        assert bot_mode_dm._agent_home(shared_agent) == str(yuki_home)
        assert bot_mode_probe.bot_mode_session_state(shared_agent)["session_kind"] is None
    finally:
        reset_hermes_home_override(token)


def test_multiplex_managed_routed_profile_ignores_unmanaged_launch_home(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    # Real topology: this agent points at the launch/default DB.
    shared_agent = _FakeAgent(home, title="Group: 1", platform="discord")
    token = set_hermes_home_override(yuki_home)
    try:
        assert bot_mode_probe.bot_mode_session_state(shared_agent)["session_kind"] == "gateway"
        assert bot_mode_dm.ensure_message_agent_tool(shared_agent) is True
    finally:
        reset_hermes_home_override(token)


def test_multiplex_cache_isolated_by_routed_profile_home(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    unlisted_home = home / "profiles" / "INVALID!"
    unlisted_home.mkdir(parents=True)

    managed = _FakeAgent(home, title="Group: 1", platform="discord")
    token = set_hermes_home_override(yuki_home)
    try:
        assert bot_mode_probe.bot_mode_session_state(managed)["session_kind"] == "gateway"
    finally:
        reset_hermes_home_override(token)

    # Same launch DB and persisted session ID, different unlisted routed home.
    unlisted = _FakeAgent(home, title="Group: 1", platform="discord")
    token = set_hermes_home_override(unlisted_home)
    try:
        assert bot_mode_probe.bot_mode_session_state(unlisted)["session_kind"] is None
    finally:
        reset_hermes_home_override(token)


def test_unmanaged_profile_gateway_session_not_routed(tmp_path):
    """Unmanaged install + unmanaged profile: a Discord chat gets nothing."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "someuser").mkdir(parents=True)
    agent = _FakeAgent(home / "profiles" / "someuser", title="Group: 1", platform="discord")
    state = bot_mode_probe.bot_mode_session_state(agent)
    assert state["managed"] is False
    assert state["session_kind"] is None


def test_unlisted_profile_path_on_managed_install_not_routed(tmp_path):
    """A managed install must not bless a path absent from profiles.list."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    managed = _FakeAgent(
        home / "profiles" / "yuki", title="Group: 1", platform="discord"
    )
    assert bot_mode_probe.bot_mode_session_state(managed)["session_kind"] == "gateway"

    unlisted_home = home / "profiles" / "INVALID!"
    unlisted_home.mkdir(parents=True)
    agent = _FakeAgent(unlisted_home, title="Group: 1", platform="discord")
    state = bot_mode_probe.bot_mode_session_state(agent)
    assert state["managed"] is True
    assert state["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False


def test_unlisted_profile_path_canonical_bot_chat_not_routed(tmp_path):
    """Install-wide management cannot bless an unlisted path's Bot Chat."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    unlisted_home = home / "profiles" / "INVALID!"
    unlisted_home.mkdir(parents=True)
    agent = _FakeAgent(unlisted_home, title="Bot Chat", platform="cli")

    assert bot_mode_probe.bot_mode_session_state(agent) == {
        "managed": True,
        "session_kind": None,
    }
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []


def test_unmanaged_profile_bot_chat_title_not_routed(tmp_path):
    """A user-titled 'Bot Chat' on an unmanaged install stays gated."""
    home = tmp_path / ".hermes"
    home.mkdir()
    agent = _FakeAgent(home, title="Bot Chat")
    state = bot_mode_probe.bot_mode_session_state(agent)
    assert state["managed"] is False
    assert state["session_kind"] is None


@pytest.mark.parametrize("platform", SELF_OWNED_SOURCES)
def test_self_owned_sessions_never_route_on_managed_install(tmp_path, platform):
    """CLI/TUI/desktop/cron/subagent/... sessions never route — even when a
    plain-titled session would otherwise look like a bot chat."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(yuki_home, title="My session", platform=platform)
    state = bot_mode_probe.bot_mode_session_state(agent)
    assert state["session_kind"] is None


@pytest.mark.parametrize("platform", BUILTIN_MESSAGING_SOURCES)
def test_explicit_builtin_messaging_platforms_route(tmp_path, platform):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(yuki_home, title="chat with JJ", platform=platform)
    assert bot_mode_probe.is_messaging_gateway_session(agent) is True
    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] == "gateway"


@pytest.mark.parametrize("platform", BUILTIN_DENIED_SOURCES)
def test_builtin_machine_and_api_platforms_are_denied(tmp_path, platform):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    agent = _FakeAgent(
        home / "profiles" / "yuki", title="external task", platform=platform
    )
    assert bot_mode_probe.is_messaging_gateway_session(agent) is False
    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] is None


def test_session_gate_is_frozen_after_first_resolution(tmp_path):
    """Disk changes mid-session cannot add/remove a tool schema and bust the
    provider's cached prefix; a new agent/session picks up the new state."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(yuki_home, title="Group: 1", platform="discord")
    first = bot_mode_probe.bot_mode_session_state(agent)
    assert first["session_kind"] == "gateway"

    # Simulate Bot Mode metadata being removed while this session lives.
    (yuki_home / "profile.yaml").unlink()
    assert bot_mode_probe.bot_mode_session_state(agent) == first
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True

    # A new session sees the current disk state and remains gated.
    new_agent = _FakeAgent(yuki_home, title="Group: 1", platform="discord")
    new_agent.session_id = "s2"
    assert bot_mode_probe.bot_mode_session_state(new_agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(new_agent) is False


def test_session_gate_survives_agent_recreation(tmp_path):
    """A recreated agent for the same persisted session keeps its first gate."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    first = _FakeAgent(yuki_home, title="Group: 1", platform="discord")
    assert bot_mode_probe.bot_mode_session_state(first)["session_kind"] == "gateway"

    (yuki_home / "profile.yaml").unlink()
    recreated = _FakeAgent(yuki_home, title="Group: 1", platform="discord")
    assert recreated.session_id == first.session_id
    assert bot_mode_probe.bot_mode_session_state(recreated)["session_kind"] == "gateway"
    assert bot_mode_dm.ensure_message_agent_tool(recreated) is True


def test_default_profile_gateway_session_routes(tmp_path):
    home = _make_hermes_home(tmp_path, managed_profiles=("default", "yuki"))
    agent = _FakeAgent(home, title="DM with JJ", platform="discord")
    state = bot_mode_probe.bot_mode_session_state(agent)
    assert state["managed"] is True
    assert state["session_kind"] == "gateway"


# ── tool injection through the new gate ──────────────────────────────────────


def test_discord_managed_session_receives_message_agent(tmp_path):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(yuki_home, title="Group: 139272514617475072", platform="discord")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    assert [t["function"]["name"] for t in agent.tools] == [bot_mode_dm.MESSAGE_AGENT_TOOL_NAME]
    assert bot_mode_dm.MESSAGE_AGENT_TOOL_NAME in agent.valid_tool_names

    # idempotent + byte-stable across turns (prompt-cache invariant)
    schema = json.dumps(agent.tools[0], sort_keys=True)
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    assert len(agent.tools) == 1
    assert json.dumps(agent.tools[0], sort_keys=True) == schema


@pytest.mark.parametrize("platform", SELF_OWNED_SOURCES)
def test_self_owned_plain_sessions_receive_no_tool(tmp_path, platform):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    agent = _FakeAgent(yuki_home, title="My session", platform=platform)
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_unmanaged_discord_session_receives_no_tool(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "someuser").mkdir(parents=True)
    agent = _FakeAgent(home / "profiles" / "someuser", title="Group: 1", platform="discord")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []


def test_bot_chat_injection_still_works(tmp_path):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    agent = _FakeAgent(home / "profiles" / "yuki", title="Bot Chat")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True


@pytest.mark.parametrize("platform", BUNDLED_MESSAGING_SOURCES)
def test_bundled_human_messaging_platform_routes(tmp_path, platform):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    agent = _FakeAgent(
        home / "profiles" / "yuki", title="chat with JJ", platform=platform
    )
    assert bot_mode_probe.is_messaging_gateway_session(agent) is True
    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] == "gateway"
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True


@pytest.mark.parametrize("platform", BUNDLED_DENIED_SOURCES)
def test_bundled_machine_or_agent_platform_is_denied(tmp_path, platform):
    """A2A/automation tasks must not gain Bot Mode teammate capabilities."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    agent = _FakeAgent(
        home / "profiles" / "yuki", title="external task", platform=platform
    )
    assert Platform(platform).value == platform  # valid bundled adapter
    assert bot_mode_probe.is_messaging_gateway_session(agent) is False
    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False


def test_every_bundled_adapter_has_explicit_security_classification():
    """A new bundled adapter fails closed until this boundary is reviewed."""
    classified = set(BUNDLED_MESSAGING_SOURCES) | set(BUNDLED_DENIED_SOURCES)
    assert Platform._scan_bundled_plugin_platforms() == classified


def test_arbitrary_source_fails_closed(tmp_path):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    agent = _FakeAgent(
        home / "profiles" / "yuki", title="External chat", platform="not-a-platform"
    )
    assert bot_mode_probe.is_messaging_gateway_session(agent) is False
    assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False


def test_config_toggle_disables_gateway_injection(tmp_path):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    agent = _FakeAgent(home / "profiles" / "yuki", title="Group: 1", platform="discord")
    agent._bot_mode_protocol = False
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []


# ── defense-in-depth dispatch gate ───────────────────────────────────────────


def test_tool_delivers_from_managed_discord_session(tmp_path, monkeypatch):
    """The dispatch gate admits the same session shape the injector does,
    resolves coder from the AGENT'S home, and ignores the shared gateway
    process's ambient default home."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki", "coder"))
    yuki_home = home / "profiles" / "yuki"
    ambient_default = tmp_path / "ambient-default"
    ambient_default.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(ambient_default))
    agent = _FakeAgent(yuki_home, title="Group: 1", platform="discord")

    captured = {}

    def fake_spawn(command, label, *, dm_file=None, task_id=None, agent=None):
        captured["label"] = label
        return json.dumps({"status": "sent", "to": label, "session_id": "proc_x"})

    monkeypatch.setattr("tools.bot_mode_dm._spawn_delivery", fake_spawn)
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="coder", message="hi", agent=agent)
    )
    assert result["status"] == "sent"
    assert captured["label"] == "@coder"


def test_tool_refuses_from_unmanaged_discord_session(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "someuser").mkdir(parents=True)
    agent = _FakeAgent(home / "profiles" / "someuser", title="Group: 1", platform="discord")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result


@pytest.mark.parametrize("platform", ("cli", "cron", "subagent", "tui"))
def test_tool_refuses_from_self_owned_plain_sessions(tmp_path, platform):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    agent = _FakeAgent(home / "profiles" / "yuki", title="My session", platform=platform)
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result


# ── prompt protocol section ──────────────────────────────────────────────────


def test_protocol_section_wording_covers_gateway_chats(tmp_path):
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki",))
    yuki_home = home / "profiles" / "yuki"
    section = bot_mode_probe.get_bot_mode_protocol_section(yuki_home, force_refresh=True)
    assert "messaging chat (Discord, Telegram, Slack, ...)" in section
    assert section.startswith("## Messaging other agents")


# ── fleet-wide audited matrix (rsi/JJ scope) ─────────────────────────────────


@pytest.mark.parametrize("profile", AUDITED_PROFILES)
def test_fleet_gateway_matrix_every_roster_profile(tmp_path, profile):
    """Exact installed topology: all 13 roster profiles route, while only the
    five customized profiles carry ``hermes-bots`` presentation metadata."""
    home = _make_hermes_home(tmp_path)
    profile_home = home if profile == "default" else home / "profiles" / profile
    if profile != "default":
        raw = (profile_home / "profile.yaml").read_text()
        assert ("hermes-bots" in raw) is (profile in AUDITED_PROFILES_WITH_BOT_META)

    # gateway session: routed
    gw = _FakeAgent(profile_home, title="Group: 1", platform="discord")
    assert bot_mode_dm.ensure_message_agent_tool(gw) is True
    assert [t["function"]["name"] for t in gw.tools] == [bot_mode_dm.MESSAGE_AGENT_TOOL_NAME]

    # canonical Bot Chat: still routed
    bc = _FakeAgent(profile_home, title="Bot Chat")
    assert bot_mode_dm.ensure_message_agent_tool(bc) is True

    # unmanaged sibling profile: not routed
    home2 = tmp_path / f".hermes-unmanaged-{profile}"
    home2.mkdir()
    if profile == "default":
        unmanaged_home = home2
    else:
        unmanaged_home = home2 / "profiles" / profile
        unmanaged_home.mkdir(parents=True)
    unmanaged = _FakeAgent(unmanaged_home, title="Group: 1", platform="discord")
    assert bot_mode_dm.ensure_message_agent_tool(unmanaged) is False
    assert unmanaged.tools == []


# ── end-to-end: real agent plumbing through system prompt build ─────────────


def test_e2e_system_prompt_carries_protocol_for_discord_session(tmp_path, monkeypatch):
    """End-to-end with the real SystemPromptBuilder against a temp HERMES_HOME:
    a managed Yuki Discord session's prompt carries the protocol + epoch;
    an unmanaged profile's Discord session does not; a CLI session does not;
    and repeated builds are byte-stable (tool list + prompt cache safe)."""
    home = _make_hermes_home(tmp_path, managed_profiles=("yuki", "researcher"))
    yuki_home = home / "profiles" / "yuki"

    from agent.system_prompt import build_system_prompt

    # The real failure topology: a shared gateway process is rooted at the
    # default home while the routed agent/session DB belongs to Yuki.
    monkeypatch.setenv("HERMES_HOME", str(home))

    def build(platform, title):
        # Mirror the real turn sequence: system prompt builds once at
        # session start (turn_context.py), then ensure_message_agent_tool
        # runs at each turn start.
        agent = _FakeAgent(
            yuki_home,
            title=title,
            platform=platform,
            session_id=f"{platform}:{title}",
        )
        prompt = build_system_prompt(agent)
        assert bot_mode_dm.ensure_message_agent_tool(agent) is True
        tools = list(agent.tools)
        return prompt, tools

    prompt, tools = build("discord", "Group: 139272514617475072")
    assert "## Messaging other agents" in prompt
    assert "Capability epoch: " in prompt
    assert [t["function"]["name"] for t in tools] == [bot_mode_dm.MESSAGE_AGENT_TOOL_NAME]

    # byte-stable rebuild (same turn shape) — cache-safe
    prompt2, tools2 = build("discord", "Group: 139272514617475072")
    assert prompt2 == prompt
    assert tools2 == tools

    # canonical Bot Chat still works end to end
    bc_prompt, bc_tools = build("discord", "Bot Chat")
    assert "## Messaging other agents" in bc_prompt
    assert [t["function"]["name"] for t in bc_tools] == [bot_mode_dm.MESSAGE_AGENT_TOOL_NAME]

    # unmanaged profile's Discord session: nothing
    home3 = tmp_path / ".hermes-unmanaged-e2e"
    home3.mkdir()
    (home3 / "profiles" / "yuki").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home3 / "profiles" / "yuki"))
    stranger = _FakeAgent(home3 / "profiles" / "yuki", title="Group: 1", platform="discord")
    prompt3 = build_system_prompt(stranger)
    assert bot_mode_dm.ensure_message_agent_tool(stranger) is False
    assert "## Messaging other agents" not in prompt3
    assert stranger.tools == []

    # self-owned CLI session on the MANAGED install: nothing (ambient gateway
    # home still differs from the routed profile home).
    monkeypatch.setenv("HERMES_HOME", str(home))
    cli_agent = _FakeAgent(yuki_home, title="My session", platform="cli")
    cli_prompt = build_system_prompt(cli_agent)
    assert bot_mode_dm.ensure_message_agent_tool(cli_agent) is False
    assert "## Messaging other agents" not in cli_prompt
    assert cli_agent.tools == []
