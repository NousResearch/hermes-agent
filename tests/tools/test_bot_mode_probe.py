"""Tests for tools/bot_mode_probe.py — the Bot Mode teammate-protocol section."""

import textwrap

import pytest

from tools import bot_mode_probe


@pytest.fixture(autouse=True)
def _fresh_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _make_bot_profile(root, name, *, managed=True, soul=None):
    d = root / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    if managed:
        (d / "profile.yaml").write_text(
            textwrap.dedent(
                """\
                ui_meta:
                  hermes-bots:
                    shape: cloud
                    color: '#8b5cf6'
                """
            ),
            encoding="utf-8",
        )
    if soul is not None:
        (d / "SOUL.md").write_text(soul, encoding="utf-8")
    return d


def test_silent_when_no_profile_is_bot_managed(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)
    assert bot_mode_probe.get_bot_mode_protocol_section(home) == ""


def test_emits_for_default_when_any_profile_is_managed(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert section.startswith("## Messaging other agents")
    # default's callable alias is @hermes, never @default
    assert "@hermes" in section
    assert "@default" not in section
    assert "@researcher" in section
    assert "message_agent" in section


def test_emits_for_named_profile_with_own_handle(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    profile_dir = _make_bot_profile(home, "coder", managed=True)

    section = bot_mode_probe.get_bot_mode_protocol_section(profile_dir)
    assert "@coder" in section
    # teammate roster excludes self, includes default (as @hermes)
    roster_block = section.split("Your teammates")[1]
    assert "`@hermes`" in roster_block
    assert "`@coder`" not in roster_block


def test_user_surface_emits_bot_mode_invocation_guidance(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)

    section = bot_mode_probe.get_bot_mode_user_protocol_section(home)

    assert section.startswith("## Bot Mode: messaging other agents")
    assert "This session can use Bot Mode" in section
    assert "ask <name>" in section
    assert "message_agent directly" in section
    assert "do not ask the user to retype special syntax" in section
    assert "`@researcher`" in section


def test_user_surface_omits_disabled_bot_mode_teammates(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    enabled = _make_bot_profile(home, "writer", managed=False)
    disabled = _make_bot_profile(home, "reviewer", managed=False)
    (enabled / "profile.yaml").write_text(
        "bot:\n  enabled: true\n",
        encoding="utf-8",
    )
    (disabled / "profile.yaml").write_text(
        "bot:\n  enabled: false\n",
        encoding="utf-8",
    )

    section = bot_mode_probe.get_bot_mode_user_protocol_section(home)

    assert "`@writer`" in section
    assert "`@reviewer`" not in section


def test_roster_lines_carry_roles(tmp_path):
    """Bots must know WHO to message: the roster carries title/description."""
    import textwrap as _tw

    home = tmp_path / ".hermes"
    home.mkdir()
    d = home / "profiles" / "researcher"
    d.mkdir(parents=True)
    (d / "profile.yaml").write_text(
        _tw.dedent(
            """\
            description: Deep research and literature review
            ui_meta:
              hermes-bots:
                title: Research Buddy
            """
        ),
        encoding="utf-8",
    )

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "`@researcher`" in section
    assert "Research Buddy" in section
    assert "Deep research and literature review" in section


def test_config_enabled_emits_without_desktop_ui_meta(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)
    (home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            agent:
              bot_mode:
                enabled: true
            """
        ),
        encoding="utf-8",
    )

    assert bot_mode_probe.is_bot_mode_managed(home) is True
    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert section.startswith("## Messaging other agents")
    assert "`@researcher`" in section


def test_config_roster_limits_local_teammates(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)
    _make_bot_profile(home, "coder", managed=False)
    (home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            agent:
              bot_mode:
                enabled: true
                roster:
                  - from: default
                    to: [coder]
            """
        ),
        encoding="utf-8",
    )

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "`@coder`" in section
    assert "`@researcher`" not in section
    assert bot_mode_probe.allowed_local_profile_names(home) == ["coder"]


@pytest.mark.parametrize("source", ["default", "caller"])
@pytest.mark.parametrize("root_config", [False, True])
def test_live_policy_uses_explicit_root_and_managed_env_refs(tmp_path, monkeypatch, source, root_config):
    home = tmp_path / "install"
    home.mkdir()
    for name in ("caller", "coder", "researcher"):
        _make_bot_profile(home, name, managed=False)
    if root_config:
        (home / "config.yaml").write_text(
            f"agent:\n  bot_mode:\n    enabled: true\n    roster:\n"
            f"      - from: {source}\n        to: [researcher]\n", encoding="utf-8",
        )
    managed = tmp_path / "managed"
    managed.mkdir()
    policy_path = managed / "config.yaml"
    policy = (
        "agent:\n  bot_mode:\n    enabled: true\n    roster:\n"
        "      - from: ${TEST_BOT_FROM}\n        to: [\"${TEST_BOT_TARGET}\"]\n"
    )
    policy_path.write_text(policy, encoding="utf-8")
    ambient = tmp_path / "unrelated-home"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("TEST_BOT_FROM", source)
    monkeypatch.setenv("TEST_BOT_TARGET", "coder")
    caller = home if source == "default" else home / "profiles" / source
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    assert bot_mode_probe.is_bot_mode_managed(caller) is True
    assert bot_mode_probe.allowed_local_profile_names(caller) == ["coder"]
    monkeypatch.setenv("TEST_BOT_TARGET", "researcher")
    assert bot_mode_probe.allowed_local_profile_names(caller) == ["researcher"]
    assert not ambient.exists()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before

    policy_path.write_text("agent:\n  bot_mode:\n    enabled: false\n", encoding="utf-8")
    assert bot_mode_probe.allowed_local_profile_names(caller) == []


@pytest.mark.parametrize("scope", ["root", "managed"])
@pytest.mark.parametrize("replacement", [
    "", "null\n", "# interrupted write\n", "agent: [broken\n", "[]\n",
    "agent: null\n", "agent:\n  bot_mode: null\n",
    'agent:\n  bot_mode:\n    enabled: "true"\n', "unreadable", "dangling",
])
def test_live_policy_never_falls_back_after_invalid_edit(tmp_path, monkeypatch, scope, replacement):
    import builtins

    home = tmp_path / "install"
    home.mkdir()
    _make_bot_profile(home, "coder")
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    path = (home if scope == "root" else managed) / "config.yaml"
    path.write_text("agent:\n  bot_mode:\n    enabled: true\n", encoding="utf-8")
    assert bot_mode_probe.allowed_local_profile_names(home) == ["coder"]
    if replacement == "unreadable":
        original_open = builtins.open

        def checked_open(file, *args, **kwargs):
            if file == path:
                raise PermissionError("policy unreadable")
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", checked_open)
    elif replacement == "dangling":
        path.unlink()
        try:
            path.symlink_to(path.parent / "absent-policy")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
    else:
        path.write_text(replacement, encoding="utf-8")
    assert bot_mode_probe.allowed_local_profile_names(home) == []
    assert bot_mode_probe.is_bot_mode_managed(home) is False


def test_silent_when_soul_already_carries_protocol(tmp_path):
    """Legacy plugin-side append — never double the section."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "coder", managed=True)
    (home / "SOUL.md").write_text(
        "# Me\n\n## Messaging other agents\nold plugin text\n", encoding="utf-8"
    )
    assert bot_mode_probe.get_bot_mode_protocol_section(home) == ""


def test_deterministic_across_calls(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    first = bot_mode_probe.get_bot_mode_protocol_section(home)
    # Even if the filesystem changes, the cached result must be byte-stable
    # for the life of the process (prompt-cache invariant).
    _make_bot_profile(home, "newbot", managed=True)
    second = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert first == second


def test_user_surface_section_is_deterministic_across_calls(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)
    first = bot_mode_probe.get_bot_mode_user_protocol_section(home)

    _make_bot_profile(home, "writer", managed=False)
    second = bot_mode_probe.get_bot_mode_user_protocol_section(home)

    assert first == second


def test_legacy_soul_protocol_does_not_hide_user_surface_bot_mode(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)
    (home / "SOUL.md").write_text(
        "# Assistant\n\n## Messaging other agents\nlegacy shellout protocol\n",
        encoding="utf-8",
    )

    section = bot_mode_probe.get_bot_mode_user_protocol_section(home)

    assert section.startswith("## Bot Mode: messaging other agents")
    assert "message_agent directly" in section


def test_never_raises_on_garbage(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    profiles = home / "profiles" / "bad"
    profiles.mkdir(parents=True)
    (profiles / "profile.yaml").write_text("ui_meta: [unclosed", encoding="utf-8")
    assert isinstance(bot_mode_probe.get_bot_mode_protocol_section(home), str)

    monkeypatch.setattr(
        bot_mode_probe, "_roster", lambda root: (_ for _ in ()).throw(OSError("boom"))
    )
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_probe.get_bot_mode_protocol_section(home) == ""


# ── capability epoch ─────────────────────────────────────────────────────────


def test_fingerprint_stable_when_nothing_changes(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    assert bot_mode_probe.capability_fingerprint(
        home
    ) == bot_mode_probe.capability_fingerprint(home)


def test_fingerprint_changes_on_each_capability_axis(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    base = bot_mode_probe.capability_fingerprint(home)

    # new skill installed
    skill = home / "skills" / "web" / "scraping"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: scraping\n---\n", encoding="utf-8")
    after_skill = bot_mode_probe.capability_fingerprint(home)
    assert after_skill != base

    # toolset pin changed
    (home / "config.yaml").write_text(
        "tools:\n  enabled_toolsets: [web]\n", encoding="utf-8"
    )
    after_tools = bot_mode_probe.capability_fingerprint(home)
    assert after_tools != after_skill

    # MCP server added
    (home / "config.yaml").write_text(
        "tools:\n  enabled_toolsets: [web]\nmcp_servers:\n  github:\n    preset: github\n",
        encoding="utf-8",
    )
    after_mcp = bot_mode_probe.capability_fingerprint(home)
    assert after_mcp != after_tools

    # SOUL edited
    (home / "SOUL.md").write_text("# New identity\n", encoding="utf-8")
    after_soul = bot_mode_probe.capability_fingerprint(home)
    assert after_soul != after_mcp

    # teammate added to the roster
    _make_bot_profile(home, "coder", managed=True)
    assert bot_mode_probe.capability_fingerprint(home) != after_soul


def test_stored_prompt_staleness(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)

    stamped = "system stuff\n\n" + bot_mode_probe.epoch_line(home)
    # unchanged surface → not stale (cache preserved)
    assert not bot_mode_probe.stored_prompt_capability_stale(stamped, home)

    # capability change → stale exactly once
    skill = home / "skills" / "new-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: new-skill\n---\n", encoding="utf-8")
    assert bot_mode_probe.stored_prompt_capability_stale(stamped, home)
    restamped = "system stuff\n\n" + bot_mode_probe.epoch_line(home)
    assert not bot_mode_probe.stored_prompt_capability_stale(restamped, home)

    # Prompts without a Bot Mode stamp are never stale by this check.
    assert not bot_mode_probe.stored_prompt_capability_stale("ordinary prompt", home)
    assert not bot_mode_probe.stored_prompt_capability_stale("", home)


def test_legacy_bot_chat_upgrade(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)

    legacy = "old prompt with no protocol and no stamp"
    # legacy Bot Chat on a managed install → upgrade once
    assert bot_mode_probe.stored_bot_chat_prompt_needs_upgrade(legacy, home)

    # a rebuilt prompt (stamped) never re-fires
    upgraded = (
        legacy
        + "\n\n"
        + bot_mode_probe.get_bot_mode_protocol_section(home)
        + "\n\n"
        + bot_mode_probe.epoch_line(home)
    )
    assert not bot_mode_probe.stored_bot_chat_prompt_needs_upgrade(upgraded, home)

    # SOUL already carries the legacy plugin-side append → probe silent →
    # no upgrade (rebuilding would loop: the new prompt would be unstamped too)
    bot_mode_probe._reset_cache_for_tests()
    (home / "SOUL.md").write_text(
        "# Me\n\n## Messaging other agents\nlegacy\n", encoding="utf-8"
    )
    assert not bot_mode_probe.stored_bot_chat_prompt_needs_upgrade(legacy, home)

    # prompt whose SOUL section rode into it → protocol heading present → no upgrade
    assert not bot_mode_probe.stored_bot_chat_prompt_needs_upgrade(
        "prompt containing\n## Messaging other agents\nfrom SOUL", home
    )

    # unmanaged install → probe silent → never upgrades
    bot_mode_probe._reset_cache_for_tests()
    home2 = tmp_path / ".hermes2"
    home2.mkdir()
    assert not bot_mode_probe.stored_bot_chat_prompt_needs_upgrade(legacy, home2)


def test_existing_user_session_gets_one_time_bot_mode_upgrade(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)
    legacy = "ordinary stored system prompt"

    assert bot_mode_probe.stored_bot_mode_user_prompt_needs_upgrade(legacy, home)
    assert bot_mode_probe.stored_bot_mode_user_prompt_needs_upgrade(
        legacy + "\n\n## Messaging other agents\nlegacy shellout protocol",
        home,
    )

    upgraded = (
        legacy
        + "\n\n"
        + bot_mode_probe.get_bot_mode_user_protocol_section(home)
        + "\n\n"
        + bot_mode_probe.epoch_line(home)
    )
    assert not bot_mode_probe.stored_bot_mode_user_prompt_needs_upgrade(
        upgraded, home
    )


# ── peer gateways (cross-machine DMs) ────────────────────────────────────────


def test_peer_paragraph_absent_without_peers(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "hermes peer dm" not in section
    assert "OTHER machines" not in section


def test_peer_paragraph_lists_registered_peers(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    (home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            bot_peers:
              spark:
                url: http://spark.lan:8377
              homelab:
                url: http://homelab.lan:8377
            """
        ),
        encoding="utf-8",
    )

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "message_agent" in section
    assert '"<peer>/<agent-name>"' in section
    assert "`homelab`" in section and "`spark`" in section
    assert "hermes peer list" in section


def test_fingerprint_changes_when_a_peer_is_registered(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)

    before = bot_mode_probe.capability_fingerprint(home)
    (home / "config.yaml").write_text(
        "bot_peers:\n  spark:\n    url: http://spark.lan:8377\n",
        encoding="utf-8",
    )
    after = bot_mode_probe.capability_fingerprint(home)
    assert before != after


def test_user_surface_omits_unreadable_metadata_teammate(tmp_path):
    """A corrupt profile.yaml fails closed: unknown authority never widens
    the callable roster (#100758 review, blocker 2)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    enabled = _make_bot_profile(home, "writer", managed=False)
    (enabled / "profile.yaml").write_text("bot:\n  enabled: true\n", encoding="utf-8")
    corrupt = _make_bot_profile(home, "reviewer", managed=False)
    (corrupt / "profile.yaml").write_text("bot: [unclosed\n", encoding="utf-8")

    section = bot_mode_probe.get_bot_mode_user_protocol_section(home)

    assert "`@writer`" in section
    assert "`@reviewer`" not in section
    assert bot_mode_probe._is_bot_enabled(corrupt) is False


def test_user_surface_telegram_uses_dollar_sigil(tmp_path):
    """Telegram sessions must never be taught @-handles: Telegram resolves
    @word as a REAL username (possibly a stranger's), so the user-visible
    sigil there is the inert $ (#100758 live-test feedback)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)

    section = bot_mode_probe.get_bot_mode_user_protocol_section(
        home, platform="telegram"
    )

    assert section.startswith("## Bot Mode: messaging other agents")
    assert "`$researcher`" in section
    assert "`@researcher`" not in section
    assert "NEVER write @-handles" in section
    assert "Telegram resolves @word as a REAL username" in section
    assert "message_agent tool accepts the same $name form" in section


def test_user_surface_platform_variants_cached_separately(tmp_path):
    """One gateway process serves CLI and Telegram sessions at once; their
    prompt sections differ by sigil and must not share a cache slot."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=False)

    cli_section = bot_mode_probe.get_bot_mode_user_protocol_section(home)
    tg_section = bot_mode_probe.get_bot_mode_user_protocol_section(
        home, platform="telegram"
    )

    assert "`@researcher`" in cli_section
    assert "NEVER write @-handles" not in cli_section
    assert "`$researcher`" in tg_section
    # Repeat calls return their own cached variants unchanged.
    assert bot_mode_probe.get_bot_mode_user_protocol_section(home) == cli_section
    assert (
        bot_mode_probe.get_bot_mode_user_protocol_section(home, platform="Telegram")
        == tg_section
    )
