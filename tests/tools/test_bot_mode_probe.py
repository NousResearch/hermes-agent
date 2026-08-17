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
    assert "`researcher`" in section
    assert "hermes profile list" in section


def test_emits_for_named_profile_with_own_handle(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    profile_dir = _make_bot_profile(home, "coder", managed=True)

    section = bot_mode_probe.get_bot_mode_protocol_section(profile_dir)
    assert "@coder" in section
    # teammate list excludes self, includes default
    assert "`default`" in section
    assert "`coder`" not in section.split("Teammates at session start:")[1]


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


def test_never_raises_on_garbage(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    profiles = home / "profiles" / "bad"
    profiles.mkdir(parents=True)
    (profiles / "profile.yaml").write_text("ui_meta: [unclosed", encoding="utf-8")
    assert isinstance(bot_mode_probe.get_bot_mode_protocol_section(home), str)

    monkeypatch.setattr(bot_mode_probe, "_roster", lambda root: (_ for _ in ()).throw(OSError("boom")))
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_probe.get_bot_mode_protocol_section(home) == ""


# ── capability epoch ─────────────────────────────────────────────────────────


def test_fingerprint_stable_when_nothing_changes(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    assert bot_mode_probe.capability_fingerprint(home) == bot_mode_probe.capability_fingerprint(home)


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
    (home / "config.yaml").write_text("tools:\n  enabled_toolsets: [web]\n", encoding="utf-8")
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

    # prompts without a stamp (every non-Bot-Chat session) are never stale
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
    upgraded = legacy + "\n\n" + bot_mode_probe.get_bot_mode_protocol_section(home) + "\n\n" + bot_mode_probe.epoch_line(home)
    assert not bot_mode_probe.stored_bot_chat_prompt_needs_upgrade(upgraded, home)

    # SOUL already carries the legacy plugin-side append → probe silent →
    # no upgrade (rebuilding would loop: the new prompt would be unstamped too)
    bot_mode_probe._reset_cache_for_tests()
    (home / "SOUL.md").write_text("# Me\n\n## Messaging other agents\nlegacy\n", encoding="utf-8")
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
    assert "hermes peer dm" in section
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
