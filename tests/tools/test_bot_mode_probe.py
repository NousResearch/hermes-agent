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


def test_soul_legacy_protocol_no_longer_suppresses_live_section(tmp_path):
    """Plugin-era SOUL append is stripped at load time; the live roster is the only copy."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "coder", managed=True)
    (home / "SOUL.md").write_text(
        "# Me\n\n## Messaging other agents\nold plugin text\n", encoding="utf-8"
    )
    assert "`@coder`" in bot_mode_probe.get_bot_mode_protocol_section(home)
    assert bot_mode_probe.strip_legacy_protocol((home / "SOUL.md").read_text()) == "# Me\n"


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

    # SOUL-era prompt (frozen roster rode in from SOUL.md, no stamp) → upgrade once
    assert bot_mode_probe.stored_bot_chat_prompt_needs_upgrade(
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


# ── mesh membership: private agents stay out of every OTHER agent's roster ───


def _profile(root, name, *, private=None, description="secret mission", title="Shadow"):
    """A Bot-Mode profile; ``private=None`` leaves the flag out entirely."""
    d = root / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"description: {description}", "ui_meta:", "  hermes-bots:", f"    title: {title}"]
    if private is not None:
        lines.append(f"    private: {private}")
    (d / "profile.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_private_agent_leaves_the_prompt_roster(tmp_path):
    """`ui_meta['hermes-bots'].private: true` → no roster line, no role text."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    _profile(home, "prediction", private="true")

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "`@researcher`" in section  # control: the public peer is still listed
    assert "`@prediction`" not in section
    assert "Shadow" not in section and "secret mission" not in section


def test_private_flag_is_per_agent(tmp_path):
    """Only the flagged agent disappears; siblings keep their line."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _profile(home, "prediction", private="true")
    _profile(home, "reviewer", private="false", description="reads diffs", title="Reviewer")

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "`@reviewer`" in section and "Reviewer" in section
    assert "`@prediction`" not in section


def test_private_flag_parsing_fails_open(tmp_path):
    """true/yes/1 honoured (bool, YAML and quoted); a typo never isolates a teammate."""
    home = tmp_path / ".hermes"
    home.mkdir()
    for name, value in (("aa", "yes"), ("bb", "1"), ("cc", "'true'"), ("dd", "maybe"), ("ee", "'TRUE'")):
        _profile(home, name, private=value)

    assert bot_mode_probe._force_private(home) is False
    assert bot_mode_probe._is_private(home / "profiles" / "aa") is True
    assert bot_mode_probe._is_private(home / "profiles" / "bb") is True
    assert bot_mode_probe._is_private(home / "profiles" / "cc") is True
    assert bot_mode_probe._is_private(home / "profiles" / "ee") is True
    assert bot_mode_probe._is_private(home / "profiles" / "dd") is False

    visible = [n for n, _d in bot_mode_probe._visible_roster(home)]
    assert visible == ["default", "dd"]


def test_force_private_empties_every_roster_without_unmanaging(tmp_path):
    """Install-wide knob outranks each agent's own (unset) choice — and Bot Mode stays ON:
    an all-private install must not read as unmanaged nor lose the protocol section."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    _make_bot_profile(home, "coder", managed=True)
    (home / "config.yaml").write_text("bots:\n  force_private: true\n", encoding="utf-8")

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert section.startswith("## Messaging other agents")
    assert "- (no teammates yet)" in section
    assert "`@researcher`" not in section and "`@coder`" not in section
    assert bot_mode_probe.is_bot_mode_managed(home) is True
    # Not a typo: an unrecognised value leaves everyone public (fail open).
    (home / "config.yaml").write_text("bots:\n  force_private: 'nope'\n", encoding="utf-8")
    assert bot_mode_probe._force_private(home) is False


def test_private_agent_itself_still_runs(tmp_path):
    """The private agent keeps its own protocol section (it still sees public peers),
    the install still counts as managed, and nothing raises on the empty roster."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    private_dir = _profile(home, "prediction", private="true")

    section = bot_mode_probe.get_bot_mode_protocol_section(private_dir)
    assert "`@researcher`" in section
    # never in a roster block — not even its own (self is always excluded)
    assert "- `@prediction`" not in section
    assert bot_mode_probe.is_bot_mode_managed(private_dir) is True
    assert bot_mode_probe.capability_fingerprint(private_dir) != "unavailable"


def test_private_remote_row_never_reaches_the_prompt(tmp_path):
    """The cross-machine relay roster loses the private peer AND its description."""
    from tools.bot_relay import write_remote_roster

    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    write_remote_roster(home, [
        {"profile": "scout", "handle": "scout", "connection_id": "ssh-vps",
         "connection_label": "VPS", "title": "Scout", "description": "watches feeds"},
        {"profile": "prediction", "handle": "prediction", "connection_id": "ssh-vps",
         "connection_label": "VPS", "title": "Shadow", "description": "secret mission", "private": True},
    ])

    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "`@scout` — on VPS — Scout — watches feeds" in section
    assert "- `@prediction`" not in section
    assert "Shadow" not in section and "secret mission" not in section


def test_flipping_private_refreshes_the_epoch(tmp_path):
    """Visibility rides the capability epoch: an eternal Bot Chat prompt stops
    carrying a peer that just went private."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher", managed=True)
    d = _profile(home, "prediction", private="false")
    public = bot_mode_probe.capability_fingerprint(home)

    _profile(home, "prediction", private="true")  # same role text, now private
    assert bot_mode_probe.capability_fingerprint(home) != public
    assert bot_mode_probe.capability_fingerprint(home) != "unavailable"
    assert "`@prediction`" not in bot_mode_probe.get_bot_mode_protocol_section(home)
