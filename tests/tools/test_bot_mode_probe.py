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


def _set_private(profile_dir, value="true"):
    """Mark an existing bot profile private, preserving its other ui_meta keys."""
    (profile_dir / "profile.yaml").write_text(
        textwrap.dedent(
            f"""\
            ui_meta:
              hermes-bots:
                shape: cloud
                private: {value}
            """
        ),
        encoding="utf-8",
    )


def test_private_agent_is_not_listed_in_the_roster(tmp_path):
    """A private agent leaves the mesh: teammates stop being told it exists."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher")
    lucky = _make_bot_profile(home, "lucky")
    _set_private(lucky)

    section = bot_mode_probe.get_bot_mode_protocol_section(home)

    assert "@researcher" in section
    assert "@lucky" not in section


def test_private_agent_still_gets_its_own_protocol_section(tmp_path):
    """Private is about what OTHERS see. The agent keeps working and keeps its own tools —
    hiding it must not silently disable Bot Mode for itself."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher")
    lucky = _make_bot_profile(home, "lucky")
    _set_private(lucky)

    section = bot_mode_probe.get_bot_mode_protocol_section(lucky)

    assert section.startswith("## Messaging other agents")
    assert "You are `@lucky`" in section
    assert "@researcher" in section


def test_an_all_private_install_does_not_look_unmanaged(tmp_path):
    """_roster also feeds _any_managed. Filtering there would switch Bot Mode off entirely
    for an install where every agent is private — including the human's own access."""
    home = tmp_path / ".hermes"
    home.mkdir()
    only = _make_bot_profile(home, "lucky")
    _set_private(only)

    assert bot_mode_probe._any_managed(home) is True
    assert bot_mode_probe.get_bot_mode_protocol_section(home).startswith("## Messaging other agents")


def test_force_private_overrides_a_public_agent(tmp_path):
    """The install-wide switch outranks each agent's own choice, never the other way round."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher")
    _make_bot_profile(home, "lucky")
    (home / "config.yaml").write_text("bots:\n  force_private: true\n", encoding="utf-8")

    section = bot_mode_probe.get_bot_mode_protocol_section(home)

    assert "@researcher" not in section
    assert "@lucky" not in section


def test_force_private_off_leaves_per_agent_choice_alone(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher")
    lucky = _make_bot_profile(home, "lucky")
    _set_private(lucky)
    (home / "config.yaml").write_text("bots:\n  force_private: false\n", encoding="utf-8")

    section = bot_mode_probe.get_bot_mode_protocol_section(home)

    assert "@researcher" in section
    assert "@lucky" not in section


@pytest.mark.parametrize("value", ["yes", "on", "1", "True"])
def test_private_accepts_hand_edited_yaml_truthies(tmp_path, value):
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher")
    lucky = _make_bot_profile(home, "lucky")
    _set_private(lucky, value)

    assert "@lucky" not in bot_mode_probe.get_bot_mode_protocol_section(home)


@pytest.mark.parametrize("value", ["false", "no", "0", "maybe", "''"])
def test_unrecognised_private_values_stay_public(tmp_path, value):
    """Fail OPEN: a typo must not silently remove an agent from the mesh."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _make_bot_profile(home, "researcher")
    lucky = _make_bot_profile(home, "lucky")
    _set_private(lucky, value)

    assert "@lucky" in bot_mode_probe.get_bot_mode_protocol_section(home)


def _set_circle(profile_dir, circle):
    (profile_dir / "profile.yaml").write_text(
        textwrap.dedent(
            f"""\
            ui_meta:
              hermes-bots:
                shape: cloud
                circle: {circle}
            """
        ),
        encoding="utf-8",
    )


def _lines(section):
    """The local roster bullets of a protocol section."""
    start = section.find("Your teammates")
    end = section.find("Teammates on OTHER")
    return section[start:end if end > 0 else None]


def test_same_circle_agents_see_each_other(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    work_a = _make_bot_profile(home, "reviewer")
    work_b = _make_bot_profile(home, "programmer")
    _set_circle(work_a, "work")
    _set_circle(work_b, "work")

    section = bot_mode_probe.get_bot_mode_protocol_section(work_a)

    assert "@programmer" in _lines(section)


def test_different_circles_do_not_see_each_other(tmp_path):
    """The whole point: work bots and hobby bots share a machine but not a mesh."""
    home = tmp_path / ".hermes"
    home.mkdir()
    work = _make_bot_profile(home, "reviewer")
    hobby = _make_bot_profile(home, "lucky")
    _set_circle(work, "work")
    _set_circle(hobby, "hobby")

    assert "@lucky" not in _lines(bot_mode_probe.get_bot_mode_protocol_section(work))
    assert "@reviewer" not in _lines(bot_mode_probe.get_bot_mode_protocol_section(hobby))


def test_unset_circle_is_the_shared_default_and_todays_behaviour(tmp_path):
    """Agents with no circle form the shared circle: they see each other exactly as before —
    and they do NOT see circled agents, nor are they seen by them."""
    home = tmp_path / ".hermes"
    home.mkdir()
    plain_a = _make_bot_profile(home, "alpha")
    _make_bot_profile(home, "beta")
    work = _make_bot_profile(home, "reviewer")
    _set_circle(work, "work")

    plain_view = _lines(bot_mode_probe.get_bot_mode_protocol_section(plain_a))
    assert "@beta" in plain_view
    assert "@reviewer" not in plain_view
    assert "@alpha" not in _lines(bot_mode_probe.get_bot_mode_protocol_section(work))


def test_private_outranks_circle(tmp_path):
    """A private agent is a circle of one, even inside a circle it named."""
    home = tmp_path / ".hermes"
    home.mkdir()
    a = _make_bot_profile(home, "reviewer")
    b = _make_bot_profile(home, "programmer")
    _set_circle(a, "work")
    (b / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    shape: cloud\n    circle: work\n    private: true\n", encoding="utf-8"
    )

    assert "@programmer" not in _lines(bot_mode_probe.get_bot_mode_protocol_section(a))


def test_force_private_outranks_circles(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    a = _make_bot_profile(home, "reviewer")
    b = _make_bot_profile(home, "programmer")
    _set_circle(a, "work")
    _set_circle(b, "work")
    (home / "config.yaml").write_text("bots:\n  force_private: true\n", encoding="utf-8")

    assert "@programmer" not in _lines(bot_mode_probe.get_bot_mode_protocol_section(a))


@pytest.mark.parametrize("value", ["''", "[]", "42", "true"])
def test_garbage_circle_fails_open_to_the_shared_circle(tmp_path, value):
    """A typo must never quietly cut an agent off: non-string / empty = the shared circle."""
    home = tmp_path / ".hermes"
    home.mkdir()
    a = _make_bot_profile(home, "alpha")
    b = _make_bot_profile(home, "beta")
    _set_circle(b, value)

    assert "@beta" in _lines(bot_mode_probe.get_bot_mode_protocol_section(a))


def test_circle_name_is_trimmed_and_capped(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    a = _make_bot_profile(home, "alpha")
    _set_circle(a, "'  work  '")
    assert bot_mode_probe._circle_of(a) == "work"
    _set_circle(a, "'" + "x" * 100 + "'")
    assert len(bot_mode_probe._circle_of(a)) == 64


def test_remote_roster_is_filtered_to_the_viewers_circle(tmp_path, monkeypatch):
    """Cross-machine rows carry `circle`; the viewer only learns about its own circle."""
    home = tmp_path / ".hermes"
    home.mkdir()
    me = _make_bot_profile(home, "reviewer")
    _set_circle(me, "work")
    rows = [
        {"profile": "programmer", "handle": "programmer", "connection_id": "mini",
         "connection_label": "mini", "title": "", "description": "", "circle": "work"},
        {"profile": "lucky", "handle": "lucky", "connection_id": "mini",
         "connection_label": "mini", "title": "", "description": "", "circle": "hobby"},
        {"profile": "plain", "handle": "plain", "connection_id": "mini",
         "connection_label": "mini", "title": "", "description": "", "circle": ""},
    ]
    monkeypatch.setattr(bot_mode_probe, "_remote_roster", lambda root: rows)

    section = bot_mode_probe.get_bot_mode_protocol_section(me)

    assert "@programmer" in section
    assert "@lucky" not in section
    assert "@plain" not in section


def test_circle_names_are_case_insensitive(tmp_path):
    """`Work` and `work` are one circle: a stray capital must never isolate an agent."""
    home = tmp_path / ".hermes"
    home.mkdir()
    a = _make_bot_profile(home, "reviewer")
    b = _make_bot_profile(home, "programmer")
    _set_circle(a, "Work")
    _set_circle(b, "work")

    assert bot_mode_probe._circle_of(a) == "work"
    assert "@programmer" in _lines(bot_mode_probe.get_bot_mode_protocol_section(a))
    assert "@reviewer" in _lines(bot_mode_probe.get_bot_mode_protocol_section(b))


def test_visible_roster_takes_no_default_viewer():
    """Cross-circle visibility must never be one forgotten argument away.

    Asserted on the signature, not on a raised error: a viewer defaulting to None still blows
    up somewhere downstream, which would make a behavioural test pass while the unfiltered
    call shape quietly came back.
    """
    import inspect

    viewer = inspect.signature(bot_mode_probe._visible_roster).parameters["viewer"]
    assert viewer.default is inspect.Parameter.empty
    assert viewer.kind is inspect.Parameter.KEYWORD_ONLY
