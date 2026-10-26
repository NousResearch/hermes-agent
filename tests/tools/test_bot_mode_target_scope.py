"""Bot Mode ``message_agent`` scope: ``bot_mode.allowed_targets`` and ``bot_mode.accept_from``.

Bot Mode is install-wide: a Bot Chat agent can message any profile on its install and any
registered peer. Two per-profile keys narrow that. ``allowed_targets`` (sender side) limits
whom an agent may message and what its roster shows; ``accept_from`` (receiver side) limits
who may message a profile. Unset keeps today's behaviour; set (even empty) fails closed.

The tests drive the real entry points: ``message_agent_tool`` (dispatch) and the Bot Chat
prompt section and capability fingerprint (``bot_mode_probe``). A refused call must leave
NOTHING behind: no admission, no process spawn, no DM file.
"""

import json
import textwrap
from pathlib import Path

import pytest

from tools import bot_mode_dm, bot_mode_probe, bot_relay


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _install(tmp_path, *, profiles=("alpha", "beta"), peers=()):
    """A scratch HERMES_HOME root with fake Bot-Mode-managed profiles (and registered peers)."""
    root = tmp_path / ".hermes"
    root.mkdir()
    for name in profiles:
        d = root / "profiles" / name
        d.mkdir(parents=True)
        (d / "profile.yaml").write_text(
            textwrap.dedent(
                f"""\
                description: {name} teammate
                ui_meta:
                  hermes-bots:
                    shape: cloud
                """
            ),
            encoding="utf-8",
        )
    if peers:
        _config(root, "bot_peers:\n" + "".join(f"  {p}:\n    url: http://{p}.lan:8377\n" for p in peers))
    return root


def _config(home: Path, text: str) -> None:
    """Append to ``<home>/config.yaml`` (a profile dir or the root)."""
    path = home / "config.yaml"
    path.write_text((path.read_text(encoding="utf-8") if path.exists() else "") + text, encoding="utf-8")


class _FakeDB:
    def __init__(self, home: Path):
        self.db_path = str(home / "state.db")

    def get_session_title(self, _sid):
        return "Bot Chat"


class _FakeAgent:
    def __init__(self, home: Path):
        self._session_db = _FakeDB(home)
        self.session_id = "sess-1"
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self.tools: list = []
        self.valid_tool_names: set = set()


class _Delivery:
    """Everything that could deliver a message. A refused call must leave all of it empty."""

    def __init__(self, monkeypatch, tmp_path):
        import tools.terminal_tool as terminal_tool_module

        self.spawns: list = []
        self.admissions: list = []
        self.dm_dir = tmp_path / "dm"
        self.dm_dir.mkdir()
        monkeypatch.setattr(bot_mode_dm, "_dm_dir", lambda: self.dm_dir)
        monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")

        def admit(profile_home, dm_file, author):
            self.admissions.append(Path(profile_home).name)
            return {"status": "queued", "delivery_id": "d-1", "profile_home": str(profile_home)}

        def spawn(command, **kwargs):
            self.spawns.append(command)
            return json.dumps({"output": "Background process started", "session_id": "proc_test1234"})

        monkeypatch.setattr(bot_mode_dm, "_admit_live_dm", admit)
        monkeypatch.setattr(terminal_tool_module, "terminal_tool", spawn)

    def clear(self):
        self.spawns.clear()
        self.admissions.clear()

    def untouched(self) -> bool:
        return not self.spawns and not self.admissions and not any(self.dm_dir.iterdir())


def _send(root, sender, target, message="status?"):
    home = root / "profiles" / sender if sender else root
    return json.loads(bot_mode_dm.message_agent_tool(target=target, message=message, agent=_FakeAgent(home)))


def _queued(result) -> bool:
    return result.get("status") == "queued"


def _refused(result) -> bool:
    return result.get("reason") == "agent_blocked" and "status" not in result


# ── allowed_targets: unset keeps today's behaviour ───────────────────────────


def test_unset_allowed_targets_delivers_to_any_teammate(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path)
    assert _queued(_send(root, "alpha", "beta"))
    assert delivery.admissions == ["beta"]


@pytest.mark.parametrize("config", [None, "model: x\n", "bot_mode:\n  turn_wait_seconds: 5\n",
                                    "# allowed_targets is documented here\nmodel: x\n"])
def test_unset_forms_read_as_none(tmp_path, config):
    home = tmp_path / "p"
    home.mkdir()
    if config is not None:
        _config(home, config)
    assert bot_mode_probe.bot_mode_scope(home, "allowed_targets") is None
    assert bot_mode_probe.bot_mode_scope(home, "accept_from") is None
    assert bot_mode_probe.scope_allows(None, "anything")


def test_unset_roster_and_epoch_are_unchanged(tmp_path):
    """No key, no change: the roster lists everyone and the epoch hash matches a config-less home."""
    root = _install(tmp_path, peers=("north",))
    home = root / "profiles" / "alpha"
    before = bot_mode_probe.capability_fingerprint(home)
    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "`@beta`" in section and "`north`" in section and '"<peer>/<agent-name>"' in section

    _config(home, "bot_mode:\n  turn_wait_seconds: 5\n")
    assert bot_mode_probe.capability_fingerprint(home) == before


# ── allowed_targets: dispatch ────────────────────────────────────────────────


def test_allowed_target_passes_and_a_forbidden_one_delivers_nothing(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, profiles=("alpha", "beta", "gamma"))
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: [gamma]\n")

    assert _queued(_send(root, "alpha", "gamma"))
    assert delivery.admissions == ["gamma"]
    delivery.clear()
    for path in delivery.dm_dir.iterdir():
        path.unlink()

    refused = _send(root, "alpha", "beta")
    assert _refused(refused)
    assert "allowed_targets" in refused["error"]
    assert refused["teammates"] == ["gamma"]
    assert "beta" not in json.dumps(refused["teammates"]) and "beta" not in refused["peers"]
    assert delivery.untouched()


def test_a_forbidden_and_a_nonexistent_target_look_the_same(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path)
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: []\n")
    forbidden, missing = _send(root, "alpha", "beta"), _send(root, "alpha", "no-such-bot")
    assert _refused(forbidden) and _refused(missing)
    assert forbidden["teammates"] == [] == missing["teammates"]
    assert forbidden["error"].replace("'beta'", "X") == missing["error"].replace("'no-such-bot'", "X")
    assert delivery.untouched()


def test_a_friendly_name_cannot_reach_a_denied_folder_id(tmp_path, monkeypatch):
    """Matching is on the ROUTED profile id, so a display name cannot bypass the list."""
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, profiles=("alpha", "beta", "gamma"))
    (root / "profiles" / "beta" / "profile.yaml").write_text(
        "display_name: Treasurer\nui_meta:\n  hermes-bots:\n    shape: cloud\n", encoding="utf-8")
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: [gamma]\n")

    assert _refused(_send(root, "alpha", "Treasurer"))
    assert _refused(_send(root, "alpha", "@beta"))
    assert delivery.untouched()


def test_entries_ignore_case_and_a_leading_at(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path)
    _config(root / "profiles" / "alpha", 'bot_mode:\n  allowed_targets: ["@Beta"]\n')
    assert _queued(_send(root, "alpha", "BETA"))
    assert delivery.admissions == ["beta"]


def test_default_profile_is_listed_by_its_hermes_handle(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path)
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: [hermes]\n")
    assert _queued(_send(root, "alpha", "hermes"))
    assert _refused(_send(root, "alpha", "beta"))
    assert len(delivery.spawns) == 1


def test_peer_targets_match_exactly(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, peers=("north", "south"))
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: [south, north/ops]\n")

    assert _queued(_send(root, "alpha", "south"))
    assert _queued(_send(root, "alpha", "north/ops"))
    assert len(delivery.spawns) == 2
    delivery.clear()
    for path in delivery.dm_dir.iterdir():
        path.unlink()

    for target in ("north/finance", "north", "south/finance"):
        refused = _send(root, "alpha", target)
        assert _refused(refused), target
        assert refused["peers"] == ["north", "south"]  # both appear in a listed entry
    assert delivery.untouched()


def test_peers_without_a_listed_entry_are_hidden_from_the_refusal(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, peers=("north", "south"))
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: [south]\n")
    refused = _send(root, "alpha", "north/finance")
    assert _refused(refused) and refused["peers"] == ["south"]
    assert delivery.untouched()


def test_empty_list_refuses_everything(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, peers=("north",))
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: []\n")
    for target in ("beta", "hermes", "north", "north/coder", "alpha"):
        assert _refused(_send(root, "alpha", target)), target
    assert delivery.untouched()


@pytest.mark.parametrize("text", [
    "bot_mode:\n  allowed_targets: [unterminated\n",  # broken YAML
    "bot_mode:\n  allowed_targets:\n",  # present but null
    "bot_mode:\n  allowed_targets: 12\n",  # not a list
    "bot_mode: oops\nallowed_targets: x\n",  # bot_mode is not a mapping
])
def test_a_present_but_malformed_key_fails_closed(tmp_path, monkeypatch, text):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path)
    _config(root / "profiles" / "alpha", text)
    assert bot_mode_probe.bot_mode_scope(root / "profiles" / "alpha", "allowed_targets") == []
    assert _refused(_send(root, "alpha", "beta"))
    assert delivery.untouched()


def test_a_scalar_entry_is_a_one_item_list(tmp_path):
    home = tmp_path / "p"
    home.mkdir()
    _config(home, "bot_mode:\n  allowed_targets: Beta\n")
    assert bot_mode_probe.bot_mode_scope(home, "allowed_targets") == ["beta"]


def test_the_scope_is_read_from_the_senders_own_profile(tmp_path, monkeypatch):
    """One profile's list never becomes another profile's policy."""
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, profiles=("alpha", "beta", "gamma"))
    _config(root / "profiles" / "alpha", "bot_mode:\n  allowed_targets: []\n")
    _config(root, "bot_mode:\n  allowed_targets: []\n")
    assert _queued(_send(root, "gamma", "beta"))
    assert delivery.admissions == ["beta"]


def test_remote_machine_targets_need_a_listed_form(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path)
    home = root / "profiles" / "alpha"
    bot_relay.write_remote_roster(home, [
        {"profile": "ops", "handle": "ops", "connection_id": "cloud-1", "connection_label": "Cloud"},
    ])
    _config(home, "bot_mode:\n  allowed_targets: [beta]\n")
    refused = _send(root, "alpha", "ops")
    assert _refused(refused)
    assert bot_relay.claim_pending_envelopes(home) == []
    assert delivery.untouched()

    (tmp_path / "second").mkdir()
    other = _install(tmp_path / "second")
    other_home = other / "profiles" / "alpha"
    bot_relay.write_remote_roster(other_home, [
        {"profile": "ops", "handle": "ops", "connection_id": "cloud-1", "connection_label": "Cloud"},
    ])
    _config(other_home, "bot_mode:\n  allowed_targets: [ops@cloud-1]\n")
    assert _queued(_send(other, "alpha", "ops"))
    assert len(bot_relay.claim_pending_envelopes(other_home)) == 1


# ── allowed_targets: what the model is told ──────────────────────────────────


def test_roster_section_lists_only_permitted_teammates(tmp_path):
    root = _install(tmp_path, profiles=("alpha", "beta", "gamma"), peers=("north", "south"))
    home = root / "profiles" / "alpha"
    _config(home, "bot_mode:\n  allowed_targets: [gamma, north/ops]\n")
    section = bot_mode_probe.get_bot_mode_protocol_section(home)

    assert "`@gamma`" in section
    assert "`@beta`" not in section and "`@hermes`" not in section
    assert "`north/ops`" in section
    assert "south" not in section
    assert '"<peer>/<agent-name>"' not in section  # the open grammar is gone


def test_roster_section_for_an_empty_list_names_no_teammate(tmp_path):
    root = _install(tmp_path, peers=("north",))
    home = root / "profiles" / "alpha"
    _config(home, "bot_mode:\n  allowed_targets: []\n")
    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "`@beta`" not in section and "`@hermes`" not in section
    assert "north" not in section
    assert "allowed_targets lists no teammate" in section


def test_remote_machine_paragraph_is_filtered(tmp_path):
    root = _install(tmp_path)
    home = root / "profiles" / "alpha"
    bot_relay.write_remote_roster(home, [
        {"profile": "ops", "handle": "ops", "connection_id": "cloud-1", "connection_label": "Cloud"},
        {"profile": "fin", "handle": "fin", "connection_id": "cloud-1", "connection_label": "Cloud"},
    ])
    _config(home, "bot_mode:\n  allowed_targets: [fin]\n")
    section = bot_mode_probe.get_bot_mode_protocol_section(home)
    assert "@fin" in section and "@ops" not in section


def test_setting_or_changing_the_list_rebuilds_the_prompt_once(tmp_path):
    root = _install(tmp_path)
    home = root / "profiles" / "alpha"
    unset = bot_mode_probe.capability_fingerprint(home)
    _config(home, "bot_mode:\n  allowed_targets: [beta]\n")
    scoped = bot_mode_probe.capability_fingerprint(home)
    assert scoped != unset
    assert bot_mode_probe.capability_fingerprint(home) == scoped
    (home / "config.yaml").write_text("bot_mode:\n  allowed_targets: []\n", encoding="utf-8")
    assert bot_mode_probe.capability_fingerprint(home) not in (unset, scoped)


# ── accept_from: receiver side ───────────────────────────────────────────────


def test_accept_from_empty_blocks_a_sibling_at_dispatch(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, profiles=("alpha", "beta", "gamma"))
    _config(root / "profiles" / "beta", "bot_mode:\n  accept_from: []\n")

    refused = _send(root, "alpha", "beta")
    assert _refused(refused) and "accept_from" in refused["error"]
    assert delivery.untouched()
    assert _queued(_send(root, "alpha", "gamma"))  # the other siblings are unaffected


def test_accept_from_lists_the_senders_it_takes(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path, profiles=("alpha", "beta", "gamma"))
    _config(root / "profiles" / "beta", "bot_mode:\n  accept_from: [Gamma]\n")

    assert _refused(_send(root, "alpha", "beta"))
    assert delivery.untouched()
    assert _queued(_send(root, "gamma", "beta"))
    assert delivery.admissions == ["beta"]


def test_accept_from_does_not_touch_a_profile_without_the_key(tmp_path, monkeypatch):
    delivery = _Delivery(monkeypatch, tmp_path)
    root = _install(tmp_path)
    _config(root / "profiles" / "beta", "model: x\n")
    assert _queued(_send(root, "alpha", "beta"))
    assert delivery.admissions == ["beta"]
