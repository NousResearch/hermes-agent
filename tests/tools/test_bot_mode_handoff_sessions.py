"""Declared handoff sessions — the scoped, opt-in extension of the Bot Mode gate.

Contract under test: an install that declares nothing keeps the canonical-only
gate exactly as before; a profile that names a session title in
``agent.bot_mode_handoff_titles`` gives THAT title the teammate-messaging surface
(send + receive) and nothing else. A teammate's declaration never reaches another
profile, and the canonical title can never be re-declared.
"""

import json
import textwrap
from pathlib import Path

import pytest

from tools import bot_live_delivery, bot_mode_dm, bot_mode_probe, bot_relay

HANDOFF = "Growth and Profit Roadmap"


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _profile_home(tmp_path, name: str | None = None, *, managed=True, handoff=()) -> Path:
    """A minimal install/profile home; ``managed`` marks it Bot-Mode-managed."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / name if name else home
    profile.mkdir(parents=True, exist_ok=True)
    marker = "ui_meta:\n  hermes-bots:\n    shape: cloud\n" if managed else ""
    body = marker
    if handoff:
        body += "agent:\n  bot_mode_handoff_titles:\n" + "".join(f"    - {json.dumps(t)}\n" for t in handoff)
    (profile / "profile.yaml").write_text(body, encoding="utf-8")
    if handoff:
        (profile / "config.yaml").write_text(body, encoding="utf-8")
    return profile


class _FakeDB:
    def __init__(self, home: Path, title: str):
        self.db_path = str(home / "state.db")
        self._title = title

    def get_session_title(self, _sid):
        return self._title


class _FakeAgent:
    def __init__(self, home: Path, title: str = "Bot Chat"):
        self._session_db = _FakeDB(home, title)
        self.session_id = "sess-1"
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self.tools: list = []
        self.valid_tool_names: set = set()


def test_undeclared_profile_keeps_the_canonical_only_gate(tmp_path):
    home = _profile_home(tmp_path)
    assert bot_mode_probe.handoff_titles(home) == frozenset()
    assert bot_mode_probe.session_has_bot_mode("Bot Chat", home) is True
    assert bot_mode_probe.session_has_bot_mode(HANDOFF, home) is False
    assert bot_mode_probe.session_has_bot_mode("", home) is False
    assert bot_mode_probe.delivery_session_title(home) == bot_mode_probe.BOT_CHAT_TITLE
    assert bot_mode_dm.message_agent_authorized(_FakeAgent(home, HANDOFF)) is False
    assert bot_mode_dm.message_agent_authorized(_FakeAgent(home, "Bot Chat")) is True


def test_declared_title_gets_the_tool_and_becomes_the_delivery_target(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF])
    assert bot_mode_probe.handoff_titles(home) == frozenset({HANDOFF})
    assert bot_mode_probe.session_has_bot_mode(HANDOFF, home) is True
    assert bot_mode_probe.delivery_session_title(home) == HANDOFF

    agent = _FakeAgent(home, HANDOFF)
    assert bot_mode_dm.message_agent_authorized(agent) is True
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    names = [t["function"]["name"] for t in agent.tools if isinstance(t, dict)]
    assert names == [bot_mode_dm.MESSAGE_AGENT_TOOL_NAME]
    assert bot_mode_dm.MESSAGE_AGENT_TOOL_NAME in agent.valid_tool_names


def test_declaration_is_profile_scoped_and_never_leaks(tmp_path):
    declaring = _profile_home(tmp_path, "marketing", handoff=[HANDOFF])
    other = _profile_home(tmp_path, "engineering")
    assert bot_mode_probe.session_has_bot_mode(HANDOFF, declaring) is True
    assert bot_mode_probe.session_has_bot_mode(HANDOFF, other) is False
    assert bot_mode_dm.message_agent_authorized(_FakeAgent(other, HANDOFF)) is False


def test_canonical_title_cannot_be_redeclared_and_junk_is_ignored(tmp_path):
    home = _profile_home(tmp_path)
    (home / "config.yaml").write_text(
        textwrap.dedent(
            f"""\
            agent:
              bot_mode_handoff_titles:
                - "  "
                - "Bot Chat"
                - {json.dumps(HANDOFF)}
                - {json.dumps(HANDOFF)}
                - 42
            """
        ),
        encoding="utf-8",
    )
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_probe.handoff_titles(home) == frozenset({HANDOFF})
    # A re-declared canonical title must not turn into a second entry point by name.
    assert bot_mode_probe.delivery_session_title(home) == HANDOFF


def test_two_declared_titles_leave_the_canonical_delivery_target(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF, "Second Roadmap"])
    assert bot_mode_probe.session_has_bot_mode(HANDOFF, home) is True
    assert bot_mode_probe.session_has_bot_mode("Second Roadmap", home) is True
    # Ambiguous inbox: the canonical Bot Chat stays the delivery target.
    assert bot_mode_probe.delivery_session_title(home) == bot_mode_probe.BOT_CHAT_TITLE


def test_declaration_edit_is_adopted_without_a_restart(tmp_path, monkeypatch):
    home = _profile_home(tmp_path)
    assert bot_mode_probe.handoff_titles(home) == frozenset()
    (home / "config.yaml").write_text(
        f"agent:\n  bot_mode_handoff_titles:\n    - {json.dumps(HANDOFF)}\n", encoding="utf-8"
    )
    # Within the cache window the previous answer stands ...
    assert bot_mode_probe.handoff_titles(home) == frozenset()
    # ... and the window is short: past it the edit is live, with no restart.
    monkeypatch.setattr(bot_mode_probe, "_HANDOFF_TTL_SECONDS", 0.0)
    assert bot_mode_probe.handoff_titles(home) == frozenset({HANDOFF})
    assert bot_mode_probe.session_has_bot_mode(HANDOFF, home) is True


def test_dispatch_refuses_an_undeclared_session(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF])
    other = _profile_home(tmp_path, "engineering")
    err = json.loads(bot_mode_dm.message_agent_tool(
        target="hermes", message="hi", agent=_FakeAgent(other, HANDOFF)))
    assert "message_agent is only available in a Bot Mode 'Bot Chat' session" in err["error"]
    assert bot_mode_dm.message_agent_tool(
        target="hermes", message="", agent=_FakeAgent(home, HANDOFF)) != ""


def _seed_session(home: Path, session_id: str, title: str) -> None:
    """A real session row with a real title (the gate's authoritative identity)."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id=session_id, source="desktop")
        db.set_session_title(session_id, title)
    finally:
        db.close()


def test_local_turn_args_address_the_declared_session_by_exact_id(tmp_path, monkeypatch):
    declaring = _profile_home(tmp_path, "marketing", handoff=[HANDOFF])
    plain = _profile_home(tmp_path, "engineering")
    _seed_session(declaring, "s-roadmap", HANDOFF)
    assert bot_mode_dm._local_turn_args(plain) == list(bot_relay.BOT_CHAT_TURN_ARGS)
    args = bot_mode_dm._local_turn_args(declaring)
    assert isinstance(args, list), args
    # Exact session id, never the title: ``-c <title>`` would resolve a numbered sibling.
    assert args == ["chat", "--in", "~", "--resume", "s-roadmap", "-Q"]
    assert HANDOFF not in args
    # Same leading shape the profile turn lock and the delivery runner match on.
    assert args[:2] == ["chat", "--in"]


def test_numbered_sibling_never_captures_a_declared_delivery(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF])
    _seed_session(home, "s-exact", HANDOFF)
    _seed_session(home, "s-numbered", f"{HANDOFF} #2")
    assert bot_live_delivery.session_id_for_title(home, HANDOFF) == "s-exact"
    args = bot_mode_dm._local_turn_args(home)
    assert args is not None and "s-exact" in args and "s-numbered" not in args


def test_absent_declared_session_refuses_instead_of_misdelivering(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF])  # declared, but no such session exists
    assert bot_live_delivery.session_id_for_title(home, HANDOFF) is None
    assert bot_mode_dm._local_turn_args(home) is None


def test_new_delivery_never_falls_back_to_the_other_chat(tmp_path, monkeypatch):
    declaring = _profile_home(tmp_path, "marketing", handoff=[HANDOFF])
    plain = _profile_home(tmp_path, "engineering")
    _seed_session(declaring, "s-roadmap", HANDOFF)
    seen = []
    handoff_owner = {"profile_home": str(declaring), "session_id": "s-roadmap", "lease_id": "l1",
                     "live_session_id": "live1"}
    canonical_owner = {"profile_home": str(declaring), "session_id": "s-botchat", "lease_id": "l2",
                       "live_session_id": "live2"}

    def fake_find_live_owner(home, title):
        seen.append(("handoff", title))
        return handoff_owner if Path(home) == declaring else None

    def fake_find_canonical_live_owner(home):
        seen.append(("canonical", None))
        return canonical_owner

    monkeypatch.setattr(bot_live_delivery, "find_live_owner", fake_find_live_owner)
    monkeypatch.setattr(bot_live_delivery, "find_canonical_live_owner", fake_find_canonical_live_owner)

    assert bot_mode_dm._delivery_live_owner(declaring) == handoff_owner
    assert seen[0] == ("handoff", HANDOFF)
    # The declared session is closed while the canonical chat is open: NO owner is selected —
    # the envelope pins the declared session, so queuing it for the other chat could never be
    # claimed. None lets the caller use its exact-ID CLI transport instead.
    seen.clear()
    monkeypatch.setattr(bot_live_delivery, "find_live_owner", lambda home, title: None)
    assert bot_mode_dm._delivery_live_owner(declaring) is None
    assert seen == []
    # A profile that declares nothing still asks the canonical lookup directly.
    seen.clear()
    assert bot_mode_dm._delivery_live_owner(plain) == canonical_owner
    assert seen == [("canonical", None)]


def test_admission_does_not_queue_for_the_other_chat(tmp_path, monkeypatch):
    declaring = _profile_home(tmp_path, "marketing", handoff=[HANDOFF])
    _seed_session(declaring, "s-roadmap", HANDOFF)
    canonical_owner = {"profile_home": str(declaring), "session_id": "s-botchat", "lease_id": "l2",
                       "live_session_id": "live2"}
    monkeypatch.setattr(bot_live_delivery, "find_live_owner", lambda home, title: None)
    monkeypatch.setattr(bot_live_delivery, "find_canonical_live_owner", lambda home: canonical_owner)
    dm_file = declaring / "dm-test.txt"
    dm_file.write_text("payload", encoding="utf-8")
    assert bot_mode_dm._admit_live_dm(declaring, str(dm_file)) is None
    # Nothing was pinned for the canonical chat, so no live intent was written.
    assert not Path(str(dm_file) + ".live.json").exists()


def test_recognition_survives_a_widened_declaration(tmp_path, monkeypatch):
    """A delivery pinned to a declared session stays claimable after the profile declares more."""
    from hermes_cli import active_sessions

    home = _profile_home(tmp_path, handoff=[HANDOFF])
    _seed_session(home, "s-roadmap", HANDOFF)
    _seed_session(home, "s-botchat", bot_mode_probe.BOT_CHAT_TITLE)
    monkeypatch.setattr(
        active_sessions, "active_session_registry_snapshot",
        lambda registry_home=None: [
            {"session_id": "s-roadmap", "lease_id": "l1", "profile_home": str(home),
             "metadata": {"bot_live_delivery_consumer": True, "live_session_id": "live1"}},
            {"session_id": "s-botchat", "lease_id": "l2", "profile_home": str(home),
             "metadata": {"bot_live_delivery_consumer": True, "live_session_id": "live2"}},
        ],
    )
    owner = bot_live_delivery.find_live_owner_for_session(home, "s-roadmap")
    assert owner is not None
    record = bot_live_delivery.deliver_to_live_owner(home, owner, "pinned to the roadmap chat",
                                                     delivery_id="c" * 32)
    # The profile now declares a SECOND title: new messages go to the canonical chat, but
    # recognition of the already-pinned envelope must not change.
    (home / "config.yaml").write_text(
        "agent:\n  bot_mode_handoff_titles:\n    - " + json.dumps(HANDOFF) +
        "\n    - " + json.dumps("Second Roadmap") + "\n", encoding="utf-8")
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_probe.delivery_session_title(home) == bot_mode_probe.BOT_CHAT_TITLE
    assert bot_live_delivery.find_live_owner_for_session(home, "s-roadmap") is not None
    assert bot_live_delivery.owner_holds_delivery(home, record) is True


def test_owner_lookup_is_generalized_over_titles(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from hermes_cli import active_sessions

    home = _profile_home(tmp_path, handoff=[HANDOFF])
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id="s-roadmap", source="desktop")
        db.set_session_title("s-roadmap", HANDOFF)
    finally:
        db.close()
    monkeypatch.setattr(
        active_sessions, "active_session_registry_snapshot",
        lambda registry_home=None: [
            {"session_id": "s-roadmap", "lease_id": "l1", "profile_home": str(home),
             "metadata": {"bot_live_delivery_consumer": True, "live_session_id": "live1"}},
        ],
    )
    entry = bot_live_delivery.find_owner_by_title(home, HANDOFF)
    assert entry and entry["lease_id"] == "l1"
    assert bot_live_delivery.find_live_owner(home, HANDOFF) == {
        "profile_home": str(home), "session_id": "s-roadmap", "lease_id": "l1", "live_session_id": "live1"}
    # A canonical Bot Chat that does not exist here resolves to nothing, not to the handoff row.
    assert bot_live_delivery.find_owner_by_title(home, bot_mode_probe.BOT_CHAT_TITLE) is None
    # An unadvertised owner never receives a mailbox delivery.
    monkeypatch.setattr(
        active_sessions, "active_session_registry_snapshot",
        lambda registry_home=None: [
            {"session_id": "s-roadmap", "lease_id": "l1", "profile_home": str(home), "metadata": {}},
        ],
    )
    assert bot_live_delivery.find_live_owner(home, HANDOFF) is None


def test_capability_fingerprint_moves_on_a_declaration_edit(tmp_path):
    home = _profile_home(tmp_path)
    before = bot_mode_probe.capability_fingerprint(home)
    (home / "config.yaml").write_text(
        f"agent:\n  bot_mode_handoff_titles:\n    - {json.dumps(HANDOFF)}\n", encoding="utf-8"
    )
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_probe.capability_fingerprint(home) != before


# ── review round 1 (gpt-6.1-sol FAIL): identity, authorization source, consumer ─────


class _TitleDB:
    """Minimal session DB double: an authoritative stored title, or a failure, plus the row
    lookup that separates "unpersisted creation" from "persisted but untitled"."""

    def __init__(self, home: Path, title: str | None | Exception, row=None, row_error=None):
        self.db_path = str(home / "state.db")
        self._title = title
        self._row = row
        self._row_error = row_error

    def get_session_title(self, _sid):
        if isinstance(self._title, Exception):
            raise self._title
        return self._title

    def get_session(self, _sid):
        if self._row_error is not None:
            raise self._row_error
        return self._row


class _IdentAgent:
    def __init__(self, home: Path, stored_title, hint, row=None, row_error=None):
        self._session_db = _TitleDB(home, stored_title, row=row, row_error=row_error)
        self.session_id = "sess-1"
        self._session_title_hint = hint
        self._bot_mode_protocol = True
        self.tools: list = []
        self.valid_tool_names: set = set()


def test_stored_title_outranks_a_stale_hint(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF])
    # Renamed away: the hint still claims the declared title, the session does not.
    assert bot_mode_dm.message_agent_authorized(_IdentAgent(home, "Some other chat", HANDOFF)) is False
    # Renamed into: the stored title is authoritative even with a stale canonical hint.
    assert bot_mode_dm.message_agent_authorized(_IdentAgent(home, HANDOFF, "Bot Chat")) is True
    assert bot_mode_dm.message_agent_authorized(_IdentAgent(home, "Bot Chat", "unrelated")) is True


def test_identity_fails_closed_when_it_cannot_be_established(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF])
    # A failed title read is unidentified, never the hint.
    assert bot_mode_dm.message_agent_authorized(
        _IdentAgent(home, RuntimeError("db gone"), HANDOFF)) is False
    # No session row at all: a genuinely unpersisted creation, where the hint IS the identity.
    assert bot_mode_dm.message_agent_authorized(_IdentAgent(home, None, HANDOFF, row=None)) is True
    # A PERSISTED row whose title was cleared/blanked: unidentified, so a matching hint must
    # not resurrect authorization ...
    assert bot_mode_dm.message_agent_authorized(
        _IdentAgent(home, "", HANDOFF, row={"id": "sess-1", "title": None})) is False
    assert bot_mode_dm.message_agent_authorized(
        _IdentAgent(home, "   ", "Bot Chat", row={"id": "sess-1", "title": ""})) is False
    # ... and an unreadable row never manufactures canonical eligibility either.
    assert bot_mode_dm.message_agent_authorized(
        _IdentAgent(home, None, "Bot Chat", row_error=RuntimeError("row gone"))) is False


def test_real_db_cleared_title_revokes_the_declared_session(tmp_path):
    """The same rule against a real SessionDB: clearing the title ends authorization."""
    from hermes_state import SessionDB

    home = _profile_home(tmp_path, handoff=[HANDOFF])
    _seed_session(home, "s-roadmap", HANDOFF)
    db = SessionDB(db_path=home / "state.db")
    try:
        assert db.get_session_title("s-roadmap") == HANDOFF

        class _Agent:
            session_id = "s-roadmap"
            _session_db = db
            _session_title_hint = HANDOFF  # a stale hint still claiming the title

        assert bot_mode_dm.message_agent_authorized(_Agent()) is True
        db.set_session_title("s-roadmap", "")  # clear it
        assert bot_mode_dm.message_agent_authorized(_Agent()) is False
        assert bot_mode_probe.session_title_for(_Agent()) == ""
        # ... and renaming into the title re-authorizes it, from the stored title.
        db.set_session_title("s-roadmap", HANDOFF)
        assert bot_mode_dm.message_agent_authorized(_Agent()) is True
    finally:
        db.close()


def test_prompt_section_and_refresh_use_the_same_authoritative_title(tmp_path):
    """The prompt section and the capability refresh must agree with the tool gate."""
    import inspect
    from pathlib import Path as _Path

    from agent import system_prompt

    home = _profile_home(tmp_path, handoff=[HANDOFF])
    declared = _IdentAgent(home, HANDOFF, "Bot Chat")             # stored title wins over the hint
    renamed_away = _IdentAgent(home, "Some other chat", HANDOFF)  # hint alone must not authorize

    assert system_prompt._bot_mode_parts(declared) != []
    assert system_prompt._bot_mode_parts(renamed_away) == []

    # The refresh probe lives in agent.conversation_loop, whose import activates the runtime
    # payload in-process (blocked by this repo's home-IO guard when imported from a test), so
    # its sharing of the resolver is asserted at the source level: it must route through
    # session_title_for and must not consult the hint directly anywhere in that probe.
    src = _Path(inspect.getsourcefile(system_prompt)).parent / "conversation_loop.py"
    body = src.read_text(encoding="utf-8")
    probe = body.split("def _bot_chat_prompt_stale", 1)[1].split("\ndef ", 1)[0]
    assert "session_title_for(agent)" in probe
    assert "_session_title_hint" not in probe


def test_declaration_never_comes_from_the_merged_config_view(tmp_path, monkeypatch):
    """An install-wide overlay (or a stale last-known-good copy) must not grant access."""
    import hermes_cli.config as hermes_config

    home = _profile_home(tmp_path)  # own config.yaml declares nothing
    monkeypatch.setattr(hermes_config, "load_config_readonly",
                        lambda *a, **k: {"agent": {"bot_mode_handoff_titles": ["Managed Roadmap"]}})
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_probe.handoff_titles(home) == frozenset()
    assert bot_mode_probe.session_has_bot_mode("Managed Roadmap", home) is False


def test_unreadable_own_config_revokes_a_previous_declaration(tmp_path):
    home = _profile_home(tmp_path, handoff=[HANDOFF])
    assert bot_mode_probe.handoff_titles(home) == frozenset({HANDOFF})
    (home / "config.yaml").write_text("agent:\n  bot_mode_handoff_titles: [\n", encoding="utf-8")
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_probe.handoff_titles(home) == frozenset()
    assert bot_mode_probe.session_has_bot_mode(HANDOFF, home) is False
    assert bot_mode_probe.delivery_session_title(home) == bot_mode_probe.BOT_CHAT_TITLE


def test_consumer_claims_a_delivery_pinned_to_the_declared_session(tmp_path, monkeypatch):
    """Admission → poll → claimed → settled receipt, through the real consumer path."""
    import contextlib

    from hermes_cli import active_sessions
    from tui_gateway import session_notifications as notif

    home = _profile_home(tmp_path, handoff=[HANDOFF])
    _seed_session(home, "s-roadmap", HANDOFF)
    monkeypatch.setattr(
        active_sessions, "active_session_registry_snapshot",
        lambda registry_home=None: [
            {"session_id": "s-roadmap", "lease_id": "lease-1", "profile_home": str(home),
             "metadata": {"bot_live_delivery_consumer": True, "live_session_id": "live-1"}},
        ],
    )
    owner = bot_live_delivery.find_live_owner_for_session(home, "s-roadmap")
    assert owner and owner["session_id"] == "s-roadmap"
    delivery_id = "a" * 32
    assert bot_live_delivery.deliver_to_live_owner(home, owner, "hello from Default",
                                                   delivery_id=delivery_id)["status"] == "queued"

    class _Lease:
        lease_id = "lease-1"
        released = False

    session = {"session_key": "s-roadmap", "agent": object(), "active_session_lease": _Lease()}
    seen: dict = {}
    # The poller body resolves these bare names from its own globals (bind_module), so
    # patch them there rather than on the module object.
    scope = notif._poll_bot_live_delivery_once.__globals__
    monkeypatch.setitem(scope, "_session_home", lambda _s: home)
    monkeypatch.setitem(scope, "_session_turn_admission", lambda _s: contextlib.nullcontext(True))
    monkeypatch.setitem(scope, "_notif_release_turn", lambda _s: None)

    def fake_submit(rid, sid, sess, text, **kwargs):
        seen.update(rid=rid, sid=sid, text=text, kwargs=kwargs)
        return True

    monkeypatch.setitem(scope, "_run_prompt_submit", fake_submit)
    assert notif._poll_bot_live_delivery_once("live-1", session) is True
    assert seen["text"] == "hello from Default"
    claimed = bot_live_delivery.read_delivery_result(home, delivery_id)
    assert claimed is not None and claimed["status"] == "claimed"
    seen["kwargs"]["terminal_callback"]({"status": "settled", "text": "received"})
    settled = bot_live_delivery.read_delivery_result(home, delivery_id)
    assert settled is not None and settled["status"] == "settled" and settled["reply"] == "received"
    # The owner is still recognised as holding its own delivery while it is open.
    assert bot_live_delivery.owner_holds_delivery(home, {"owner": owner})


def test_consumer_does_not_adopt_a_canonical_pinned_delivery(tmp_path, monkeypatch):
    """Preserved behaviour: only the chat a delivery was pinned to may claim it."""
    import contextlib

    from hermes_cli import active_sessions
    from tui_gateway import session_notifications as notif

    home = _profile_home(tmp_path, handoff=[HANDOFF])
    _seed_session(home, "s-roadmap", HANDOFF)
    _seed_session(home, "s-botchat", bot_mode_probe.BOT_CHAT_TITLE)
    monkeypatch.setattr(
        active_sessions, "active_session_registry_snapshot",
        lambda registry_home=None: [
            {"session_id": "s-botchat", "lease_id": "lease-bot", "profile_home": str(home),
             "metadata": {"bot_live_delivery_consumer": True, "live_session_id": "live-bot"}},
            {"session_id": "s-roadmap", "lease_id": "lease-1", "profile_home": str(home),
             "metadata": {"bot_live_delivery_consumer": True, "live_session_id": "live-1"}},
        ],
    )
    canonical = bot_live_delivery.find_canonical_live_owner(home)
    assert canonical is not None
    delivery_id = "b" * 32
    bot_live_delivery.deliver_to_live_owner(home, canonical, "for the canonical chat",
                                            delivery_id=delivery_id)

    class _Lease:
        lease_id = "lease-1"
        released = False

    session = {"session_key": "s-roadmap", "agent": object(), "active_session_lease": _Lease()}
    # The poller body resolves these bare names from its own globals (bind_module), so
    # patch them there rather than on the module object.
    scope = notif._poll_bot_live_delivery_once.__globals__
    monkeypatch.setitem(scope, "_session_home", lambda _s: home)
    monkeypatch.setitem(scope, "_session_turn_admission", lambda _s: contextlib.nullcontext(True))
    monkeypatch.setitem(scope, "_notif_release_turn", lambda _s: None)
    monkeypatch.setitem(scope, "_run_prompt_submit", lambda *a, **k: True)
    assert notif._poll_bot_live_delivery_once("live-1", session) is False
    untouched = bot_live_delivery.read_delivery_result(home, delivery_id)
    assert untouched is not None and untouched["status"] == "queued"
