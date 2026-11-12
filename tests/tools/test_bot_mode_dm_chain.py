"""``message_agent``'s chain gate through the real tool entry (``tools/bot_mode_dm_chain.py``): the
recipient identity it compares, notify-only replies on every transport, released no-sends (#137027)."""

import json

import pytest

from agent import bot_chain
from gateway.bot_loop_guard import BotLoopGuardSettings
from tools import bot_mode_dm, bot_relay
from tools.bot_mode_probe import NO_REPLY_MARKER

# _fresh_probe_cache is the autouse probe-cache reset; importing it applies it here too.
from tests.tools.test_bot_mode_dm import _FakeAgent, _canonical_target, _capture_spawn, _fresh_probe_cache, _managed_home


@pytest.fixture(autouse=True)
def chains(monkeypatch):
    monkeypatch.setattr(bot_chain, "_CHAINS", {})
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(max_hops=4))


def _bot_turn(agent, author_id, message="hi", **envelope):
    agent._turn_author = {"id": author_id, "name": author_id.rsplit("/", 1)[-1][4:], "is_bot": True, **envelope}
    bot_chain.on_turn_start(agent, agent._turn_author, None, message)


def test_recipients_are_keyed_by_the_identity_their_turns_carry(tmp_path):
    """Greptile 4467193900: ``@hermes`` is the default bot, whose turns carry ``bot:default``; two peers'
    same-named agents stay two recipients; a relay row is ``bot:<connection>/<profile>``."""
    from tools.bot_mode_dm_chain import recipient_key

    home = _managed_home(tmp_path, teammates=("researcher",), peers=("peer-a", "peer-b"))
    # The sender's own relay folder, which delivery reads too.
    bot_relay.write_remote_roster(home / "profiles" / "researcher", [
        {"profile": "ops", "handle": "ops", "connection_id": "mini", "connection_label": "Mini"}])
    sender = _sender(home / "profiles" / "researcher")

    assert recipient_key("hermes", sender) == recipient_key("@default", sender) == "bot:default"
    assert recipient_key("peer-a/worker", sender) != recipient_key("peer-b/worker", sender)
    assert recipient_key("ops@mini", sender) == recipient_key("ops", sender) == "bot:mini/ops"


def test_a_peer_reply_is_keyed_as_the_peers_author_id(tmp_path):
    """A peer dm reaches us as ``bot:<its hostname>/<profile>``; answering it by peer name is the reply."""
    from tools.bot_mode_dm_chain import recipient_key

    home = _managed_home(tmp_path, peers=("spark",))
    sender = _sender(home, author={"id": "bot:spark/default", "name": "hermes", "is_bot": True})
    assert recipient_key("spark", sender) == recipient_key("spark/default", sender) == "bot:spark/default"
    assert recipient_key("spark/other", sender) == "peer:spark/other"


def _sender(home, author=None):
    from types import SimpleNamespace

    from tools import bot_mode_probe as probe

    agent = _FakeAgent(home)
    agent._turn_author = author
    root = probe._hermes_root(home)
    return SimpleNamespace(home=str(home), root=root, me=probe._profile_name(home), roster_homes=dict(probe._roster(root)),
                           peers=probe._peers(root), agent=agent)


def test_several_sends_in_one_bot_turn_hit_the_cap(tmp_path, monkeypatch):
    """Greptile 4467193907 through the tool: distinct sends inside one bot turn advance the hop."""
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(max_hops=2))
    home = _managed_home(tmp_path, teammates=("researcher", "scribe", "qa"))
    _canonical_target(monkeypatch, *(home / "profiles" / n for n in ("researcher", "scribe", "qa")))
    _capture_spawn(monkeypatch)
    agent = _FakeAgent(home)
    _bot_turn(agent, "bot:researcher", hops=1)

    first = json.loads(bot_mode_dm.message_agent_tool("scribe", "one", agent=agent))
    second = json.loads(bot_mode_dm.message_agent_tool("qa", "two", agent=agent))
    assert not first.get("error") and "max_hops" in second["error"]


def test_an_offline_relay_target_can_be_retried_after_it_reconnects(tmp_path, monkeypatch):
    """Greptile 4467193919: the definite no-send hands its ask back, so the same message is not a duplicate."""
    home = _managed_home(tmp_path)
    _capture_spawn(monkeypatch)
    row = {"profile": "default", "handle": "hermes", "connection_id": "mini", "connection_label": "Mini"}
    bot_relay.write_remote_roster(home, [{**row, "online": False}])
    agent = _FakeAgent(home)
    _bot_turn(agent, "bot:researcher")

    offline = json.loads(bot_mode_dm.message_agent_tool("hermes@mini", "ping", agent=agent))
    bot_relay.write_remote_roster(home, [{**row, "online": True}])
    retry = json.loads(bot_mode_dm.message_agent_tool("hermes@mini", "ping", agent=agent))

    assert offline["reason"] == "runtime_offline"
    assert retry["status"] == "queued", retry


def test_a_relay_reply_carries_the_no_reply_marker_and_the_receiver_honours_it(tmp_path, monkeypatch):
    """Greptile 4467193937: the Desktop relay rebuilds the author from id/name/is_bot only, so the
    reply flag rides in the message text (#144789's marker) and the receiving turn reads it there."""
    home = _managed_home(tmp_path)
    calls = _capture_spawn(monkeypatch)
    bot_relay.write_remote_roster(home, [
        {"profile": "default", "handle": "hermes", "connection_id": "mini", "connection_label": "Mini"}])
    agent = _FakeAgent(home)
    _bot_turn(agent, "bot:mini/default")  # mini's default messaged us through the relay

    ack = json.loads(bot_mode_dm.message_agent_tool("hermes@mini", "42", agent=agent))
    (envelope,) = bot_relay.claim_pending_envelopes(home)
    assert ack["expects_reply"] is False and calls == []  # notify-only: no waiter
    assert f"{NO_REPLY_MARKER} 42" in envelope["message"]

    # The receiving side: its turn author is what hermes_agent/tui_gateway/methods_bot_relay.py rebuilds.
    receiver = _FakeAgent(home / "elsewhere")
    receiver.session_id = "receiver"
    _bot_turn(receiver, bot_relay.delivery_turn_author("default", "hermes", "laptop")["id"],
              message=envelope["message"])
    assert "needs no answer" in bot_chain.admit_send(receiver, ["bot:laptop/default"], "thanks").refused[
        "bot:laptop/default"]


def test_a_disabled_guard_leaves_message_agent_as_main_has_it(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(enabled=False, max_hops=1))
    home = _managed_home(tmp_path, teammates=("researcher", "scribe"))
    authority = _canonical_target(monkeypatch, *(home / "profiles" / n for n in ("researcher", "scribe")))
    _capture_spawn(monkeypatch)
    agent = _FakeAgent(home)
    _bot_turn(agent, "bot:researcher", hops=5)

    for _ in range(3):
        assert not json.loads(bot_mode_dm.message_agent_tool("scribe", "same", agent=agent)).get("error")
    assert len(authority.calls) == 3 and NO_REPLY_MARKER not in authority.calls[0][1]["message"]
