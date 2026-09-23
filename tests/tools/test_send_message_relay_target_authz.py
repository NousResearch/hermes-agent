"""P5(a): `send_message` cannot silently name an arbitrary relay target.

The `target` tool parameter is free-form (`'platform:chat_id'`), so before
this guard a model could name ANY chat id and the gateway would emit an
outbound relay frame for it — authenticating the sender while never
authorizing the destination. These tests drive the REAL `send_message_tool`
entrypoint through the REAL production wiring (`gateway.relay.egress`,
`gateway.channel_directory`, `gateway.relay.relay_fronted_platforms`) against
a temp HERMES_HOME; nothing under test is constructed by the test itself.
"""

from __future__ import annotations

import json

import pytest

from gateway.config import Platform
from tools.send_message_tool import send_message_tool

ATTESTED_CHAT = "111111111111111111"
ARBITRARY_CHAT = "999999999999999999"
HOME_CHAT = "222222222222222222"


@pytest.fixture
def relay_env(tmp_path, monkeypatch):
    """A gateway whose ONLY reachable Discord destinations are attested.

    Mirrors the production shape: `GATEWAY_RELAY_PLATFORMS` is the deploy
    stamp `gateway.relay.relay_fronted_platforms()` reads, the channel
    directory json is the file `channel_directory.load_directory()` reads, and
    no live native adapter exists in this process (so the relay owns egress
    for `discord`, exactly as `gateway/delivery.resolve_delivery_transport`
    decides it).
    """
    import gateway.channel_directory as cd

    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://connector.example/relay")
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setenv("GATEWAY_RELAY_BOT_IDS", json.dumps({"discord": {"botId": "b1"}}))

    directory = tmp_path / "channel_directory.json"
    directory.write_text(
        json.dumps(
            {
                "updated_at": None,
                "platforms": {
                    "discord": [
                        {"id": ATTESTED_CHAT, "name": "bot-home", "type": "channel"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cd, "DIRECTORY_PATH", directory)
    monkeypatch.setattr(cd, "CHANNEL_ALIASES_PATH", tmp_path / "channel_aliases.json")
    # No gateway-session origins for discord in this temp home.
    monkeypatch.setattr(cd, "_build_from_sessions", lambda _platform: [])
    return directory


def _send(target: str, sent):
    """Invoke the real tool, recording any egress it attempts."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import asyncio

    discord_cfg = SimpleNamespace(enabled=True, token="t", extra={})
    config = SimpleNamespace(
        platforms={Platform.DISCORD: discord_cfg},
        get_home_channel=lambda _p: SimpleNamespace(chat_id=HOME_CHAT),
    )

    async def _record(platform, pconfig, chat_id, message, **kwargs):
        sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "tools.interrupt.is_interrupted", return_value=False
    ), patch("model_tools._run_async", side_effect=lambda c: asyncio.run(c)), patch(
        "tools.send_message_tool._send_to_platform", side_effect=_record
    ), patch(
        "gateway.mirror.mirror_to_session", return_value=False
    ):
        return json.loads(
            send_message_tool(
                {"action": "send", "target": target, "message": "hello"}
            )
        )


def test_arbitrary_relay_chat_id_is_refused_and_never_egresses(relay_env):
    """The whole observable: refused, naming THAT target, and ZERO egress."""
    sent: list[str] = []
    result = _send(f"discord:{ARBITRARY_CHAT}", sent)

    assert result == {
        "error": (
            f"Refusing to send to unattested relay target 'discord:{ARBITRARY_CHAT}': "
            "this gateway has no record of that destination. Use "
            "send_message(action='list') to see the targets it can reach."
        )
    }
    assert sent == []


def test_attested_directory_chat_id_still_sends(relay_env):
    """The guard must not destroy the feature: an attested chat goes through."""
    sent: list[str] = []
    result = _send(f"discord:{ATTESTED_CHAT}", sent)

    assert result == {"success": True, "message_id": "m1"}
    assert sent == [ATTESTED_CHAT]


def test_home_channel_is_attested(relay_env):
    """The operator-configured home channel is a provenance, not a guess."""
    sent: list[str] = []
    result = _send("discord", sent)

    assert result["success"] is True
    assert sent == [HOME_CHAT]


def test_session_origin_chat_is_attested(relay_env, monkeypatch):
    """A chat this gateway actually holds a session in is reachable."""
    import gateway.channel_directory as cd

    monkeypatch.setattr(
        cd,
        "_build_from_sessions",
        lambda platform: (
            [{"id": ARBITRARY_CHAT, "name": "seen", "type": "channel"}]
            if platform == "discord"
            else []
        ),
    )
    sent: list[str] = []
    result = _send(f"discord:{ARBITRARY_CHAT}", sent)

    assert result["success"] is True
    assert sent == [ARBITRARY_CHAT]


def test_platform_not_fronted_by_relay_is_untouched(relay_env, monkeypatch):
    """Non-relay platforms keep their own adapters' authorization, unchanged."""
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "telegram")
    monkeypatch.setenv(
        "GATEWAY_RELAY_BOT_IDS", json.dumps({"telegram": {"botId": "b1"}})
    )
    sent: list[str] = []
    result = _send(f"discord:{ARBITRARY_CHAT}", sent)

    assert result["success"] is True
    assert sent == [ARBITRARY_CHAT]


def test_live_native_adapter_takes_precedence_over_the_relay_guard(
    relay_env, monkeypatch
):
    """A platform served by a live NATIVE adapter here is not a relay egress.

    Same precedence `gateway/delivery.resolve_delivery_transport` applies: a
    concrete native adapter always wins over the relay, so this guard must not
    fire for it.
    """
    from types import SimpleNamespace

    import gateway.run

    runner = SimpleNamespace(adapters={Platform.DISCORD: object()})
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: runner)
    sent: list[str] = []
    result = _send(f"discord:{ARBITRARY_CHAT}", sent)

    assert result["success"] is True
    assert sent == [ARBITRARY_CHAT]


def test_react_refuses_an_arbitrary_relay_target(relay_env):
    """Reactions are outbound acts too — same floor, same refusal."""
    result = json.loads(
        send_message_tool(
            {
                "action": "react",
                "target": f"discord:{ARBITRARY_CHAT}",
                "emoji": "👍",
            }
        )
    )
    assert result == {
        "error": (
            f"Refusing to send to unattested relay target 'discord:{ARBITRARY_CHAT}': "
            "this gateway has no record of that destination. Use "
            "send_message(action='list') to see the targets it can reach."
        )
    }

# ── B-1: the guard must authorize the RESOLVED destination ──────────────────
#
# Slack `@handle` / `U...` targets are internal PSEUDO-ids
# (`user_name:ben`, `user:U...`) until `_resolve_slack_user_target` opens the
# DM and returns the real `D...` conversation. Provenances only ever hold
# resolved ids, so authorizing the pseudo-id compares a handle against a set
# of channel ids and refuses every Slack DM — an OUTAGE caused by a security
# fix. Review round 1 found this; reproduced before fixing.
#
# These tests are the falsifiable floor for the guard's POSITION: they pass
# only while authorization happens AFTER resolution.

SLACK_DM = "D01234567AB"
SLACK_USER = "U01234567AB"


@pytest.fixture
def slack_relay_env(tmp_path, monkeypatch):
    """A relay-fronted Slack gateway whose attested destination is a DM id."""
    import gateway.channel_directory as cd

    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://connector.example/relay")
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
    monkeypatch.setenv("GATEWAY_RELAY_BOT_IDS", json.dumps({"slack": {"botId": "b1"}}))

    directory = tmp_path / "channel_directory.json"
    directory.write_text(
        json.dumps(
            {
                "updated_at": None,
                # The DM conversation id — what resolution produces, and the
                # only form any provenance ever stores.
                "platforms": {"slack": [{"id": SLACK_DM, "name": "ben", "type": "im"}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cd, "DIRECTORY_PATH", directory)
    monkeypatch.setattr(cd, "CHANNEL_ALIASES_PATH", tmp_path / "channel_aliases.json")
    monkeypatch.setattr(cd, "_build_from_sessions", lambda _platform: [])
    return directory


def _send_slack(target: str, sent, *, resolves_to: str | None = SLACK_DM):
    """Invoke the real tool with the REAL Slack resolution step in the path.

    Only `conversations.open` is faked (a network call). The ordering of the
    guard against the resolver is production's.
    """
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    slack_cfg = SimpleNamespace(enabled=True, token="xoxb-t", extra={})
    config = SimpleNamespace(
        platforms={Platform.SLACK: slack_cfg},
        get_home_channel=lambda _p: SimpleNamespace(chat_id=SLACK_DM),
    )

    async def _record(platform, pconfig, chat_id, message, **kwargs):
        sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    async def _resolve(_token, target_ref):
        # Stands in for the Slack API call only; returns what production's
        # resolver returns — the opened DM channel id.
        return (resolves_to, None)

    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "tools.interrupt.is_interrupted", return_value=False
    ), patch("model_tools._run_async", side_effect=lambda c: asyncio.run(c)), patch(
        "tools.send_message_tool._send_to_platform", side_effect=_record
    ), patch(
        "tools.send_message_tool._resolve_slack_user_target", side_effect=_resolve
    ), patch(
        "gateway.mirror.mirror_to_session", return_value=False
    ):
        return json.loads(
            send_message_tool({"action": "send", "target": target, "message": "hello"})
        )


@pytest.mark.parametrize(
    "target",
    [f"slack:@ben", f"slack:{SLACK_USER}", f"slack:<@{SLACK_USER}>"],
)
def test_slack_user_targets_resolve_then_authorize(slack_relay_env, target):
    """An attested DM must SEND regardless of which alias names it.

    Fails if the guard runs before resolution: the pseudo-id
    (`user_name:ben` / `user:U...`) is not in any provenance, so the send is
    refused and `sent` stays empty.
    """
    sent: list[str] = []
    result = _send_slack(target, sent)

    assert result == {"success": True, "message_id": "m1"}
    # The whole observable: it egressed, and to the RESOLVED destination.
    assert sent == [SLACK_DM]


def test_slack_user_target_resolving_to_unattested_dm_is_refused(slack_relay_env):
    """Moving the guard must not disable it.

    A handle that resolves to a DM this gateway cannot attest is still
    refused — and the refusal names the RESOLVED id, which is the destination
    that was actually authorized.
    """
    sent: list[str] = []
    unattested = "D99999999XX"
    result = _send_slack("slack:@stranger", sent, resolves_to=unattested)

    assert result == {
        "error": (
            f"Refusing to send to unattested relay target 'slack:{unattested}': "
            "this gateway has no record of that destination. Use "
            "send_message(action='list') to see the targets it can reach."
        )
    }
    assert sent == []


# ── the guard must FAIL CLOSED on its own fault ─────────────────────────────
#
# Round-2 review: `_authorize_relay_target` wrapped BOTH the import and the
# call in one `except Exception: return None`, and None means AUTHORIZED. So
# any runtime bug inside the guard silently switched the whole P5(a) boundary
# off — the most expensive possible failure mode for an authorization check.


def test_guard_fault_refuses_rather_than_authorizing(relay_env, monkeypatch):
    """A guard that cannot answer must refuse, and must not egress."""
    import gateway.relay.egress as eg
    from tools import send_message_tool as smt

    def _boom(*_a, **_k):
        raise RuntimeError("bug inside the guard")

    monkeypatch.setattr(eg, "authorize_relay_target", _boom)

    denial = smt._authorize_relay_target("discord", ATTESTED_CHAT)
    assert denial is not None, "a faulting guard authorized the send"
    assert "authorization check failed" in denial

    # And end to end: nothing may egress.
    sent: list[str] = []
    result = _send(f"discord:{ATTESTED_CHAT}", sent)
    assert "error" in result
    assert sent == []


def test_missing_gateway_package_still_allows(relay_env, monkeypatch):
    """The tolerated case survives: no gateway ⇒ no relay egress to authorize.

    This is the distinction the original code collapsed. Keeping it tested
    stops a future "make it fail closed" change from breaking the CLI-only
    install.
    """
    import builtins

    from tools import send_message_tool as smt

    real_import = builtins.__import__

    def _no_gateway(name, *a, **k):
        if name.startswith("gateway.relay.egress"):
            # The shape real absence takes: ModuleNotFoundError WITH a name.
            # A bare ImportError is not something a missing module produces,
            # and treating it as absence was a fail-open (round 4, blocker 1).
            raise ModuleNotFoundError("no gateway package", name="gateway")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_gateway)
    assert smt._authorize_relay_target("discord", ARBITRARY_CHAT) is None


# ── fail-open boundaries: ABSENCE is not FAULT ─────────────────────────────


def test_route_discovery_fault_refuses_rather_than_authorizing(relay_env, monkeypatch):
    """A fault while determining relay routing must DENY, not fall through.

    `_relay_fronted` used to swallow every exception and return an empty set,
    which `relay_routed_platform` reads as "not relay-routed" — skipping the
    guard entirely. Review injected a discovery fault and watched an
    unattested target get authorized.
    """
    import gateway.relay as gr
    from tools.send_message_tool import _authorize_relay_target

    def boom():
        raise RuntimeError("config unreadable while listing fronted platforms")

    monkeypatch.setattr(gr, "relay_fronted_platforms", boom)
    denial = _authorize_relay_target("discord", "999")
    assert denial is not None
    assert "could not be" in denial


def test_missing_relay_module_still_authorizes(relay_env, monkeypatch):
    """The converse: genuine ABSENCE must keep working (no gateway ⇒ no relay).

    Without this, "fail closed on faults" would silently become "refuse
    everything in CLI/cron contexts", which is the outage the original broad
    except was there to avoid.
    """
    import gateway.relay as gr
    from tools.send_message_tool import _authorize_relay_target

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    assert _authorize_relay_target("discord", "999") is None


def test_relay_module_import_fault_refuses(monkeypatch):
    """A module that EXISTS but fails to import is a fault, not an absence."""
    import builtins

    import tools.send_message_tool as smt

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "gateway.relay.egress":
            raise RuntimeError("broken dependency inside an installed gateway")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    denial = smt._authorize_relay_target("discord", "999")
    assert denial is not None
    assert "could not be" in denial


@pytest.mark.parametrize("configured", ["discord", "Discord", "DISCORD", " discord "])
def test_relay_fronted_matching_is_case_insensitive(relay_env, monkeypatch, configured):
    """Review round 3, finding 3 — an attestation bypass on a string compare.

    `relay_routed_platform` lowercases the REQUESTED name but `_relay_fronted`
    returned configured names verbatim, so a platform configured as "Discord"
    missed the membership test and looked native — skipping the guard entirely.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {configured})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: set())
    assert eg.authorize_relay_target("discord", "999") is not None


@pytest.mark.parametrize("requested", ["Discord", "DISCORD", " discord "])
def test_requested_platform_name_is_also_normalised(relay_env, monkeypatch, requested):
    """The OTHER half of round 3, finding 3 — and it was never covered.

    The test above varies the CONFIGURED name while always requesting
    lowercase "discord", so it only pins `_relay_fronted`'s normalisation.
    Removing `.lower()` from the REQUESTED name in `relay_routed_platform`
    (and in `authorize_relay_target`) therefore survived the whole suite.

    SCOPE, precisely: `send_message` itself cannot reach this, because
    `_resolve_tool_target` lowercases the platform at
    tools/send_message_tool.py:47 before the guard is called. This pins the
    HELPERS' own contract for every other caller — the gateway lanes, and any
    future entry point that does not pre-normalise. Both functions are public
    within the package and must not assume a lowercased argument.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: set())

    assert eg.relay_routed_platform(requested) is True
    assert eg.authorize_relay_target(requested, "999") is not None


@pytest.mark.parametrize("requested", ["Discord", "DISCORD"])
def test_mixed_case_request_still_reaches_attested_targets(relay_env, monkeypatch, requested):
    """Control: normalising the requested name must not refuse real traffic.

    The attestation store is keyed by the LOWERCASE platform. If the lookup in
    `attested_relay_targets` / `authorize_relay_target` stops normalising, a
    mixed-case request misses its own attested set and is refused — an OUTAGE
    for legitimate traffic rather than a bypass. Both directions matter, so the
    lookup name is asserted, not just the routing decision.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    seen = []

    def _attested(platform):
        seen.append(platform)
        return {"999"} if platform == "discord" else set()

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", _attested)

    assert eg.authorize_relay_target(requested, "999") is None
    # The store was queried with the NORMALISED name.
    assert seen and all(p == "discord" for p in seen), seen


@pytest.mark.parametrize("requested", ["Discord", "DISCORD", " discord "])
def test_attested_lookup_normalises_before_querying_the_sources(relay_env, monkeypatch, requested):
    """`attested_relay_targets` does its OWN normalisation, and every other
    test monkeypatches this function away — so that `.lower()` was covered by
    nothing. Dropping it made a mixed-case platform find an EMPTY attested set
    (its sources are keyed lowercase), refusing legitimate traffic.

    Asserted against the real function with only its leaf sources stubbed.
    """
    import gateway.relay.egress as eg

    monkeypatch.setattr(eg, "_home_channel_id", lambda n: None)
    monkeypatch.setattr(eg, "_directory_ids", lambda n: {"999"} if n == "discord" else set())
    monkeypatch.setattr(eg, "_session_ids", lambda n: set())

    assert eg.attested_relay_targets(requested) == {"999"}


def test_attested_target_still_allowed_when_config_case_differs(relay_env, monkeypatch):
    """Control: normalizing must not start refusing legitimate traffic."""
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"Discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"999"})
    assert eg.authorize_relay_target("discord", "999") is None


def test_nested_dependency_importerror_refuses(monkeypatch):
    """Review round 3, finding 2 — `except ImportError` was still fail-open.

    An ImportError naming a NESTED module means an installed gateway failed to
    load (broken dependency). That is a fault, not "there is no relay here",
    and returning None means authorized.
    """
    import builtins

    import tools.send_message_tool as smt

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "gateway.relay.egress":
            raise ModuleNotFoundError(
                "No module named 'gateway.relay.dependency'",
                name="gateway.relay.dependency",
            )
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    denial = smt._authorize_relay_target("discord", "999")
    assert denial is not None


def test_absent_gateway_module_importerror_still_authorizes(monkeypatch):
    """Control: genuine absence (CLI/cron) must keep working."""
    import builtins

    import tools.send_message_tool as smt

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "gateway.relay.egress":
            raise ModuleNotFoundError("No module named 'gateway'", name="gateway")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert smt._authorize_relay_target("discord", "999") is None


def test_nameless_importerror_refuses(monkeypatch):
    """Round 4, blocker 1 — a bare ImportError is a FAULT, not absence.

    Genuine absence raises ModuleNotFoundError with `.name` set (verified
    against the interpreter). A plain ImportError therefore comes from an
    import hook or a module that failed while initializing, and authorizing on
    it means any such fault silently disables the boundary.
    """
    import builtins

    import tools.send_message_tool as smt

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "gateway.relay.egress":
            raise ImportError("something went wrong during init")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert smt._authorize_relay_target("discord", "999") is not None


def test_session_attestation_does_not_invent_a_matrix_room_prefix(monkeypatch):
    """Round 4, blocker 2 — the attestation set must not FABRICATE ids.

    `_session_ids` split every id on the first colon to recover "chat" from
    "chat:thread". Matrix room ids contain a colon natively
    (`!room:server.org`), so the split attested a bare `!room` that no session
    ever used — the guard vouching for a destination on its own invention.
    """
    import gateway.channel_directory as cd
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"matrix"})
    # A Matrix room with NO thread: the colon is part of the address itself.
    monkeypatch.setattr(
        cd,
        "_build_from_sessions",
        lambda p: [{"id": "!owned:server.org", "thread_id": None}],
    )

    # The real session id is still attested...
    assert eg.authorize_relay_target("matrix", "!owned:server.org") is None
    # ...but the invented prefix is not.
    assert eg.authorize_relay_target("matrix", "!owned") is not None


def test_session_attestation_still_recovers_a_slack_thread_parent(monkeypatch):
    """Control: thread-parent recovery must keep working.

    Slack session ids are genuinely `chat:thread` and the connector authorizes
    the CHAT, so failing to recover the parent would refuse legitimate replies.
    """
    import gateway.channel_directory as cd
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"slack"})
    # A genuinely thread-qualified entry carries thread_id separately, which is
    # how the parent is recovered — no string guessing.
    monkeypatch.setattr(
        cd,
        "_build_from_sessions",
        lambda p: [{"id": "C123:1700000000.1", "thread_id": "1700000000.1"}],
    )

    assert eg.authorize_relay_target("slack", "C123") is None


def test_egress_module_own_import_boundary_fails_closed(monkeypatch):
    """Round 4, non-blocking finding: `gateway/relay/egress.py` has its OWN
    import boundary (`_relay_fronted` -> `from gateway.relay import ...`), and
    the existing nested-ImportError test intercepts the EARLIER import in
    tools/send_message_tool.py, so this one was never exercised.
    """
    import builtins

    import gateway.relay.egress as eg

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "gateway.relay" and "relay_fronted_platforms" in (kw.get("fromlist") or a[2] if len(a) > 2 else []):
            raise ModuleNotFoundError("broken dep", name="gateway.relay.broken_dep")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(eg.RelayRouteUnknown):
        eg._relay_fronted()


def test_matrix_thread_parent_is_recovered_without_splitting_the_room_id(monkeypatch):
    """The case the allow-list could never have handled correctly.

    A Matrix room id contains a colon AND the session can be thread-qualified:
    `!room:server.org:$thread`. Splitting on the FIRST colon yields `!room`
    (invented); using the structured `thread_id` yields the real room.
    """
    import gateway.channel_directory as cd
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"matrix"})
    monkeypatch.setattr(
        cd,
        "_build_from_sessions",
        lambda p: [{"id": "!room:server.org:$thr", "thread_id": "$thr"}],
    )

    assert eg.authorize_relay_target("matrix", "!room:server.org") is None
    assert eg.authorize_relay_target("matrix", "!room") is not None


# ── round 5 ────────────────────────────────────────────────────────────────


def test_disabled_native_adapter_is_not_treated_as_native(monkeypatch):
    """R5-1: the guard and the delivery router must not disagree about routing.

    `resolve_delivery_transport` ignores a native adapter whose config is
    DISABLED and routes over Relay. `_has_live_native_adapter` treated mere
    presence in the adapter map as native, so the guard skipped authorization
    for a send that actually went over the relay.
    """
    from types import SimpleNamespace

    import gateway.relay.egress as eg
    from gateway.config import Platform

    monkeypatch.setattr(
        eg,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.DISCORD: object()}),
        raising=False,
    )
    import gateway.run as gr_run

    monkeypatch.setattr(gr_run, "_gateway_runner_ref", eg._gateway_runner_ref, raising=False)
    monkeypatch.setattr(
        eg,
        "load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.DISCORD: SimpleNamespace(enabled=False)}
        ),
        raising=False,
    )
    import gateway.config as gc

    monkeypatch.setattr(
        gc,
        "load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.DISCORD: SimpleNamespace(enabled=False)}
        ),
    )
    assert eg._has_live_native_adapter("discord") is False


def test_enabled_native_adapter_is_still_native(monkeypatch):
    """Control: an ENABLED native adapter must keep bypassing the relay guard."""
    from types import SimpleNamespace

    import gateway.config as gc
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    ref = lambda: SimpleNamespace(adapters={Platform.DISCORD: object()})  # noqa: E731
    monkeypatch.setattr(gr_run, "_gateway_runner_ref", ref, raising=False)
    monkeypatch.setattr(
        gc,
        "load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.DISCORD: SimpleNamespace(enabled=True)}
        ),
    )
    assert eg._has_live_native_adapter("discord") is True


def test_arbitrary_thread_under_an_attested_parent_is_refused(relay_env, monkeypatch):
    """R5-2: on Discord the THREAD is the REST destination.

    `POST /channels/{thread_id}/messages` — so an attested parent channel must
    not vouch for a caller-supplied thread the gateway has never seen.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"111"})
    assert eg.authorize_relay_target("discord", "111", "999") is not None


def test_attested_thread_is_allowed(relay_env, monkeypatch):
    """Control: a thread the gateway HAS a provenance for must still send.

    Only the BOUND `chat:thread` form counts. This test used to also accept a
    bare `{"111", "999"}` and so pinned a real defect: an unrelated attested
    chat whose id equalled the requested thread id authorized that thread. The
    bound form is what `_session_entry_id` actually records, so nothing
    legitimate is lost.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"111", "111:999"})
    assert eg.authorize_relay_target("discord", "111", "999") is None


def test_unrelated_attested_chat_does_not_vouch_for_a_thread(relay_env, monkeypatch):
    """An attested chat id equal to the requested THREAD id proves nothing.

    Reproduced against the pre-fix code: attested `{"-100A", "7"}` plus a
    request for thread `7` under parent `-100A` returned None (authorized),
    though no session ever existed in that thread. `7` is a sibling CHAT, not
    a thread of `-100A`.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"-100A", "7"})
    denial = eg.authorize_relay_target("discord", "-100A", "7")
    assert denial is not None and "no record of" in denial


def test_a_thread_attested_under_another_parent_is_refused(relay_env, monkeypatch):
    """`B:55` must not authorize thread 55 under parent A."""
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(
        eg, "attested_relay_targets", lambda p: {"-100A", "-100B", "-100B:55"}
    )
    assert eg.authorize_relay_target("discord", "-100A", "55") is not None
    # Control: the same thread under its OWN parent still sends.
    assert eg.authorize_relay_target("discord", "-100B", "55") is None


def test_tool_guard_forwards_the_thread_id(monkeypatch):
    """The wrapper must PASS thread_id, not just accept it.

    Mutating `_authorize_relay_target` to drop the argument survived every
    other test here — they all call `authorize_relay_target` directly, so
    nothing observed what the tool wrapper forwards. Same gap as the caller
    findings: testing the callee never proves the caller uses it.
    """
    import tools.send_message_tool as smt

    seen = {}

    def fake_authorize(platform_name, chat_id, thread_id=None):
        seen["args"] = (platform_name, chat_id, thread_id)
        return None

    import gateway.relay.egress as eg

    monkeypatch.setattr(eg, "authorize_relay_target", fake_authorize)
    smt._authorize_relay_target("discord", "111", "999")
    assert seen["args"] == ("discord", "111", "999")


def test_config_read_fault_refuses_rather_than_assuming_native(monkeypatch):
    """R6-1: my own R5-1 fix reintroduced the bypass it was closing.

    `_has_live_native_adapter` caught a `load_gateway_config()` failure and
    returned True, so a config fault declared the platform native while the
    ROUTER — reading the real config — would send over the relay. Routing we
    cannot determine is UNKNOWN, and unknown must refuse.
    """
    from types import SimpleNamespace

    import gateway.config as gc
    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.DISCORD: object()}),
        raising=False,
    )
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: set())

    def boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(gc, "load_gateway_config", boom)
    assert eg.authorize_relay_target("discord", "999") is not None


def test_healthy_config_with_enabled_native_still_bypasses_the_guard(monkeypatch):
    """Control: the guard must not start refusing native platforms."""
    from types import SimpleNamespace

    import gateway.config as gc
    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.DISCORD: object()}),
        raising=False,
    )
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: set())
    monkeypatch.setattr(
        gc,
        "load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.DISCORD: SimpleNamespace(enabled=True)}
        ),
    )
    assert eg.authorize_relay_target("discord", "999") is None


def test_guard_uses_the_live_adapter_not_stale_env_discovery(monkeypatch):
    """R7-1: the guard and the delivery router read DIFFERENT snapshots.

    `resolve_delivery_transport` asks the CONNECTED relay adapter
    (`fronts_platform`, from the handshake identity set). The guard rebuilt
    routing from `GATEWAY_RELAY_PLATFORMS`. When that env is stale or
    momentarily empty the guard said "native", the router sent over the relay,
    and the guard was skipped for a relay send.
    """
    from types import SimpleNamespace

    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    class _LiveRelay:
        def fronts_platform(self, p):
            return getattr(p, "value", p) == "discord"

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.RELAY: _LiveRelay()}),
        raising=False,
    )
    # Env discovery is empty/stale — the disagreement condition.
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: set())

    assert eg.relay_routed_platform("discord") is True
    assert eg.authorize_relay_target("discord", "999") is not None


def test_guard_falls_back_to_config_when_no_live_adapter(monkeypatch):
    """Control: with no live runner (CLI/cron) the config path must still work."""
    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run

    monkeypatch.setattr(gr_run, "_gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: set())

    assert eg.authorize_relay_target("discord", "999") is not None


def test_a_live_adapter_that_cannot_answer_is_a_fault_not_an_absence(monkeypatch):
    """A LIVE relay adapter whose `fronts_platform()` raises must fail closed.

    This was a real bypass. `_live_relay_fronted` caught every exception and
    returned None, which means "no live adapter — use the config snapshot". So
    a live adapter that could not report its routing PLUS an empty/stale config
    snapshot made the guard conclude "not relay-routed" and authorize an
    unattested destination, while `resolve_delivery_transport` asks that same
    adapter and still routes over the relay. Measured before the fix:
    `relay_routed=False`, verdict `None` for chat `999`.
    """
    from types import SimpleNamespace

    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    class Exploding:
        def fronts_platform(self, platform):
            raise RuntimeError("descriptor lookup failed")

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.RELAY: Exploding()}),
        raising=False,
    )
    # An EMPTY config snapshot: the fallback the fault used to reach.
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    with pytest.raises(eg.RelayRouteUnknown):
        eg._live_relay_fronted()
    with pytest.raises(eg.RelayRouteUnknown):
        eg.relay_routed_platform("discord")

    # And the tool-facing guard must refuse rather than authorize.
    from tools.send_message_tool import _authorize_relay_target

    denial = _authorize_relay_target("discord", "999")
    assert denial is not None
    assert "could not" in denial or "unavailable" in denial or "refus" in denial.lower()


def test_a_missing_relay_adapter_is_still_an_absence(monkeypatch):
    """Positive control for the split above: genuine ABSENCE keeps the config path.

    A runner with no relay adapter at all must NOT raise — otherwise the fix
    above would have turned every native-only deployment into a hard failure.
    """
    from types import SimpleNamespace

    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run

    monkeypatch.setattr(
        gr_run, "_gateway_runner_ref", lambda: SimpleNamespace(adapters={}), raising=False
    )
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    assert eg._live_relay_fronted() is None
    assert eg.relay_routed_platform("discord") is True
    assert eg.authorize_relay_target("discord", "123") is None
    assert eg.authorize_relay_target("discord", "999") is not None


def test_a_raising_fronts_platform_attribute_is_a_fault_not_an_absence(monkeypatch):
    """Reading the attribute can raise, and that is still a FAULT.

    Reviewer finding, reproduced before fixing. `fronts_platform` may be a
    property or descriptor, so the LOOKUP itself can raise — and the lookup
    used to sit inside the absence handler, which returned None and degraded
    to the config snapshot:

        live=None, routed=False, verdict=None   ← unattested target authorized

    The earlier test only made an already-retrieved METHOD raise, so it could
    not catch this. Only `relay is None` is absence now; everything about a
    present adapter, attribute access included, is a fault.
    """
    from types import SimpleNamespace

    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    class RaisingAttr:
        @property
        def fronts_platform(self):
            raise RuntimeError("attribute access failed")

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.RELAY: RaisingAttr()}),
        raising=False,
    )
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    with pytest.raises(eg.RelayRouteUnknown):
        eg._live_relay_fronted()
    with pytest.raises(eg.RelayRouteUnknown):
        eg.relay_routed_platform("discord")


def test_an_adapter_without_fronts_platform_is_a_fault(monkeypatch):
    """A present adapter missing the method entirely cannot answer either.

    It used to return None (→ config fallback). A relay adapter that cannot
    say what it fronts is a broken adapter, not an absent one.
    """
    from types import SimpleNamespace

    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.RELAY: object()}),
        raising=False,
    )
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    with pytest.raises(eg.RelayRouteUnknown):
        eg._live_relay_fronted()


def test_an_adapter_registry_that_cannot_be_read_is_a_fault(monkeypatch):
    """Reviewer finding, round 2: `.get()` on the registry can raise too.

    This was the FIFTH boundary in this one function where "something went
    wrong" became `None`, i.e. "no live adapter, use the config snapshot".
    Probed before the fix: relay_present=True, live=None, routed=False,
    verdict=None — unattested `discord:999` authorized.

    The function is now inverted: each `return None` sits behind an explicit
    narrow check, and anything else raises. That is why this test and the one
    below are grouped with the other fault cases rather than patching a sixth
    boundary.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    class HostileRegistry(dict):
        def get(self, key, default=None):
            raise RuntimeError("registry read failed")

    class Runner:
        def __init__(self):
            self.adapters = HostileRegistry({Platform.RELAY: object()})

    monkeypatch.setattr(gr_run, "_gateway_runner_ref", Runner, raising=False)
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    with pytest.raises(eg.RelayRouteUnknown):
        eg._live_relay_fronted()
    with pytest.raises(eg.RelayRouteUnknown):
        eg.relay_routed_platform("discord")


def test_an_unreadable_adapters_attribute_is_a_fault(monkeypatch):
    """`.adapters` may itself be a property that raises."""
    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run

    class Hostile:
        @property
        def adapters(self):
            raise RuntimeError("adapters unavailable")

    monkeypatch.setattr(gr_run, "_gateway_runner_ref", Hostile, raising=False)
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    with pytest.raises(eg.RelayRouteUnknown):
        eg._live_relay_fronted()


def test_a_runner_with_no_relay_key_is_still_an_absence(monkeypatch):
    """Control: a runner holding only NATIVE adapters is an absence, not a fault.

    Without this control the inversion above could have turned every
    native-only gateway into a hard failure.
    """
    from types import SimpleNamespace

    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.TELEGRAM: object()}),
        raising=False,
    )
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"discord"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    assert eg._live_relay_fronted() is None
    assert eg.authorize_relay_target("discord", "123") is None
    assert eg.authorize_relay_target("discord", "999") is not None


def test_a_healthy_live_adapter_is_unaffected_by_the_fault_handling(monkeypatch):
    """Liveness control: the ordinary path must still answer from the adapter.

    Every other test here drives a failure. This one proves the fault handling
    did not swallow the success case: a healthy adapter's own answer wins, the
    attested target sends, and the unattested one is refused.
    """
    from types import SimpleNamespace

    import gateway.relay as gr
    import gateway.relay.egress as eg
    import gateway.run as gr_run
    from gateway.config import Platform

    class Healthy:
        def fronts_platform(self, platform):
            return str(getattr(platform, "value", "")).lower() == "discord"

    monkeypatch.setattr(
        gr_run,
        "_gateway_runner_ref",
        lambda: SimpleNamespace(adapters={Platform.RELAY: Healthy()}),
        raising=False,
    )
    # The config snapshot DISAGREES on purpose: the live adapter must win.
    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    assert eg._live_relay_fronted() == {"discord"}
    assert eg.relay_routed_platform("discord") is True
    assert eg.authorize_relay_target("discord", "123") is None
    assert eg.authorize_relay_target("discord", "999") is not None


def test_a_broken_gateway_import_inside_the_live_probe_is_a_fault(monkeypatch):
    """A nested ImportError in the live probe must not degrade to the config path.

    Found by spot-check, not by the reviewer. `_live_relay_fronted` imports
    `gateway.config` and `gateway.run` inside its own try; before this, ANY
    ImportError there returned None, which means "no live adapter — use the
    config snapshot". So a broken installation plus an empty snapshot
    authorized an unattested destination. Probed with a healthy-adapter
    positive control in the same run:

        healthy_routed=True,  healthy_refuses_unattested=True
        fault_routed=False,   fault_verdict=None   ← authorized

    `_relay_fronted` below already made exactly this distinction for its own
    import; this is the same rule one function up.
    """
    import builtins

    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: set())
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "gateway.config":
            raise ImportError("cannot import name Platform from gateway.config")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)

    with pytest.raises(eg.RelayRouteUnknown):
        eg._live_relay_fronted()


def test_a_genuinely_absent_gateway_package_is_still_an_absence(monkeypatch):
    """Control for the test above: real absence must stay benign.

    `ModuleNotFoundError` naming the gateway package itself means there is no
    relay egress to authorize. If this raised, a CLI-only install would fail
    every send instead of using its native credential.
    """
    import builtins

    import gateway.relay.egress as eg

    real_import = builtins.__import__

    def absent(name, *args, **kwargs):
        if name == "gateway.run":
            raise ModuleNotFoundError("No module named 'gateway'", name="gateway")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)

    assert eg._live_relay_fronted() is None


# ── the Telegram @handle exemption must not cover a NATIVE send ────────────


def test_handle_exemption_withdrawn_when_a_native_token_exists(monkeypatch):
    """The exemption's justification is "the connector authorizes this".

    That is false when the gateway holds its own Telegram token:
    `_send_to_platform` calls `_send_telegram(pconfig.token, ...)` directly and
    no connector is involved. Measured before the fix: the numeric control was
    refused while `@unattested_public` was DELIVERED with the gateway's own
    credential.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"telegram"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})
    monkeypatch.setattr(eg, "_has_native_credential", lambda p: True)

    assert eg.authorize_relay_target("telegram", "@unattested_public") is not None
    # Controls: the ordinary guard is unchanged in both directions.
    assert eg.authorize_relay_target("telegram", "999") is not None
    assert eg.authorize_relay_target("telegram", "123") is None


def test_handle_exemption_survives_when_only_the_connector_can_send(monkeypatch):
    """The converse control — without it the fix is just "refuse everything".

    Relay-only is the configuration the exemption exists for: the connector
    holds the token and resolves the handle, so it authorizes the destination.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"telegram"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})
    monkeypatch.setattr(eg, "_has_native_credential", lambda p: False)

    assert eg.authorize_relay_target("telegram", "@public_channel") is None
    # A resolved destination stays fully guarded even in this mode.
    assert eg.authorize_relay_target("telegram", "999") is not None


def test_native_credential_probe_fault_withdraws_the_exemption(monkeypatch):
    """A fault must not GRANT an exemption — the safe direction is to withdraw
    it and fall back to the ordinary attestation check."""
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"telegram"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})

    import gateway.config as gcfg

    def _boom(*a, **kw):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(gcfg, "load_gateway_config", _boom)
    assert eg._has_native_credential("telegram") is True
    assert eg.authorize_relay_target("telegram", "@anything") is not None


# ── one config snapshot for authorization AND dispatch ─────────────────────


def test_dispatch_snapshot_token_withdraws_the_exemption(monkeypatch):
    """Authorization must use the SAME token dispatch will send with.

    Reviewer probe: the guard reloaded config independently, so a transition
    where the authorization snapshot had no token but the retained dispatch
    snapshot did produced a native send to an unattested handle.
    """
    import gateway.relay as gr
    import gateway.relay.egress as eg

    monkeypatch.setattr(gr, "relay_fronted_platforms", lambda: {"telegram"})
    monkeypatch.setattr(eg, "attested_relay_targets", lambda p: {"123"})
    # The independent reload is STALE and says connector-only.
    monkeypatch.setattr(eg, "_has_native_credential", lambda p: False)

    # Dispatch will use this token, so the exemption must not be granted.
    assert eg.authorize_relay_target(
        "telegram", "@unattested", native_token="123456:dispatch-token"
    ) is not None
    # Converse: a genuinely tokenless dispatch still exempts the handle.
    assert eg.authorize_relay_target("telegram", "@public", native_token=None) is None
    # Controls unchanged in both modes.
    assert eg.authorize_relay_target("telegram", "999", native_token=None) is not None
    assert eg.authorize_relay_target("telegram", "123", native_token="t") is None


def test_tool_guard_forwards_the_dispatch_token(monkeypatch):
    """CALLER-LEVEL. The test above drives the callee directly, so it passes
    even if the tool never forwards its snapshot — the gap that has produced
    four blockers on this branch. This asserts the wiring itself.
    """
    import tools.send_message_tool as smt

    seen = {}

    def _spy(platform_name, chat_id, thread_id=None, *, native_token=smt._TOKEN_UNSET):
        seen["native_token"] = native_token
        return None

    monkeypatch.setattr("gateway.relay.egress.authorize_relay_target", _spy)
    smt._authorize_relay_target("telegram", "@x", None, native_token="123:tok")
    assert seen["native_token"] == "123:tok"


def test_a_caller_that_omits_the_snapshot_does_not_get_the_exemption(monkeypatch):
    """A forgotten argument must not silently look like "no native token"."""
    import tools.send_message_tool as smt

    seen = {}

    def _spy(platform_name, chat_id, thread_id=None, **kwargs):
        seen["kwargs"] = kwargs
        return None

    monkeypatch.setattr("gateway.relay.egress.authorize_relay_target", _spy)
    smt._authorize_relay_target("telegram", "@x")
    # No native_token forwarded at all -> the guard runs its own probe.
    assert "native_token" not in seen["kwargs"]


def test_tool_guard_forwards_thread_id(monkeypatch):
    """CALLER-LEVEL: the tool must pass thread_id INTO the guard.

    thread_id is part of the DESTINATION — on Discord the thread is the literal
    REST target, so an attested parent must not vouch for an arbitrary thread.
    Every other test in this file calls `authorize_relay_target` directly, so
    they all pass even when the tool drops the argument on the floor: the
    mutation `_authorize_relay_target(platform_name, chat_id, None, ...)`
    survived the ENTIRE tests/tools suite (146 passed).
    """
    import tools.send_message_tool as smt

    seen = {}

    def _spy(platform_name, chat_id, thread_id=None, **kwargs):
        seen["thread_id"] = thread_id
        # Refuse, so the send stops here and the test asserts only the wiring.
        return "refused-for-test"

    from gateway.config import Platform, PlatformConfig

    monkeypatch.setattr(smt, "_authorize_relay_target", _spy)
    monkeypatch.setattr(
        smt, "_resolve_tool_target", lambda target: ("discord", "C1", "T99", None)
    )
    # Get past config resolution so execution actually reaches the guard.
    monkeypatch.setattr(
        smt,
        "_resolve_platform_config",
        lambda name, config: (
            Platform.DISCORD,
            PlatformConfig(enabled=True, token="t", extra={}),
            None,
            None,
        ),
    )

    smt._handle_send({"target": "discord:C1:T99", "message": "hi"})

    assert seen.get("thread_id") == "T99", "the tool dropped thread_id before the guard"
