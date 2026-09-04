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
            raise ImportError("no gateway package")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_gateway)
    assert smt._authorize_relay_target("discord", ARBITRARY_CHAT) is None
