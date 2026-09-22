"""Matrix administration uses an opted-in session and its current owner."""

import asyncio
import importlib
import json
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from gateway.session_identity import RoutingIdentity
from hermes_cli.config import atomic_config_write
from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.tools_config import _get_platform_tools
from hermes_constants import get_hermes_home
from tools.registry import registry

importlib.import_module("tools.matrix_room_tool")
importlib.import_module("tools.matrix_pin_tool")
importlib.import_module("tools.matrix_read_tool")


def _raw_api(state, event):
    """Serve raw state and event requests from *state* and *event* fakes."""
    async def request(_method, path, *, query_params=None, **_kwargs):
        room_id, _, rest = unquote(str(path)).removeprefix("/_matrix/client/v3/rooms/").partition("/")
        kind, _, rest = rest.partition("/")
        if kind == "event":
            return await event(room_id, rest)
        event_type, _, state_key = rest.partition("/")
        return await state(room_id, event_type, state_key, **(query_params or {}))

    return SimpleNamespace(request=request)


# Only a hung test reaches this bound; every wait below ends on an event.
_HANG_TIMEOUT = 30


@pytest.fixture(autouse=True)
def _no_plugins(monkeypatch):
    # Each test has a fresh HERMES_HOME, so every toolset selection would rescan the plugin manifests.
    monkeypatch.setattr("hermes_cli.plugins.PluginManager.discover_and_load", lambda self, force=False: None)


@pytest.mark.parametrize("platform", ["matrix", "telegram", "cli"])
@pytest.mark.parametrize("enabled", [False, True])
def test_matrix_admin_discovery_requires_explicit_matrix_selection(platform, enabled, monkeypatch, tmp_path):
    from model_tools import get_tool_definitions

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    for profile, opt_in in (("A", enabled), ("B", not enabled), ("A", enabled)):
        home = tmp_path / profile
        home.mkdir(exist_ok=True)
        config = {"platform_toolsets": {platform: [f"hermes-{platform}", "matrix_read"]}}
        if opt_in:
            config["platform_toolsets"][platform].append("matrix_admin")
        atomic_config_write(home / "config.yaml", config)
        token = set_hermes_home_override(home)
        try:
            selected = _get_platform_tools(load_user_config_effective(), platform, include_default_mcp_servers=False)
            schemas = get_tool_definitions(list(selected), quiet_mode=True, skip_tool_search_assembly=True)
        finally:
            reset_hermes_home_override(token)
        names = {schema["function"]["name"] for schema in schemas
                 if schema["function"]["name"].startswith("matrix_")}
        expected = {"matrix_read"} if platform == "matrix" else set()
        if opt_in and platform == "matrix":
            expected |= {"matrix_room_admin", "matrix_pin"}
        assert names == expected



@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "invite", "leave", "forget", "redact", "pin", "unpin"])
@pytest.mark.parametrize("change", ["none", "client", "crypto", "room", "actor", "opt_in", "session", "profile", "power", "server", "queued_client", "revoked_during_write", "routed", "routed_runtime_off", "routed_transport_off", "routed_owner"])
async def test_admin_rechecks_owner_after_permission_reads(action, change):
    from hermes_constants import set_hermes_home_override
    from plugins.platforms.matrix.adapter import MatrixAdapter

    sdk = pytest.importorskip("mautrix.types")
    if change.startswith("routed"):
        await _routed_admin_sequence(action, change)
        return
    home = get_hermes_home()
    atomic_config_write(home / "config.yaml", {"platform_toolsets": {"matrix": ["matrix_admin"]}})
    writes = []
    room = "!room:server"
    actor = "@alice:server"
    blocked = asyncio.Event()
    resume = asyncio.Event()
    client = SimpleNamespace(crypto=object())
    adapter = object.__new__(MatrixAdapter)
    adapter._pin_state_lock = asyncio.Lock()
    adapter._client = client
    adapter._user_id = "@bot:server"
    adapter._owner_profile = None
    adapter._joined_rooms = {room} if action != "forget" else set()
    adapter._allowed_room_ids = set()
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter.set_authorization_check(lambda user, chat_type, chat_id: user == actor)

    async def state(room_id, kind, state_key="", **kwargs):
        if kind == "m.room.member":
            if state_key == actor:
                blocked.set()
                await resume.wait()
                if change == "session":
                    set_session_vars(platform="matrix", chat_id=room, user_id=actor,
                                     session_key="session-B", transport_adapter=adapter,
                                     transport_loop=asyncio.get_running_loop())
                if change == "profile":
                    set_hermes_home_override(home / "other")
            return {"membership": "leave" if action == "forget" and state_key != actor else "join"}
        if kind == "m.room.power_levels":
            return {"users": {actor: 0 if change == "power" else 100, adapter._user_id: 100}, "invite": 50}
        if kind == "m.room.create":
            return {"type": "m.room.create", "sender": actor, "content": {"room_version": "11"}}
        if kind == "m.room.pinned_events":
            return {"pinned": ["$target"] if action == "unpin" else []}
        return {}

    class Forbidden(Exception):
        errcode = "M_FORBIDDEN"

    async def mutate(*args, **kwargs):
        if change == "server":
            raise Forbidden("server denies the operation")
        writes.append((asyncio.get_running_loop(), get_hermes_home(), args, kwargs))
        if change == "revoked_during_write":
            adapter.set_authorization_check(lambda user, chat_type, chat_id: False)
        return "!created:server" if action == "create" else "$redaction"

    client.get_event = AsyncMock(return_value={"room_id": room, "event_id": "$target", "sender": "@bob:server"})
    client.api = _raw_api(state, client.get_event)
    for method in ("create_room", "invite_user", "leave_room", "forget_room", "redact", "send_state_event"):
        setattr(client, method, mutate)
    identity = RoutingIdentity("default", "default", home, home, multiplexed=False, transport=weakref.ref(adapter))
    tokens = set_session_vars(platform="matrix", chat_id=room, user_id=actor,
                              session_key="session-A", transport_adapter=adapter,
                              transport_loop=asyncio.get_running_loop(), routing_identity=identity)
    args = {"action": action, "user_id": "@bob:server", "event_id": "$target", "encrypted": True}
    tool = "matrix_pin" if action in {"pin", "unpin"} else "matrix_room_admin"
    def dispatch():
        result = registry.dispatch(tool, args)
        assert isinstance(result, str)
        return json.loads(result)

    if change == "queued_client":
        # The worker creates the change after capturing its owner and before
        # scheduling it on the gateway loop.
        for method in ("administer_matrix_room", "change_matrix_pin"):
            def replace_client_then_create(*args, _create=getattr(adapter, method), **kwargs):
                adapter._client = SimpleNamespace(**client.__dict__)
                return _create(*args, **kwargs)

            setattr(adapter, method, replace_client_then_create)

    task = asyncio.create_task(asyncio.to_thread(dispatch))
    try:
        assert registry.get_entry(tool) is not None
        if change == "queued_client":
            resume.set()
            result = await asyncio.wait_for(task, timeout=_HANG_TIMEOUT)
            assert (result, writes) == ({"error": "Matrix session or client ownership changed"}, [])
            return
        barrier = asyncio.create_task(blocked.wait())
        await asyncio.wait_for(asyncio.wait({task, barrier}, return_when=asyncio.FIRST_COMPLETED), timeout=_HANG_TIMEOUT)
        barrier.cancel()
        if change == "crypto":
            client.crypto = object()
        if change == "client":
            adapter._client = SimpleNamespace()
        if change == "room":
            adapter._is_allowed_matrix_room_event.return_value = False
        if change == "actor":
            adapter.set_authorization_check(lambda user, chat_type, chat_id: False)
        if change == "opt_in":
            atomic_config_write(home / "config.yaml", {"platform_toolsets": {"matrix": []}})
        resume.set()
        result = await asyncio.wait_for(task, timeout=_HANG_TIMEOUT)
    finally:
        resume.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        clear_session_vars(tokens)
    allowed = change in {"none", "revoked_during_write"} or change == "power" and action == "create"
    joined = set() if action == "forget" or allowed and action == "leave" else {room}
    if allowed:
        results = {
            "create": {"action": "create", "room_id": "!created:server", "encrypted": True},
            "invite": {"action": "invite", "room_id": room, "user_id": "@bob:server"},
            "leave": {"action": "leave", "room_id": room},
            "forget": {"action": "forget", "room_id": room},
            "redact": {"action": "redact", "room_id": room, "event_id": "$target",
                       "redaction_event_id": "$redaction"},
            "pin": {"pinned": ["$target"], "state_event_id": "$redaction"},
            "unpin": {"pinned": [], "state_event_id": "$redaction"},
        }
        calls = {
            "create": ((), {"name": None, "topic": None, "invitees": [], "is_direct": False,
                             "preset": sdk.RoomCreatePreset.PRIVATE, "initial_state": [{"type": "m.room.encryption",
                                 "state_key": "", "content": {"algorithm": "m.megolm.v1.aes-sha2"}}]}),
            "invite": ((room, "@bob:server"), {"reason": None}),
            "leave": ((room,), {"reason": None, "raise_not_in_room": True}),
            "forget": ((room,), {}),
            "redact": ((room, "$target"), {"reason": None}),
            "pin": ((room, "m.room.pinned_events", {"pinned": ["$target"]}), {}),
            "unpin": ((room, "m.room.pinned_events", {"pinned": []}), {}),
        }
        expected = results[action]
        if change == "revoked_during_write" and tool == "matrix_room_admin":
            expected = {**expected, "warning": "Matrix requester is not authorized for this room"}
        assert (result, writes, adapter._joined_rooms) == (
            expected, [(asyncio.get_running_loop(), home, *calls[action])], joined,
        )
    elif change == "server":
        if tool == "matrix_pin":
            expected = {"error": "Matrix pin update was rejected", "errcode": "M_FORBIDDEN",
                        "message": "server denies the operation"}
        else:
            expected = {"error": "Matrix administration failed: Forbidden", "errcode": "M_FORBIDDEN",
                        "message": "server denies the operation"}
        assert (result, writes, adapter._joined_rooms) == (expected, [], joined)
    else:
        ownership = {"error": "Matrix session or client ownership changed"}
        denials = {
            "client": ownership, "crypto": ownership, "session": ownership, "profile": ownership,
            "room": {"error": "Matrix room is not allowed"},
            "actor": {"error": "Matrix requester is not authorized for this room"},
            "opt_in": {"error": "Enable matrix_admin for Matrix in both the runtime and transport profiles in hermes tools"},
            "power": {
                "error": {
                    "invite": "Matrix requester lacks permission to invite users",
                    "leave": "Matrix requester lacks permission to remove the bot from this room",
                    "forget": "Matrix requester lacks permission to remove the bot from this room",
                    "redact": "Matrix requester lacks permission to redact this event",
                    "pin": "Matrix requester lacks permission to change pins",
                    "unpin": "Matrix requester lacks permission to change pins",
                }.get(action),
                "required": 50, "level": 0,
            },
        }
        assert (result, writes, adapter._joined_rooms) == (denials[change], [], joined)


async def _routed_admin_sequence(action, change):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session import SessionContext, SessionSource
    from gateway.session_context import get_session_routing_identity, get_session_transport
    from plugins.platforms.matrix.adapter import MatrixAdapter

    root = get_hermes_home()
    homes = {profile: root / profile for profile in ("A", "B")}
    for home in homes.values():
        home.mkdir()
    room, actor = "!room:server", "@alice:server"
    writes = []
    observations = []
    runner = object.__new__(GatewayRunner)
    runner._gateway_loop = asyncio.get_running_loop()
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "A"
    adapter = object.__new__(MatrixAdapter)
    adapter._user_id = "@bot-A:server"
    adapter._owner_profile = "A"
    adapter.gateway_runner = runner
    adapter._allowed_room_ids = set()
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter.set_authorization_check(lambda user, chat_type, chat_id: user == actor)
    other_bot = object.__new__(MatrixAdapter)
    other_bot._user_id = "@bot-B:server"
    runner._profile_adapters = {"B": {Platform.MATRIX: other_bot}}
    tool = "matrix_pin" if action in {"pin", "unpin"} else "matrix_room_admin"
    args = {"action": action, "user_id": "@bob:server", "event_id": "$target"}

    for profile in ("A", "B", "A"):
        for home in homes.values():
            atomic_config_write(home / "config.yaml", {"platform_toolsets": {"matrix": ["matrix_admin"]}})
        runner.adapters = {Platform.MATRIX: adapter}
        adapter._joined_rooms = {room} if action != "forget" else set()
        adapter._pin_state_lock = asyncio.Lock()
        client = SimpleNamespace(crypto=object())
        adapter._client = client
        blocked, resume = asyncio.Event(), asyncio.Event()

        async def state(room_id, kind, state_key="", **kwargs):
            if kind == "m.room.member":
                if state_key == actor:
                    blocked.set()
                    await resume.wait()
                return {"membership": "leave" if action == "forget" and state_key != actor else "join"}
            if kind == "m.room.power_levels":
                return {"users": {actor: 100, adapter._user_id: 100}}
            if kind == "m.room.create":
                return {"type": "m.room.create", "sender": actor, "content": {"room_version": "11"}}
            if kind == "m.room.pinned_events":
                return {"pinned": ["$target"] if action == "unpin" else []}
            return {}

        async def mutate(*args, **kwargs):
            writes.append((profile, get_hermes_home(), get_session_routing_identity(), get_session_transport()[0]))
            return "!created:server" if action == "create" else "$redaction"

        client.get_event = AsyncMock(return_value={"room_id": room, "event_id": "$target", "sender": "@bob:server"})
        client.api = _raw_api(state, client.get_event)
        for method in ("create_room", "invite_user", "leave_room", "forget_room", "redact", "send_state_event"):
            setattr(client, method, mutate)
        source = SessionSource(platform=Platform.MATRIX, chat_id=room, chat_type="group", user_id=actor,
                               profile=profile)
        identity = RoutingIdentity("A", profile, homes["A"], homes[profile], transport=weakref.ref(adapter))
        setattr(source, "_identity", identity)
        context = SessionContext(source=source, connected_platforms=[], home_channels={})
        with _profile_runtime_scope(homes[profile], hydrate_secrets=False):
            tokens = runner._set_session_env(context)
            task = asyncio.create_task(asyncio.to_thread(registry.dispatch, tool, args))
            barrier = asyncio.create_task(blocked.wait())
            try:
                await asyncio.wait_for(asyncio.wait({task, barrier}, return_when=asyncio.FIRST_COMPLETED), timeout=_HANG_TIMEOUT)
                assert blocked.is_set(), await task
                if change in {"routed_runtime_off", "routed_transport_off"}:
                    disabled = homes[profile] if change == "routed_runtime_off" else homes["A"]
                    atomic_config_write(disabled / "config.yaml", {"platform_toolsets": {"matrix": []}})
                if change == "routed_owner":
                    runner.adapters = {Platform.MATRIX: other_bot}
                resume.set()
                raw_result = await asyncio.wait_for(task, timeout=_HANG_TIMEOUT)
                assert isinstance(raw_result, str)
                result = json.loads(raw_result)
                observations.append((profile, result, get_hermes_home()))
            finally:
                resume.set()
                task.cancel()
                barrier.cancel()
                await asyncio.gather(task, barrier, return_exceptions=True)
                runner._clear_session_env(tokens)
        if change == "routed":
            results = {
                "create": {"action": "create", "room_id": "!created:server", "encrypted": False},
                "invite": {"action": "invite", "room_id": room, "user_id": "@bob:server"},
                "leave": {"action": "leave", "room_id": room},
                "forget": {"action": "forget", "room_id": room},
                "redact": {"action": "redact", "room_id": room, "event_id": "$target", "redaction_event_id": "$redaction"},
                "pin": {"pinned": ["$target"], "state_event_id": "$redaction"},
                "unpin": {"pinned": [], "state_event_id": "$redaction"},
            }
            assert (result, writes[-1]) == (results[action], (profile, homes["A"], identity, adapter))
        else:
            error = "Matrix session or client ownership changed" if change == "routed_owner" else "Enable matrix_admin for Matrix in both the runtime and transport profiles in hermes tools"
            assert (result, writes) == ({"error": error}, [])
    expected = results[action] if change == "routed" else {"error": error}
    assert observations == [(profile, expected, homes[profile]) for profile in ("A", "B", "A")]


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplexed", [False, True], ids=["standalone", "multiplexed"])
@pytest.mark.parametrize("resumed", [False, True], ids=["first-turn", "resumed"])
async def test_admin_uses_identity_bound_for_gateway_turn(multiplexed, resumed):
    import dataclasses
    from datetime import datetime

    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, build_session_context
    from plugins.platforms.matrix.adapter import MatrixAdapter

    home = get_hermes_home()
    atomic_config_write(home / "config.yaml", {"platform_toolsets": {"matrix": ["matrix_admin"]}})
    room, actor = "!room:server", "@alice:server"
    runner = object.__new__(GatewayRunner)
    runner._gateway_loop = asyncio.get_running_loop()
    runner.config = GatewayConfig(multiplex_profiles=multiplexed)
    runner._primary_profile_name = "default"
    runner._profile_adapters = {}
    adapter = object.__new__(MatrixAdapter)
    adapter.platform = Platform.MATRIX
    adapter.gateway_runner = runner
    adapter._owner_profile = None
    adapter._user_id = "@bot:server"
    adapter._joined_rooms = {room}
    adapter._allowed_room_ids = set()
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter.set_authorization_check(lambda user, chat_type, chat_id: user == actor)
    runner.adapters = {Platform.MATRIX: adapter}
    writes = []

    async def state(room_id, kind, state_key="", **kwargs):
        if kind == "m.room.member":
            return {"membership": "join"}
        if kind == "m.room.power_levels":
            return {"users": {actor: 100, adapter._user_id: 100}}
        if kind == "m.room.create":
            return {"type": "m.room.create", "sender": actor, "content": {"room_version": "11"}}
        return {}

    async def invite_user(*args, **kwargs):
        writes.append((get_hermes_home(), args, kwargs))

    adapter._client = SimpleNamespace(crypto=None, api=_raw_api(state, AsyncMock()), invite_user=invite_user)
    source = adapter.build_source(room, chat_name="Room", chat_type="group", user_id=actor)
    adapter._canonicalize(source)
    if multiplexed:
        runner._admit_primary_source(source, home)
    now = datetime.now()
    entry = SessionEntry("session-A", "id-A", now, now, origin=dataclasses.replace(source)) if resumed else None
    tokens = runner._set_session_env(build_session_context(source, runner.config, entry))
    try:
        raw = await asyncio.to_thread(
            registry.dispatch, "matrix_room_admin", {"action": "invite", "user_id": "@bob:server"},
        )
    finally:
        runner._clear_session_env(tokens)
    assert (json.loads(raw), writes) == (
        {"action": "invite", "room_id": room, "user_id": "@bob:server"},
        [(home, (room, "@bob:server"), {"reason": None})],
    )


ROOM, ACTOR, BOT = "!room:server", "@alice:server", "@bot:server"


def _bind_admin_room(power, *, members=(ACTOR, BOT, "@bob:server"), bot_membership="join",
                     room_version="11", creator="@creator:server", sender="@bob:server",
                     allowed_users=(ACTOR,), gate=None):
    """Bind a Matrix session for ACTOR in ROOM and return the adapter and its recorded writes."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    home = get_hermes_home()
    atomic_config_write(home / "config.yaml", {"platform_toolsets": {"matrix": ["matrix_admin"]}})
    adapter = object.__new__(MatrixAdapter)
    adapter._pin_state_lock = asyncio.Lock()
    adapter._user_id = BOT
    adapter._owner_profile = None
    adapter._joined_rooms = {ROOM} if bot_membership == "join" else set()
    adapter._allowed_room_ids = set()
    adapter._allowed_user_ids = set(allowed_users)
    adapter._dm_rooms = {}
    adapter._room_identities = {}
    adapter._room_identity_cached_at = {}
    adapter._is_dm_room = AsyncMock(return_value=len(members) == 2 and BOT in members)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._get_room_members = AsyncMock(return_value=set(members))
    adapter.set_authorization_check(lambda user, chat_type, chat_id: user in allowed_users)
    writes = []

    async def state(room_id, kind, state_key="", **kwargs):
        if kind == "m.room.member":
            return {"membership": bot_membership if state_key == BOT else "join"}
        if kind == "m.room.power_levels":
            return power
        if kind == "m.room.create":
            return {"type": "m.room.create", "sender": creator, "content": {"room_version": room_version}}
        return {}

    def recorder(method):
        async def mutate(*args, **kwargs):
            writes.append((method, args, kwargs))
            if gate is not None:
                gate[0].set()
                await gate[1].wait()
            return "!created:server" if method == "create_room" else "$redaction"
        return mutate

    client = SimpleNamespace(crypto=None)
    client.get_event = AsyncMock(return_value={"room_id": ROOM, "event_id": "$target", "sender": sender})
    client.api = _raw_api(state, client.get_event)
    client.get_account_data = AsyncMock(return_value={"@carol:server": ["!old:server"]})
    for method in ("create_room", "invite_user", "leave_room", "forget_room", "redact", "set_account_data",
                   "send_state_event"):
        setattr(client, method, recorder(method))
    adapter._client = client
    identity = RoutingIdentity("default", "default", home, home, multiplexed=False, transport=weakref.ref(adapter))
    tokens = set_session_vars(platform="matrix", chat_id=ROOM, user_id=ACTOR, session_key="session-A",
                              transport_adapter=adapter,
                              transport_loop=asyncio.get_running_loop(), routing_identity=identity)
    return adapter, writes, tokens


async def _dispatch_admin(tokens, args):
    try:
        return json.loads(await asyncio.to_thread(registry.dispatch, "matrix_room_admin", args))
    finally:
        clear_session_vars(tokens)


_REMOVE_REFUSAL = {"error": "Matrix requester lacks permission to remove the bot from this room",
                   "required": 50, "level": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["leave", "forget"])
@pytest.mark.parametrize(("room", "level", "room_version", "refusal"), [
    ("dm", 0, "11", None),
    ("group", 0, "11", _REMOVE_REFUSAL),
    ("group", 50, "11", None),
    ("group", 0, "12", None),
], ids=["dm", "group-below-kick", "group-at-kick", "group-v12-creator"])
async def test_removing_the_bot_needs_kick_power_outside_a_dm(action, room, level, room_version, refusal):
    pytest.importorskip("mautrix.types")
    others = () if room == "dm" else ("@bob:server",)
    members = (ACTOR, *others) + ((BOT,) if action == "leave" else ())
    power = {"users": {ACTOR: level, BOT: 100}}
    _, writes, tokens = _bind_admin_room(
        power, members=members, bot_membership="join" if action == "leave" else "leave",
        room_version=room_version, creator=ACTOR if room_version == "12" else "@creator:server",
    )
    result = await _dispatch_admin(tokens, {"action": action})
    method = "leave_room" if action == "leave" else "forget_room"
    expected_writes = [] if refusal else [(method, (ROOM,), {"reason": None, "raise_not_in_room": True}
                                           if action == "leave" else {})]
    assert (result, writes) == (refusal or {"action": action, "room_id": ROOM}, expected_writes)


_REDACT_REFUSAL = {"error": "Matrix requester lacks permission to redact this event", "required": 50, "level": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("sender", "power", "refusal"), [
    (ACTOR, {"users": {BOT: 100}, "events": {"m.room.redaction": 50}}, _REDACT_REFUSAL),
    (ACTOR, {"users": {ACTOR: 50, BOT: 100}, "events": {"m.room.redaction": 50}}, None),
    (ACTOR, {"users": {BOT: 100}, "events_default": 50}, _REDACT_REFUSAL),
    (ACTOR, {"users": {BOT: 100}}, None),
    ("@bob:server", {"users": {BOT: 100}, "redact": 0, "events": {"m.room.redaction": 50}}, _REDACT_REFUSAL),
    ("@bob:server", {"users": {BOT: 100}, "redact": 50}, _REDACT_REFUSAL),
    (BOT, {"users": {BOT: 100}, "redact": 50}, _REDACT_REFUSAL),
    (BOT, {"users": {ACTOR: 50, BOT: 100}, "redact": 50, "events": {"m.room.redaction": 50}}, None),
], ids=["own-below-event-level", "own-at-event-level", "own-below-events-default", "own-default-levels",
        "other-below-event-level", "other-below-redact", "bot-below-redact", "bot-at-both-levels"])
async def test_redaction_needs_the_redaction_event_level_and_redact_for_other_senders(sender, power, refusal):
    pytest.importorskip("mautrix.types")
    _, writes, tokens = _bind_admin_room(power, sender=sender)
    result = await _dispatch_admin(tokens, {"action": "redact", "event_id": "$target"})
    expected = refusal or {"action": "redact", "room_id": ROOM, "event_id": "$target",
                           "redaction_event_id": "$redaction"}
    assert (result, writes) == (expected, [] if refusal else [("redact", (ROOM, "$target"), {"reason": None})])


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "next_step"), [
    ("create", "Ask the user whether the room was created before retrying, so that no duplicate room is created"),
    ("invite", "Ask the user whether the invite arrived before retrying"),
    ("leave", "Ask the user whether the bot is still in the room before retrying"),
    ("forget", "Forget changes only the bot's account, so retrying it is harmless"),
    ("redact", "Read the event with matrix_read kind=event before retrying"),
], ids=["create", "invite", "leave", "forget", "redact"])
async def test_interrupt_after_the_write_was_sent_reports_an_unknown_outcome(action, next_step):
    pytest.importorskip("mautrix.types")
    from tools.interrupt import set_interrupt

    started, release = asyncio.Event(), asyncio.Event()
    adapter, writes, tokens = _bind_admin_room(
        {"users": {ACTOR: 100, BOT: 100}}, bot_membership="leave" if action == "forget" else "join",
        gate=(started, release),
    )
    worker = {}

    def dispatch():
        worker["id"] = threading.get_ident()
        return registry.dispatch("matrix_room_admin", {"action": action, "user_id": "@bob:server",
                                                       "event_id": "$target"})

    task = asyncio.create_task(asyncio.to_thread(dispatch))
    try:
        await asyncio.wait_for(started.wait(), timeout=_HANG_TIMEOUT)
        set_interrupt(True, worker["id"])
        result = json.loads(await asyncio.wait_for(task, timeout=_HANG_TIMEOUT))
    finally:
        release.set()
        set_interrupt(False, worker.get("id"))
        await asyncio.gather(task, return_exceptions=True)
        clear_session_vars(tokens)
    assert (result, len(writes), adapter._joined_rooms) == (
        {"error": "Matrix administration interrupted after the change was sent to the homeserver",
         "outcome": "unknown", "next_step": next_step},
        1, set() if action in {"leave", "forget"} else {ROOM},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("invite", "refusal"), [
    ([ACTOR], None),
    ([ACTOR, "@friend:server"], None),
    ([ACTOR, "@stranger:server", "@friend:server", "@other:elsewhere"],
     {"error": "Matrix room creation can invite only the requester and authorised users",
      "unauthorised": ["@stranger:server", "@other:elsewhere"]}),
], ids=["requester", "authorised", "unauthorised"])
async def test_created_rooms_invite_only_the_requester_and_authorised_users(invite, refusal):
    sdk = pytest.importorskip("mautrix.types")
    _, writes, tokens = _bind_admin_room({"users": {ACTOR: 100, BOT: 100}}, allowed_users=(ACTOR, "@friend:server"))
    result = await _dispatch_admin(tokens, {"action": "create", "invite": invite})
    expected_writes = [] if refusal else [("create_room", (), {
        "name": None, "topic": None, "invitees": invite, "is_direct": False,
        "preset": sdk.RoomCreatePreset.PRIVATE, "initial_state": [],
    })]
    assert (result, writes) == (
        refusal or {"action": "create", "room_id": "!created:server", "encrypted": False}, expected_writes,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invite", [[], [ACTOR], [ACTOR, "@friend:server"]], ids=["none", "one", "two"])
async def test_direct_rooms_have_one_invitee_and_are_recorded_in_m_direct(invite):
    sdk = pytest.importorskip("mautrix.types")
    adapter, writes, tokens = _bind_admin_room({"users": {ACTOR: 100, BOT: 100}},
                                               allowed_users=(ACTOR, "@friend:server"))
    result = await _dispatch_admin(tokens, {"action": "create", "invite": invite, "is_direct": True})
    if len(invite) != 1:
        assert (result, writes, adapter._dm_rooms) == (
            {"error": "is_direct requires exactly one invitee"}, [], {},
        )
        return
    assert (result, writes, adapter._dm_rooms) == (
        {"action": "create", "room_id": "!created:server", "encrypted": False},
        [("create_room", (), {"name": None, "topic": None, "invitees": invite, "is_direct": True,
                              "preset": sdk.RoomCreatePreset.PRIVATE, "initial_state": []}),
         ("set_account_data", ("m.direct", {"@carol:server": ["!old:server"], ACTOR: ["!created:server"]}), {})],
        {"!created:server": True},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "invite", "leave", "forget", "redact", "pin"])
async def test_profile_files_are_read_off_the_gateway_loop(action, monkeypatch):
    pytest.importorskip("mautrix.types")
    import agent.secret_scope
    from plugins.platforms.matrix import room_admin

    calls = []
    secret_reads = []
    enabled = room_admin._enabled
    build_secret_scope = agent.secret_scope.build_profile_secret_scope

    def counted(*args):
        calls.append(threading.get_ident())
        return enabled(*args)

    def recorded_secret_scope(*args, **kwargs):
        secret_reads.append(threading.get_ident())
        return build_secret_scope(*args, **kwargs)

    monkeypatch.setattr(room_admin, "_enabled", counted)
    monkeypatch.setattr(agent.secret_scope, "build_profile_secret_scope", recorded_secret_scope)
    _, writes, tokens = _bind_admin_room({"users": {ACTOR: 100, BOT: 100}},
                                         bot_membership="leave" if action == "forget" else "join")
    tool = "matrix_pin" if action == "pin" else "matrix_room_admin"
    try:
        raw = await asyncio.to_thread(registry.dispatch, tool, {"action": action, "user_id": "@bob:server",
                                                                "event_id": "$target"})
    finally:
        clear_session_vars(tokens)
    loop_thread = threading.get_ident()
    assert (json.loads(raw).get("error"), len(writes), len(calls), loop_thread in calls,
            bool(secret_reads), loop_thread in secret_reads) == (None, 1, 2, False, True, False)
