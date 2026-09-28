"""Native tool deliveries mirror the canonical destination in the owning profile."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource, SessionStore
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools.send_message_tool import send_message_tool


class _MissingEncryption(Exception):
    errcode = "M_NOT_FOUND"


@pytest.mark.parametrize("target_kind", ["alias", "home"])
@pytest.mark.parametrize("payload_kind", ["text", "media"])
@pytest.mark.parametrize("rejected", [False, True])
@pytest.mark.parametrize("alice_exists", [False, True])
def test_alias_send_mirrors_the_canonical_participant_thread(
    tmp_path, monkeypatch, target_kind, payload_kind, rejected, alice_exists
):
    import hermes_state
    import model_tools
    from agent.secret_scope import set_multiplex_active
    from gateway.session_context import set_session_vars, clear_session_vars

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(model_tools, "_run_async", asyncio.run)
    monkeypatch.setattr("plugins.platforms.matrix.adapter.time.time", lambda: 1_800_000_000)
    runner = object.__new__(GatewayRunner)
    runner._primary_profile_name = "default"
    runner._profile_adapters = {}
    owners = {}
    alias = "#reports:remote.test"
    for profile in ("default", "secondary"):
        target_home = home if profile == "default" else home / "profiles" / profile
        target_home.mkdir(parents=True)
        (target_home / "config.yaml").write_text(json.dumps({
            "thread_sessions_per_user": True,
            "platforms": {"matrix": {
                "enabled": True, "token": "token",
                "home_channel": {"platform": "matrix", "chat_id": f"{alias}/$root"},
                "extra": {"e2ee_mode": "off"},
            }},
        }), encoding="utf-8")
        with _profile_runtime_scope(target_home, {}):
            config = load_gateway_config()
            config.multiplex_profiles = True
            adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"e2ee_mode": "off"}))
            room = f"!{profile}:remote.test"
            adapter._client = SimpleNamespace(
                resolve_room_alias=AsyncMock(
                    return_value=SimpleNamespace(room_id=room, servers=["route.test"])
                ),
                join_room=AsyncMock(return_value=room),
                get_state_event=AsyncMock(side_effect=_MissingEncryption()),
                send_message_event=AsyncMock(return_value="$sent"),
                upload_media=AsyncMock(return_value="mxc://remote.test/report"),
                crypto=None,
            )
            adapter._refresh_dm_cache = AsyncMock()
            adapter._user_id = "@bot:remote.test"
            adapter._get_room_members = AsyncMock(return_value={adapter._user_id, "@alice:remote.test", "@bob:remote.test"})
            adapter._get_room_member_profiles = AsyncMock(return_value=None)
            adapter._get_room_state_value = AsyncMock(return_value=None)
            if rejected:
                if payload_kind == "text":
                    adapter._client.send_message_event.side_effect = ValueError("send rejected")
                else:
                    adapter._client.upload_media.side_effect = ValueError("upload rejected")
            adapter.gateway_runner = runner
            adapter.set_owner_profile(profile)
            adapter.set_session_store(SessionStore(target_home / "sessions", config))
            source = SessionSource(
                platform=Platform.MATRIX, chat_id=room, thread_id="$root",
                chat_type="group", user_id="@alice:remote.test", profile=profile,
            )
            alice = adapter._session_store.get_or_create_session(source) if alice_exists else None
            bob = adapter._session_store.get_or_create_session(SessionSource(
                platform=Platform.MATRIX, chat_id=room, thread_id="$root",
                chat_type="group", user_id="@bob:remote.test", profile=profile,
            ))
            owners[profile] = (target_home, adapter, room, alice, bob)
    runner.adapters = {Platform.MATRIX: owners["default"][1]}
    runner._profile_adapters["secondary"] = {Platform.MATRIX: owners["secondary"][1]}
    runner.config = config
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    set_multiplex_active(True)
    attachment = tmp_path / "report.txt"
    attachment.write_text("Report attachment", encoding="utf-8")
    for index, profile in enumerate(("default", "secondary", "default")):
        target_home, adapter, room, alice, bob = owners[profile]
        with _profile_runtime_scope(target_home, {}):
            if alice is not None:
                adapter._session_store._db.update_system_prompt(alice.session_id, "Cached system prefix")
                adapter._session_store.append_to_transcript(alice.session_id, {
                    "role": "user", "content": f"Request report {index} for {profile}",
                })
            before = deepcopy(adapter._session_store.load_transcript(alice.session_id)) if alice else []
            tokens = set_session_vars(platform="matrix", user_id="@alice:remote.test")
            try:
                brief = f"Report {index} for {profile}"
                message = brief if payload_kind == "text" else f"{brief}\nMEDIA:{attachment}"
                target = "matrix" if target_kind == "home" else f"matrix:{alias}/$root"
                result = json.loads(send_message_tool({
                    "target": target,
                    "message": message,
                }))
            finally:
                clear_session_vars(tokens)
            if alice is not None:
                assert adapter._session_store._db.get_session(alice.session_id)["system_prompt"] == "Cached system prefix"
            if rejected:
                original_target = f"{alias}/$root" if target_kind == "home" else target
                assert original_target in result["error"]
                assert ("send rejected" if payload_kind == "text" else "upload rejected") in result["error"]
                if alice is not None:
                    assert adapter._session_store.load_transcript(alice.session_id) == before
                assert adapter._session_store.load_transcript(bob.session_id) == []
                continue
            assert result.get("success"), result
            store = adapter._session_store
            if alice is None:
                assert not result.get("mirrored"), result
                assert store.load_transcript(bob.session_id) == []
                assert list(store._entries) == [bob.session_key]
                continue
            assert [(turn["role"], turn["content"]) for turn in store.load_transcript(alice.session_id)] == [
                turn
                for prior in range(index + 1)
                if ("secondary" if prior == 1 else "default") == profile
                for turn in [
                    ("user", f"Request report {prior} for {profile}"),
                    ("assistant", f"Report {prior} for {profile}"),
                ]
            ]
            assert store.load_transcript(bob.session_id) == []
            args = adapter._client.send_message_event.await_args.args
            assert (str(args[0]), args[2]["m.relates_to"]) == (
                room, {
                    "rel_type": "m.thread", "event_id": "$root", "is_falling_back": True,
                    "m.in_reply_to": {"event_id": "$sent" if payload_kind == "media" or index == 2 else "$root"},
                },
            )


@pytest.mark.parametrize("qualified", [False, True])
def test_qualified_aliases_and_short_directory_names_use_distinct_routes(monkeypatch, qualified):
    from tools.send_message_targets import resolve_send_target
    from unittest.mock import Mock

    alias = "#reports:remote.test"
    directory_room = "!directory:remote.test"
    directory = Mock(return_value=f"{directory_room}/$root")
    monkeypatch.setattr("gateway.channel_directory.resolve_channel_name", directory)
    target = f"{alias}/$root" if qualified else "#reports"
    assert resolve_send_target("matrix", target) == (
        alias if qualified else directory_room, "$root", None,
    )
    if qualified:
        directory.assert_not_called()
        return
    directory.assert_called_once_with("matrix", target)
