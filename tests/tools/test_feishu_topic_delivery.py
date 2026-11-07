"""Feishu topic anchors survive session discovery and plugin delivery (#37509)."""

import asyncio
import json
from importlib import import_module
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("store", ["json", "db"])
@pytest.mark.parametrize("named", [False, True], ids=["explicit", "named"])
@pytest.mark.parametrize("with_document", [False, True], ids=["text", "document"])
def test_session_topic_anchor_reaches_reply_api(tmp_path, monkeypatch, store, named, with_document):
    from gateway.channel_directory import build_channel_directory, load_directory
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    from hermes_state import SessionDB
    from tools.send_message_tool import send_message_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json")
    discover_plugins()
    entry = platform_registry.get("feishu")
    assert entry is not None and entry.standalone_sender_fn is not None
    adapter_type = import_module(entry.standalone_sender_fn.__module__).FeishuAdapter
    platform = Platform("feishu")
    origin = {
        "platform": "feishu", "chat_id": "oc_chat", "chat_name": "Agent topics",
        "thread_id": "omt_topic", "message_id": "om_root",
    }
    if store == "json":
        sessions = tmp_path / "sessions"
        sessions.mkdir(exist_ok=True)
        (sessions / "sessions.json").write_text(json.dumps({
            "feishu-topic": {"origin": origin, "chat_type": "group"},
        }), encoding="utf-8")
    else:
        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session("feishu-topic", "feishu", user_id="ou_user")
            db.record_gateway_session_peer(
                "feishu-topic", source="feishu", user_id="ou_user",
                session_key="agent:main:feishu:group:oc_chat:omt_topic",
                chat_id="oc_chat", chat_type="group", thread_id="omt_topic",
                display_name="Agent topics", origin_json=json.dumps(origin),
            )
        finally:
            db.close()

    asyncio.run(build_channel_directory({platform: SimpleNamespace()}))
    config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda *a, **kw: True)
    requests = []

    def reply(request):
        requests.append(("reply", request))
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_reply"))

    def create(request):
        requests.append(("create", request))
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_created"))

    def upload(request):
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(file_key="file_test"))

    def list_messages(request):
        requests.append(("list", request))
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=[]))

    client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message=SimpleNamespace(reply=reply, create=create, list=list_messages),
        file=SimpleNamespace(create=upload),
    )))
    monkeypatch.setattr(adapter_type, "_build_lark_client", lambda self, domain: client)
    target = "Agent topics / topic omt_topic (group)" if named else "oc_chat:omt_topic"
    message = "hello topic"
    if with_document:
        document = tmp_path / "report.pdf"
        document.write_bytes(b"%PDF-1.4 fixture")
        message += f"\nMEDIA:{document}"

    result = json.loads(send_message_tool({"target": f"feishu:{target}", "message": message}))

    assert result.get("success"), result
    assert requests and all(kind == "reply" for kind, _ in requests), requests
    assert {request.message_id for _, request in requests} == {origin["message_id"]}
    assert all(request.request_body.reply_in_thread for _, request in requests)
    assert len(requests) == (2 if with_document else 1)
    assert load_directory()["platforms"]["feishu"][0]["message_id"] == origin["message_id"]


@pytest.mark.parametrize("anchor_source", ["directory", "caller", "missing"])
def test_topic_metadata_precedence_and_unlisted_fallback(tmp_path, monkeypatch, anchor_source):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    from tools.send_message_tool import _send_to_platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "channel_directory.json"
    monkeypatch.setattr("gateway.channel_directory.DIRECTORY_PATH", path)
    if anchor_source != "missing":
        path.write_text(json.dumps({"platforms": {"feishu": [{
            "id": "oc_chat:omt_topic", "name": "Topic", "thread_id": "omt_topic",
            "message_id": "om_old", "reply_to_message_id": "om_directory",
        }]}}), encoding="utf-8")
    discover_plugins()
    entry = platform_registry.get("feishu")
    assert entry is not None and entry.standalone_sender_fn is not None
    adapter_type = import_module(entry.standalone_sender_fn.__module__).FeishuAdapter
    requests = []

    def deliver(kind, request):
        requests.append((kind, request))
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_sent"))

    client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(
        reply=lambda request: deliver("reply", request),
        create=lambda request: deliver("create", request),
    ))))
    monkeypatch.setattr(adapter_type, "_build_lark_client", lambda self, domain: client)
    metadata = {"reply_to_message_id": "om_caller"} if anchor_source == "caller" else None
    original_metadata = dict(metadata) if metadata is not None else None
    result = asyncio.run(_send_to_platform(
        "feishu", PlatformConfig(), "oc_chat", "hello", thread_id="omt_topic", send_metadata=metadata,
    ))

    assert result.get("success"), result
    assert metadata == original_metadata
    assert len(requests) == 1
    kind, request = requests[0]
    if anchor_source == "missing":
        assert kind == "create"
        assert request.request_body.receive_id == "omt_topic"
        assert request.receive_id_type == "thread_id"
    else:
        assert kind == "reply"
        assert request.message_id == ("om_caller" if anchor_source == "caller" else "om_directory")
        assert request.request_body.reply_in_thread
