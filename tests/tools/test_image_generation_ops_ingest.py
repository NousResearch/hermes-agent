import json
import sys
import types

from tools import image_generation_tool as image_tool


class _FakeProvider:
    name = "openai-codex"

    def generate(self, **kwargs):
        return {
            "success": True,
            "image": "/tmp/generated.png",
            "provider": self.name,
            "model": kwargs.get("model"),
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
        }


def test_plugin_success_reports_generate_image_ingest(monkeypatch):
    captured = []

    monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "openai-codex")
    monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: "gpt-image-2-high")
    monkeypatch.setattr(image_tool, "_report_image_success_to_agent_ops", lambda result, *, prompt, aspect_ratio: captured.append((result, prompt, aspect_ratio)))

    registry_module = types.ModuleType("agent.image_gen_registry")
    registry_module.get_provider = lambda name: _FakeProvider()
    plugins_module = types.ModuleType("hermes_cli.plugins")
    plugins_module._ensure_plugins_discovered = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "agent.image_gen_registry", registry_module)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins_module)

    result = json.loads(image_tool._dispatch_to_plugin_provider("draw a lighthouse", "portrait"))

    assert result["success"] is True
    assert captured == [(result, "draw a lighthouse", "portrait")]


def test_reporter_is_best_effort_when_disabled(monkeypatch):
    monkeypatch.delenv("IMAGE_GEN_AGENT_OPS_INGEST_ENABLED", raising=False)
    monkeypatch.delenv("OPS_CONSOLE_SOCKET", raising=False)
    monkeypatch.delenv("INGEST_TOKEN", raising=False)

    calls = []
    monkeypatch.setattr(image_tool, "_post_agent_ops_ingest", lambda payload, *, socket_path, token, idempotency_key: calls.append(payload))

    image_tool._report_image_success_to_agent_ops(
        {"success": True, "image": "/tmp/generated.png", "provider": "openai-codex", "model": "gpt-image-2-high"},
        prompt="draw a lighthouse",
        aspect_ratio="portrait",
    )

    assert calls == []


def test_payload_marks_hermes_as_producer_not_legacy_generate_image(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-123")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "feishu")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-456")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "thread-789")

    payload, idempotency_key = image_tool._build_agent_ops_image_ingest_payload(
        {"success": True, "image": "/tmp/generated.png", "provider": "openai-codex", "model": "gpt-image-2-high"},
        prompt="draw a lighthouse",
        aspect_ratio="portrait",
    )

    assert payload["projectKey"] == "generate-image"
    assert payload["runId"] == "hermes-img-bot-sess-123"
    assert idempotency_key == "generate-image:hermes-img-bot-sess-123"
    item = payload["items"][0]
    assert item["type"] == "generated-image"
    assert item["manualReviewStatus"] == "pending"
    assert item["securityReviewStatus"] == "skipped"
    assert item["data"]["source"] == "hermes-img-bot-image-generate"
    assert item["data"]["legacyGenerateImageServiceStatus"] == "offline-standby"
    assert item["data"]["legacyGenerateImageServiceNote"] == "Legacy generate-image service is offline and retained only as standby; do not route production image generation through it."
    assert item["data"]["platform"] == "feishu"
    assert item["data"]["chatId"] == "chat-456"
    assert item["data"]["threadId"] == "thread-789"
