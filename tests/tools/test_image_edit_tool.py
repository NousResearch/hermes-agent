import json

from tools import image_edit_tool


class _Provider:
    name = "fake-edit"

    def __init__(self):
        self.calls = []

    def supports_edit(self):
        return True

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "success": True,
            "image": "/tmp/edited.png",
            "model": kwargs.get("model", "fake-model"),
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "provider": self.name,
        }


class _UnsupportedProvider:
    name = "fake-no-edit"

    def supports_edit(self):
        return False

    def generate(self, *args, **kwargs):
        raise AssertionError("generate should not be called for image_edit")


def test_schema_accepts_prompt_and_reference_images():
    schema = image_edit_tool.IMAGE_EDIT_SCHEMA
    props = schema["parameters"]["properties"]
    assert schema["name"] == "image_edit"
    assert "prompt" in schema["parameters"]["required"]
    assert "image" in props
    assert "images" in props
    assert "reference_images" in props
    assert "references" in props
    assert "aspect_ratio" in props
    assert "9:16" in props["aspect_ratio"]["enum"]


def test_handler_requires_prompt_and_image():
    missing_prompt = image_edit_tool._handle_image_edit({"image": "https://example.test/a.png"})
    assert "prompt is required" in missing_prompt

    missing_image = image_edit_tool._handle_image_edit({"prompt": "make it blue"})
    assert "image is required" in missing_image


def test_handler_routes_to_configured_provider(monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "fake-edit")
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_model", lambda: "configured-model")
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: provider if name == "fake-edit" else None)

    result = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "make it blue",
        "images": ["https://example.test/source.png"],
        "aspect_ratio": "1:1",
        "size": "1024x1024",
        "quality_tier": "high",
    }))

    assert result["success"] is True
    assert result["provider"] == "fake-edit"
    assert provider.calls == [{
        "prompt": "make it blue",
        "image": "https://example.test/source.png",
        "aspect_ratio": "1:1",
        "model": "configured-model",
        "size": "1024x1024",
        "quality_tier": "high",
        "images": ["https://example.test/source.png"],
    }]


def test_handler_preserves_images_order_to_provider(monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "fake-edit")
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: provider if name == "fake-edit" else None)

    images = [
        "https://example.test/01.png",
        "https://example.test/02.png",
        "https://example.test/03.png",
    ]
    result = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "compose in this order",
        "images": images,
    }))

    assert result["success"] is True
    assert provider.calls[0]["image"] == images[0]
    assert provider.calls[0]["images"] == images


def test_handler_preserves_reference_images_order_to_provider(monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "fake-edit")
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: provider if name == "fake-edit" else None)

    reference_images = [
        "https://example.test/ref-01.png",
        "https://example.test/ref-02.png",
        "https://example.test/ref-01.png",
    ]
    result = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "compose in this reference order",
        "reference_images": reference_images,
    }))

    assert result["success"] is True
    assert provider.calls[0]["image"] == reference_images[0]
    assert provider.calls[0]["images"] == reference_images


def test_handler_preserves_references_alias_order_to_provider(monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "fake-edit")
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: provider if name == "fake-edit" else None)

    references = [
        "https://example.test/url1.png",
        "https://example.test/url2.png",
        "https://example.test/url1.png",
    ]
    result = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "compose in references order",
        "references": references,
    }))

    assert result["success"] is True
    assert provider.calls[0]["image"] == references[0]
    assert provider.calls[0]["images"] == references


def test_handler_reports_provider_without_edit(monkeypatch):
    provider = _UnsupportedProvider()
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "fake-no-edit")
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: provider if name == "fake-no-edit" else None)

    result = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "make it blue",
        "image": "https://example.test/source.png",
    }))

    assert result["success"] is False
    assert result["error_type"] == "unsupported"
    assert "does not support image editing" in result["error"]


def test_handler_reports_unconfigured_provider(monkeypatch):
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: None)

    result = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "make it blue",
        "image": "https://example.test/source.png",
    }))

    assert result["success"] is False
    assert result["error_type"] == "provider_not_configured"


def test_check_requirements_only_when_provider_supports_edit(monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "fake-edit")
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: provider if name == "fake-edit" else None)

    assert image_edit_tool.check_image_edit_requirements() is True


def test_check_requirements_rejects_non_edit_provider(monkeypatch):
    provider = _UnsupportedProvider()
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "fake-no-edit")
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: provider if name == "fake-no-edit" else None)

    assert image_edit_tool.check_image_edit_requirements() is False


def test_image_edit_is_opt_in_image_gen_tool_not_core():
    import toolsets

    assert "image_edit" in toolsets.TOOLSETS["image_gen"]["tools"]
    assert "image_edit" not in toolsets._HERMES_CORE_TOOLS


def test_legacy_image_tools_include_image_edit():
    import model_tools

    assert "image_edit" in model_tools._LEGACY_TOOLSET_MAP["image_tools"]
