from __future__ import annotations

import json
import sys
from types import SimpleNamespace


class _FakeImages:
    def __init__(self):
        self.generate_kwargs = None
        self.edit_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(data=[SimpleNamespace(b64_json="aGVsbG8=", revised_prompt="revised")])

    def edit(self, **kwargs):
        self.edit_kwargs = kwargs
        return SimpleNamespace(data=[SimpleNamespace(b64_json="aGVsbG8=", revised_prompt=None)])


def _install_fake_openai(monkeypatch, images):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.images = images

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    return created


def test_openai_provider_forwards_generate_parameters(monkeypatch, tmp_path):
    from plugins.image_gen import openai as openai_image

    images = _FakeImages()
    created = _install_fake_openai(monkeypatch, images)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_image, "_resolve_model", lambda: ("gpt-image-2-medium", {"quality": "medium"}))
    monkeypatch.setattr(openai_image, "save_b64_image", lambda *a, **k: tmp_path / "out.webp")

    result = openai_image.OpenAIImageGenProvider().generate(
        "draw a cat",
        "landscape",
        size="2560x1440",
        quality="high",
        n=2,
        background="opaque",
        output_format="webp",
        output_compression=72,
        moderation="low",
        seed=123,
    )

    assert created == [{}]
    assert images.generate_kwargs == {
        "model": "gpt-image-2",
        "prompt": "draw a cat",
        "size": "2560x1440",
        "n": 2,
        "quality": "high",
        "background": "opaque",
        "output_format": "webp",
        "output_compression": 72,
        "moderation": "low",
    }
    assert result["success"] is True
    assert result["size"] == "2560x1440"
    assert result["quality"] == "high"
    assert result["output_format"] == "webp"
    assert result["revised_prompt"] == "revised"


def test_openai_provider_uses_edit_when_image_is_supplied(monkeypatch, tmp_path):
    from plugins.image_gen import openai as openai_image

    source = tmp_path / "input.png"
    source.write_bytes(b"fake image")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"fake mask")

    images = _FakeImages()
    _install_fake_openai(monkeypatch, images)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_image, "_resolve_model", lambda: ("gpt-image-2-medium", {"quality": "medium"}))
    monkeypatch.setattr(openai_image, "save_b64_image", lambda *a, **k: tmp_path / "edited.png")

    result = openai_image.OpenAIImageGenProvider().generate(
        "make the jacket red",
        "square",
        image=str(source),
        mask=str(mask),
        input_fidelity="high",
        size="1024x1024",
        quality="high",
        n=1,
        background="opaque",
        output_format="png",
        output_compression=90,
    )

    assert images.generate_kwargs is None
    assert images.edit_kwargs["model"] == "gpt-image-2"
    assert images.edit_kwargs["prompt"] == "make the jacket red"
    assert images.edit_kwargs["image"].name == str(source)
    assert images.edit_kwargs["image"].closed is True
    assert images.edit_kwargs["mask"].name == str(mask)
    assert images.edit_kwargs["mask"].closed is True
    assert images.edit_kwargs["input_fidelity"] == "high"
    assert images.edit_kwargs["size"] == "1024x1024"
    assert images.edit_kwargs["quality"] == "high"
    assert images.edit_kwargs["background"] == "opaque"
    assert result["operation"] == "edit"


def test_custom_provider_uses_openai_compatible_images_api(monkeypatch, tmp_path):
    from tools import image_generation_tool
    import hermes_cli.runtime_provider as runtime_provider

    images = _FakeImages()
    created = _install_fake_openai(monkeypatch, images)
    monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "custom:yuna")
    monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: "gpt-image-2")

    def fake_named_custom(name):
        assert name == "custom:yuna"
        return {
            "name": "yuna",
            "base_url": "https://yuna.example/v1",
            "api_key": "sk-yuna",
        }

    monkeypatch.setattr(runtime_provider, "_get_named_custom_provider", fake_named_custom)
    monkeypatch.setattr(image_generation_tool, "save_b64_image", lambda *a, **k: tmp_path / "custom.png", raising=False)

    payload = image_generation_tool._dispatch_to_plugin_provider(
        "draw a cat",
        "square",
        {"size": "1024x1024", "quality": "high", "output_format": "png"},
    )

    result = json.loads(payload)
    assert result["success"] is True
    assert result["message"] == "Image generation completed"
    assert result["parameters"] == {
        "prompt": "draw a cat",
        "aspect_ratio": "square",
        "size": "1024x1024",
        "quality": "high",
        "output_format": "png",
        "model": "gpt-image-2",
    }
    assert result["provider"] == "custom:yuna"
    assert created == [{"api_key": "sk-yuna", "base_url": "https://yuna.example/v1"}]
    assert images.generate_kwargs["model"] == "gpt-image-2"
    assert images.generate_kwargs["prompt"] == "draw a cat"
    assert images.generate_kwargs["size"] == "1024x1024"
    assert images.generate_kwargs["quality"] == "high"
