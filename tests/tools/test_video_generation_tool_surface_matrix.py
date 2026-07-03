"""Tool-surface routing matrix: every (provider, model, modality) combo.

This is the integration test for the question Teknium asked: regardless
of which provider+model the user picks and whether they pass an
image_url or not, does the tool surface route correctly to the right
endpoint with the right payload shape?

Drives ``_handle_video_generate(args)`` end-to-end — config write →
config read → registry lookup → provider.generate() → outbound HTTP/SDK
call. Stubs fal_client and httpx so we observe routing without hitting
the network.
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Any, Dict, List

import pytest
import yaml


@pytest.fixture(autouse=True)
def _reset_registry():
    from agent import video_gen_registry
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


@pytest.fixture
def matrix_env(tmp_path, monkeypatch):
    """Set up HERMES_HOME, stub fal_client + httpx, force plugin discovery."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    fal_calls: List[Dict[str, Any]] = []
    xai_calls: List[Dict[str, Any]] = []

    # fal_client stub
    fake_fal = types.ModuleType("fal_client")
    def _subscribe(endpoint, arguments=None, with_logs=False):
        fal_calls.append({"endpoint": endpoint, "arguments": arguments})
        return {"video": {"url": f"https://fake-fal/{endpoint.replace('/','_')}.mp4"}}
    fake_fal.subscribe = _subscribe  # type: ignore

    class _FalHandle:
        def __init__(self, result):
            self._result = result
        def get(self):
            return self._result

    def _submit(endpoint, arguments=None, headers=None):
        fal_calls.append({"endpoint": endpoint, "arguments": arguments})
        return _FalHandle({"video": {"url": f"https://fake-fal/{endpoint.replace('/','_')}.mp4"}})
    fake_fal.submit = _submit  # type: ignore

    monkeypatch.setitem(__import__("sys").modules, "fal_client", fake_fal)

    # httpx stub for xAI
    import httpx
    class _Resp:
        def __init__(self, p, s=200):
            self.status_code = s
            self._p = p
            self.text = json.dumps(p)
        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore
        def json(self):
            return self._p
    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def post(self, url, headers=None, json=None, timeout=None):
            xai_calls.append({"url": url, "json": json})
            return _Resp({"request_id": "req-1"})
        async def get(self, url, headers=None, timeout=None):
            payload = xai_calls[-1]["json"]
            storage_options = payload.get("storage_options") or {}
            return _Resp({
                "status": "done",
                "video": {
                    "url": "https://xai-cdn/out.mp4",
                    "duration": 8,
                    "file_output": {
                        "file_id": "file-123",
                        "filename": storage_options.get("filename", "out.mp4"),
                        "public_url": "https://xai-files.example/out.mp4",
                        "public_url_expires_at": 1234567890,
                    },
                },
                "model": payload.get("model", "grok-imagine-video"),
            })
    import plugins.video_gen.xai as xai_plugin
    monkeypatch.setattr(xai_plugin.httpx, "AsyncClient", lambda: _Client())
    async def _no_sleep(*a, **k): return None
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    # Reset FAL plugin's lazy fal_client cache so it picks up the stub
    from plugins.video_gen import fal as fal_plugin
    fal_plugin._fal_client = None

    # Force discovery
    from hermes_cli.plugins import _ensure_plugins_discovered
    _ensure_plugins_discovered(force=True)

    return tmp_path, fal_calls, xai_calls


def _invoke_tool(home, cfg: dict, args: dict, tool_name: str = "video_generate") -> dict:
    """Write config, invoke the registered tool handler, return parsed JSON."""
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    import hermes_cli.config as cfg_mod
    if hasattr(cfg_mod, "_invalidate_load_config_cache"):
        cfg_mod._invalidate_load_config_cache()

    from tools.registry import discover_builtin_tools, registry
    if tool_name not in registry._tools:
        discover_builtin_tools()
    handler = registry._tools[tool_name].handler
    return json.loads(handler(args))


# ─────────────────────────────────────────────────────────────────────────
# FAL: every family × {text-only, text+image}
# ─────────────────────────────────────────────────────────────────────────

# We parametrize over the catalog so the test discovers new families
# automatically. If someone adds 'sora-2' to FAL_FAMILIES, this matrix
# picks it up — no test changes needed beyond confirming the endpoints.
def _all_fal_families():
    from plugins.video_gen.fal import FAL_FAMILIES
    return list(FAL_FAMILIES.keys())


@pytest.mark.parametrize("family_id", _all_fal_families())
def test_fal_text_only_routes_to_text_endpoint(matrix_env, family_id):
    home, fal_calls, _ = matrix_env
    from plugins.video_gen.fal import FAL_FAMILIES

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "fal", "model": family_id}},
        {"prompt": "a dog running"},
    )

    assert result["success"] is True, f"{family_id}: {result.get('error')}"
    assert result["modality"] == "text"
    assert result["provider"] == "fal"

    # Outbound endpoint must be the family's text endpoint
    assert len(fal_calls) == 1
    endpoint = fal_calls[0]["endpoint"]
    assert endpoint == FAL_FAMILIES[family_id]["text_endpoint"]

    # Payload must NOT contain any image-shaped key
    payload = fal_calls[0]["arguments"] or {}
    image_keys = [k for k in payload if "image" in k and "url" in k]
    assert not image_keys, f"{family_id} text-only leaked image keys: {image_keys}"


@pytest.mark.parametrize("family_id", _all_fal_families())
def test_fal_text_plus_image_routes_to_image_endpoint(matrix_env, family_id):
    home, fal_calls, _ = matrix_env
    from plugins.video_gen.fal import FAL_FAMILIES

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "fal", "model": family_id}},
        {"prompt": "animate this dog", "image_url": "https://example.com/dog.png"},
    )

    assert result["success"] is True, f"{family_id}: {result.get('error')}"
    assert result["modality"] == "image"
    assert result["provider"] == "fal"

    # Outbound endpoint must be the family's image endpoint
    assert len(fal_calls) == 1
    endpoint = fal_calls[0]["endpoint"]
    assert endpoint == FAL_FAMILIES[family_id]["image_endpoint"]

    # Payload must contain the right image key (may be image_url or
    # start_image_url depending on the family's image_param_key)
    payload = fal_calls[0]["arguments"] or {}
    expected_image_key = FAL_FAMILIES[family_id].get("image_param_key") or "image_url"
    assert payload.get(expected_image_key) == "https://example.com/dog.png", (
        f"{family_id} text+image missing {expected_image_key} in payload "
        f"(keys: {sorted(payload.keys())})"
    )


# ─────────────────────────────────────────────────────────────────────────
# xAI: text-only / text+image both go to /videos/generations
# (xAI uses one endpoint with an optional 'image' field, not separate URLs)
# ─────────────────────────────────────────────────────────────────────────

def test_xai_text_only_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {"prompt": "a dog running"},
    )
    assert result["success"] is True
    assert result["modality"] == "text"
    assert result["provider"] == "xai"

    assert len(xai_calls) == 1
    assert xai_calls[0]["url"].endswith("/videos/generations")
    payload = xai_calls[0]["json"] or {}
    assert payload["model"] == "grok-imagine-video"
    assert "image" not in payload
    assert "reference_images" not in payload
    assert payload["storage_options"]["public_url"] is True
    assert "expires_after" not in payload["storage_options"]
    assert result["video"] == "https://xai-files.example/out.mp4"
    assert result["public_url"] == "https://xai-files.example/out.mp4"
    assert result.get("temporary_url") == "https://xai-cdn/out.mp4"


def test_xai_text_plus_image_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {"prompt": "animate this", "image_url": "https://example.com/img.png"},
    )
    assert result["success"] is True
    assert result["modality"] == "image"
    assert result["provider"] == "xai"

    assert len(xai_calls) == 1
    assert xai_calls[0]["url"].endswith("/videos/generations")
    payload = xai_calls[0]["json"] or {}
    assert payload["model"] == "grok-imagine-video-1.5"
    assert payload["image"] == {"url": "https://example.com/img.png"}


def test_xai_image_to_video_rejects_bare_file_id_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "animate this robot waving",
            "image_url": "file_03eb65b1-aa97-482f-9ef0-b04f9172ea00",
        },
    )
    assert result["success"] is False
    assert result.get("error_type") == "invalid_image_url"
    assert len(xai_calls) == 0


def test_xai_reference_to_video_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "put the jacket from the reference on the runway model",
            "reference_image_urls": [
                "https://example.com/model.png",
                "https://example.com/jacket.png",
            ],
            "duration": 15,
        },
    )
    assert result["success"] is True
    assert result["modality"] == "reference"
    assert result["provider"] == "xai"

    payload = xai_calls[0]["json"] or {}
    assert xai_calls[0]["url"].endswith("/videos/generations")
    assert payload["model"] == "grok-imagine-video"
    assert payload["duration"] == 10
    assert payload["reference_images"] == [
        {"url": "https://example.com/model.png"},
        {"url": "https://example.com/jacket.png"},
    ]


def test_xai_reference_to_video_rejects_bare_file_ids_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "use these references for a robot product shot",
            "reference_image_urls": [
                "file_03eb65b1-aa97-482f-9ef0-b04f9172ea00",
                "file_54b48d6d-28ad-4982-9d72-bd3ac677c9bc",
            ],
        },
    )
    assert result["success"] is False
    assert result.get("error_type") == "invalid_reference_image_urls"
    assert len(xai_calls) == 0


def test_xai_video_edit_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "make the sky stormy",
            "video_url": "https://example.com/source.mp4",
        },
        tool_name="xai_video_edit",
    )
    assert result["success"] is True
    assert result["modality"] == "edit"

    payload = xai_calls[0]["json"] or {}
    assert xai_calls[0]["url"].endswith("/videos/edits")
    assert payload["model"] == "grok-imagine-video"
    assert payload["video"] == {"url": "https://example.com/source.mp4"}
    assert "duration" not in payload
    assert "aspect_ratio" not in payload
    assert "resolution" not in payload


def test_xai_video_extend_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "the camera pulls back to reveal the city",
            "video_url": "https://example.com/source.mp4",
            "duration": 15,
        },
        tool_name="xai_video_extend",
    )
    assert result["success"] is True
    assert result["modality"] == "extend"

    payload = xai_calls[0]["json"] or {}
    assert xai_calls[0]["url"].endswith("/videos/extensions")
    assert payload["model"] == "grok-imagine-video"
    assert payload["video"] == {"url": "https://example.com/source.mp4"}
    assert payload["duration"] == 10


def test_xai_video_edit_rejects_bare_file_id_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "make the sky stormy",
            "video_url": "file-123",
        },
        tool_name="xai_video_edit",
    )
    assert result.get("success") is not True
    assert "error" in result
    assert "url" in result["error"].lower()
    assert len(xai_calls) == 0


def test_xai_video_extend_rejects_bare_file_id_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "continue into a sunrise",
            "video_url": "file_25ac1c31-d6d8-48b2-8504-a97d282310c4",
        },
        tool_name="xai_video_extend",
    )
    assert result.get("success") is not True
    assert "error" in result
    assert "url" in result["error"].lower()
    assert len(xai_calls) == 0


def test_xai_explicit_model_override_via_tool_surface(matrix_env):
    home, _, xai_calls = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "xai"}},
        {
            "prompt": "animate this",
            "image_url": "https://example.com/img.png",
            "model": "grok-imagine-video",
        },
    )
    assert result["success"] is True

    payload = xai_calls[0]["json"] or {}
    assert payload["model"] == "grok-imagine-video"
    assert payload["image"] == {"url": "https://example.com/img.png"}


# ─────────────────────────────────────────────────────────────────────────
# tool-level `model` arg overrides config
# ─────────────────────────────────────────────────────────────────────────

def test_tool_model_arg_overrides_config(matrix_env):
    """When the tool call passes model=, it wins over video_gen.model in config."""
    home, fal_calls, _ = matrix_env

    # Config picks pixverse-v6, but tool call says veo3.1
    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "fal", "model": "pixverse-v6"}},
        {"prompt": "a dog", "model": "veo3.1"},
    )

    assert result["success"] is True
    assert result["model"] == "veo3.1"
    # Outbound endpoint reflects the override, not config
    assert fal_calls[0]["endpoint"] == "fal-ai/veo3.1"


def test_tool_model_arg_with_image_url_routes_to_override_image_endpoint(matrix_env):
    """model= override on text+image goes to the override family's image endpoint."""
    home, fal_calls, _ = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "fal", "model": "pixverse-v6"}},
        {
            "prompt": "animate this",
            "image_url": "https://example.com/i.png",
            "model": "kling-v3-4k",
        },
    )

    assert result["success"] is True
    assert result["model"] == "kling-v3-4k"
    assert fal_calls[0]["endpoint"] == "fal-ai/kling-video/v3/4k/image-to-video"
    # Kling 4K uses start_image_url
    assert fal_calls[0]["arguments"].get("start_image_url") == "https://example.com/i.png"
    assert "image_url" not in fal_calls[0]["arguments"]
