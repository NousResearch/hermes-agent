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


def _t2v_fal_families():
    """Families that advertise a text-to-video endpoint."""
    from plugins.video_gen.fal import FAL_FAMILIES
    return [fid for fid, meta in FAL_FAMILIES.items() if meta.get("text_endpoint")]


def _i2v_only_fal_families():
    """Families that only animate an existing image (no text_endpoint)."""
    from plugins.video_gen.fal import FAL_FAMILIES
    return [fid for fid, meta in FAL_FAMILIES.items() if not meta.get("text_endpoint")]


@pytest.mark.parametrize("family_id", _t2v_fal_families())
def test_fal_text_only_routes_to_text_endpoint(matrix_env, family_id):
    home, fal_calls, _ = matrix_env
    from plugins.video_gen.fal import FAL_FAMILIES

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "fal", "model": family_id}},
        {"prompt": "a dog running"},
    )

    # Image-only families (e.g. gemini-omni-flash) must reject text-only
    # jobs with a clean modality error instead of submitting anywhere.
    if not FAL_FAMILIES[family_id].get("text_endpoint"):
        assert result["success"] is False, family_id
        assert result.get("error_type") == "modality_unsupported", result
        assert not fal_calls, f"{family_id} submitted despite no text endpoint"
        return

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


@pytest.mark.parametrize("family_id", _i2v_only_fal_families())
def test_fal_i2v_only_family_refuses_text_only(matrix_env, family_id):
    """An i2v-only family must refuse a text-only call rather than guess an endpoint."""
    home, fal_calls, _ = matrix_env

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "fal", "model": family_id}},
        {"prompt": "a dog running"},
    )

    assert result["success"] is False, f"{family_id} has no text-to-video route"
    assert result.get("error_type") == "modality_unsupported"
    assert not fal_calls, f"{family_id} must not reach FAL for an unsupported modality"


def _i2v_fal_families():
    """Every family that can animate an existing image."""
    from plugins.video_gen.fal import FAL_FAMILIES
    return [fid for fid, meta in FAL_FAMILIES.items() if meta.get("image_endpoint")]


@pytest.mark.parametrize("family_id", _i2v_fal_families())
def test_fal_image_to_video_routes_to_image_endpoint(matrix_env, family_id):
    home, fal_calls, _ = matrix_env
    from plugins.video_gen.fal import FAL_FAMILIES

    result = _invoke_tool(
        home,
        {"video_gen": {"provider": "fal", "model": family_id}},
        {"prompt": "animate this", "image_url": "https://example.com/i.png"},
    )

    meta = FAL_FAMILIES[family_id]
    assert result["success"] is True, f"{family_id}: {result.get('error')}"
    assert result["modality"] == "image"
    assert len(fal_calls) == 1
    assert fal_calls[0]["endpoint"] == meta["image_endpoint"]

    # The image must land under the family's declared key and no other
    # (kling v3 4k wants start_image_url; sending both would be a 422).
    payload = fal_calls[0]["arguments"] or {}
    image_key = meta.get("image_param_key") or "image_url"
    assert payload.get(image_key) == "https://example.com/i.png"
    other_keys = [k for k in payload if "image" in k and "url" in k and k != image_key]
    assert not other_keys, f"{family_id} sent extra image keys: {other_keys}"


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
