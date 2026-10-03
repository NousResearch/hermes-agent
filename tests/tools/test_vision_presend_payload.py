"""Pre-send payload budget for the auxiliary vision path. Regression for #8120.

`vision_analyze` has two request paths. The native fast path bakes a bounded
data URL into the tool result; the auxiliary path (`vision_analyze_tool`, used
whenever the main model is not vision-capable) used to hand the provider the
full-resolution base64 of whatever file it was pointed at, with its only
downscale gated on the provider *rejecting* the payload as too large. A
provider that is merely slow never rejects, so the request ran to the vision
call timeout. These tests pin the contract that every image `vision_analyze`
puts on the wire is bounded by the same embed budget, on either path.
"""

import base64
import io
import json
import random

import pytest
from PIL import Image

from tools.vision_tools import (
    _EMBED_MAX_DIMENSION,
    _image_to_base64_data_url,
    vision_analyze_tool,
)
from tools.vision_tools_history_budget import resolve_embed_target_bytes


def _noisy_png(path, width, height):
    """A real, decodable PNG of incompressible pixels, so the fixture is over the
    byte budget on its own. Random bytes do not compress away, which is how the
    "multi-megabyte local image" case in #8120 is reproduced."""
    image = Image.frombytes(
        "RGB", (width, height), random.Random(20260412).randbytes(width * height * 3))
    image.save(path, format="PNG")
    return path


def _pixel_dense_but_small_png(path, width, height, block=32):
    """The shape the #8120 reporter actually hit: a screenshot that is CHEAP in
    bytes (a few flat blocks) but dense in pixels, so it sails under any byte
    budget and only the long-edge cap can bound it. Nearest-neighbour upscaling
    from a small noise tile keeps the encoded size tiny."""
    tile = Image.frombytes(
        "RGB", (width // block, height // block),
        random.Random(20260412).randbytes((width // block) * (height // block) * 3))
    tile.resize((width, height), Image.NEAREST).save(path, format="PNG")
    return path


def _wire_payloads(sent):
    """Every data URL handed to the vision provider by the captured call."""
    return [
        part["image_url"]["url"]
        for call in sent
        for message in call["messages"]
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


def _decoded_size(data_url):
    raw = base64.b64decode(data_url.partition(",")[2])
    with Image.open(io.BytesIO(raw)) as image:
        return image.size


@pytest.fixture
def captured_vision_calls(monkeypatch):
    """Record what reaches the provider instead of calling it."""
    from tools import vision_tools

    sent = []

    async def _fake_call_llm(**kwargs):
        sent.append(kwargs)
        response = type("R", (), {})()
        choice = type("C", (), {})()
        choice.message = type("M", (), {"content": "a picture", "reasoning_content": None})()
        response.choices = [choice]
        return response

    monkeypatch.setattr(vision_tools, "async_call_llm", _fake_call_llm)
    return sent


@pytest.mark.asyncio
async def test_aux_path_does_not_send_full_resolution_local_image(
    tmp_path, captured_vision_calls
):
    """A local image taller than the vision embed long-edge cap must reach the
    provider downscaled, and the result must say so. The fixture is deliberately
    under the byte budget and over the long-edge cap, so this pins the bound that
    a 171 KB screenshot trips. On the unfixed path the provider receives the 2:1
    file untouched and the call is left to time out."""
    src = _pixel_dense_but_small_png(
        tmp_path / "tall.png", 1400, _EMBED_MAX_DIMENSION + 600)
    # Guard the premise: this case is only about the long edge if bytes already fit.
    assert len(_image_to_base64_data_url(src)) <= resolve_embed_target_bytes()

    result = json.loads(await vision_analyze_tool(str(src), "describe this"))

    assert result["success"] is True
    payloads = _wire_payloads(captured_vision_calls)
    assert payloads, "no image reached the vision provider"
    for data_url in payloads:
        assert max(_decoded_size(data_url)) <= _EMBED_MAX_DIMENSION
        assert len(data_url) <= resolve_embed_target_bytes()
    # The bound is a contract, not silence: the model is told pixels were dropped.
    assert "downscaled" in result["analysis"]


@pytest.mark.asyncio
async def test_aux_and_native_paths_share_one_embed_budget(
    tmp_path, captured_vision_calls
):
    """The same local file must not cost more on the auxiliary path than on the
    native one. This fixture is the multi-megabyte case: incompressible pixels
    put the base64 far over the byte budget, so the byte bound is what has to
    bind here (a 2268x1500 file left as 13.6 MB of base64 is what a vision call
    with a 120 s default timeout cannot carry)."""
    from tools.vision_tools import _vision_analyze_native

    src = _noisy_png(tmp_path / "wide.png", _EMBED_MAX_DIMENSION + 700, 1500)
    # Guard the premise: this case is only about bytes if the long edge alone would fit.
    assert max(Image.open(src).size) > _EMBED_MAX_DIMENSION

    await vision_analyze_tool(str(src), "describe this")
    aux_payload = _wire_payloads(captured_vision_calls)

    captured_vision_calls.clear()
    native = await _vision_analyze_native(str(src), "describe this")
    assert isinstance(native, dict)
    native_payload = [
        part["image_url"]["url"]
        for part in native["content"]
        if part.get("type") == "image_url"
    ]

    assert aux_payload and native_payload
    for data_url in aux_payload + native_payload:
        assert len(data_url) <= resolve_embed_target_bytes()
        assert max(_decoded_size(data_url)) <= _EMBED_MAX_DIMENSION
