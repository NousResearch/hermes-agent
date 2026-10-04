"""Hard wall-time cap for the whole vision_analyze call chain.

Invariant: _handle_vision_analyze always returns within its configured hard
timeout, even when every sub-step hangs. Sub-step timeouts stack (download 30s +
encode + LLM 120s + retry paths), so a wedged provider could previously exceed
any bound with no cancellation point (observed 362s wedge).
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from tools import vision_tools
from tools.vision_tools import _handle_vision_analyze


@pytest.mark.asyncio
async def test_wedged_provider_returns_within_hard_timeout(monkeypatch):
    """A sub-step that never completes must not hang the tool call.

    Simulates the observed failure: provider call (or download) wedges with no
    sub-step timeout firing — the chain has no outer bound without the fix.
    """
    async def _wedged_analysis(image_url, full_prompt, model, task_id=None, region=None):
        await asyncio.Event().wait()  # never completes

    monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: False)
    monkeypatch.setattr(vision_tools, "vision_analyze_tool", _wedged_analysis)
    monkeypatch.setattr(vision_tools, "_vision_hard_timeout", lambda: 0.2)

    start = time.monotonic()
    result = await _handle_vision_analyze({"image_url": "https://x/y.png", "question": "?"})
    elapsed = time.monotonic() - start

    data = json.loads(result)
    assert data["success"] is False
    assert "timed out" in data["error"].lower()
    assert elapsed < 5, f"tool call took {elapsed:.1f}s with a 0.2s cap — no outer bound"