"""Invariants for the native-embed history budgets (#112095).

A native ``vision_analyze`` result is re-sent on every later API call, so the per-embed byte budget
must follow ``vision.embed_target_bytes`` instead of a hardcoded 256 KB.
"""
from __future__ import annotations

import asyncio
import random

import pytest

from hermes_cli.config import get_config_path
from tools import vision_tools_history_budget as budget
from tools.vision_tools import _vision_analyze_native

PIL = pytest.importorskip("PIL.Image")


def _write_config(text: str) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _png(path, size=(16, 16), noisy=False):
    img = PIL.new("RGB", size, (200, 40, 40))
    if noisy:
        rnd = random.Random(7)
        img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                     for _ in range(size[0] * size[1])])
    img.save(path)
    return str(path)


def _load(image, region=None):
    return asyncio.get_event_loop().run_until_complete(_vision_analyze_native(image, "q", region=region))


def _embed_len(result) -> int:
    return len(next(p["image_url"]["url"] for p in result["content"] if p.get("type") == "image_url"))


class TestEmbedTargetBytes:
    def test_native_embed_follows_configured_budget(self, tmp_path):
        """A 400x400 noisy PNG (~160 KB base64) rides under the 256 KB default untouched, and is
        shrunk once ``vision.embed_target_bytes`` drops to 64 KiB."""
        dense = _png(tmp_path / "dense.png", size=(400, 400), noisy=True)
        default_len = _embed_len(_load(dense))
        assert 65536 < default_len <= budget._DEFAULT_EMBED_TARGET_BYTES

        _write_config("vision:\n  embed_target_bytes: 65536\n")
        assert _embed_len(_load(dense)) <= 65536

    @pytest.mark.parametrize("raw, expected", [
        ("not-a-number", budget._DEFAULT_EMBED_TARGET_BYTES),
        ("true", budget._DEFAULT_EMBED_TARGET_BYTES),
        ("1", budget._MIN_EMBED_TARGET_BYTES),
        (str(64 * 1024 * 1024), budget._MAX_EMBED_TARGET_BYTES),
    ])
    def test_bad_or_extreme_values_are_clamped_to_the_safe_range(self, raw, expected):
        _write_config(f"vision:\n  embed_target_bytes: {raw}\n")
        assert budget.resolve_embed_target_bytes() == expected
