"""Invariants for the native-embed history budgets (#112095).

A native ``vision_analyze`` result is re-sent on every later API call, so (1) a delegated subagent
may not embed the same image without limit and (2) the per-embed byte budget must follow
``vision.embed_target_bytes`` instead of a hardcoded 256 KB.
"""
from __future__ import annotations

import asyncio
import json
import random

import pytest

from agent.delegation_context import delegated_child_context
from hermes_cli.config import get_config_path
from tools import vision_tools_history_budget as budget
from tools.vision_tools import _vision_analyze_native

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture(autouse=True)
def _fresh_counters():
    with budget._repeat_lock:
        budget._repeat_counts.clear()
    yield
    with budget._repeat_lock:
        budget._repeat_counts.clear()


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


def _embedded(result) -> bool:
    return isinstance(result, dict) and result.get("_multimodal") is True


def _embed_len(result) -> int:
    return len(next(p["image_url"]["url"] for p in result["content"] if p.get("type") == "image_url"))


class TestRepeatCap:
    def test_delegated_subagent_is_refused_after_three_loads_of_one_image(self, tmp_path):
        """Full loads and region crops of the same file share one counter; the refusal names the
        knob and tells the model to answer from what it already has (no fourth embed)."""
        shot = _png(tmp_path / "shot.png")
        with delegated_child_context("child-session"):
            assert _embedded(_load(shot))
            assert _embedded(_load(shot, region=[0, 0, 8, 8]))
            assert _embedded(_load(shot))
            refused = _load(shot, region=[4, 4, 12, 12])
            other = _load(_png(tmp_path / "other.png"))
        assert isinstance(refused, str)
        payload = json.loads(refused)
        assert payload["success"] is False
        assert "already been loaded" in payload["error"] and "max_calls_per_image" in payload["error"]
        assert _embedded(other), "a different image in the same session is not affected"

    def test_parallel_batch_on_one_image_cannot_overshoot_the_cap(self, tmp_path):
        """The executor runs a tool batch concurrently: 6 simultaneous loads of one image with cap 3
        must yield exactly 3 embeds — the slot is reserved atomically, not check-then-record."""
        import contextvars
        import threading

        shot = _png(tmp_path / "shot.png")
        results = [None] * 6

        def one(i):
            results[i] = _embedded(asyncio.new_event_loop().run_until_complete(_vision_analyze_native(shot, "q")))

        with delegated_child_context("child-parallel"):
            threads = [threading.Thread(target=contextvars.copy_context().run, args=(one, i)) for i in range(6)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        assert sum(results) == 3
        assert budget._repeat_counts[("child-parallel", budget._image_key(shot))] == 3

    def test_failed_embed_releases_its_reserved_slot(self, tmp_path):
        """A refused/failed load (missing file) must not burn one of the three slots."""
        shot = _png(tmp_path / "shot.png")
        with delegated_child_context("child-release"):
            for _ in range(3):
                assert json.loads(_load(str(tmp_path / "missing.png")))["success"] is False
            assert all(_embedded(_load(shot)) for _ in range(3))
        assert ("child-release", budget._image_key(str(tmp_path / "missing.png"))) not in budget._repeat_counts

    def test_main_agent_is_unlimited_unless_configured(self, tmp_path):
        shot = _png(tmp_path / "shot.png")
        assert all(_embedded(_load(shot)) for _ in range(5))

        _write_config("vision:\n  max_calls_per_image: 1\n")
        with budget._repeat_lock:
            budget._repeat_counts.clear()
        assert _embedded(_load(shot))
        assert json.loads(_load(shot))["success"] is False


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
