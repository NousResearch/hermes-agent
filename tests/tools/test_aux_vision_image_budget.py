"""Aux vision requests are downscaled BEFORE the wire, not after a rejection (#135497).

An aux ``vision_analyze`` call is one-shot: a 9 MB plate sent at full size costs ~30 s of
ingest latency and blows the 45 s auxiliary timeout, so the "send full-res, downscale only if
the provider rejects >20 MB" order never gets a second chance. Measured on the tiered flash
lane: 9 MB at full size 30.0 s, the same image at 1568 px 4.6 s.
"""
from __future__ import annotations

import base64
import io
import random

import pytest

from tools.vision_tools import _aux_vision_budget, _aux_vision_data_url

PIL = pytest.importorskip("PIL.Image")


def _png(path, size=(1800, 1400), noisy=False):
    img = PIL.new("RGB", size, (200, 40, 40))
    if noisy:
        rnd = random.Random(11)
        img.putdata([
            (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
            for _ in range(size[0] * size[1])
        ])
    img.save(path, format="PNG")
    return path


def _long_edge(data_url: str) -> int:
    with PIL.open(io.BytesIO(base64.b64decode(data_url.partition(",")[2]))) as img:
        return max(img.size)


def test_small_image_is_sent_byte_identical(tmp_path):
    path = _png(tmp_path / "small.png", size=(40, 40))
    url = _aux_vision_data_url(path, "image/png", 1024 * 1024, 1568)

    assert url is not None
    assert url.startswith("data:image/png;base64,")
    assert _long_edge(url) == 40


def test_oversized_image_fits_the_budget_and_the_dimension_cap(tmp_path):
    path = _png(tmp_path / "big.png", noisy=True)
    budget = 200_000
    url = _aux_vision_data_url(path, "image/png", budget, 800)

    assert url is not None
    assert len(url) <= budget
    assert _long_edge(url) <= 800


def test_budget_never_returns_a_payload_larger_than_the_source(tmp_path):
    path = _png(tmp_path / "wide.png", size=(1900, 300), noisy=True)
    raw_len = (path.stat().st_size * 4) // 3
    url = _aux_vision_data_url(path, "image/png", 10, 1568)

    assert url is not None
    assert len(url) <= raw_len


def test_budget_defaults_are_bounded():
    budget, dimension = _aux_vision_budget()
    assert 0 < budget <= 8 * 1024 * 1024
    assert 0 < dimension <= 2048
