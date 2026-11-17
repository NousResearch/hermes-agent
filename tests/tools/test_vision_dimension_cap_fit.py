"""The resizer's two limits are satisfied by one descending scan (#124509).

Pinned here: both limits hold whenever any candidate fits; the 64 px edge is a preference, never a
reason to return a file the cap forbids; the answer never shrinks when the budget grows and by at most
1% when the cap grows; an image inside both limits is untouched; and the scan stays inside its
allowance, which is pinned as a literal so a widened one fails here.
"""
from __future__ import annotations

import base64
import io
import itertools
import random

import pytest

from tools import vision_tools
from tools.vision_tools import _resize_image_for_vision

PILImage = pytest.importorskip("PIL.Image")  # the PIL.Image module, so PILImage.Image is the class

CAP = 1568
KIB = 1024
MIB = 1024 * 1024


def _img(path, size, noisy: bool = False, fmt: str = "JPEG"):
    img = PILImage.new("RGB", size)
    if noisy:
        rnd = random.Random(7)
        img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                     for _ in range(size[0] * size[1])])
    img.save(path, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return path


def _call(path, budget, cap=None, mime_type="image/jpeg", scale_out=None, force_jpeg=True):
    """Return (decoded size, payload length, decoded mime)."""
    url = _resize_image_for_vision(path, mime_type, max_base64_bytes=budget,
                                   max_dimension=cap, scale_out=scale_out, force_jpeg=force_jpeg)
    raw = base64.b64decode(url.split(",", 1)[1])
    return PILImage.open(io.BytesIO(raw)).size, len(url), url.split(";", 1)[0].split(":", 1)[1]


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """The noisy sources, built once: generating them dominates the file's runtime otherwise."""
    root = tmp_path_factory.mktemp("sources")
    return {
        "grid": _img(root / "grid.jpg", (2048, 1536), noisy=True),
        "photo": _img(root / "photo.jpg", (3200, 2400), noisy=True),
    }


@pytest.fixture
def encodes(monkeypatch):
    """Count encoder calls, so the search's cost is pinned rather than described."""
    counter = {"n": 0}
    original = PILImage.Image.save

    def counting(self, fp, *args, **kwargs):
        counter["n"] += 1
        return original(self, fp, *args, **kwargs)

    monkeypatch.setattr(PILImage.Image, "save", counting)
    return counter


@pytest.fixture
def resizes(monkeypatch):
    """Count image resizes: one per new candidate, repeats hit the probe cache."""
    counter = {"n": 0}
    original = PILImage.Image.resize

    def counting(self, *args, **kwargs):
        counter["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PILImage.Image, "resize", counting)
    return counter


# -- the cap costs its own overflow, not a halving step --------------------------------

def test_cap_overflow_is_fitted_not_halved(tmp_path):
    """A source one pixel over the cap is fitted to the cap."""
    src = _img(tmp_path / "over.jpg", (1569, 1177))
    assert _call(src, MIB, CAP)[0] == (1568, 1176)


def test_large_photo_is_fitted_to_the_cap(tmp_path):
    """A 1568px encoding fits the byte budget, so the photo reaches the model at the cap."""
    src = _img(tmp_path / "big.jpg", (4000, 3000))
    assert _call(src, MIB, CAP)[0] == (1568, 1176)


def test_modest_image_over_the_cap_is_fitted_not_halved(tmp_path):
    """A 1200px source under a 1000px cap comes back at 1000, not at 600."""
    src = _img(tmp_path / "modest.jpg", (1200, 800))
    scale_out: dict = {}
    size, _, _ = _call(src, MIB, 1000, scale_out=scale_out)
    assert size == (1000, 667)
    assert scale_out == {"orig_width": 1200, "orig_height": 800,
                         "new_width": 1000, "new_height": 667}


def test_png_without_force_jpeg_is_fitted_to_the_cap(tmp_path):
    """The PNG path has no quality ladder, so a scale-only scan has to reach the cap."""
    src = _img(tmp_path / "big.png", (2000, 1500), noisy=True, fmt="PNG")
    size, _, mime = _call(src, 8 * MIB, CAP, mime_type="image/png", force_jpeg=False)
    assert (size, mime) == ((1568, 1176), "image/png")


# -- the cap is a hard limit, the edge floor only a preference -------------------------

def test_thin_strip_is_returned_inside_the_cap(tmp_path):
    """A strip the cap cuts to two pixels tall is returned inside the cap."""
    src = _img(tmp_path / "strip.jpg", (60000, 64))
    scale_out: dict = {}
    size, payload_len, _ = _call(src, 256 * KIB, CAP, scale_out=scale_out)
    assert size == (1568, 2)
    assert payload_len <= 256 * KIB and payload_len < 4 * KIB
    assert scale_out == {"orig_width": 60000, "orig_height": 64,
                         "new_width": 1568, "new_height": 2}


def test_one_pixel_high_strip_is_returned_inside_the_cap(tmp_path):
    """The degenerate form of the same case: a one-pixel source has no candidate above the floor."""
    src = _img(tmp_path / "line.jpg", (4000, 1))
    size, payload_len, _ = _call(src, 256 * KIB, CAP)
    assert size == (1568, 1) and payload_len <= 256 * KIB


def test_an_unsatisfiable_budget_still_returns_a_measured_encoding(tmp_path, resizes):
    """No candidate can satisfy this budget, so the answer is the last one measured, not the source.

    The smallest JPEG this encoder produces is over a kilobyte, so a 200-character budget is
    unsatisfiable by construction. The scan measures two candidates above the edge floor, stops there
    and hands back the smaller one instead of the source payload (``raw_payload_len`` below).
    """
    assert vision_tools._RESIZE_MAX_PROBES == 24
    src = _img(tmp_path / "small.jpg", (200, 80))
    raw_payload_len = len(base64.b64encode(src.read_bytes())) + 32
    resizes["n"] = 0
    size, payload_len, _ = _call(src, 200, None)  # 200 characters is under every JPEG encoding
    assert payload_len < raw_payload_len
    assert size == (170, 68)  # the last candidate measured, two steps below the source
    assert resizes["n"] <= 2  # one candidate above the floor, one below it, pinned as a literal


# -- the byte budget is met by scale, not by halving -----------------------------------

def test_byte_bound_photo_keeps_the_scale_the_budget_allows(sources):
    """Under a byte budget the scan keeps the largest fitting scale, not a halved one."""
    budget = 380 * KIB
    size, payload_len, _ = _call(sources["photo"], budget, CAP)
    assert payload_len <= budget
    assert size[0] >= 1392, size  # the largest fitting scale for this source, within the 1% band


def test_no_candidate_above_the_answer_is_left_unmeasured(tmp_path, monkeypatch):
    """The encoder reports the top candidate as over budget and the band below it as fitting, so the
    scan must measure the larger candidate before it accepts the smaller one.
    """
    src = _img(tmp_path / "notch.jpg", (1600, 1200))
    real_save = PILImage.Image.save
    probed: list[tuple[int, int]] = []

    def fake_save(self, fp, *args, **kwargs):
        probed.append(self.size)
        real_save(self, fp, *args, **kwargs)
        width = self.size[0]
        # over the budget at the top, cheap in the band below it, over it again further down
        fp.write(b"\x00" * (900 * KIB if width > 1560 or width < 0.80 * 1600 else 200))

    monkeypatch.setattr(PILImage.Image, "save", fake_save)
    size, payload_len, _ = _call(src, 256 * KIB, CAP)
    assert probed[0] == (1568, 1176)  # the cap fit was measured, not assumed
    assert 0.80 * 1600 <= size[0] <= 1560  # the band above the failures
    assert payload_len <= 256 * KIB


# -- both limits hold by construction, and grow with the knobs -------------------------

def test_result_respects_the_byte_budget(sources, tmp_path):
    """No path may hand back more than the budget while a useful scale fits."""
    cases = [((2048, 1536), 90 * KIB, 1024),
             ((4096, 2560), 26 * KIB, 1024),
             ((3200, 2400), 380 * KIB, CAP),
             ((60000, 64), 256 * KIB, CAP)]
    for size, budget, cap in cases:
        if size == (2048, 1536):
            src = sources["grid"]
        elif size == (3200, 2400):
            src = sources["photo"]
        else:
            src = _img(tmp_path / f"b_{size[0]}x{size[1]}.jpg", size, noisy=True)
        dims, payload_len, _ = _call(src, budget, cap)
        assert payload_len <= budget, (size, dims, payload_len, budget)
        assert max(dims) <= cap


def test_the_answer_grows_with_the_budget(sources):
    """Monotone in the budget, exactly: a bigger budget never yields fewer pixels."""
    for cap in (800, 1024, CAP):
        areas = []
        for budget in (60 * KIB, 200 * KIB, 600 * KIB):
            (w, h), payload_len, _ = _call(sources["grid"], budget, cap)
            assert payload_len <= budget and max(w, h) <= cap
            areas.append(w * h)
        assert areas[0] > 0
        assert areas == sorted(areas), (cap, areas)


def test_a_larger_cap_never_costs_less_than_one_percent(sources):
    """Monotone in the cap up to the refinement resolution."""
    for budget in (60 * KIB, 200 * KIB, 600 * KIB):
        areas = []
        for cap in (800, 1024, 1280, CAP):
            (w, h), payload_len, _ = _call(sources["grid"], budget, cap)
            assert max(w, h) <= cap and payload_len <= budget
            areas.append(w * h)
        for (i, previous), (j, current) in itertools.combinations(enumerate(areas), 2):
            assert current >= 0.99 * previous, (budget, i, j, areas)


def test_a_larger_cap_gets_a_larger_embed(sources):
    """The cap is the ceiling on how much survives, so raising it shows up in the answer."""
    budget = 700 * KIB
    small, small_len, _ = _call(sources["photo"], budget, CAP)
    large, large_len, _ = _call(sources["photo"], budget, 2000)
    assert small_len <= budget and large_len <= budget
    assert max(large) > max(small), (small, large)
    assert max(small) >= 1500  # within the refinement resolution of the 1568 px cap


# -- untouched and bounded-work paths --------------------------------------------------

def test_image_within_both_caps_is_untouched(tmp_path, encodes, resizes):
    src = _img(tmp_path / "small.jpg", (1200, 800))
    encodes["n"] = resizes["n"] = 0  # writing the fixture is not the resizer's work
    size, payload_len, _ = _call(src, MIB, CAP)
    assert size == (1200, 800)
    assert payload_len == len(vision_tools._image_to_base64_data_url(src, mime_type="image/jpeg"))
    assert (encodes["n"], resizes["n"]) == (0, 0)


def test_common_case_costs_one_encode(tmp_path, encodes, resizes):
    """The cap-bound case is settled by the first candidate: one encode, not a ladder."""
    src = _img(tmp_path / "one.jpg", (4000, 3000))
    encodes["n"] = resizes["n"] = 0
    assert _call(src, MIB, CAP)[0] == (1568, 1176)
    assert (encodes["n"], resizes["n"]) == (1, 1)


def test_search_work_is_bounded(sources, encodes, resizes):
    """A byte-bound case does the work its bracket implies, and no more than the allowance."""
    encodes["n"] = resizes["n"] = 0
    _call(sources["grid"], 90 * KIB, 1024)
    assert resizes["n"] == 6  # 3 candidates above the edge floor, 3 refinement probes
    assert encodes["n"] == 18  # every candidate walks the whole JPEG quality ladder here
    assert resizes["n"] <= 27  # the allowance ceiling: 24 candidates + 3 refinement probes


def test_the_worst_grid_cell_keeps_its_answer_and_its_price(sources, encodes):
    """The priciest cell of the cap/budget grid: 30 encoder calls buy 644x483 where 60 KB binds."""
    encodes["n"] = 0
    size, payload_len, _ = _call(sources["grid"], 60 * KIB, CAP)
    assert size == (644, 483)
    assert payload_len <= 60 * KIB
    assert encodes["n"] == 30  # the grid's worst cell, pinned so a slower scan shows up here


def test_scale_out_reports_original_and_chosen_size(tmp_path):
    src = _img(tmp_path / "scale.jpg", (4000, 3000))
    scale_out: dict = {}
    _call(src, MIB, CAP, scale_out=scale_out)
    assert scale_out == {"orig_width": 4000, "orig_height": 3000,
                         "new_width": 1568, "new_height": 1176}
