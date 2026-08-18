"""Zero-rect AX bounds must read as 'unknown', never as a clickable position.

Live QA (Aug 2026): KDE/Qt apps report [0,0,0,0] for elements that are
perfectly clickable by index (all of kcalc's radio buttons). A
plausible-looking zero rect invites coordinate=[0,0] derivation.
"""
from tools.computer_use.backend import UIElement
from tools.computer_use.tool import (
    _bounds_unknown,
    _element_to_dict,
    _format_elements,
)


def _el(bounds, index=2, role="radio button", label="Deg"):
    return UIElement(index=index, role=role, label=label, bounds=tuple(bounds), app="")


def test_zero_rect_is_unknown():
    assert _bounds_unknown((0, 0, 0, 0)) is True


def test_real_rects_are_known():
    assert _bounds_unknown((0, 0, 640, 29)) is False
    assert _bounds_unknown((820, 433, 620, 19)) is False


def test_malformed_bounds_fail_open():
    assert _bounds_unknown(None) is False
    assert _bounds_unknown(("x", 0, 0, 0)) is False


def test_json_bounds_nulled_for_zero_rect():
    out = _element_to_dict(_el((0, 0, 0, 0)))
    assert out["bounds"] is None
    assert out["index"] == 2  # index targeting stays intact


def test_json_bounds_preserved_for_real_rect():
    out = _element_to_dict(_el((10, 20, 30, 40)))
    assert out["bounds"] == [10, 20, 30, 40]


def test_summary_line_says_bounds_unknown():
    lines = _format_elements([_el((0, 0, 0, 0))])
    assert "bounds-unknown" in lines[0]
    assert "click by element index" in lines[0]
    assert "(0, 0, 0, 0)" not in lines[0]


def test_summary_line_normal_for_real_rect():
    lines = _format_elements([_el((10, 20, 30, 40))])
    assert "bounds-unknown" not in lines[0]
    assert "(10, 20, 30, 40)" in lines[0]
