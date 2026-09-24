"""Zero-rect AX bounds must read as 'unknown', never as a clickable position.

Live QA (Aug 2026): KDE/Qt apps report [0,0,0,0] for elements that are
perfectly clickable by index (all of kcalc's radio buttons). A
plausible-looking zero rect invites coordinate=[0,0] derivation.
"""
from tools.computer_use.backend import UIElement
from tools.computer_use.tool import (
    _bounds_unknown,
    _element_to_dict,
)


def _el(bounds, index=2, role="radio button", label="Deg"):
    return UIElement(index=index, role=role, label=label, bounds=tuple(bounds), app="")






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




