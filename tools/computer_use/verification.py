"""Bounded predicate contract from cua-driver-rs-v0.21.0's verification.rs.

Hermes excludes empty predicates/selectors instead of sending vacuous evidence.
"""
from typing import Any
import math

_BOUNDS_PROPERTIES: dict[str, Any] = {key: {"type": "number"} for key in ("x", "y", "width", "height")}
_BOUNDS_PROPERTIES["tolerance_px"] = {"type": "number", "minimum": 0, "maximum": 100}

EXPECT_SCHEMA = {
    "type": "array", "minItems": 1, "maxItems": 8,
    "description": "Required for verify_state: 1–8 predicates combined with AND. Use separate element/window entries. Unknown never implies success.",
    "items": {
        "type": "object", "minProperties": 1, "additionalProperties": False,
        "properties": {
            "element": {
                "type": "object", "additionalProperties": False, "required": ["selector"],
                "properties": {
                    "selector": {
                        "type": "object", "minProperties": 1, "additionalProperties": False,
                        "properties": {key: {"type": "string", "minLength": 1}
                                       for key in ("role", "label_contains")},
                        "description": "Match trusted accessibility elements by role and/or label substring; provide at least one nonblank field.",
                    },
                    "exists": {"type": "boolean", "enum": [True], "description": "Only true is supported; element absence cannot be proven."},
                    "value_equals": {"type": ["string", "null"]},
                    "enabled": {"type": ["boolean", "null"]},
                    "selected": {"type": ["boolean", "null"]},
                },
            },
            "window": {
                "type": "object", "minProperties": 1, "additionalProperties": False,
                "properties": {
                    "exists": {"type": ["boolean", "null"]},
                    "bounds": {
                        "type": "object", "additionalProperties": False,
                        "required": ["x", "y", "width", "height"], "properties": _BOUNDS_PROPERTIES,
                    },
                },
            },
        },
    },
}


def valid_verification_expect(expect: Any) -> bool:
    """Validate the bounded shape without importing an optional schema library."""
    if not isinstance(expect, list) or not 1 <= len(expect) <= 8:
        return False
    for item in expect:
        if not isinstance(item, dict) or not item or item.keys() - {"element", "window"}:
            return False
        if "element" in item:
            element = item["element"]
            if not isinstance(element, dict) or element.keys() - {"selector", "exists", "value_equals", "enabled", "selected"}:
                return False
            selector = element.get("selector")
            if (not isinstance(selector, dict) or not selector or selector.keys() - {"role", "label_contains"}
                    or any(not isinstance(v, str) or not v.strip() for v in selector.values())):
                return False
            if "exists" in element and element["exists"] is not True:
                return False
            if element.get("value_equals") is not None and not isinstance(element["value_equals"], str):
                return False
            if any(element.get(k) is not None and not isinstance(element[k], bool) for k in ("enabled", "selected")):
                return False
        if "window" in item:
            window = item["window"]
            if not isinstance(window, dict) or not window or window.keys() - {"exists", "bounds"}:
                return False
            if window.get("exists") is not None and not isinstance(window["exists"], bool):
                return False
            if "bounds" in window:
                bounds = window["bounds"]
                if (not isinstance(bounds, dict) or not {"x", "y", "width", "height"} <= bounds.keys()
                        or bounds.keys() - _BOUNDS_PROPERTIES.keys()
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds.values())
                        or not 0 <= bounds.get("tolerance_px", 1) <= 100):
                    return False
            elif window.get("exists") is None:
                return False
    return True
