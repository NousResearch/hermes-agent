"""Page-controlled JSON must not crash browser output redaction (#137067)."""

import json

import pytest

from tools.browser_cdp_tool import _redact_cdp_output

SECRET = "sk-" + "a" * 30
WALKERS = [_redact_cdp_output]


def _nest(leaf, levels: int, *, as_dict: bool):
    for _ in range(levels):
        leaf = {"x": leaf} if as_dict else [leaf]
    return leaf


@pytest.mark.parametrize("walker", WALKERS)
@pytest.mark.parametrize("as_dict", [False, True])
def test_hostile_deep_page_value_is_cut_off_not_raised(walker, as_dict):
    # Deeper than the interpreter recursion limit: RecursionError before the fix.
    blob = json.dumps(walker(_nest(SECRET, 2_000, as_dict=as_dict)))
    assert SECRET not in blob


@pytest.mark.parametrize("walker", WALKERS)
def test_deepest_dom_chrome_returns_is_not_truncated(walker):
    # Chrome refuses DOM.getDocument past ~150 nested elements; each element is
    # node -> children[] -> node, so that is ~300 JSON levels and must survive whole.
    node = {"nodeId": 1, "nodeName": "#text", "nodeValue": "hello"}
    for i in range(150):
        node = {"nodeId": i + 2, "nodeName": "DIV", "children": [node]}
    assert walker(node) == node
