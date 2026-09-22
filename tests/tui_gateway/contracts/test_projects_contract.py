"""Contract regression: the sidebar's archived-tree request must pass the wire contract.

The frontend sends ``include_archived`` on ``projects.tree``; before t_844e931e the
contract lacked the field and ``validate_params`` rejected the request before the
handler ran — the single-profile tree never loaded. The one check no handler
performs is the unknown-key rejection, so the test goes through ``validate_params``.
"""

from tui_gateway.contracts.registry import METHODS, validate_params


def test_projects_tree_contract_accepts_include_archived():
    params, error = validate_params(
        METHODS["projects.tree"],
        {"profile": "default", "preview_limit": 3, "include_archived": True},
    )
    assert error is None, error
    assert params is not None
    assert params["include_archived"] is True


def test_projects_tree_contract_still_rejects_unknown_keys():
    params, error = validate_params(METHODS["projects.tree"], {"profile": "default", "gibberish": 1})
    assert params is None
    assert error is not None and "gibberish" in error
