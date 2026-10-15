"""Regression: a project's sidebar parent is part of the projects.tree wire contract.

``project_tree._assign_parent_projects`` stamps ``parentId`` onto every node, and these result models
forbid unknown fields — so with the field undeclared, every ``projects.tree`` reply logged a contract
violation and the desktop lost the nesting it had just been sent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tui_gateway import project_tree as pt
from tui_gateway.contracts.projects_pets import ProjectTreeNode
from tui_gateway.contracts.registry import METHODS, check_result


def _project(pid: str, name: str, folders: list[str], **over):
    row = {
        "id": pid,
        "name": name,
        "primary_path": folders[0] if folders else None,
        "archived": False,
        "folders": [{"path": p, "is_primary": i == 0} for i, p in enumerate(folders)],
    }
    row.update(over)
    return row


def test_project_parent_id_survives_result_validation():
    parent = _project("p_parent", "Parent", ["/www/app"])
    child = _project("p_child", "Child", ["/www/app/vendor/lib"], parent_id="p_parent")

    tree = pt.build_tree([parent, child], [], [], resolve=None, hydrate=False)
    payload = {
        "projects": tree["projects"],
        "active_id": None,
        "scoped_session_ids": tree["scoped_session_ids"],
    }

    check_result(METHODS["projects.tree"], payload)

    by_id = {node["id"]: node for node in payload["projects"]}
    assert by_id["p_child"]["parentId"] == "p_parent"
    # Top level is an explicit null on the wire, not an absent key.
    assert by_id["p_parent"]["parentId"] is None


def test_project_node_keeps_strict_result_validation():
    for extra in ({"parentId": 7}, {"unknown_field": []}):
        with pytest.raises(ValidationError):
            ProjectTreeNode.model_validate({"id": "p1", "label": "P", **extra})
