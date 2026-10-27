"""Regression: the two wire contracts the desktop's repository discovery depends on.

Both models are closed on purpose — params reject an unknown key with `4000`, results log a violation —
so a field the client sends (or the handler returns) that they do not declare is not a cosmetic error:
it fails the call. The desktop's `repoDiscoveryPolicyFromConfig` puts `nested` on the wire, and
`Project.to_dict()` carries `parent_id`; undeclared, every `projects.record_repos` was rejected before
the server could record a scan, and every `projects.list` reply was logged as a violation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hermes_cli.projects_db import Project
from tui_gateway.contracts.projects_pets import ProjectInfo, RepoDiscoveryPolicyParams
from tui_gateway.contracts.registry import METHODS, check_result, validate_params

# Exactly what `repoDiscoveryPolicyFromConfig` builds, short keys and all.
DESKTOP_POLICY = {"enabled": True, "nested": True, "roots": [], "exclude_paths": []}


def test_desktop_discovery_policy_is_accepted_as_record_repos_params():
    _, problem = validate_params(
        METHODS["projects.record_repos"], {"discovery_policy": DESKTOP_POLICY, "repos": []}
    )

    assert problem is None


def test_record_repos_still_rejects_an_unknown_policy_key():
    _, problem = validate_params(
        METHODS["projects.record_repos"], {"discovery_policy": {**DESKTOP_POLICY, "nestedd": True}}
    )

    assert problem is not None
    assert "discovery_policy.nestedd" in problem


def test_record_repos_echoes_the_policy_it_recorded_under():
    """`record_repos` returns the effective policy; `_repo_discovery_policy` reports `nested`, so the
    result model has to declare it too or the echo is itself a violation."""
    check_result(
        METHODS["projects.record_repos"],
        {
            "accepted": True,
            "discovery_policy": {"enabled": True, "nested": True, "roots": [], "exclude_paths": []},
            "repos": [],
        },
    )


def test_projects_list_payload_matches_the_stored_row():
    row = Project(id="p_child", slug="child", name="Child", created_at=1, parent_id="p_parent").to_dict()

    check_result(METHODS["projects.list"], {"projects": [row], "active_id": "p_child"})

    assert row["parent_id"] == "p_parent"


def test_scanned_repo_node_is_marked_discovered():
    """The sidebar keeps session-less rows only when the disk scan found them, so the node that
    *is* one has to say so on the wire."""
    from tui_gateway import project_tree as pt

    tree = pt.build_tree([], [], [{"label": "lib", "root": "/www/app/vendor/lib"}], resolve=None, hydrate=False)

    check_result(
        METHODS["projects.tree"],
        {"projects": tree["projects"], "active_id": None, "scoped_session_ids": tree["scoped_session_ids"]},
    )

    node = tree["projects"][0]
    assert node["id"] == "/www/app/vendor/lib"
    assert node["discovered"] is True
    assert node["isAuto"] is True


def test_project_info_keeps_strict_result_validation():
    for extra in ({"parent_id": 7}, {"unknown_field": []}):
        with pytest.raises(ValidationError):
            ProjectInfo.model_validate({"id": "p1", "slug": "s", "name": "N", "created_at": 1, **extra})


def test_every_policy_key_the_backend_reads_is_declared():
    """`_repo_discovery_policy` accepts both spellings; the model must not be the narrower half."""
    for key in ("enabled", "nested", "roots", "exclude_paths"):
        assert key in RepoDiscoveryPolicyParams.model_fields, key
