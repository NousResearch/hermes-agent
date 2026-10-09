"""Regression: ``parent_id`` on ``projects.create`` is part of the wire contract.

The desktop's create dialog puts ``parent_id`` on the params exactly when the new project is a
subproject (`store/projects.ts`, ``projectParams({…, parent_id: input.parentId})``). Params models are
closed on purpose — ``validate_params`` answers ``4000`` with the key path for an undeclared key — so
with the field missing from the model, every subproject create was rejected before the handler ran and
the desktop reported it as a client/backend version skew. Plain creates kept working because the key
is absent rather than null when there is no parent.

``tests/tui_gateway/test_projects_rpc.py`` could not catch this: its ``_call`` helper invokes the
handler directly, which is the one path that does not validate params.
"""

from __future__ import annotations

from tui_gateway.contracts.projects_pets import ProjectsCreateParams
from tui_gateway.contracts.registry import METHODS, validate_params

# Exactly what the desktop sends for a subproject (`store/projects.ts` create call).
DESKTOP_CREATE = {
    "name": "Child",
    "folders": ["/www/app/vendor/lib"],
    "primary_path": "/www/app/vendor/lib",
    "slug": "child",
    "description": None,
    "icon": None,
    "color": None,
    "board_slug": None,
    "parent_id": "p_parent",
    "use": False,
}


def test_desktop_subproject_create_is_accepted_as_create_params():
    _, problem = validate_params(METHODS["projects.create"], DESKTOP_CREATE)

    assert problem is None


def test_top_level_create_is_accepted_with_an_explicit_empty_parent():
    _, problem = validate_params(METHODS["projects.create"], {**DESKTOP_CREATE, "parent_id": ""})

    assert problem is None


def test_create_still_rejects_an_unknown_key():
    _, problem = validate_params(METHODS["projects.create"], {**DESKTOP_CREATE, "parentId": "p_parent"})

    assert problem is not None
    assert "parentId" in problem


def test_every_create_field_the_handler_reads_is_declared():
    """``methods_projects.projects.create`` passes these through ``_pick``; the model must not be the
    narrower half."""
    for key in ("slug", "primary_path", "description", "icon", "color", "board_slug", "parent_id"):
        assert key in ProjectsCreateParams.model_fields, key
