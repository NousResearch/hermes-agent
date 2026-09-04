"""Dependency-plan validation for adaptive subagent scheduling."""

import pytest

from tools.delegation_graph import build_dependency_plan


def test_batch_without_metadata_keeps_flat_scheduler():
    plan, error = build_dependency_plan(
        [{"goal": "Research alpha"}, {"goal": "Research beta"}]
    )

    assert error is None
    assert plan.enabled is False
    assert plan.components == ((0, 1),)


def test_dependencies_form_clusters_and_preserve_declared_order():
    plan, error = build_dependency_plan(
        [
            {"id": "alpha", "goal": "Research alpha"},
            {"id": "beta", "goal": "Research beta"},
            {
                "id": "combine",
                "goal": "Combine both findings",
                "depends_on": ["alpha", "beta"],
            },
            {"id": "independent", "goal": "Write an independent note"},
        ]
    )

    assert error is None
    assert plan.enabled is True
    assert plan.task_ids == ("alpha", "beta", "combine", "independent")
    assert plan.dependencies[2] == (0, 1)
    assert plan.components == ((0, 1, 2), (3,))
    assert plan.dependency_ids(2) == ("alpha", "beta")


def test_ids_without_edges_keep_flat_scheduler():
    plan, error = build_dependency_plan(
        [
            {"id": "one", "goal": "Complete the first task"},
            {"id": "two", "goal": "Complete the second task"},
        ]
    )

    assert error is None
    assert plan.enabled is False
    assert plan.components == ((0, 1),)


@pytest.mark.parametrize("dependency", [[], None])
def test_empty_dependencies_without_ids_keep_flat_scheduler(dependency):
    plan, error = build_dependency_plan([
        {"goal": "one", "depends_on": dependency},
        {"id": "optional-label", "goal": "two", "depends_on": []},
    ])
    assert error is None
    assert not plan.enabled
    assert plan.components == ((0, 1),)


@pytest.mark.parametrize("dependency", ["one", [1], False, {}, ""])
def test_malformed_dependencies_are_not_silently_flattened(dependency):
    plan, error = build_dependency_plan([{"goal": "one", "depends_on": dependency}])
    assert not plan.enabled
    assert "array of task-id strings" in error


def test_partial_ids_are_rejected_instead_of_silently_flattened():
    _plan, error = build_dependency_plan(
        [
            {"id": "one", "goal": "Complete the first task"},
            {"goal": "Complete the second task", "depends_on": ["one"]},
        ]
    )

    assert "Add an id to every task" in error


def test_unknown_dependency_is_rejected():
    _plan, error = build_dependency_plan(
        [
            {
                "id": "one",
                "goal": "Complete the first task",
                "depends_on": ["missing"],
            }
        ]
    )

    assert "unknown task id 'missing'" in error


def test_cycle_is_rejected_before_any_child_runs():
    _plan, error = build_dependency_plan(
        [
            {"id": "one", "goal": "Complete first", "depends_on": ["two"]},
            {"id": "two", "goal": "Complete second", "depends_on": ["one"]},
        ]
    )

    assert "Dependency cycle detected" in error
    assert "one" in error
    assert "two" in error
