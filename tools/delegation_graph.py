"""Pure dependency planning for ``delegate_task`` batches.

The model may opt a batch into dependency-aware scheduling by assigning every
task a stable ``id`` and, where needed, a ``depends_on`` list.  This module is
deliberately free of agent/runtime imports: it validates that declaration,
rejects cycles, and derives independent connected components for the async
dispatcher.  Batches without dependency metadata retain the historical flat
parallel behavior with no graph-scheduler overhead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class DependencyPlan:
    """Validated execution plan for one delegation batch."""

    enabled: bool
    task_ids: Tuple[str, ...]
    dependencies: Tuple[Tuple[int, ...], ...]
    components: Tuple[Tuple[int, ...], ...]

    def dependency_ids(self, task_index: int) -> Tuple[str, ...]:
        return tuple(self.task_ids[i] for i in self.dependencies[task_index])


def _flat_plan(task_count: int) -> DependencyPlan:
    indices = tuple(range(task_count))
    return DependencyPlan(
        enabled=False,
        task_ids=tuple(f"task-{i + 1}" for i in indices),
        dependencies=tuple(() for _ in indices),
        components=(indices,) if indices else (),
    )


def _connected_components(
    dependencies: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, ...], ...]:
    adjacency = [set() for _ in dependencies]
    for child, parents in enumerate(dependencies):
        for parent in parents:
            adjacency[child].add(parent)
            adjacency[parent].add(child)

    unseen = set(range(len(dependencies)))
    components: List[Tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component = []
        unseen.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


def build_dependency_plan(
    tasks: Sequence[Dict[str, Any]],
) -> tuple[DependencyPlan, Optional[str]]:
    """Validate task dependency metadata and derive independent components.

    Dependency scheduling is opt-in through a non-empty ``depends_on`` list.
    IDs alone are labels; omitted, null, or empty dependency lists keep the
    historical flat batch. Malformed dependency declarations still fail closed
    rather than accidentally running work before its prerequisites.
    """

    task_count = len(tasks)
    has_dependencies = False
    for index, task in enumerate(tasks):
        raw_dependencies = task.get("depends_on")
        if raw_dependencies is None:
            continue
        if not isinstance(raw_dependencies, list) or any(
            not isinstance(item, str) for item in raw_dependencies
        ):
            return _flat_plan(task_count), (
                f"Task {index} depends_on must be an array of task-id strings."
            )
        has_dependencies = has_dependencies or bool(raw_dependencies)
    if not has_dependencies:
        return _flat_plan(task_count), None

    task_ids: List[str] = []
    id_to_index: Dict[str, int] = {}
    for index, task in enumerate(tasks):
        raw_id = task.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return _flat_plan(task_count), (
                f"Task {index} must define a non-empty 'id' because this batch "
                "uses dependency-aware scheduling. Add an id to every task, "
                "or remove all dependency edges to use a flat batch."
            )
        task_id = raw_id.strip()
        if not _TASK_ID_RE.fullmatch(task_id):
            return _flat_plan(task_count), (
                f"Task {index} has invalid id {task_id!r}. Use 1-64 letters, "
                "numbers, dots, underscores, or hyphens, beginning with a "
                "letter or number."
            )
        if task_id in id_to_index:
            return _flat_plan(task_count), (
                f"Duplicate task id {task_id!r}. Every task id must be unique."
            )
        id_to_index[task_id] = index
        task_ids.append(task_id)

    dependencies: List[Tuple[int, ...]] = []
    for index, task in enumerate(tasks):
        raw_dependencies = task.get("depends_on", [])
        if raw_dependencies is None:
            raw_dependencies = []
        seen = set()
        resolved: List[int] = []
        for raw_dependency in raw_dependencies:
            dependency_id = raw_dependency.strip()
            if dependency_id not in id_to_index:
                return _flat_plan(task_count), (
                    f"Task {task_ids[index]!r} depends on unknown task id "
                    f"{dependency_id!r}."
                )
            dependency_index = id_to_index[dependency_id]
            if dependency_index == index:
                return _flat_plan(task_count), (
                    f"Task {task_ids[index]!r} cannot depend on itself."
                )
            if dependency_index not in seen:
                seen.add(dependency_index)
                resolved.append(dependency_index)
        dependencies.append(tuple(resolved))

    # Kahn's algorithm validates the directed graph.  Component derivation
    # below treats edges as undirected only for independent delivery grouping.
    indegree = [len(parents) for parents in dependencies]
    children: List[List[int]] = [[] for _ in dependencies]
    for child, parents in enumerate(dependencies):
        for parent in parents:
            children[parent].append(child)
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != task_count:
        cyclic_ids = [
            task_ids[index] for index, degree in enumerate(indegree) if degree > 0
        ]
        return _flat_plan(task_count), (
            "Dependency cycle detected involving: " + ", ".join(cyclic_ids) + "."
        )

    dependency_tuple = tuple(dependencies)
    return DependencyPlan(
        enabled=True,
        task_ids=tuple(task_ids),
        dependencies=dependency_tuple,
        components=_connected_components(dependency_tuple),
    ), None
