"""Line-interval coverage helpers for read_file context deduplication."""

from __future__ import annotations


def merge_intervals(
    intervals: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Merge an inclusive interval into sorted inclusive intervals."""
    merged: list[tuple[int, int]] = []
    for s, e in sorted([*intervals, (start, end)]):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def first_uncovered_interval(
    intervals: list[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Return the first inclusive sub-interval not covered, or None."""
    cursor = start
    for s, e in intervals:
        if e < cursor:
            continue
        if s > cursor:
            return cursor, min(end, s - 1)
        cursor = max(cursor, e + 1)
        if cursor > end:
            return None
    return (cursor, end) if cursor <= end else None


def get_path_coverage(task_data: dict, resolved_path: str, mtime: float) -> dict:
    """Return path coverage state, resetting intervals when mtime changes."""
    coverage_by_path = task_data.setdefault("path_coverage", {})
    coverage = coverage_by_path.get(resolved_path)
    if coverage is None or coverage.get("mtime") != mtime:
        coverage = {"mtime": mtime, "intervals": []}
        coverage_by_path[resolved_path] = coverage
    else:
        coverage.setdefault("intervals", [])
    return coverage


def record_path_coverage(
    task_data: dict,
    resolved_path: str,
    mtime: float,
    offset: int,
    limit: int,
    max_intervals: int,
) -> None:
    """Record the returned read_file line interval for an unchanged path."""
    start, end = offset, offset + limit - 1
    coverage = get_path_coverage(task_data, resolved_path, mtime)
    merged = merge_intervals(coverage.get("intervals", []), start, end)
    # If a file is too fragmented, forget old coverage rather than growing
    # task-local state forever. The safe degradation is one future resend.
    coverage["intervals"] = [(start, end)] if len(merged) > max_intervals else merged


def cap_path_coverage(task_data: dict, max_paths: int) -> None:
    """Bound the number of paths that retain interval coverage."""
    coverage_by_path = task_data.get("path_coverage")
    if coverage_by_path is None or len(coverage_by_path) <= max_paths:
        return
    for _ in range(len(coverage_by_path) - max_paths):
        try:
            coverage_by_path.pop(next(iter(coverage_by_path)))
        except (StopIteration, KeyError):
            break
