from tools.execution_target import ExecutionTarget, infer_execution_target, normalize_routed_toolsets
from tools.registry import ToolEntry


def _entry(name: str, toolset: str) -> ToolEntry:
    return ToolEntry(
        name=name,
        toolset=toolset,
        schema={},
        handler=lambda args: "{}",
        check_fn=None,
        requires_env=[],
        is_async=False,
        description="",
        emoji="",
    )


def test_split_runtime_disabled_routes_everything_to_server():
    assert infer_execution_target(_entry("read_file", "file"), enabled=False) is ExecutionTarget.SERVER


def test_read_only_file_tools_route_local_when_enabled():
    routed = normalize_routed_toolsets(["file"])
    assert infer_execution_target(_entry("read_file", "file"), enabled=True, routed_toolsets=routed) is ExecutionTarget.LOCAL
    assert infer_execution_target(_entry("search_files", "file"), enabled=True, routed_toolsets=routed) is ExecutionTarget.LOCAL


def test_mutating_file_tools_stay_server_even_when_file_toolset_routed():
    routed = normalize_routed_toolsets("file")
    assert infer_execution_target(_entry("write_file", "file"), enabled=True, routed_toolsets=routed) is ExecutionTarget.SERVER
    assert infer_execution_target(_entry("patch", "file"), enabled=True, routed_toolsets=routed) is ExecutionTarget.SERVER


def test_non_file_toolsets_stay_server():
    routed = normalize_routed_toolsets(["terminal", "file"])
    assert routed == frozenset({"file"})
    assert infer_execution_target(_entry("terminal", "terminal"), enabled=True, routed_toolsets=routed) is ExecutionTarget.SERVER


def test_unknown_toolsets_do_not_advertise_effective_routing():
    assert normalize_routed_toolsets(["terminal", "unknown"]) == frozenset()
