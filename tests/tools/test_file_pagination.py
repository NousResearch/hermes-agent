"""Non-finite JSON pagination inputs fall back to the normal file-tool defaults."""

import json

import pytest

from tools.file_operations_common import (
    normalize_read_pagination,
    normalize_search_pagination,
)


@pytest.mark.parametrize("normalize", [normalize_read_pagination, normalize_search_pagination])
@pytest.mark.parametrize("field", ["offset", "limit"])
@pytest.mark.parametrize("number", ["1e309", "-1e309", "NaN"])
def test_nonfinite_pagination_uses_default(normalize, field, number):
    """Overflowing JSON numbers must fall back without changing the other bound."""
    args = {"offset": 2, "limit": 3}
    args.update(json.loads(f'{{"{field}": {number}}}'))
    expected = normalize(**{key: value for key, value in args.items() if key != field})
    assert normalize(**args) == expected


@pytest.mark.parametrize("tool", ["read_file", "search_files"])
@pytest.mark.parametrize("field", ["offset", "limit"])
@pytest.mark.parametrize("number", ["1e309", "-1e309"])
def test_registry_pagination_reads_real_files(tmp_path, monkeypatch, tool, field, number):
    """Schema-bypassing JSON must still produce file results through real dispatch."""
    from tools import file_tools
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    from tools.registry import registry

    path = tmp_path / "sample.txt"
    path.write_text("".join(f"needle {i}\n" for i in range(60)), encoding="utf-8")
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    ops = ShellFileOperations(env, cwd=str(tmp_path))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: ops)
    monkeypatch.setattr(file_tools, "_read_tracker", {})
    args = {"path": str(path)}
    if tool == "search_files":
        args["pattern"] = "needle"
    try:
        expected = json.loads(registry.dispatch(tool, args, task_id="pagination-default"))
        assert "error" not in expected
        assert expected.get("content") or expected.get("total_count")
        args.update(json.loads(f'{{"{field}": {number}}}'))
        actual = json.loads(registry.dispatch(tool, args, task_id="pagination-nonfinite"))
        assert actual == expected
    finally:
        env.cleanup()
