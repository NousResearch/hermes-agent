import json
from types import SimpleNamespace


def test_patch_tool_cache_clear_receives_task_id(monkeypatch, tmp_path):
    import tools.file_tools as file_mod
    import tools.execution_targets as targets_mod

    targets_mod.set_execution_target_config_source({"terminal": {"backend": "local"}})
    path = tmp_path / "sample.txt"
    path.write_text("before")
    observed = []

    class FakeFileOps:
        def patch_replace(self, resolved_path, old_string, new_string, replace_all):
            assert resolved_path == str(path)
            assert (old_string, new_string, replace_all) == ("before", "after", False)
            return SimpleNamespace(to_dict=lambda: {"output": "Done!"})

    def legacy_cache_getter(task_id):
        observed.append(task_id)
        return FakeFileOps()

    monkeypatch.setattr(file_mod, "_get_file_ops", legacy_cache_getter)

    result = file_mod.patch_tool(
        path=str(path),
        old_string="before",
        new_string="after",
        task_id="legacy-task",
    )

    if isinstance(result, str):
        result = json.loads(result)
    assert not result.get("error")
    assert observed == ["legacy-task"]
    targets_mod.set_execution_target_config_source(None)
