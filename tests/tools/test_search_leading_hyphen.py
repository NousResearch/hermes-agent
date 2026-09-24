from tools.file_operations import ShellFileOperations
from tools.file_operations_common import ExecuteResult


class _Env:
    cwd = "/workspace"


def _ops():
    return ShellFileOperations(_Env())


def test_rg_content_search_terminates_options_before_leading_hyphen_pattern(monkeypatch):
    ops = _ops()
    captured = {}

    def fake_pipeline(parts, *args, **kwargs):
        captured["parts"] = parts
        return ExecuteResult(stdout="", exit_code=1)

    monkeypatch.setattr(ops, "_run_search_pipeline", fake_pipeline)
    ops._search_with_rg("--literal", "/workspace", None, 50, 0, "content", 0, rg_executable="rg")

    parts = captured["parts"]
    separator = parts.index("--")
    assert parts[separator + 1] == ops._escape_shell_arg("--literal")
    assert parts[separator + 2] == ops._escape_native_tool_arg("/workspace")


def test_rg_content_search_keeps_normal_pattern_after_option_separator(monkeypatch):
    ops = _ops()
    captured = {}

    def fake_pipeline(parts, *args, **kwargs):
        captured["parts"] = parts
        return ExecuteResult(stdout="", exit_code=1)

    monkeypatch.setattr(ops, "_run_search_pipeline", fake_pipeline)
    ops._search_with_rg("needle", "/workspace", "*.py", 50, 0, "files_only", 2, rg_executable="rg")

    parts = captured["parts"]
    separator = parts.index("--")
    assert "-C" in parts
    assert "--glob" in parts
    assert "-l" in parts
    assert parts[separator + 1] == ops._escape_shell_arg("needle")
    assert parts[separator + 2] == ops._escape_native_tool_arg("/workspace")