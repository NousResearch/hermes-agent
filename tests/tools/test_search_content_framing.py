"""Search locations must not be inferred from separator-shaped file content."""

import json
import shlex
import shutil
from unittest.mock import Mock

import pytest

from tools import file_operations, file_tools, terminal_tool
from tools.environments.local import LocalEnvironment
from tools.file_operations import ExecuteResult, ShellFileOperations
from tools.file_operations_search import _parse_search_output


@pytest.fixture(scope="module")
def local_env(tmp_path_factory):
    return LocalEnvironment(cwd=str(tmp_path_factory.mktemp("search-framing")))


@pytest.fixture(params=["rg-default", "rg-shell", "grep",
                        pytest.param("grep-pruned", marks=pytest.mark.macos_only)])
def search(request, tmp_path, monkeypatch, local_env):
    engine = "grep" if request.param.startswith("grep") else "rg"
    executable = shutil.which(engine)
    if executable is None:
        pytest.skip(f"{engine} is not installed")
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1" if request.param == "rg-default" else "0")
    monkeypatch.setattr(local_env, "cwd", str(tmp_path))
    ops = ShellFileOperations(local_env, cwd=str(tmp_path))
    monkeypatch.setattr(ops, "_has_command", lambda command: command == engine)
    if engine == "rg":
        ops._rg_resolution_cache["rg"] = executable
    pipeline = Mock(wraps=ops._run_search_pipeline)
    native = Mock(wraps=ops._run_rg_native)
    monkeypatch.setattr(ops, "_run_search_pipeline", pipeline)
    monkeypatch.setattr(ops, "_run_rg_native", native)
    task_id = request.node.nodeid
    env_id = terminal_tool._resolve_container_task_id(task_id)
    monkeypatch.setitem(terminal_tool._active_environments, env_id, local_env)
    monkeypatch.setitem(file_tools._file_ops_cache, env_id, ops)
    if request.param == "grep-pruned":
        monkeypatch.setattr(file_operations, "_HOME", str(tmp_path))
        protected = tmp_path / "Downloads"
        protected.mkdir()
        (protected / "ignored.txt").write_text("NEEDLE\n", encoding="utf-8")

    def run(**kwargs):
        pipeline.reset_mock()
        native.reset_mock()
        if request.param == "grep-pruned":
            kwargs["path"] = str(tmp_path)
        result = json.loads(file_tools.search_tool(task_id=task_id, **kwargs))
        pipeline.assert_called_once()
        argv = shlex.split(" ".join(pipeline.call_args.args[0]))
        if request.param == "grep-pruned":
            assert argv[0] == "find"
            assert argv[argv.index("-exec") + 1] == "grep"
        else:
            assert argv[0] == ("grep" if engine == "grep" else executable)
        assert native.called is (engine == "rg" and ops._native_read_enabled())
        return result

    return run


@pytest.mark.parametrize("filename", ["notes.txt", "notes-12-name.txt", "sub dir/notes-12-name.txt"])
@pytest.mark.parametrize("neighbor", ["plain context", "release-2026-final", "release-2026-final:42:status",
                                      "", "--", "  release-2026-final  "])
def test_context_locations_round_trip(search, tmp_path, filename, neighbor):
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = [neighbor, "NEEDLE-123-tail:88:end", neighbor]
    path.write_text("\n".join(contents) + "\n", encoding="utf-8")
    original = path.read_bytes()

    result = search(pattern="NEEDLE", path=str(path), context=1)

    assert "error" not in result
    assert result["total_count"] == len(contents)
    assert result["matches"] == [
        {"path": str(path), "line": number, "content": text}
        for number, text in enumerate(contents, 1)
    ]
    without_context = search(pattern="NEEDLE", path=str(path), context=0)
    assert "error" not in without_context, without_context
    assert without_context["matches"] == [{"path": str(path), "line": 2, "content": contents[1]}]
    page = search(pattern="NEEDLE", path=str(path), context=1, offset=1, limit=1)
    assert "error" not in page, page
    assert page["matches"] == without_context["matches"]
    assert page["truncated"] is True
    assert path.read_bytes() == original


@pytest.mark.parametrize("path", ["dir/notes-12-name.txt", r"C:\work\notes-12-name.txt",
                                 "rg: notes-12-name.txt", "dir/notes-12-name:34: with space.txt"])
@pytest.mark.parametrize("exit_code", [0, 2, 124, 141])
def test_framed_records_preserve_partial_results(path, exit_code):
    before = "release-2026-final:42:status"
    after = "tail-17-end "
    output = f"{path}\x001-{before}\n{path}\x002:NEEDLE\n--\n{path}\x003-{after}\n"
    if exit_code == 2:
        output += "grep: missing.txt: No such file or directory\n"
    elif exit_code == 124:
        output += "[Command timed out after 60s]\n"

    result = _parse_search_output(ExecuteResult(output, exit_code), "content", 2, 1, 1)

    assert result.error is None
    assert [(m.path, m.line_number, m.content) for m in result.matches] == [
        (path, 2, "NEEDLE"), (path, 3, after)
    ]
    assert result.total_count == 3
    assert result.truncated is (exit_code == 124)
    assert result.limit_reason == ("search_timeout" if exit_code == 124 else None)
