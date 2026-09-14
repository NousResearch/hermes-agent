"""Synthetic result-level regressions for regex arguments that resemble options.

Real LocalEnvironment, rg and grep; no mocked search output. The pruned lane
exercises find/grep on POSIX with a synthetic exclusion, not macOS TCC itself.
"""

import json
import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

LANES = ("rg-native", "rg-shell", "grep", "grep-pruned")
MODES = ("content", "count", "files_only")
pytestmark = pytest.mark.linux_only


@pytest.fixture(scope="module")
def local_env(tmp_path_factory):
    return LocalEnvironment(cwd=str(tmp_path_factory.mktemp("pattern-arguments")))


@pytest.fixture
def ops(tmp_path, local_env, monkeypatch, lane):
    required = ("rg",) if lane.startswith("rg") else ("grep", "find")
    if not all(shutil.which(cmd) for cmd in required):
        pytest.skip("Real search executable unavailable: " + ", ".join(required))
    local_env.cwd = str(tmp_path)
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1" if lane == "rg-native" else "0")
    instance = ShellFileOperations(local_env)
    if lane.startswith("grep"):
        resolve = instance._resolve_command
        monkeypatch.setattr(
            instance,
            "_resolve_command",
            lambda cmd: None if cmd == "rg" else resolve(cmd),
        )
    if lane == "grep-pruned":
        excluded = tmp_path / "excluded"
        excluded.mkdir()
        (excluded / "sentinel.txt").write_text("--example-option\nneedle one\n")
        monkeypatch.setattr(
            instance, "_protected_prune_paths", lambda path: [str(excluded)]
        )
    native_calls, commands = [], []
    native, execute = instance._run_rg_native, instance._exec

    def record_native(*args, **kwargs):
        native_calls.append(args)
        return native(*args, **kwargs)

    def record_shell(command, *args, **kwargs):
        commands.append(command)
        return execute(command, *args, **kwargs)

    monkeypatch.setattr(instance, "_run_rg_native", record_native)
    monkeypatch.setattr(instance, "_exec", record_shell)
    yield instance
    if lane == "rg-native":
        assert native_calls, "The native transport must actually execute"
        assert not any("pipefail" in cmd for cmd in commands)
    else:
        assert not native_calls
        assert any("pipefail" in cmd for cmd in commands)
    if lane == "grep-pruned":
        assert any("-exec grep" in cmd for cmd in commands)


def assert_result(result, mode, path, expected_lines):
    assert result.error is None
    if mode == "content":
        assert [(m.path, m.line_number, m.content) for m in result.matches] == [
            (str(path), line, text) for line, text in expected_lines
        ]
    elif mode == "count":
        assert result.counts == {str(path): len(expected_lines)}
    else:
        assert result.files == [str(path)]


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "pattern", ("--example-option", "--version", "-e", "--", "-", "->", "-77")
)
def test_literal_option_patterns(ops, tmp_path, mode, pattern):
    path = tmp_path / "sample.txt"
    path.write_text(pattern + "\nplain control\n")
    result = ops.search(
        pattern, path=str(tmp_path), output_mode=mode, file_glob="*.txt"
    )
    assert_result(result, mode, path, [(1, pattern)])


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "pattern,lines",
    [
        ("--accent|--surface", ["--accent: blue;", "--surface: white;"]),
        ("needle (one|two)", ["needle one", "needle two"]),
    ],
)
def test_regex_semantics_and_successful_controls(ops, tmp_path, mode, pattern, lines):
    path = tmp_path / "sample.css"
    path.write_text("\n".join(lines) + "\nplain control\n")
    (tmp_path / "ignored.txt").write_text("\n".join(lines) + "\n")
    result = ops.search(
        pattern, path=str(tmp_path), output_mode=mode, file_glob="*.css"
    )
    assert_result(result, mode, path, list(enumerate(lines, 1)))


@pytest.mark.parametrize("lane", ("rg-native", "rg-shell", "grep"))
def test_invalid_regex_is_rejected(ops, tmp_path):
    # The pruned find/grep lane already folds grep errors into an empty result;
    # changing that separate error-propagation contract is not part of this fix.
    (tmp_path / "sample.txt").write_text("plain control\n")
    result = ops.search("(", path=str(tmp_path))
    assert result.error
    assert not result.matches


@pytest.mark.parametrize("lane", LANES)
def test_no_match_control(ops, tmp_path):
    (tmp_path / "sample.txt").write_text("plain control\n")
    result = ops.search("absent_synthetic_needle", path=str(tmp_path))
    assert result.error is None
    assert result.total_count == 0


@pytest.mark.parametrize("lane", ("rg-native", "rg-shell"))
@pytest.mark.parametrize(
    "pattern,text,hidden,hint",
    [
        ("--accent", "--Accent", False, "case-insensitive"),
        ("--hidden-needle", "--hidden-needle", True, "hidden"),
        ("--value[0]", "--value[0]", False, "literal"),
    ],
)
def test_zero_match_probes(ops, tmp_path, pattern, text, hidden, hint):
    root = tmp_path / ".hidden" if hidden else tmp_path
    root.mkdir(exist_ok=True)
    (root / "sample.txt").write_text(text + "\n")
    result = ops.search(pattern, path=str(tmp_path))
    assert result.error is None
    assert result.total_count == 0
    assert hint in (result.warning or "")


@pytest.mark.parametrize("lane", LANES)
def test_registered_search_tool(ops, tmp_path, monkeypatch):
    from tools import file_tools
    from tools.registry import registry

    path = tmp_path / "sample.txt"
    path.write_text("--example-option\nplain control\n")
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: ops)
    result = json.loads(
        registry.dispatch(
            "search_files",
            {"pattern": "--example-option", "path": str(path)},
            task_id="synthetic-pattern-arguments",
        )
    )
    assert not result.get("error")
    assert result["matches"] == [
        {"path": str(path), "line": 1, "content": "--example-option"}
    ]
