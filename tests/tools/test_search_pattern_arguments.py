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
pytestmark = pytest.mark.platforms("linux")


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
    "pattern", ("--example-option", "--version", "-e", "--", "-", "->", "-77", "--accent|--surface")
)
def test_option_shaped_patterns_are_searched_as_regex(ops, tmp_path, mode, pattern):
    path = tmp_path / "sample.txt"
    text = pattern.split("|")[0]
    path.write_text(text + "\nplain control\n")
    result = ops.search(
        pattern, path=str(tmp_path), output_mode=mode, file_glob="*.txt"
    )
    assert_result(result, mode, path, [(1, text)])


@pytest.mark.parametrize("lane", ("rg-native", "rg-shell"))
@pytest.mark.parametrize(
    "pattern,text,hidden,hint",
    [
        ("--accent", "--Accent", False, "case-insensitive"),
        ("--hidden-needle", "--hidden-needle", True, "hidden"),
        ("--value[0]", "--value[0]", False, "literal"),
    ],
)
def test_zero_match_probes_accept_option_shaped_patterns(ops, tmp_path, pattern, text, hidden, hint):
    root = tmp_path / ".hidden" if hidden else tmp_path
    root.mkdir(exist_ok=True)
    (root / "sample.txt").write_text(text + "\n")
    result = ops.search(pattern, path=str(tmp_path))
    assert result.error is None
    assert result.total_count == 0
    assert hint in (result.warning or "")
