"""Log-derived recovery cases: absent commands and invalid external responses."""

import json
import subprocess

import pytest

from tools.terminal_hints import annotate_failure, annotate_masked_success


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("command", [
    "forge_nonexistent_utility; true",
    "bash -c 'forge_nonexistent_utility; true'",
    "forge_nonexistent_utility || true",
])
def test_shell_success_does_not_hide_missing_command(command):
    result = subprocess.run(["/bin/sh", "-c", command], text=True, capture_output=True, timeout=5, check=False)
    assert result.returncode == 0
    hint = annotate_masked_success(command, result.stdout + result.stderr)
    assert hint and "forge_nonexistent_utility" in hint
    assert "exit" in hint.lower()


@pytest.mark.parametrize("command", [
    "cat synthetic.log | tail -3",
    "grep 'not found' synthetic.log | head",
    "printf 'sh: missing: not found' | cat",
    "python3 script.py --label ';'",
])
def test_missing_command_text_in_read_output_is_not_an_execution_claim(command):
    assert annotate_masked_success(command, "sh: missing: not found\n") is None


@pytest.mark.parametrize("output, required", [
    ("shodan.exception.APIError: Invalid search query", "query"),
    ("json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)", "response"),
    ("nmap: unrecognized option '--synthetic-invalid'", "help"),
    ("zsh:1: command not found: forge_missing", "forge_missing"),
])
def test_actual_failure_diagnostic_gives_bounded_recovery(output, required):
    hint = annotate_failure("python3 synthetic.py", 1, output)
    assert hint and required in hint.lower()


def test_embedded_diagnostic_in_source_text_is_not_misclassified():
    assert annotate_failure("compiler", 1, '    print("shodan.exception.APIError: Invalid search query")') is None
    assert annotate_failure("compiler", 1, "JSONDecodeError documentation example") is None


def test_nmap_unresolved_target_is_not_treated_as_a_host_down_result():
    output = ('Failed to resolve "fixture-a.invalid,fixture-b.invalid".\n'
              'WARNING: No targets were specified, so 0 hosts scanned.\n')
    hint = annotate_masked_success("nmap -sT fixture-a.invalid,fixture-b.invalid", output)
    assert hint and "target" in hint.lower() and "down" in hint.lower()
    assert annotate_masked_success("cat synthetic-nmap.log", output) is None


@pytest.mark.platforms("posix")
def test_real_registry_result_preserves_exit_status_and_exposes_masked_failure(tmp_path, monkeypatch):
    import tools.terminal_tool  # registers the real handler
    from tools.registry import registry

    monkeypatch.chdir(tmp_path)
    result = json.loads(registry.dispatch(
        "terminal", {"command": "forge_nonexistent_utility; true", "timeout": 5},
        task_id="forge-masked-registry",
    ))
    assert result["exit_code"] == 0
    assert result["error"] is None
    assert "forge_nonexistent_utility" in result["hint"]
