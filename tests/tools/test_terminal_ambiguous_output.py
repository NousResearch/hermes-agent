from __future__ import annotations

import json


def test_cmd_banner_only_exit_zero_is_ambiguous_no_machine_evidence(tmp_path):
    from tools.terminal_tool import _classify_terminal_result

    banner = (
        "Microsoft Windows [Version 10.0.19045.0000]\n"
        "(c) Microsoft Corporation. All rights reserved.\n\n"
        "C:\\Users\\Admin>"
    )

    classification = _classify_terminal_result(
        command="cmd /c hermes chat --resume session --query-file prompt.txt",
        output=banner,
        returncode=0,
        cwd=str(tmp_path),
        env_type="local",
    )

    assert classification["classification"] == "ambiguous_no_machine_evidence"
    assert classification["reason"] == "windows_cmd_banner_only"


def test_parser_output_exit_zero_is_parser_or_launch_failure(tmp_path):
    from tools.terminal_tool import _classify_terminal_result

    classification = _classify_terminal_result(
        command="python -m hermes_cli.main chat --oneshot --resume abc",
        output="usage: hermes chat [-h]\nhermes chat: error: unrecognized arguments: --resume abc\n",
        returncode=0,
        cwd=str(tmp_path),
        env_type="local",
    )

    assert classification["classification"] == "parser_or_launch_failure"
    assert classification["reason"] == "parser_or_launch_output"


def test_missing_expected_output_artifact_is_not_ok(tmp_path):
    from tools.terminal_tool import _classify_terminal_result

    artifact = tmp_path / "bundle.zip"

    classification = _classify_terminal_result(
        command=f"python build_archive.py --out {artifact}",
        output="archive complete\n",
        returncode=0,
        cwd=str(tmp_path),
        env_type="local",
    )

    assert classification["classification"] == "artifact_missing"
    assert classification["reason"] == "expected_artifact_missing"
    assert classification["missing_artifacts"] == [str(artifact)]


def test_replacement_decoded_output_is_classified_as_decode_recovered(tmp_path):
    from tools.terminal_tool import _classify_terminal_result

    classification = _classify_terminal_result(
        command="python emit_cp1250.py",
        output="out: Za\ufffd\ufffd\ufffd\ufffd\n",
        returncode=0,
        cwd=str(tmp_path),
        env_type="local",
    )

    assert classification["classification"] == "decode_recovered"
    assert classification["reason"] == "replacement_characters_present"


def test_terminal_tool_includes_result_classification(monkeypatch, tmp_path):
    import tools.terminal_tool as terminal_tool

    class FakeEnv:
        cwd = str(tmp_path)
        env = {}

        def execute(self, command, **kwargs):
            return {
                "output": "Microsoft Windows [Version 10.0.19045.0000]\nC:\\Users\\Admin>",
                "returncode": 0,
            }

    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": str(tmp_path),
            "timeout": 30,
            "local_persistent": False,
            "host_cwd": None,
        },
    )
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, has_host_access=False: {"approved": True},
    )

    payload = json.loads(terminal_tool.terminal_tool("cmd /c wrapper", task_id="t1"))

    assert payload["exit_code"] == 0
    assert payload["result_classification"] == "ambiguous_no_machine_evidence"
    assert payload["classification_reason"] == "windows_cmd_banner_only"