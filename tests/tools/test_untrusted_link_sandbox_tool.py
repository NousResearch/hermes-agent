import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import untrusted_link_sandbox_tool as tool


def make_sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    (root / "bin").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "quarantine" / "downloads").mkdir(parents=True)
    (root / "docker-compose.yml").write_text("services: {}\n")
    for wrapper in tool.REQUIRED_WRAPPERS:
        path = root / "bin" / wrapper
        path.write_text("#!/usr/bin/env bash\n")
        path.chmod(0o755)
    return root


def test_tool_available_with_expected_stack(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))

    assert tool._tool_available() is True


def test_tool_unavailable_when_any_required_wrapper_is_missing(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    (root / "bin" / "audit-url").unlink()
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))

    assert tool._tool_available() is False


def test_toolset_is_cli_configurable():
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS

    assert "untrusted_link_sandbox" in {name for name, _, _ in CONFIGURABLE_TOOLSETS}


def test_host_quarantine_path_is_converted(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    target = root / "quarantine" / "downloads" / "sample.txt"
    target.write_text("hello")
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))

    assert tool._containerish_target(str(target)) == "/quarantine/downloads/sample.txt"


def test_triage_invokes_wrapper_and_reads_summary(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    report = root / "reports" / "run.md"
    summary = root / "reports" / "run.json"
    report.write_text("# Report\n\nUseful excerpt\n")
    summary.write_text(json.dumps({"target": "https://example.com", "type": "url", "report_path": "/reports/run.md"}))
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))

    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout="/reports/run.md\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = json.loads(tool.untrusted_link_triage("https://example.com", deep=True))

    assert captured["args"] == ["./bin/triage", "--deep", "https://example.com"]
    assert captured["cwd"] == str(root)
    assert result["success"] is True
    assert result["report_path_host"] == str(report)
    assert result["summary"]["type"] == "url"
    assert result["summary"]["report_path_host"] == str(report)
    assert "Useful excerpt" in result["report_excerpt"]


def test_audit_repo_wrapper(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout="/reports/repo.md\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = json.loads(tool.audit_untrusted_repo("https://github.com/octocat/Hello-World"))

    assert captured["args"] == ["./bin/audit-repo", "https://github.com/octocat/Hello-World"]
    assert result["success"] is True


def test_inspect_download_converts_host_path(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    target = root / "quarantine" / "downloads" / "sample.txt"
    target.write_text("hello")
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout="/reports/download.md\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    json.loads(tool.inspect_untrusted_download(str(target)))

    assert captured["args"] == ["./bin/inspect-download", "/quarantine/downloads/sample.txt"]


def test_untrusted_tools_are_available_in_noninteractive_profiles():
    from toolsets import resolve_toolset

    expected = {
        "untrusted_link_triage",
        "audit_untrusted_url",
        "audit_untrusted_repo",
        "inspect_untrusted_download",
    }

    assert expected.issubset(resolve_toolset("hermes-cli"))
    assert expected.issubset(resolve_toolset("hermes-api-server"))
    assert expected.issubset(resolve_toolset("hermes-acp"))

    # Newer branches define a dedicated cron platform toolset; older Azure main
    # may not. When present, cron should expose the same safety triage tools.
    from toolsets import TOOLSETS

    if "hermes-cron" in TOOLSETS:
        assert expected.issubset(resolve_toolset("hermes-cron"))


def test_bin_triage_contract_uses_last_report_line_and_bounded_output(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    report = root / "reports" / "contract.md"
    summary = root / "reports" / "contract.json"
    report.write_text("# Contract report\n")
    summary.write_text(json.dumps({"verdict": "clean", "risk_level": "low", "report_path": "/reports/contract.md"}))
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))
    captured = {}

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        stdout = "noise before report\n/reports/old.md\n" + ("x" * 5000) + "\n/reports/contract.md\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="e" * 5000)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = json.loads(tool.untrusted_link_triage("https://example.com", timeout_seconds=9999))

    assert captured["timeout"] == tool.MAX_TIMEOUT_SECONDS
    assert result["report_path"] == "/reports/contract.md"
    assert result["report_path_host"] == str(report)
    assert result["summary"]["summary_path"] == str(summary)
    assert result["summary"]["verdict"] == "clean"
    assert len(result["stdout_tail"]) == 4000
    assert len(result["stderr_tail"]) == 4000


def test_bin_triage_contract_failure_without_report_is_json(monkeypatch, tmp_path):
    root = make_sandbox(tmp_path)
    monkeypatch.setenv("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(root))

    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=2, stdout="validation failed\n", stderr="bad target\n")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = json.loads(tool.untrusted_link_triage("not-a-target", timeout_seconds=1))

    assert result["success"] is False
    assert result["exit_code"] == 2
    assert result["report_path"] is None
    assert result["summary"] == {}
    assert result["stdout_tail"] == "validation failed"
    assert result["stderr_tail"] == "bad target"
