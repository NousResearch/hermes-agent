"""Machine consumers must not parse a terminal display preview as complete JSON."""

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.code_execution_tool import generate_hermes_tools_module
from tools.environments.base_output import _BoundedOutputCollector, _finalize_wait_result
from tools.terminal_tool_result import finalize_foreground_result


def test_collector_reports_loss_without_a_spill():
    collector = _BoundedOutputCollector(100)
    collector.append(json.dumps({"text": "x" * 300}))
    result = _finalize_wait_result(collector, collector.render(), 0)
    assert result["output_truncated"] is True
    assert result["output_total_chars"] == collector.total_chars
    assert "full_output_path" not in result


def test_suffix_truncation_is_reported():
    collector = _BoundedOutputCollector(100)
    collector.append("x" * 95)
    result = _finalize_wait_result(collector, collector.render(suffix="[timeout]"), 1)
    assert result["output_truncated"] is True


def test_final_output_cap_reports_loss(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    raw = json.dumps({"text": "x" * 300})
    result = json.loads(finalize_foreground_result(
        command="example", result={"output": raw, "returncode": 0},
        env=SimpleNamespace(cwd=str(tmp_path)), env_type="local",
        effective_task_id="json-preview", task_id=None, session_id=None,
        session_key="json-preview", workdir=None, command_cwd=str(tmp_path),
        approval_note=None,
    ))
    assert result["output_truncated"] is True
    assert result["output_total_chars"] == len(raw)


@pytest.mark.parametrize("transport", ["uds", "file"])
def test_json_helper_rejects_truncated_result_even_if_preview_parses(transport):
    namespace = {}
    exec(generate_hermes_tools_module([], transport=transport), namespace)
    parse = namespace["json_parse"]
    # strict=False can otherwise silently accept a notice inside a JSON string.
    preview = '{"text":"head\n[OUTPUT TRUNCATED]\ntail"}'
    assert json.loads(preview, strict=False)
    with pytest.raises(ValueError, match="truncated"):
        parse({"output": preview, "output_truncated": True, "exit_code": 0})
    assert parse({"output": '{"ok":true}', "exit_code": 0}) == {"ok": True}
    assert parse('\ufeff{"text":"a\nb"}') == {"text": "a\nb"}


@pytest.mark.parametrize("transport", ["uds", "file"])
def test_json_helper_rejects_a_bare_output_string_carrying_the_truncation_notice(transport):
    """Callers that pass ``result["output"]`` (not the whole result) must not get corrupted data
    either: the head/tail notice landed inside a SHA-256 string and strict=False parsed a short,
    damaged manifest as if it were complete (independent reproduction on #128137)."""
    import hashlib

    from tools.tool_output_truncate import truncate_head_tail

    namespace = {}
    exec(generate_hermes_tools_module([], transport=transport), namespace)
    parse = namespace["json_parse"]
    manifest = [{"path": f"f{i}", "sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(292)]
    raw = json.dumps(manifest)
    preview = truncate_head_tail(raw, 2000)
    corrupted = json.loads(preview, strict=False)  # the silent-corruption shape
    assert len(corrupted) < len(manifest)
    assert any("TRUNCATED" in entry.get("sha256", "") for entry in corrupted if isinstance(entry, dict))

    with pytest.raises(ValueError, match="truncation notice"):
        parse(preview)
    assert parse(raw) == manifest
    # Valid JSON that merely quotes the notice (escaped newlines) is still data, not a preview.
    quoted = json.dumps({"log": truncate_head_tail("y" * 5000, 100)})
    assert "TRUNCATED" in parse(quoted)["log"]


@pytest.mark.platforms("posix")
def test_registered_terminal_json_spill_roundtrip(monkeypatch, tmp_path):
    import tools.terminal_tool  # registers the actual handler
    from tools.registry import registry

    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 2000)
    command = shlex.quote(sys.executable) + " -c " + shlex.quote(
        "import json; print(json.dumps({'text': 'x' * 6000, 'last': 42}))"
    )
    result = json.loads(registry.dispatch("terminal", {"command": command},
                                          task_id="json-spill-roundtrip"))
    assert result["exit_code"] == 0, result
    assert result["output_truncated"] is True
    full = json.loads(Path(result["full_output_path"]).read_text())
    assert full == {"text": "x" * 6000, "last": 42}


@pytest.mark.platforms("posix")
def test_execute_code_rejects_preview_and_recovers_complete_json(monkeypatch):
    from tools.code_execution_tool import execute_code
    from tools.code_kernel import shutdown_all_kernels

    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 2000)
    command = shlex.quote(sys.executable) + " -c " + shlex.quote(
        "import json; print(json.dumps({'text': 'x' * 6000}))"
    )
    code = f'''
from hermes_tools import terminal, json_parse, shell_quote
result = terminal({command!r})
try:
    json_parse(result)
except ValueError as exc:
    assert "truncated" in str(exc)
else:
    raise AssertionError("accepted a truncated preview")
try:
    json_parse(result["output"])
except ValueError as exc:
    assert "truncation notice" in str(exc)
else:
    raise AssertionError("accepted a truncated preview passed as a bare string")
script = "import json; from pathlib import Path; print(len(json.loads(Path(" + repr(result['full_output_path']) + ").read_text())['text']))"
recovered = terminal({shlex.quote(sys.executable)!r} + " -c " + shell_quote(script))
assert json_parse(recovered) == 6000, recovered
print("complete JSON recovered")
'''
    try:
        result = json.loads(execute_code(code, task_id="json-preview-rpc", enabled_tools=["terminal"]))
        assert result["status"] == "success", result
        assert "complete JSON recovered" in result["output"]
    finally:
        shutdown_all_kernels()
