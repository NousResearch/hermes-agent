"""Execute the helper usage taught by the public schema and recovery hints."""

import json
import re

import pytest

from tools import code_execution_tool
from tools.code_kernel import shutdown_all_kernels
from tools.registry import registry


PROBE = '''
from __future__ import annotations
{imports}
import json
print(json.dumps([json_parse('{{"value": 7}}')["value"],
                  shell_quote("two words"), retry(lambda: "ok", delay=0)]))
'''


@pytest.fixture(autouse=True)
def local_kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("code_execution:\n  mode: strict\n", encoding="utf-8")
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def run(code, reset=False):
    return json.loads(registry.dispatch("execute_code", {"code": code, "reset": reset},
                                        task_id="helper-contract", enabled_tools=[]))


def test_schema_helper_instructions_work_across_kernel_lifetimes():
    description = registry.get_schema("execute_code")["description"]
    imports = "\n".join(re.findall(r"`(from hermes_tools import [\w, ]+)`", description))
    for reset, reused in ((False, False), (False, True), (True, False)):
        result = run(PROBE.format(imports=imports), reset=reset)
        assert result["status"] == "success", result
        assert json.loads(result["output"]) == [7, "'two words'", "ok"]
        assert result["kernel"]["reused"] is reused


def test_missing_helper_recovery_instructions_execute():
    for helper in ("json_parse", "shell_quote", "retry"):
        failed = run(f"print({helper})", reset=True)
        assert failed["status"] == "error", failed
        instruction = re.search(r"from hermes_tools import \w+", failed["hint"])
        assert instruction, failed
        recovered = run(instruction.group() + f"\nprint(callable({helper}))")
        assert recovered["status"] == "success", recovered
        assert recovered["output"].strip() == "True"
        skew = code_execution_tool._sandbox_failure_hint(
            f"ImportError: cannot import name '{helper}' from 'hermes_tools'")
        assert instruction.group() in skew
