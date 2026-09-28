"""Regression for #113421: execute_code must not silently bypass the config.yaml guard.

write_file/patch refuse the active profile's config.yaml via
``_check_sensitive_path``; execute_code ran arbitrary Python with a plain
``open(path, "w")`` and no gate. Static analysis of the script cannot hold,
so execute_code snapshots the protected file before dispatch and fails loudly
when the cell mutated it.
"""

import json

import pytest

import hermes_cli.config as hc
from tools.file_tools_write_guards import (
    hermes_config_mutated,
    snapshot_hermes_config_state,
)


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\n"
        "approvals:\n  mode: manual\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    yield home
    hc._LOAD_CONFIG_CACHE.clear()


def test_snapshot_quiet_when_untouched(config_home):
    before = snapshot_hermes_config_state()
    assert before and before["exists"]
    changed, _ = hermes_config_mutated(before)
    assert changed is False


def test_snapshot_detects_append(config_home):
    before = snapshot_hermes_config_state()
    cfg = config_home / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "\nhooks:\n  x: 1\n", encoding="utf-8")
    changed, detail = hermes_config_mutated(before)
    assert changed is True
    assert detail and "config" in detail.lower()


def test_snapshot_detects_delete(config_home):
    before = snapshot_hermes_config_state()
    (config_home / "config.yaml").unlink()
    changed, detail = hermes_config_mutated(before)
    assert changed is True
    assert detail and "deleted" in detail.lower()


def test_execute_code_reports_direct_config_write(config_home):
    from tools.code_execution_tool import execute_code

    cfg = (config_home / "config.yaml").as_posix()
    code = (
        f"p = {cfg!r}\n"
        "s = open(p, encoding='utf-8').read()\n"
        "open(p, 'w', encoding='utf-8').write(s + '\\nhooks:\\n  x: 1\\n')\n"
        "print('wrote')\n"
    )
    raw = execute_code(code=code, task_id="test-113421-mutate", reset=True)
    body = json.loads(raw)
    assert body.get("status") == "error", f"config write must fail loudly, got: {raw[:300]}"
    assert "config" in (body.get("error") or "").lower()


def test_execute_code_clean_script_still_succeeds(config_home):
    from tools.code_execution_tool import execute_code

    raw = execute_code(code="print(1 + 1)", task_id="test-113421-clean", reset=True)
    body = json.loads(raw)
    assert body.get("status") in {"ok", "success"}, f"clean script must succeed, got: {raw[:300]}"
    assert "2" in (body.get("output") or "")
