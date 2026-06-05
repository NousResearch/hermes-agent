"""Tests for tools/workflow_storage.py, tools/routine_storage.py and
tools/workflow_routine_tools.py (the save_workflow + create_routine
pivot — hermes-agent-202606-028 / desktop-202606-514)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Clear any bound-folder leakage from the host shell.
    monkeypatch.delenv("AGENTE_BOUND_FOLDER", raising=False)
    yield tmp_path


def _import_modules():
    import tools.routine_storage as routines_mod
    import tools.workflow_routine_tools as tool_mod
    import tools.workflow_storage as workflows_mod

    importlib.reload(workflows_mod)
    importlib.reload(routines_mod)
    importlib.reload(tool_mod)
    return workflows_mod, routines_mod, tool_mod


def test_save_workflow_writes_yaml_with_hebrew(hermes_home):
    workflows, _, _ = _import_modules()
    saved = workflows.save_workflow(
        {
            "id": "wf-urgent-wa",
            "name_he": "סינון הודעות דחופות מוואטסאפ",
            "description_he": "הודעות שמכילות 'דחוף' הופכות לכרטיס חדש",
            "trigger_kind": "wa_incoming_message",
            "triage_keywords_he": ["דחוף", "חירום"],
            "actions": [
                {"kind": "create_ticket", "title_template": "פנייה דחופה"}
            ],
        }
    )
    assert saved["id"] == "wf-urgent-wa"
    assert saved["version"] == "1"
    assert saved["trigger"] == {"kind": "wa_incoming_message"}
    assert saved["triage"] == {"keywords_he": ["דחוף", "חירום"]}

    path: Path = hermes_home / "workflows" / "wf-urgent-wa.yaml"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert "דחוף" in raw
    parsed = yaml.safe_load(raw)
    assert parsed["actions"][0]["kind"] == "create_ticket"


def test_save_workflow_mirrors_to_bound_folder(hermes_home, tmp_path, monkeypatch):
    bound = tmp_path / "bound-workspace"
    monkeypatch.setenv("AGENTE_BOUND_FOLDER", str(bound))
    workflows, _, _ = _import_modules()
    # Apply the adapter's mirror wrapper (the logic+env handling was moved out of core
    # storage into the thin adapter; tests exercise it explicitly here).
    from gateway.agente_desktop_adapter import workflow_routine_bridge as br
    workflows.save_workflow = br._wrap_save_with_mirror(
        workflows.save_workflow, br._bound_workflows_dir
    )
    workflows.save_workflow(
        {
            "id": "wf-mirror",
            "name_he": "x",
            "trigger_kind": "manual",
        }
    )
    canonical = hermes_home / "workflows" / "wf-mirror.yaml"
    mirror = bound / "office" / "workflows" / "wf-mirror.yaml"
    assert canonical.is_file()
    assert mirror.is_file()
    assert yaml.safe_load(canonical.read_text()) == yaml.safe_load(mirror.read_text())


def test_save_workflow_rejects_unknown_trigger_kind(hermes_home):
    workflows, _, _ = _import_modules()
    with pytest.raises(ValueError):
        workflows.save_workflow(
            {"id": "bad", "name_he": "x", "trigger_kind": "nope"}
        )


def test_save_workflow_rejects_unsafe_id(hermes_home):
    workflows, _, _ = _import_modules()
    with pytest.raises(ValueError):
        workflows.save_workflow(
            {"id": "../escape", "name_he": "x", "trigger_kind": "manual"}
        )


def test_save_workflow_handler_returns_success_json(hermes_home):
    _, _, tools = _import_modules()
    out = tools.save_workflow_handler(
        id="wf-1",
        name_he="שלום",
        trigger_kind="manual",
    )
    payload = json.loads(out)
    assert payload["success"] is True
    assert payload["workflow"]["id"] == "wf-1"


def test_save_workflow_handler_surfaces_validation_error(hermes_home):
    _, _, tools = _import_modules()
    out = tools.save_workflow_handler(id="", name_he="x", trigger_kind="manual")
    payload = json.loads(out)
    assert payload.get("success") is False
    assert "error" in payload


def test_create_routine_persists_yaml_record(hermes_home, monkeypatch):
    # Avoid touching the real cron module — return a stable fake job id so we
    # exercise the storage path deterministically.
    _, routines, tools = _import_modules()

    def fake_create_cron_job(routine_id, workflow_id, name_he, cron_schedule):
        return "fake-cron-id"

    monkeypatch.setattr(
        tools, "_create_cron_job_for_routine", fake_create_cron_job
    )

    out = tools.create_routine_handler(
        id="rt-hourly",
        workflow_id="wf-urgent-wa",
        name_he="כל שעה",
        cron_schedule="0 * * * *",
        natural_language_schedule_he="בכל שעה עגולה",
    )
    payload = json.loads(out)
    assert payload["success"] is True
    record = payload["routine"]
    assert record["id"] == "rt-hourly"
    assert record["workflow_id"] == "wf-urgent-wa"
    assert record["cron_schedule"] == "0 * * * *"
    assert record["cron_job_id"] == "fake-cron-id"
    assert record["natural_language_schedule_he"] == "בכל שעה עגולה"

    path = hermes_home / "routines" / "rt-hourly.yaml"
    assert path.is_file()
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["workflow_id"] == "wf-urgent-wa"


def test_create_routine_handler_requires_fields(hermes_home, monkeypatch):
    _, _, tools = _import_modules()
    monkeypatch.setattr(
        tools, "_create_cron_job_for_routine", lambda **_kw: None
    )
    out = tools.create_routine_handler(
        id="",
        workflow_id="wf",
        name_he="x",
        cron_schedule="0 * * * *",
    )
    payload = json.loads(out)
    assert payload.get("success") is False


def test_new_tools_registered_in_registry():
    """Importing the module registers both handlers under toolset 'workflows'."""
    import tools.workflow_routine_tools  # noqa: F401 — import for side effects
    from tools.registry import registry

    save_entry = registry.get_entry("save_workflow")
    routine_entry = registry.get_entry("create_routine")
    assert save_entry is not None and save_entry.toolset == "workflows"
    assert routine_entry is not None and routine_entry.toolset == "workflows"


def test_new_tools_in_toolsets_and_legacy_map():
    from toolsets import _HERMES_CORE_TOOLS, TOOLSETS
    from model_tools import _LEGACY_TOOLSET_MAP

    for name in ("save_workflow", "create_routine"):
        assert name in _HERMES_CORE_TOOLS
        assert name in TOOLSETS["workflows"]["tools"]
        assert name in _LEGACY_TOOLSET_MAP["workflow_routine_tools"]
