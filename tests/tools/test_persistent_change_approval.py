"""Tests for the opt-in "approval-required persistent changes" switch (#110429).

``agent.require_persistent_change_approval`` (default **false**) arms the write-approval gate
for every control-file subsystem at once — memory (MEMORY.md / USER.md), skills, and
config.yaml. With it on, no control file changes without an accepted approval: the write is
staged in ``<HERMES_HOME>/pending/<subsystem>/`` and replayed only by
``hermes pending approve <id>`` (or ``/memory approve`` / ``/skills approve``).

Covers: default-off is a no-op, the switch arms all three subsystems, a gated write leaves
the control file byte-identical, approve applies, reject discards, and the CLI review surface
(``hermes pending``) lists/approves/rejects across subsystems.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_pca_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _write_raw_config(hermes_home, **sections):
    """Merge ``{section: {key: value}}`` into the profile's config.yaml (bypassing the gate)."""
    path = Path(hermes_home) / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw if isinstance(raw, dict) else {}
    for section, values in sections.items():
        raw.setdefault(section, {}).update(values)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _enable_switch(hermes_home):
    _write_raw_config(hermes_home, agent={"require_persistent_change_approval": True})


def _config_yaml(hermes_home) -> str:
    path = Path(hermes_home) / "config.yaml"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _memory_md(hermes_home) -> str:
    path = Path(hermes_home) / "memories" / "MEMORY.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ---------------------------------------------------------------------------
# The switch itself
# ---------------------------------------------------------------------------

def test_switch_defaults_to_false_in_schema():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["agent"]["require_persistent_change_approval"] is False


def test_switch_off_leaves_every_gate_off(hermes_home):
    from tools import write_approval as wa
    assert wa.persistent_change_approval_required() is False
    for subsystem in (wa.MEMORY, wa.SKILLS, wa.CONFIG):
        assert wa.write_approval_enabled(subsystem) is False


def test_switch_arms_every_subsystem_at_once(hermes_home):
    from tools import write_approval as wa
    _enable_switch(hermes_home)
    assert wa.persistent_change_approval_required() is True
    for subsystem in (wa.MEMORY, wa.SKILLS, wa.CONFIG):
        assert wa.write_approval_enabled(subsystem) is True


def test_switch_is_read_per_call_not_frozen(hermes_home):
    """A mid-session flip must take effect without a restart (and without a module reload)."""
    from tools import write_approval as wa
    assert wa.persistent_change_approval_required() is False
    _enable_switch(hermes_home)
    assert wa.persistent_change_approval_required() is True


def test_per_subsystem_flag_still_works_without_switch(hermes_home):
    """The global switch is additive: memory.write_approval alone still gates memory only."""
    from tools import write_approval as wa
    _write_raw_config(hermes_home, memory={"write_approval": True})
    assert wa.write_approval_enabled(wa.MEMORY) is True
    assert wa.write_approval_enabled(wa.SKILLS) is False
    assert wa.write_approval_enabled(wa.CONFIG) is False


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------

def test_config_write_unchanged_when_switch_off(hermes_home):
    """Regression guard: with the switch off, `hermes config set` writes exactly as before."""
    from hermes_cli import config as cfg
    from tools import write_approval as wa

    cfg.set_config_value("agent.run_budget_seconds", "42")
    assert "42" in _config_yaml(hermes_home)
    assert wa.all_pending() == []


def test_config_write_staged_when_switch_on(hermes_home, capsys):
    from hermes_cli import config as cfg
    from tools import write_approval as wa

    cfg.set_config_value("agent.run_budget_seconds", "42")  # seed a real value
    _enable_switch(hermes_home)

    cfg.set_config_value("agent.run_budget_seconds", "99")

    # Nothing landed: config.yaml still holds the seeded value and the change waits for review.
    on_disk = yaml.safe_load(_config_yaml(hermes_home)) or {}
    assert on_disk["agent"]["run_budget_seconds"] == 42
    records = wa.all_pending()
    assert len(records) == 1
    assert records[0]["subsystem"] == wa.CONFIG
    assert records[0]["payload"] == {"key": "agent.run_budget_seconds", "value": "99", "force": False}
    assert "Staged for approval" in capsys.readouterr().out


def test_config_approve_applies_and_drains_queue(hermes_home):
    from hermes_cli import config as cfg
    from tools import write_approval as wa

    cfg.set_config_value("agent.run_budget_seconds", "42")
    _enable_switch(hermes_home)
    cfg.set_config_value("agent.run_budget_seconds", "99")

    record = wa.all_pending()[0]
    result = wa.approve_pending(wa.CONFIG, record["id"])
    assert result["success"] is True, result
    assert wa.all_pending() == []
    assert "99" in _config_yaml(hermes_home)


def test_config_reject_leaves_file_untouched(hermes_home):
    from hermes_cli import config as cfg
    from tools import write_approval as wa

    cfg.set_config_value("agent.run_budget_seconds", "42")
    _enable_switch(hermes_home)
    cfg.set_config_value("agent.run_budget_seconds", "99")
    before = _config_yaml(hermes_home)

    record = wa.all_pending()[0]
    assert wa.discard_pending(wa.CONFIG, record["id"]) is True

    assert wa.all_pending() == []
    assert _config_yaml(hermes_home) == before
    assert "99" not in _config_yaml(hermes_home)


def test_config_unset_is_gated_too(hermes_home):
    """Removing a key is a config.yaml change; it must not be an un-approved way around the gate."""
    from hermes_cli import config as cfg
    from tools import write_approval as wa

    cfg.set_config_value("agent.run_budget_seconds", "42")
    _enable_switch(hermes_home)
    before = _config_yaml(hermes_home)

    cfg.unset_config_value("agent.run_budget_seconds")
    assert _config_yaml(hermes_home) == before
    records = wa.all_pending()
    assert len(records) == 1 and records[0]["payload"] == {"key": "agent.run_budget_seconds", "unset": True}

    assert wa.approve_pending(wa.CONFIG, records[0]["id"])["success"] is True
    assert "run_budget_seconds" not in _config_yaml(hermes_home)


def test_staged_config_value_is_masked_in_the_record(hermes_home):
    """A credential-shaped value must not be echoed in the staged summary."""
    from hermes_cli import config as cfg
    from tools import write_approval as wa

    _enable_switch(hermes_home)
    cfg.set_config_value("model.api_key", "sk-super-secret-value")

    record = wa.all_pending()[0]
    assert "sk-super-secret-value" not in record["summary"]
    # The payload still carries the real value so approval can replay it exactly.
    assert record["payload"]["value"] == "sk-super-secret-value"


# ---------------------------------------------------------------------------
# memory (MEMORY.md / USER.md) — the switch drives the pre-existing gate
# ---------------------------------------------------------------------------

def test_memory_write_staged_when_switch_on(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _enable_switch(hermes_home)
    store = MemoryStore()
    store.load_from_disk()

    result = json.loads(memory_tool("add", "memory", "switch staged this", store=store))
    assert result.get("staged") is True, result
    assert result["pending_id"]
    assert store.memory_entries == []
    assert "switch staged this" not in _memory_md(hermes_home)
    assert wa.pending_count(wa.MEMORY) == 1


def test_memory_approve_writes_to_disk(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _enable_switch(hermes_home)
    staging = MemoryStore()
    staging.load_from_disk()
    result = json.loads(memory_tool("add", "memory", "approved by hand", store=staging))
    assert result.get("staged") is True, result

    store = MemoryStore()
    store.load_from_disk()
    assert wa.approve_pending(wa.MEMORY, result["pending_id"], memory_store=store)["success"] is True

    assert wa.pending_count(wa.MEMORY) == 0
    assert "approved by hand" in _memory_md(hermes_home)


def test_memory_reject_discards(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _enable_switch(hermes_home)
    store = MemoryStore()
    store.load_from_disk()
    result = json.loads(memory_tool("add", "memory", "never mind", store=store))

    assert wa.discard_pending(wa.MEMORY, result["pending_id"]) is True
    assert wa.pending_count(wa.MEMORY) == 0
    assert _memory_md(hermes_home) == ""


# ---------------------------------------------------------------------------
# skills — same switch, same queue
# ---------------------------------------------------------------------------

_SKILL_MD = "---\nname: gated-skill\ndescription: A skill behind the gate\n---\n# Gated\n"


def _skill_path(hermes_home) -> Path:
    return Path(hermes_home) / "skills" / "gated-skill" / "SKILL.md"


def test_skill_write_staged_when_switch_on(hermes_home):
    from tools import write_approval as wa
    from tools.skill_manager_tool import skill_manage

    _enable_switch(hermes_home)
    result = json.loads(skill_manage("create", "gated-skill", content=_SKILL_MD))
    assert result.get("staged") is True, result
    assert not _skill_path(hermes_home).exists()
    assert wa.pending_count(wa.SKILLS) == 1


def test_skill_approve_creates_the_skill(hermes_home):
    from tools import write_approval as wa
    from tools.skill_manager_tool import skill_manage

    _enable_switch(hermes_home)
    result = json.loads(skill_manage("create", "gated-skill", content=_SKILL_MD))
    assert result.get("staged") is True, result

    applied = wa.approve_pending(wa.SKILLS, result["pending_id"])
    assert applied["success"] is True, applied
    assert _skill_path(hermes_home).exists()
    assert "A skill behind the gate" in _skill_path(hermes_home).read_text(encoding="utf-8")


def test_skill_write_unchanged_when_switch_off(hermes_home):
    from tools import write_approval as wa
    from tools.skill_manager_tool import skill_manage

    result = json.loads(skill_manage("create", "gated-skill", content=_SKILL_MD))
    assert result.get("success") is True, result
    assert result.get("staged") is None
    assert _skill_path(hermes_home).exists()
    assert wa.pending_count(wa.SKILLS) == 0


# ---------------------------------------------------------------------------
# Review surface: `hermes pending`
# ---------------------------------------------------------------------------

def _pending_args(argv):
    import argparse
    from hermes_cli.subcommands.pending import build_pending_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_pending_parser(subparsers)
    args = parser.parse_args(["pending"] + argv)
    return args.func(args)


def test_cli_lists_and_approves_across_subsystems(hermes_home, capsys):
    from hermes_cli import config as cfg
    from tools import write_approval as wa

    cfg.set_config_value("agent.run_budget_seconds", "42")
    _enable_switch(hermes_home)
    cfg.set_config_value("agent.run_budget_seconds", "99")
    pending_id = wa.all_pending()[0]["id"]

    assert _pending_args([]) == 0
    out = capsys.readouterr().out
    assert pending_id in out and "config" in out

    assert _pending_args(["show", pending_id]) == 0
    assert "agent.run_budget_seconds" in capsys.readouterr().out

    assert _pending_args(["approve", pending_id]) == 0
    assert "99" in _config_yaml(hermes_home)
    assert wa.all_pending() == []


def test_cli_reject_all_discards_everything(hermes_home, capsys):
    from hermes_cli import config as cfg
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    cfg.set_config_value("agent.run_budget_seconds", "42")
    _enable_switch(hermes_home)
    cfg.set_config_value("agent.run_budget_seconds", "99")
    store = MemoryStore()
    store.load_from_disk()
    memory_tool("add", "memory", "also staged", store=store)
    before = _config_yaml(hermes_home)

    assert _pending_args(["reject", "all"]) == 0
    capsys.readouterr()

    assert wa.all_pending() == []
    assert _config_yaml(hermes_home) == before
    assert _memory_md(hermes_home) == ""


def test_cli_status_reports_the_switch(hermes_home, capsys):
    _enable_switch(hermes_home)
    assert _pending_args(["status"]) == 0
    out = capsys.readouterr().out
    assert "agent.require_persistent_change_approval = on" in out
    assert "config  gate=on" in out


def test_cli_approve_unknown_id_fails(hermes_home, capsys):
    assert _pending_args(["approve", "deadbeef"]) == 1
    assert "No pending write" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Shared command handler (`/memory`, `/skills`) still replays through the same dispatcher
# ---------------------------------------------------------------------------

def test_shared_handler_approves_a_staged_config_write(hermes_home):
    """handle_pending_subcommand(config, ...) uses the shared apply_pending dispatcher."""
    from hermes_cli import config as cfg
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    cfg.set_config_value("agent.run_budget_seconds", "42")
    _enable_switch(hermes_home)
    cfg.set_config_value("agent.run_budget_seconds", "99")

    out = handle_pending_subcommand(wa.CONFIG, ["approve", "all"])
    assert "Approved 1" in out, out
    assert wa.pending_count(wa.CONFIG) == 0
    assert "99" in _config_yaml(hermes_home)


def test_shared_handler_reports_the_global_switch(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    _enable_switch(hermes_home)
    out = handle_pending_subcommand(wa.MEMORY, [])
    assert "on (via agent.require_persistent_change_approval)" in out, out
