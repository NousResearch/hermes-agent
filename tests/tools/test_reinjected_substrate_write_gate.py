"""HERMES_HOME reinjected/unattended substrates share the always-ask gate.

Regression for the parity gap #117818 describes: the protected-instruction gate
exempted the whole ~/.hermes tree claiming the area was covered by "config.yaml
hard-block, mirror guard, write_approval" — but none of those guards sit on the
generic write_file/patch path, so memories/**, cron/jobs.json,
webhook_subscriptions.json, scripts/**, and skills/** were writable headless
while their content is auto-reinjected into future agent context or acts
unattended. These tests pin the four substrate shapes GATED (same one-operation,
no-yolo, fail-closed semantics as AGENTS.md), and pin what stays exempt: home
files off the auto-reinjection path.
"""

import json

import pytest


@pytest.fixture(autouse=True)
def _gate_on(monkeypatch):
    import tools.file_tools_write_guards as ft
    monkeypatch.setattr(ft, "_protected_instruction_config", lambda: (True, []))
    yield


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME bound via the per-turn override, real-shaped dirs."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    home = tmp_path / ".hermes"
    for d in ("memories", "cron", "skills/demo", "scripts"):
        (home / d).mkdir(parents=True)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def approvals(monkeypatch):
    # Isolate from a polluting -q/cron parent shell: with these set, the
    # approval context correctly fails closed WITHOUT asking, and the
    # "callback was asked" assertion would spuriously fail.
    for var in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION",
                "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION"):
        monkeypatch.delenv(var, raising=False)
    from tools.terminal_tool import set_approval_callback
    state = {"calls": [], "answer": "deny"}

    def cb(command, description, **kwargs):
        state["calls"].append({"command": command, "description": description, **kwargs})
        return state["answer"]

    set_approval_callback(cb)
    yield state
    set_approval_callback(None)


def _write(path, content="poisoned"):
    from tools.file_tools import write_file_tool
    return json.loads(write_file_tool(str(path), content))


# ---- gated substrates ------------------------------------------------------

@pytest.mark.parametrize("rel", [
    "memories/MEMORY.md",
    "memories/USER.md",
    "cron/jobs.json",
    "webhook_subscriptions.json",
    "scripts/watchdog.sh",
    "skills/demo/SKILL.md",
    "skills/demo/helper.py",
], ids=["memory", "user", "jobs", "webhooks", "cron_script", "skill_md", "skill_py"])
def test_substrate_write_requires_approval(hermes_home, approvals, rel):
    target = hermes_home / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    res = _write(target)
    assert res.get("error") and "BLOCKED" in res["error"], res
    assert not target.exists()
    assert len(approvals["calls"]) == 1
    # Same contract as AGENTS.md: one-operation grant, nothing persisted.
    assert approvals["calls"][0]["allow_session"] is False


def test_substrate_write_approve_lands(hermes_home, approvals):
    approvals["answer"] = "once"
    target = hermes_home / "memories/MEMORY.md"
    res = _write(target, "approved memory")
    assert not res.get("error"), res
    assert target.read_text(encoding="utf-8") == "approved memory"


def test_substrate_headless_fails_closed(hermes_home, monkeypatch):
    """No callback (cron/-q/worker): block without asking, don't hang."""
    from tools.terminal_tool import set_approval_callback
    set_approval_callback(None)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    target = hermes_home / "cron/jobs.json"
    res = _write(target)
    assert res.get("error") and "no interactive user" in res["error"]
    assert not target.exists()


def test_kill_switch_disables_substrate_gate(hermes_home, approvals, monkeypatch):
    """Same security.protected_instruction_files switch governs both halves."""
    import tools.file_tools_write_guards as ft
    monkeypatch.setattr(ft, "_protected_instruction_config", lambda: (False, []))
    res = _write(hermes_home / "memories/MEMORY.md", "ok")
    assert not res.get("error"), res
    assert approvals["calls"] == []


# ---- what stays exempt -----------------------------------------------------

def test_home_direct_files_stay_exempt(hermes_home, approvals):
    """Home files off the auto-reinjection path keep the plain exemption —
    including a root-level SOUL.md (already basename-gated nowhere here since
    the exemption precedes basename matching) and kanban/session state."""
    for rel in ("config-backup-notes.md", "kanban/report.md", "logs/x.txt"):
        target = hermes_home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        res = _write(target, "fine")
        assert not res.get("error"), rel
    assert approvals["calls"] == []


def test_sibling_home_is_active_home_scoped(hermes_home, approvals, tmp_path):
    """Scope decision (documented, pins the design): substrate coverage keys off
    the ACTIVE home — the gate protects the store THIS session reinjects.
    Another profile's same-OS-user tree is not exempted as "this home", and not
    gated as a substrate either: it is neither on this session's auto-reinjection
    path nor project-local ``.hermes`` config, and the cross-profile guard was
    deliberately removed upstream. That store's OWN sessions gate it on write."""
    other = tmp_path / "other-profile"
    (other / "memories").mkdir(parents=True)
    res = _write(other / "memories/MEMORY.md", "their store")
    assert not res.get("error"), res
    assert approvals["calls"] == []
    # ...while THIS home's store of the identical shape stays gated.
    res = _write(hermes_home / "memories/MEMORY.md", "mine")
    assert res.get("error") and "BLOCKED" in res["error"]
    assert len(approvals["calls"]) == 1


def test_reason_labels_are_relative_paths(hermes_home):
    import tools.file_tools_write_guards as ft
    home = str(hermes_home)
    assert ft._reinjected_substrate_label(f"{home}/memories/MEMORY.md", home) == "memories/MEMORY.md"
    assert ft._reinjected_substrate_label(f"{home}/cron/jobs.json", home) == "cron/jobs.json"
    assert ft._reinjected_substrate_label(f"{home}/scripts/x.sh", home) == "scripts/x.sh"
    assert ft._reinjected_substrate_label(f"{home}/webhook_subscriptions.json", home) == "webhook_subscriptions.json"
    # negatives: home dir itself, outside, bare "cron" dir, cron output files
    assert ft._reinjected_substrate_label(home, home) is None
    assert ft._reinjected_substrate_label("/etc/passwd", home) is None
    assert ft._reinjected_substrate_label(f"{home}/cron", home) is None
    assert ft._reinjected_substrate_label(f"{home}/cron/output/run1.txt", home) is None
    assert ft._reinjected_substrate_label(f"{home}/kanban/kanban.db", home) is None
