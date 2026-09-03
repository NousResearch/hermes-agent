"""Adversarial Blocker 1 tests — process-boundary contract (review #100694).

Every attack below must fail mechanically:

    valid_grant=NO
    qga_create_filesystem_calls=0

Attack matrix (mandate §9 Track E):

1.  direct issue_grant from execute_code (child process)
2.  write arbitrary grant JSON (write_file surface)
3.  copy valid grant
4.  change grant_id
5.  change filename UUID
6.  extend expires_at
7.  change issued_at
8.  change authorization_source
9.  replace receipt_id
10. replay same human receipt
11. issue second grant from same receipt
12. terminal hermes grant issue (hardline + CLI hard-fail)
13. terminal pty=true (hardline)
14. yolo
15. approvals.mode=off
16. cron permissive
17. smart approval / aux LLM
18. subagent (child process provider)
19. child process provider
20. restart provider generation

Legitimate-flow witness (the ONLY issuer is the consumer process):

    running Hermes process B
    -> model requests governed operation
    -> mocked REAL human approve_once
    -> B observes incarnation
    -> B issues grant with authority B
    -> B claims grant
    -> exactly one qga_create_filesystem call
    -> completed

Isolation proofs:

- parent Hermes PID != execute_code kernel PID
- execute_code cannot retrieve the parent's process authority secret
- the secret is not in os.environ / /proc/self/environ / subprocess env
- repr(provider) / exceptions / logs never leak the secret
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import destructive_grants as dg
from tools import grant_authority as ga
from tools.governed_mkfs_tool import _handle_governed_mkfs

from tests.tools.test_grant_authority_lifecycle import (
    _INCARNATION,
    _make_receipt,
    _mock_approval,
    _mock_qga,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _issue(**kw):
    from tests.tools.test_grant_authority_lifecycle import _issue as _base_issue

    return _base_issue(**kw)


def _call(**kw):
    from tests.tools.test_grant_authority_lifecycle import _call as _base_call

    return _base_call(**kw)


def _grant_path(grant_id):
    return dg._grant_path(grant_id)


def _read_grant(grant_id):
    return json.loads(_grant_path(grant_id).read_text())


def _write_grant(grant_id, data):
    dg._ensure_dir()
    _grant_path(grant_id).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# G1 — import direct / globals access from a child process
# ---------------------------------------------------------------------------


def test_child_process_has_different_authority_generation():
    """A child process (execute_code kernel) imports its own provider with a
    DIFFERENT secret and generation: it cannot authenticate parent grants."""
    parent_gen = ga.get_authority().generation
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from tools import grant_authority as ga; "
        "print(ga.get_authority().generation)" % os.getcwd()
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    child_gen = out.stdout.strip()
    assert child_gen
    assert child_gen != parent_gen


def test_child_process_cannot_verify_parent_grant():
    """A grant minted in the parent is rejected by a child process provider
    (generation mismatch => authentication failure)."""
    grant = _issue()
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from tools import destructive_grants as dg\n"
        "try:\n"
        "    dg.load_grant(%r)\n"
        "    print('LOADED')\n"
        "except dg.GrantDeniedError:\n"
        "    print('DENIED')\n"
        % (os.getcwd(), grant.grant_id)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "DENIED"


def test_child_process_cannot_issue_valid_grant():
    """issue_grant from a child process produces a grant the PARENT rejects."""
    code = (
        "import sys, time, uuid; sys.path.insert(0, %r); "
        "from tools import destructive_grants as dg; "
        "from tools import grant_authority as ga; "
        "from tools.grant_authority import HumanApprovalReceipt, _store_receipt; "
        "r = _store_receipt(HumanApprovalReceipt("
        "receipt_id='rcpt-'+uuid.uuid4().hex, request_id='r'*32, "
        "request_digest='d'*64, session_id='sess-1', turn_id='t', "
        "tool_call_id='tc', operation='CREATE_FILESYSTEM', vm_id='101', "
        "device='/dev/sdb1', fs_type='ext4', label='X', "
        "incarnation_product_uuid='uuid-A', incarnation_boot_id='boot-A', "
        "incarnation_hostname='storage-guest', "
        "issued_at=time.time(), expires_at=time.time()+600)); "
        "g = dg.issue_grant(operation='CREATE_FILESYSTEM', vm_id='101', "
        "hostname='storage-guest', device='/dev/sdb1', fs_type='ext4', label='X', "
        "authorization_subject='operator', session_id='sess-1', "
        "receipt_id=r.receipt_id, incarnation_product_uuid='uuid-A', "
        "incarnation_boot_id='boot-A', incarnation_hostname='storage-guest'); "
        "print(g.grant_id)"
        % os.getcwd()
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    child_grant_id = out.stdout.strip()
    assert child_grant_id
    # The PARENT must reject the child-minted grant (different key).
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(child_grant_id)


# ---------------------------------------------------------------------------
# G2 — forge file (write_file surface)
# ---------------------------------------------------------------------------


def test_forge_arbitrary_grant_json_denied():
    """write_file fabricating a grant JSON with a fake tag is rejected."""
    forged = {
        "grant_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "operation": "CREATE_FILESYSTEM",
        "vm_id": "101",
        "hostname": "storage-guest",
        "device": "/dev/sdb1",
        "fs_type": "ext4",
        "label": "X",
        "authorization_subject": "operator",
        "authorization_source": "USER",
        "session_id": "sess-1",
        "issued_at": time.time(),
        "expires_at": time.time() + 600,
        "nonce": "0" * 32,
        "auth_tag": "0" * 64,
        "authority_generation": ga.get_authority().generation,
        "receipt_id": "rcpt-" + "0" * 28,
        "schema_version": 1,
        "authorization_evidence": {},
        "incarnation_product_uuid": "uuid-A",
        "incarnation_boot_id": "boot-A",
        "incarnation_hostname": "storage-guest",
    }
    _write_grant(forged["grant_id"], forged)
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(forged["grant_id"])


def test_forge_with_copied_tag_from_other_grant_denied():
    """Copying a valid grant's tag onto a forged payload is rejected."""
    valid = _issue()
    forged = _read_grant(valid.grant_id)
    forged["grant_id"] = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    forged["device"] = "/dev/sdc1"  # different target
    _write_grant(forged["grant_id"], forged)
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(forged["grant_id"])


def test_forge_ttl_extension_denied():
    grant = _issue()
    data = _read_grant(grant.grant_id)
    data["expires_at"] = data["expires_at"] + 3600
    _write_grant(grant.grant_id, data)
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(grant.grant_id)


def test_forge_issued_at_change_denied():
    grant = _issue()
    data = _read_grant(grant.grant_id)
    data["issued_at"] = data["issued_at"] - 100
    _write_grant(grant.grant_id, data)
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(grant.grant_id)


def test_forge_authorization_source_denied():
    grant = _issue()
    data = _read_grant(grant.grant_id)
    data["authorization_source"] = "MODEL"
    _write_grant(grant.grant_id, data)
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(grant.grant_id)


def test_forge_receipt_id_denied():
    grant = _issue()
    data = _read_grant(grant.grant_id)
    data["receipt_id"] = "rcpt-" + "9" * 28
    _write_grant(grant.grant_id, data)
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(grant.grant_id)


def test_forge_clone_under_new_filename_denied():
    grant = _issue()
    data = _read_grant(grant.grant_id)
    new_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    data["grant_id"] = new_id
    _write_grant(new_id, data)
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(new_id)


# ---------------------------------------------------------------------------
# G3 — receipt replay
# ---------------------------------------------------------------------------


def test_receipt_replay_denied():
    """The same human receipt cannot issue a second grant."""
    receipt = _make_receipt()
    g1 = _issue(receipt=receipt)
    assert g1.receipt_id == receipt.receipt_id
    with pytest.raises(dg.GrantError):
        _issue(receipt=receipt)


def test_receipt_serialized_replay_denied():
    """Replaying the serialized receipt payload (as the model would see it in
    the grant file) cannot mint a second grant: the receipt store is
    process-local and the id is consumed."""
    receipt = _make_receipt()
    g1 = _issue(receipt=receipt)
    serialized = g1.authorization_evidence  # what the model can read
    assert serialized["receipt_id"] == receipt.receipt_id
    # The model cannot re-insert the receipt into the store: consume_receipt
    # only accepts ids present in the process-local store.
    with pytest.raises(dg.GrantError):
        dg.issue_grant(
            operation="CREATE_FILESYSTEM",
            vm_id="101", hostname="storage-guest", device="/dev/sdb1",
            fs_type="ext4", label="X",
            authorization_subject="operator", session_id="sess-1",
            receipt_id=str(serialized["receipt_id"]),
            incarnation_product_uuid="uuid-A",
            incarnation_boot_id="boot-A", incarnation_hostname="storage-guest",
        )


# ---------------------------------------------------------------------------
# G4 — environment / logs / repr leak
# ---------------------------------------------------------------------------


def test_secret_not_in_environment():
    code = (
        "import sys, os; sys.path.insert(0, %r); "
        "from tools import grant_authority as ga; "
        "a = ga.get_authority(); "
        "print(any('grant' in k.lower() or 'authority' in k.lower() "
        "for k in os.environ)); "
        "print(repr(a)); "
        "print(hasattr(a, '_secret'))"
        % os.getcwd()
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == "False"  # no grant/authority env var
    assert "generation" in lines[1] and "_secret" not in lines[1]
    assert lines[2] == "True"  # attribute exists but repr never shows it


def test_secret_not_in_proc_self_environ():
    env = os.environ.copy()
    env.pop("HERMES_YOLO_MODE", None)
    code = (
        "import sys, os\n"
        "sys.path.insert(0, %r)\n"
        "from tools import grant_authority as ga\n"
        "a = ga.get_authority()\n"
        "secret = a._secret\n"
        "data = open('/proc/self/environ','rb').read()\n"
        "print(secret in data)\n"
        % os.getcwd()
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
        env=env,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_secret_not_in_subprocess_environment():
    """A child spawned by the parent does not inherit the secret."""
    code = (
        "import sys, os, subprocess\n"
        "sys.path.insert(0, %r)\n"
        "from tools import grant_authority as ga\n"
        "a = ga.get_authority()\n"
        "secret = a._secret\n"
        "out = subprocess.run([sys.executable, '-c', "
        "'import os; print(repr(os.environ))'], "
        "capture_output=True, text=True, env=os.environ.copy())\n"
        "print(secret in out.stdout.encode())\n"
        % os.getcwd()
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# G5 — restart semantics
# ---------------------------------------------------------------------------


def test_restart_invalidates_live_grants():
    """A new provider generation (restart) rejects previously minted grants."""
    grant = _issue()
    assert dg.load_grant(grant.grant_id)  # valid under current provider
    ga.reset_authority_for_tests()  # simulate process restart
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(grant.grant_id)


def test_restart_denies_consumption():
    """A grant minted before a restart cannot be claimed/consumed after it:
    the authority generation check rejects it before any effect."""
    grant = _issue()
    ga.reset_authority_for_tests()
    with pytest.raises(dg.GrantDeniedError):
        dg.claim_grant(grant.grant_id, "exec-after-restart")


# ---------------------------------------------------------------------------
# J — terminal / yolo / mode=off / cron bypass
# ---------------------------------------------------------------------------


def test_terminal_grant_issue_hardline_blocked():
    from tools.approval import detect_hardline_command

    for cmd in (
        "hermes grant issue --operation CREATE_FILESYSTEM --vm 101 --device /dev/sdb1 --fs ext4 --label X --session s1",
        "sudo hermes grant issue --vm 1",
        "echo hi; hermes grant issue --vm 1",
        "hermes grant issue --vm 1 --json",
    ):
        is_hl, desc = detect_hardline_command(cmd)
        assert is_hl, f"hardline missed: {cmd!r}"


def test_terminal_pty_grant_issue_hardline_blocked():
    from tools.approval import detect_hardline_command

    # pty=true does not change the command string: still hardline-blocked.
    is_hl, _ = detect_hardline_command(
        "hermes grant issue --operation CREATE_FILESYSTEM --vm 101 --device /dev/sdb1 --fs ext4 --label X --session s1"
    )
    assert is_hl


def test_yolo_cannot_bypass_grant_issue_hardline(monkeypatch):
    from tools.approval import (
        check_all_command_guards,
        check_dangerous_command,
        detect_hardline_command,
    )

    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    cmd = "hermes grant issue --operation CREATE_FILESYSTEM --vm 101 --device /dev/sdb1 --fs ext4 --label X --session s1"
    assert detect_hardline_command(cmd)[0] is True
    r1 = check_dangerous_command(cmd, "local")
    assert r1["approved"] is False and r1.get("hardline") is True
    r2 = check_all_command_guards(cmd, "local")
    assert r2["approved"] is False and r2.get("hardline") is True


def test_approvals_mode_off_cannot_bypass_grant_issue_hardline(monkeypatch):
    import tools.approval as approval_mod
    from tools.approval import check_all_command_guards

    monkeypatch.setattr(approval_mod, "_get_approval_mode", lambda: "off")
    cmd = "hermes grant issue --operation CREATE_FILESYSTEM --vm 101 --device /dev/sdb1 --fs ext4 --label X --session s1"
    r = check_all_command_guards(cmd, "local")
    assert r["approved"] is False and r.get("hardline") is True


def test_cron_approve_cannot_bypass_grant_issue_hardline(monkeypatch):
    import tools.approval as approval_mod
    from tools.approval import check_all_command_guards

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(approval_mod, "_get_cron_approval_mode", lambda: "approve")
    cmd = "hermes grant issue --operation CREATE_FILESYSTEM --vm 101 --device /dev/sdb1 --fs ext4 --label X --session s1"
    r = check_all_command_guards(cmd, "local")
    assert r["approved"] is False and r.get("hardline") is True


def test_grant_issue_list_revoke_not_hardline():
    """list/revoke/audit stay usable (read-only / revocation)."""
    from tools.approval import detect_hardline_command

    for cmd in ("hermes grant list", "hermes grant revoke abc", "hermes grant audit"):
        assert detect_hardline_command(cmd)[0] is False, cmd


# ---------------------------------------------------------------------------
# J — approval gate refuses bypass contexts
# ---------------------------------------------------------------------------


def test_approval_gate_refuses_yolo(monkeypatch):
    import tools.approval as approval_mod
    from tools.grant_authority import ReceiptError, request_destructive_grant_approval

    monkeypatch.setattr(approval_mod, "is_approval_bypass_active", lambda: True)
    with pytest.raises(ReceiptError):
        request_destructive_grant_approval(
            operation="CREATE_FILESYSTEM", vm_id="101", device="/dev/sdb1",
            fs_type="ext4", label="X", session_id="sess-1", ttl_seconds=600,
        )


def test_approval_gate_refuses_cron(monkeypatch):
    import tools.approval as approval_mod
    from tools.grant_authority import ReceiptError, request_destructive_grant_approval

    monkeypatch.setattr(approval_mod, "is_approval_bypass_active", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_cron_approval_context", lambda: True)
    with pytest.raises(ReceiptError):
        request_destructive_grant_approval(
            operation="CREATE_FILESYSTEM", vm_id="101", device="/dev/sdb1",
            fs_type="ext4", label="X", session_id="sess-1", ttl_seconds=600,
        )


def test_approval_gate_refuses_unattended(monkeypatch):
    import tools.approval as approval_mod
    from tools.grant_authority import ReceiptError, request_destructive_grant_approval

    monkeypatch.setattr(approval_mod, "is_approval_bypass_active", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_unattended_platform_approval_context", lambda: True)
    with pytest.raises(ReceiptError):
        request_destructive_grant_approval(
            operation="CREATE_FILESYSTEM", vm_id="101", device="/dev/sdb1",
            fs_type="ext4", label="X", session_id="sess-1", ttl_seconds=600,
        )


def test_approval_gate_refuses_single_query(monkeypatch):
    import tools.approval as approval_mod
    from tools.grant_authority import ReceiptError, request_destructive_grant_approval

    monkeypatch.setattr(approval_mod, "is_approval_bypass_active", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_single_query_approval_context", lambda: True)
    with pytest.raises(ReceiptError):
        request_destructive_grant_approval(
            operation="CREATE_FILESYSTEM", vm_id="101", device="/dev/sdb1",
            fs_type="ext4", label="X", session_id="sess-1", ttl_seconds=600,
        )


def test_approval_gate_refuses_no_human_surface(monkeypatch):
    import tools.approval as approval_mod
    from tools.grant_authority import ReceiptError, request_destructive_grant_approval

    monkeypatch.setattr(approval_mod, "is_approval_bypass_active", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval_mod, "_is_gateway_approval_context", lambda: False)
    with pytest.raises(ReceiptError):
        request_destructive_grant_approval(
            operation="CREATE_FILESYSTEM", vm_id="101", device="/dev/sdb1",
            fs_type="ext4", label="X", session_id="sess-1", ttl_seconds=600,
        )


# ---------------------------------------------------------------------------
# E — legitimate-flow witness (the ONLY issuer is the consumer process)
# ---------------------------------------------------------------------------


def test_legitimate_flow_in_process_witness():
    """The documented production flow: the running Hermes process observes
    the incarnation, the human approves once, the SAME process mints the
    grant with its own authority, claims it, and executes exactly one
    qga_create_filesystem call -> completed."""
    exec_calls = []
    with _mock_approval(), _mock_qga(exec_side_effect=lambda: exec_calls.append(1)):
        result = _call()
    assert result["decision"] == "ALLOW"
    assert result["outcome"] == "completed"
    assert len(exec_calls) == 1
    # The grant was minted by THIS process: its generation is the consumer
    # generation (the handler's issue_grant used get_authority()).
    assert result["grant_id"]
    grant = dg._grant_path(result["grant_id"])
    assert not grant.exists()  # settled: removed from the pool


def test_human_denial_no_grant_no_qga():
    """human decision = deny => grant_created=NO, qga_calls=0."""
    exec_calls = []
    with _mock_approval(decision="deny"), _mock_qga(
        exec_side_effect=lambda: exec_calls.append(1)
    ):
        result = _call()
    assert result["decision"] == "DENY"
    assert result["reason"] == "human_approval_denied"
    assert exec_calls == []
    assert dg.list_grants() == []


def test_approval_unavailable_no_grant_no_qga():
    """Approval transport lost / no human surface => grant_created=NO,
    qga_calls=0."""
    import tools.approval as approval_mod
    from tools.grant_authority import ReceiptError

    def _no_surface(**kwargs):
        raise ReceiptError("no human surface available")

    exec_calls = []
    with patch("tools.governed_mkfs_tool.request_destructive_grant_approval",
               side_effect=_no_surface), _mock_qga(
        exec_side_effect=lambda: exec_calls.append(1)
    ):
        result = _call()
    assert result["decision"] == "DENY"
    assert result["reason"] == "human_approval_denied"
    assert exec_calls == []
    assert dg.list_grants() == []


def test_incarnation_replacement_between_approval_and_effect():
    """observe A -> human approves A -> live target becomes B -> sink
    re-read = B => DENY, qga_calls=0."""
    exec_calls = []
    with _mock_approval(), _mock_qga(
        identity={"hostname": "storage-guest", "boot_id": "boot-A",
                  "product_uuid": "uuid-A"},
        identity_after={"hostname": "storage-guest", "boot_id": "boot-B",
                        "product_uuid": "uuid-A"},
        exec_side_effect=lambda: exec_calls.append(1),
    ):
        result = _call()
    assert result["decision"] == "DENY"
    assert result["reason"] == "incarnation_changed"
    assert exec_calls == []


# ---------------------------------------------------------------------------
# J — end-to-end: forged grant never reaches qga_create_filesystem
# ---------------------------------------------------------------------------


def test_forged_grant_never_reaches_qga():
    """A forged grant file is denied at load; qga_create_filesystem is never
    called.  The governed handler no longer accepts an external grant_id at
    all: the schema has no grant_id parameter, so a forged grant cannot even
    be referenced by the model-facing tool."""
    forged_id = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    forged = {
        "grant_id": forged_id,
        "operation": "CREATE_FILESYSTEM",
        "vm_id": "101",
        "hostname": "storage-guest",
        "device": "/dev/sdb1",
        "fs_type": "ext4",
        "label": "X",
        "authorization_subject": "operator",
        "authorization_source": "USER",
        "session_id": "sess-1",
        "issued_at": time.time(),
        "expires_at": time.time() + 600,
        "nonce": "0" * 32,
        "auth_tag": "0" * 64,
        "authority_generation": ga.get_authority().generation,
        "receipt_id": "rcpt-" + "0" * 28,
        "schema_version": 1,
        "authorization_evidence": {},
        "incarnation_product_uuid": "uuid-A",
        "incarnation_boot_id": "boot-A",
        "incarnation_hostname": "storage-guest",
    }
    _write_grant(forged_id, forged)
    # The forged file is rejected by the integrity check.
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(forged_id)
    # The model-facing schema has no grant_id: the handler cannot be pointed
    # at the forged file at all.
    from tools.governed_mkfs_tool import GOVERNED_MKFS_SCHEMA

    assert "grant_id" not in GOVERNED_MKFS_SCHEMA["parameters"]["properties"]
    assert "grant_id" not in GOVERNED_MKFS_SCHEMA["parameters"]["required"]


def test_child_minted_grant_never_reaches_qga():
    """A grant minted by a child process is rejected by the parent's
    authority; the governed path never consumes it."""
    code = (
        "import sys, time, uuid; sys.path.insert(0, %r); "
        "from tools import destructive_grants as dg; "
        "from tools.grant_authority import HumanApprovalReceipt, _store_receipt; "
        "r = _store_receipt(HumanApprovalReceipt("
        "receipt_id='rcpt-'+uuid.uuid4().hex, request_id='r'*32, "
        "request_digest='d'*64, session_id='sess-1', turn_id='t', "
        "tool_call_id='tc', operation='CREATE_FILESYSTEM', vm_id='101', "
        "device='/dev/sdb1', fs_type='ext4', label='X', "
        "incarnation_product_uuid='uuid-A', incarnation_boot_id='boot-A', "
        "incarnation_hostname='storage-guest', "
        "issued_at=time.time(), expires_at=time.time()+600)); "
        "g = dg.issue_grant(operation='CREATE_FILESYSTEM', vm_id='101', "
        "hostname='storage-guest', device='/dev/sdb1', fs_type='ext4', label='X', "
        "authorization_subject='operator', session_id='sess-1', "
        "receipt_id=r.receipt_id, incarnation_product_uuid='uuid-A', "
        "incarnation_boot_id='boot-A', incarnation_hostname='storage-guest'); "
        "print(g.grant_id)"
        % os.getcwd()
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    child_grant_id = out.stdout.strip()
    # The parent rejects the child-minted grant before any effect.
    with pytest.raises(dg.GrantDeniedError):
        dg.load_grant(child_grant_id)
