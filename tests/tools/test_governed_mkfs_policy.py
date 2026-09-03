"""Policy tests for the governed destructive capability (mandate §20).

Process-boundary contract (review #100694, 2026-09-02): the model requests
(vm_id, device, fs_type, label) in a live session; the handler captures the
guest incarnation, asks the human for an explicit one-shot approval of the
EXACT tuple, mints the grant in-process, claims it, and runs the governed
workflow.  The QGA transport is mocked at the ``tools.qga_structured``
boundary so the policy engine is exercised end-to-end without touching a
real VM.

PASS
    human approve_once + correct tuple + empty target + all prechecks PASS
    -> filesystem creation allowed, grant settled completed

DENY
    no human approval (deny / no surface)
    receipt mismatch (device / vm / fs / label / session / incarnation)
    root device (never issuable)
    mounted target / existing filesystem / existing signature
    TOCTOU identity change
    execution failure -> INDETERMINATE (never retry)
    postcheck mismatch -> INDETERMINATE
"""

import json
import time
import uuid
from unittest.mock import patch

import pytest

from tools import destructive_grants as dg
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


def _clean_prechecks(device="/dev/sdb1"):
    """A fully green precheck payload (mandate §13 all PASS)."""
    return {
        "vm_id": "101",
        "device": device,
        "device_exists": True,
        "is_block_device": True,
        "major_minor": "8:17",
        "size_bytes": 137438953472,
        "parent": "sdb",
        "mounted": False,
        "swap": False,
        "filesystem_existing": False,
        "filesystem_signature": False,
        "lvm_member": False,
        "mdraid_member": False,
        "holders": [],
        "docker_use": False,
        "fstab_use": False,
    }


def _exec_ok(device="/dev/sdb1", fs_type="ext4", label="DATA"):
    return {
        "operation": "CREATE_FILESYSTEM",
        "vm_id": "101",
        "device": device,
        "fs_type": fs_type,
        "label": label,
        "guest_argv": [dg.FS_TYPE_TO_BINARY[fs_type], "-L", label, device],
        "exit_code": 0,
        "out_data": "",
        "err_data": "",
    }


def _postcheck_ok(fs_type="ext4", label="DATA"):
    return {
        "exit_code": 0,
        "filesystem": fs_type,
        "label": label,
        "uuid": "11111111-2222-3333-4444-555555555555",
    }


def _call(*, device="/dev/sdb1", fs_type="ext4", label="DATA",
          vm_id="101", session_id="sess-1"):
    return json.loads(
        _handle_governed_mkfs(
            {
                "vm_id": vm_id,
                "device": device,
                "fs_type": fs_type,
                "label": label,
            },
            session_id=session_id,
        )
    )


# ---------------------------------------------------------------------------
# PASS
# ---------------------------------------------------------------------------


class TestPass:
    def test_valid_go_full_tuple_allows(self):
        with _mock_approval(), _mock_qga():
            result = _call()
        assert result["decision"] == "ALLOW"
        assert result["capability_consumed"] is True
        assert result["fs_type"] == "ext4"
        assert result["label"] == "DATA"
        assert result["uuid"] == "11111111-2222-3333-4444-555555555555"
        # The grant was minted in-process and settled: no live grant remains.
        assert dg.list_grants() == []

    def test_grant_file_permissions_are_0600(self):
        grant = _issue()
        path = dg._grant_path(grant.grant_id)
        assert (path.stat().st_mode & 0o777) == 0o600
        # Directory is 0700.
        assert (path.parent.stat().st_mode & 0o777) == 0o700


# ---------------------------------------------------------------------------
# DENY — request validation
# ---------------------------------------------------------------------------


class TestDenyNoGo:
    def test_missing_vm_id_denied(self):
        with _mock_approval(), _mock_qga():
            result = json.loads(
                _handle_governed_mkfs(
                    {"device": "/dev/sdb1", "fs_type": "ext4", "label": "DATA"},
                    session_id="sess-1",
                )
            )
        assert result["decision"] == "DENY"
        assert result["reason"] == "vm_id_required"

    def test_missing_device_denied(self):
        with _mock_approval(), _mock_qga():
            result = json.loads(
                _handle_governed_mkfs(
                    {"vm_id": "101", "fs_type": "ext4", "label": "DATA"},
                    session_id="sess-1",
                )
            )
        assert result["decision"] == "DENY"
        assert result["reason"] == "device_required"

    def test_missing_session_denied(self):
        with _mock_approval(), _mock_qga():
            result = json.loads(
                _handle_governed_mkfs(
                    {"vm_id": "101", "device": "/dev/sdb1",
                     "fs_type": "ext4", "label": "DATA"},
                    session_id="",
                )
            )
        assert result["decision"] == "DENY"
        assert result["reason"] == "session_id_required"


class TestDenyHumanApproval:
    def test_human_deny_no_grant_no_qga(self):
        exec_calls = []
        with _mock_approval(decision="deny"), _mock_qga(
            exec_side_effect=lambda: exec_calls.append(1)
        ):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["reason"] == "human_approval_denied"
        assert exec_calls == []
        assert dg.list_grants() == []

    def test_no_human_surface_no_grant_no_qga(self):
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


class TestDenyApprovalMismatch:
    """The receipt is bound to the EXACT tuple the human approved; a
    mismatch between the request and the receipt denies issuance."""

    def test_receipt_device_mismatch_denied(self):
        # The mock approval binds the receipt to what the handler observed;
        # a receipt for a different device cannot mint the grant.
        receipt = _make_receipt(device="/dev/sdc1")
        with patch(
            "tools.governed_mkfs_tool.request_destructive_grant_approval",
            return_value=receipt,
        ), _mock_qga():
            result = _call(device="/dev/sdb1")
        assert result["decision"] == "DENY"
        assert result["reason"] == "grant_issue_failed"

    def test_receipt_vm_mismatch_denied(self):
        receipt = _make_receipt(vm_id="149")
        with patch(
            "tools.governed_mkfs_tool.request_destructive_grant_approval",
            return_value=receipt,
        ), _mock_qga():
            result = _call(vm_id="101")
        assert result["decision"] == "DENY"
        assert result["reason"] == "grant_issue_failed"

    def test_receipt_fs_mismatch_denied(self):
        receipt = _make_receipt(fs_type="xfs")
        with patch(
            "tools.governed_mkfs_tool.request_destructive_grant_approval",
            return_value=receipt,
        ), _mock_qga():
            result = _call(fs_type="ext4")
        assert result["decision"] == "DENY"
        assert result["reason"] == "grant_issue_failed"

    def test_receipt_label_mismatch_denied(self):
        receipt = _make_receipt(label="OTHER")
        with patch(
            "tools.governed_mkfs_tool.request_destructive_grant_approval",
            return_value=receipt,
        ), _mock_qga():
            result = _call(label="DATA")
        assert result["decision"] == "DENY"
        assert result["reason"] == "grant_issue_failed"

    def test_receipt_session_mismatch_denied(self):
        receipt = _make_receipt(session_id="sess-2")
        with patch(
            "tools.governed_mkfs_tool.request_destructive_grant_approval",
            return_value=receipt,
        ), _mock_qga():
            result = _call(session_id="sess-1")
        assert result["decision"] == "DENY"
        assert result["reason"] == "grant_issue_failed"

    def test_receipt_incarnation_mismatch_denied(self):
        """The human approved generation A; the handler observed generation B
        -> the receipt cannot mint the grant (identity observed before
        approval == identity approved == grant identity)."""
        receipt = _make_receipt(incarnation={
            "hostname": "storage-guest", "boot_id": "boot-A",
            "product_uuid": "uuid-A",
        })
        with patch(
            "tools.governed_mkfs_tool.request_destructive_grant_approval",
            return_value=receipt,
        ), _mock_qga(
            identity={"hostname": "storage-guest", "boot_id": "boot-B",
                      "product_uuid": "uuid-A"},
        ):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["reason"] == "grant_issue_failed"


class TestDenyRootDevice:
    def test_root_device_never_issuable(self):
        with pytest.raises(dg.GrantError):
            _issue(device="/dev/sda1")
        with pytest.raises(dg.GrantError):
            _issue(device="/dev/sda")
        with pytest.raises(dg.GrantError):
            _issue(device="/dev/vda1")
        with pytest.raises(dg.GrantError):
            _issue(device="/dev/nvme0n1")


# ---------------------------------------------------------------------------
# DENY — prechecks
# ---------------------------------------------------------------------------


class TestDenyMounted:
    def test_mounted_target_denied(self):
        prechecks = _clean_prechecks()
        prechecks["mounted"] = True
        prechecks["mount_target"] = "/var/lib/docker"
        with _mock_approval(), _mock_qga(prechecks=prechecks):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["reason"] == "prechecks_failed"
        assert "mounted=YES" in result["failures"]


class TestDenyExistingFilesystem:
    def test_existing_fs_denied(self):
        prechecks = _clean_prechecks()
        prechecks["filesystem_existing"] = True
        with _mock_approval(), _mock_qga(prechecks=prechecks):
            result = _call()
        assert result["decision"] == "DENY"
        assert "filesystem_existing=YES" in result["failures"]


class TestDenyExistingSignature:
    def test_existing_signature_denied(self):
        prechecks = _clean_prechecks()
        prechecks["filesystem_signature"] = True
        with _mock_approval(), _mock_qga(prechecks=prechecks):
            result = _call()
        assert result["decision"] == "DENY"
        assert "filesystem_signature=YES" in result["failures"]


class TestPrecheckPartuuidSemantics:
    """blkid returns rc=0 with PARTUUID= for ANY GPT partition, even empty.

    PARTUUID is partition-table identity, not a data signature.  Only a real
    filesystem TYPE= (or a wipefs hit) must set filesystem_signature.
    """

    def _run_prechecks(self, blkid_out, wipefs_out=""):
        import tools.qga_structured as qs

        def _fake_exec(vm_id, argv, timeout=None):
            cmd = argv[0]
            if cmd == "lsblk":
                out = json.dumps({
                    "blockdevices": [{
                        "name": "sdb1", "type": "part", "maj:min": "8:17",
                        "size": "128G", "mountpoints": [], "fstype": None,
                    }]
                })
                return {"exit_code": 0, "out_data": out, "err_data": ""}
            if cmd == "findmnt":
                return {"exit_code": 1, "out_data": "", "err_data": ""}
            if cmd == "blkid":
                return {"exit_code": 0, "out_data": blkid_out, "err_data": ""}
            if cmd == "wipefs":
                return {"exit_code": 0, "out_data": wipefs_out, "err_data": ""}
            if cmd == "grep" and "swaps" in argv:
                return {"exit_code": 1, "out_data": "", "err_data": ""}
            if cmd == "pvs":
                return {"exit_code": 1, "out_data": "", "err_data": ""}
            if cmd == "cat":
                return {"exit_code": 0, "out_data": "", "err_data": ""}
            if cmd == "ls":
                return {"exit_code": 0, "out_data": "", "err_data": ""}
            if cmd == "bash":
                return {"exit_code": 0, "out_data": "", "err_data": ""}
            return {"exit_code": 0, "out_data": "", "err_data": ""}

        with patch.object(qs, "_qm_guest_exec", side_effect=_fake_exec):
            return qs.qga_prechecks("101", "/dev/sdb1")

    def test_empty_gpt_partition_partuuid_only_is_not_signature(self):
        checks = self._run_prechecks(
            blkid_out='/dev/sdb1: PARTUUID="c675ed5f-8323-45a3-b8d0-4e62097a1a09"'
        )
        assert checks["filesystem_signature"] is False
        assert checks["device_exists"] is True
        assert checks["is_block_device"] is True

    def test_real_filesystem_type_is_signature(self):
        checks = self._run_prechecks(
            blkid_out='/dev/sdb1: UUID="abc" TYPE="ext4" PARTUUID="c675ed5f"'
        )
        assert checks["filesystem_signature"] is True

    def test_wipefs_hit_is_signature_even_without_blkid_type(self):
        checks = self._run_prechecks(
            blkid_out='',
            wipefs_out="DEVICE OFFSET TYPE UUID LABEL\nsdb1 0x438 ext4",
        )
        assert checks["filesystem_signature"] is True


# ---------------------------------------------------------------------------
# DENY — TOCTOU
# ---------------------------------------------------------------------------


class TestDenyToctou:
    def test_identity_changed_between_precheck_and_action(self):
        prechecks = _clean_prechecks()
        recheck = _clean_prechecks()
        recheck["major_minor"] = "8:18"  # device identity changed
        with _mock_approval(), _mock_qga(prechecks=prechecks, recheck=recheck):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["reason"] == "toctou_identity_changed"

    def test_mounted_between_precheck_and_action(self):
        prechecks = _clean_prechecks()
        recheck = _clean_prechecks()
        recheck["mounted"] = True
        with _mock_approval(), _mock_qga(prechecks=prechecks, recheck=recheck):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["reason"] == "toctou_recheck_failed"


# ---------------------------------------------------------------------------
# DENY — execution failure (INDETERMINATE, never retry)
# ---------------------------------------------------------------------------


class TestDenyExecutionFailure:
    def test_nonzero_exit_indeterminate_and_grant_settled(self):
        exec_result = _exec_ok()
        exec_result["exit_code"] = 1
        exec_result["err_data"] = "mkfs.ext4: Device or resource busy"
        with _mock_approval(), _mock_qga(exec_result=exec_result):
            result = _call()
        assert result["decision"] == "INDETERMINATE"
        assert result["reason"] == "execution_exit_nonzero"
        assert result["outcome"] == "indeterminate"
        # A mutation may have happened: the grant is settled indeterminate and
        # NEVER returns to the pool (no blind retry).
        assert dg.list_grants() == []

    def test_postcheck_fs_mismatch_indeterminate(self):
        with _mock_approval(), _mock_qga(postcheck=_postcheck_ok(fs_type="xfs")):
            result = _call()
        assert result["decision"] == "INDETERMINATE"
        assert result["reason"] == "postcheck_fs_mismatch"
        assert result["outcome"] == "indeterminate"


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_audit_records_issue_verify_consume_without_secrets(self):
        with _mock_approval(), _mock_qga():
            _call()
        entries = dg.read_audit_trail()
        events = [e["event"] for e in entries]
        assert "grant_issued" in events
        assert "grant_claimed" in events
        assert "grant_verified" in events
        assert "grant_settled" in events
        # No secrets in the trail.
        blob = json.dumps(entries)
        assert "nonce" not in blob

    def test_deny_is_audited(self):
        with _mock_approval(decision="deny"), _mock_qga():
            _call()
        entries = dg.read_audit_trail()
        assert any(e["event"] == "grant_denied" for e in entries)


# ---------------------------------------------------------------------------
# Grant integrity (unit level)
# ---------------------------------------------------------------------------


class TestGrantIntegrity:
    def test_tampered_grant_file_denied(self):
        grant = _issue()
        path = dg._grant_path(grant.grant_id)
        data = json.loads(path.read_text())
        data["device"] = "/dev/sdc1"
        path.write_text(json.dumps(data))
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant(grant.grant_id)

    def test_revoked_grant_denied(self):
        grant = _issue()
        assert dg.revoke_grant(grant.grant_id) is True
        with pytest.raises(dg.GrantNotFoundError):
            dg.load_grant(grant.grant_id)
