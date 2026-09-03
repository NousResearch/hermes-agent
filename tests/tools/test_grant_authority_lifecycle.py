"""RED tests — review #100694 blockers 1-3 (andrexibiza, 2026-09-01/02).

Process-boundary contract (review #100694, 2026-09-02):

    model requests (vm_id, device, fs_type, label) in a live session
    -> process captures trusted guest identity
    -> human approves the EXACT observed tuple + incarnation
    -> grant minted IN THE CONSUMER PROCESS (same authority generation)
    -> atomic claim -> sink fencing -> prechecks -> TOCTOU -> exec ->
       postcheck -> settlement

B1  Authority provenance + grant integrity:
    - issuance requires a correlated human approval receipt (one-shot)
    - the receipt binds operation, target, session AND incarnation
    - identity observed BEFORE approval == identity approved == grant
    - clone under a second UUID is denied (path-ID == payload-ID)
    - TTL extension / evidence swap / incarnation swap are denied
    - the CLI cannot mint a grant (hard-fail, no local issuance)

B2  Claim/settlement lifecycle:
    - atomic claim BEFORE the first effect; one winner under concurrency
    - verify rejects claimed grants (no replay, no double-claim)
    - settlement is bound to the claiming execution_id
    - pre-effect failure -> DENY + outcome failed_pre_effect (no return to pool)
    - post-effect failure -> INDETERMINATE + outcome indeterminate (never retry)
    - success -> ALLOW + outcome completed

B3  Durable generation fencing:
    - incarnation captured at issue, revalidated at the sink
    - ABA witness: grant for generation A refused on generation B
    - boot_id part of the TOCTOU identity snapshot
"""

import json
import threading
import time
import uuid
from unittest.mock import patch

import pytest

from tools import destructive_grants as dg
from tools.governed_mkfs_tool import _handle_governed_mkfs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INCARNATION = {
    "hostname": "storage-guest",
    "boot_id": "boot-A",
    "product_uuid": "uuid-A",
}


def _make_receipt(
    *,
    operation="CREATE_FILESYSTEM",
    vm_id="101",
    device="/dev/sdb1",
    fs_type="ext4",
    label="DATA",
    session_id="sess-1",
    ttl=600,
    incarnation=None,
):
    """Mint a correlated human approval receipt in the process-local store
    (the same store the approval layer writes to)."""
    from tools.grant_authority import HumanApprovalReceipt, _store_receipt

    inc = incarnation if incarnation is not None else _INCARNATION
    now = time.time()
    return _store_receipt(
        HumanApprovalReceipt(
            receipt_id="rcpt-" + uuid.uuid4().hex,
            request_id="req-11111111111111111111111111111111",
            request_digest="d" * 64,
            session_id=session_id,
            turn_id="turn-1",
            tool_call_id="tool-1",
            operation=operation,
            vm_id=vm_id,
            device=device,
            fs_type=fs_type,
            label=label,
            incarnation_product_uuid=inc["product_uuid"],
            incarnation_boot_id=inc["boot_id"],
            incarnation_hostname=inc["hostname"],
            issued_at=now,
            expires_at=now + ttl,
        )
    )


def _issue(
    *,
    device="/dev/sdb1",
    fs_type="ext4",
    label="DATA",
    vm_id="101",
    hostname="storage-guest",
    subject="operator",
    session_id="sess-1",
    ttl=600,
    receipt=None,
    incarnation=None,
):
    receipt = receipt if receipt is not None else _make_receipt(
        device=device, fs_type=fs_type, label=label, vm_id=vm_id,
        session_id=session_id, ttl=ttl, incarnation=incarnation,
    )
    return dg.issue_grant(
        operation="CREATE_FILESYSTEM",
        vm_id=vm_id,
        hostname=hostname,
        device=device,
        fs_type=fs_type,
        label=label,
        authorization_subject=subject,
        session_id=session_id,
        receipt_id=receipt.receipt_id,
        incarnation_product_uuid=(incarnation or _INCARNATION)["product_uuid"],
        incarnation_boot_id=(incarnation or _INCARNATION)["boot_id"],
        incarnation_hostname=(incarnation or _INCARNATION)["hostname"],
        ttl_seconds=ttl,
    )


def _clean_prechecks(device="/dev/sdb1"):
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
    """Drive the governed handler end-to-end (approval -> issue -> claim ->
    workflow) exactly as the model-facing tool does."""
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


def _mock_approval(decision="once", incarnation=None):
    """Patch the human approval gate: return a receipt bound to the EXACT
    incarnation the handler observed (the mock mirrors the real gate, which
    receives the observed identity and binds it into the receipt)."""
    from tools.grant_authority import HumanApprovalReceipt, _store_receipt

    def _fake_request(**kwargs):
        if decision != "once":
            from tools.grant_authority import ReceiptError

            raise ReceiptError("human approval not granted")
        now = time.time()
        return _store_receipt(
            HumanApprovalReceipt(
                receipt_id="rcpt-" + uuid.uuid4().hex,
                request_id="req-" + uuid.uuid4().hex,
                request_digest="d" * 64,
                session_id=kwargs["session_id"],
                turn_id="turn-1",
                tool_call_id="tool-1",
                operation=kwargs["operation"],
                vm_id=kwargs["vm_id"],
                device=kwargs["device"],
                fs_type=kwargs["fs_type"],
                label=kwargs["label"],
                incarnation_product_uuid=kwargs["incarnation_product_uuid"],
                incarnation_boot_id=kwargs["incarnation_boot_id"],
                incarnation_hostname=kwargs["incarnation_hostname"],
                issued_at=now,
                expires_at=now + kwargs["ttl_seconds"],
            )
        )

    return patch("tools.governed_mkfs_tool.request_destructive_grant_approval",
                 side_effect=_fake_request)


def _mock_qga(prechecks=None, recheck=None, exec_result=None, postcheck=None,
              identity=None, identity_after=None, exec_side_effect=None):
    """Patch the structured QGA boundary with deterministic payloads.

    ``qga_prechecks`` is called twice (precheck then TOCTOU recheck); when
    ``recheck`` is provided it is returned on the second call.  ``identity``
    is the incarnation payload returned by ``qga_guest_identity`` at the
    initial capture; ``identity_after`` (if given) is returned on the second
    (sink) read — used to witness an incarnation replacement between
    approval and effect.
    """
    prechecks = prechecks if prechecks is not None else _clean_prechecks()
    recheck = recheck if recheck is not None else prechecks
    exec_result = exec_result if exec_result is not None else _exec_ok()
    postcheck = postcheck if postcheck is not None else _postcheck_ok()
    identity = identity if identity is not None else dict(_INCARNATION)
    identity_after = identity_after if identity_after is not None else identity
    pre_calls = {"n": 0}
    id_calls = {"n": 0}

    def _prechecks(vm_id, device):
        pre_calls["n"] += 1
        return recheck if pre_calls["n"] >= 2 else prechecks

    def _identity(vm_id):
        id_calls["n"] += 1
        return dict(identity_after if id_calls["n"] >= 2 else identity)

    def _exec(vm_id, device, fs_type, label):
        if exec_side_effect is not None:
            exec_side_effect()
        return exec_result

    def _postcheck(vm_id, device, fs_type, label):
        if callable(postcheck):
            return postcheck(vm_id, device, fs_type, label)
        return postcheck

    return patch.multiple(
        "tools.governed_mkfs_tool",
        qga_prechecks=_prechecks,
        qga_guest_identity=_identity,
        qga_create_filesystem=_exec,
        qga_postcheck=_postcheck,
    )


# ---------------------------------------------------------------------------
# B1 — authority provenance and grant integrity
# ---------------------------------------------------------------------------


class TestB1IssuanceEvidence:
    def test_issue_requires_receipt(self):
        with pytest.raises(dg.GrantError):
            dg.issue_grant(
                operation="CREATE_FILESYSTEM",
                vm_id="101", hostname="storage-guest", device="/dev/sdb1",
                fs_type="ext4", label="X",
                authorization_subject="operator", session_id="sess-1",
                receipt_id="no-such-receipt",
                incarnation_product_uuid="uuid-A",
                incarnation_boot_id="boot-A", incarnation_hostname="storage-guest",
            )

    def test_issue_rejects_unknown_receipt(self):
        with pytest.raises(dg.GrantError):
            dg.issue_grant(
                operation="CREATE_FILESYSTEM",
                vm_id="101", hostname="storage-guest", device="/dev/sdb1",
                fs_type="ext4", label="X",
                authorization_subject="operator", session_id="sess-1",
                receipt_id="rcpt-00000000000000000000000000000000",
                incarnation_product_uuid="uuid-A",
                incarnation_boot_id="boot-A", incarnation_hostname="storage-guest",
            )

    def test_issue_rejects_mismatched_receipt(self):
        # Receipt for a different device: issuance must be denied.
        receipt = _make_receipt(device="/dev/sdc1")
        with pytest.raises(dg.GrantError):
            _issue(receipt=receipt)

    def test_issue_rejects_receipt_session_mismatch(self):
        """A receipt approved for session A must never mint a grant for
        session B (session-scoped authority)."""
        receipt = _make_receipt(session_id="sess-A")
        with pytest.raises(dg.GrantError):
            _issue(receipt=receipt, session_id="sess-B")
        # The receipt is consumed only on success: a denied issuance must
        # not burn the receipt (the human decision stays usable).
        grant = _issue(receipt=receipt, session_id="sess-A")
        assert grant.session_id == "sess-A"

    def test_issue_rejects_receipt_incarnation_mismatch(self):
        """A receipt approved for generation A must never mint a grant for
        generation B (identity observed before approval == grant identity)."""
        receipt = _make_receipt(incarnation={
            "hostname": "storage-guest", "boot_id": "boot-A",
            "product_uuid": "uuid-A",
        })
        with pytest.raises(dg.GrantError):
            _issue(receipt=receipt, incarnation={
                "hostname": "storage-guest", "boot_id": "boot-B",
                "product_uuid": "uuid-A",
            })

    def test_grant_expiry_never_exceeds_receipt_expiry(self):
        """grant.expires_at <= receipt.expires_at always (no implicit
        authority renewal at issuance)."""
        receipt = _make_receipt(ttl=60)
        grant = _issue(receipt=receipt, ttl=600)  # requested TTL > approved
        assert grant.expires_at <= receipt.expires_at
        assert grant.expires_at - grant.issued_at <= 60

    def test_expired_receipt_cannot_issue(self):
        """An already-expired human approval cannot mint a grant."""
        receipt = _make_receipt(ttl=-10)  # expires in the past
        with pytest.raises(dg.GrantError):
            _issue(receipt=receipt)

    def test_issue_requires_incarnation(self):
        with pytest.raises(dg.GrantError):
            dg.issue_grant(
                operation="CREATE_FILESYSTEM",
                vm_id="101", hostname="storage-guest", device="/dev/sdb1",
                fs_type="ext4", label="X",
                authorization_subject="operator", session_id="sess-1",
                receipt_id=_make_receipt().receipt_id,
                incarnation_product_uuid="",
                incarnation_boot_id="boot-A", incarnation_hostname="storage-guest",
            )

    def test_issue_requires_product_uuid(self):
        with pytest.raises(dg.GrantError):
            dg.issue_grant(
                operation="CREATE_FILESYSTEM",
                vm_id="101", hostname="storage-guest", device="/dev/sdb1",
                fs_type="ext4", label="X",
                authorization_subject="operator", session_id="sess-1",
                receipt_id=_make_receipt().receipt_id,
                incarnation_product_uuid="uuid-A",
                incarnation_boot_id="", incarnation_hostname="storage-guest",
            )

    def test_receipt_is_persisted_and_bound(self):
        grant = _issue()
        assert grant.authorization_evidence["request_id"] == "req-11111111111111111111111111111111"
        assert grant.incarnation_boot_id == "boot-A"
        # Reload from disk: auth tag still valid (no tamper).
        loaded = dg.load_grant(grant.grant_id)
        assert loaded.auth_tag == grant.auth_tag
        assert loaded.authority_generation == grant.authority_generation

    def test_receipt_is_one_shot(self):
        """The same receipt cannot issue a second grant (replay denied)."""
        receipt = _make_receipt()
        g1 = _issue(receipt=receipt)
        assert g1.receipt_id == receipt.receipt_id
        with pytest.raises(dg.GrantError):
            _issue(receipt=receipt)  # already consumed


class TestB1CloneAndTamper:
    def _clone(self, grant_id, new_id, rewrite_payload_id=True):
        src = dg._grant_path(grant_id)
        dst = dg._grant_path(new_id)
        data = json.loads(src.read_text())
        if rewrite_payload_id:
            data["grant_id"] = new_id
        dst.write_text(json.dumps(data))

    def test_clone_under_new_uuid_denied(self):
        grant = _issue()
        self._clone(grant.grant_id, "99999999-9999-4999-8999-999999999999")
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant("99999999-9999-4999-8999-999999999999")

    def test_clone_with_unchanged_payload_id_denied(self):
        grant = _issue()
        self._clone(grant.grant_id, "99999999-9999-4999-8999-999999999999",
                    rewrite_payload_id=False)
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant("99999999-9999-4999-8999-999999999999")

    def test_ttl_extension_denied(self):
        grant = _issue()
        path = dg._grant_path(grant.grant_id)
        data = json.loads(path.read_text())
        data["expires_at"] = data["expires_at"] + 3600
        path.write_text(json.dumps(data))
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant(grant.grant_id)

    def test_evidence_swap_denied(self):
        grant = _issue()
        path = dg._grant_path(grant.grant_id)
        data = json.loads(path.read_text())
        data["authorization_evidence"] = dict(
            data["authorization_evidence"], principal="Mallory"
        )
        path.write_text(json.dumps(data))
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant(grant.grant_id)

    def test_receipt_swap_denied(self):
        grant = _issue()
        path = dg._grant_path(grant.grant_id)
        data = json.loads(path.read_text())
        data["receipt_id"] = "rcpt-" + "2" * 28
        path.write_text(json.dumps(data))
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant(grant.grant_id)

    def test_incarnation_swap_denied(self):
        grant = _issue()
        path = dg._grant_path(grant.grant_id)
        data = json.loads(path.read_text())
        data["incarnation_boot_id"] = "boot-B"
        path.write_text(json.dumps(data))
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant(grant.grant_id)

    def test_product_uuid_swap_denied(self):
        grant = _issue()
        path = dg._grant_path(grant.grant_id)
        data = json.loads(path.read_text())
        data["incarnation_product_uuid"] = "uuid-B"
        path.write_text(json.dumps(data))
        with pytest.raises(dg.GrantDeniedError):
            dg.load_grant(grant.grant_id)


# ---------------------------------------------------------------------------
# B2 — claim/settlement lifecycle
# ---------------------------------------------------------------------------


class TestB2Claim:
    def test_claim_is_atomic_and_exclusive(self):
        grant = _issue()
        claimed = dg.claim_grant(grant.grant_id, "exec-1")
        assert claimed.claimed_by_execution == "exec-1"
        with pytest.raises(dg.GrantInFlightError):
            dg.claim_grant(grant.grant_id, "exec-2")

    def test_verify_rejects_claimed_grant(self):
        grant = _issue()
        dg.claim_grant(grant.grant_id, "exec-1")
        with pytest.raises(dg.GrantInFlightError):
            dg.verify_grant(
                grant.grant_id,
                operation="CREATE_FILESYSTEM", vm_id="101", device="/dev/sdb1",
                fs_type="ext4", label="DATA", session_id="sess-1",
            )

    def test_settle_requires_matching_execution(self):
        grant = _issue()
        dg.claim_grant(grant.grant_id, "exec-1")
        with pytest.raises(dg.GrantError):
            dg.settle_grant(grant.grant_id, "exec-2", "completed")

    def test_settle_completed_removes_from_pool(self):
        grant = _issue()
        dg.claim_grant(grant.grant_id, "exec-1")
        dg.settle_grant(grant.grant_id, "exec-1", "completed")
        assert dg.get_grant_state(grant.grant_id) == "settled:completed"
        with pytest.raises(dg.GrantConsumedError):
            dg.load_grant(grant.grant_id)
        assert all(g.grant_id != grant.grant_id for g in dg.list_grants())

    def test_settle_failed_pre_effect_removes_from_pool(self):
        grant = _issue()
        dg.claim_grant(grant.grant_id, "exec-1")
        dg.settle_grant(grant.grant_id, "exec-1", "failed_pre_effect")
        assert dg.get_grant_state(grant.grant_id) == "settled:failed_pre_effect"
        with pytest.raises(dg.GrantConsumedError):
            dg.load_grant(grant.grant_id)

    def test_settle_indeterminate_removes_from_pool(self):
        grant = _issue()
        dg.claim_grant(grant.grant_id, "exec-1")
        dg.settle_grant(grant.grant_id, "exec-1", "indeterminate")
        assert dg.get_grant_state(grant.grant_id) == "settled:indeterminate"
        with pytest.raises(dg.GrantConsumedError):
            dg.load_grant(grant.grant_id)

    def test_invalid_outcome_rejected(self):
        grant = _issue()
        dg.claim_grant(grant.grant_id, "exec-1")
        with pytest.raises(dg.GrantError):
            dg.settle_grant(grant.grant_id, "exec-1", "maybe")


class TestB2Concurrency:
    def test_concurrent_callers_single_winner(self):
        """Two concurrent model calls racing on the SAME grant: one winner,
        exactly one qga_create_filesystem call.  The claim is the atomic
        reservation point (review #100694 blocker 2)."""
        from tools import governed_mkfs_tool as gmt

        shared_grant = _issue()
        barrier = threading.Barrier(2)
        real_claim = gmt.claim_grant
        exec_calls = []

        def racing_claim(grant_id, execution_id):
            barrier.wait(timeout=5)
            return real_claim(grant_id, execution_id)

        results = {}

        def run():
            results[threading.get_ident()] = _call()

        # Both callers race on the same grant: issue_grant is patched to
        # return the shared grant, so the claim is the single contention
        # point.  Patch ONCE in the main thread: the mock must stay active
        # for the whole race, including while the winner runs the
        # post-claim flow.
        with _mock_approval(), _mock_qga(exec_side_effect=lambda: exec_calls.append(1)):
            with patch.object(gmt, "issue_grant", return_value=shared_grant):
                with patch.object(gmt, "claim_grant", side_effect=racing_claim):
                    t1 = threading.Thread(target=run)
                    t2 = threading.Thread(target=run)
                    t1.start()
                    t2.start()
                    t1.join(10)
                    t2.join(10)
        assert not t1.is_alive() and not t2.is_alive()

        decisions = sorted(r["decision"] for r in results.values())
        assert decisions == ["ALLOW", "DENY"]
        assert len(exec_calls) == 1
        loser = [r for r in results.values() if r["decision"] == "DENY"][0]
        # The loser is denied either at the claim (claim_lost) or, if the
        # winner already settled the grant before the loser reached the
        # claim, as a replay (grant_denied).  Both are legitimate DENYs.
        assert loser["reason"] in ("claim_lost", "grant_denied")


class TestB2SettlementOutcomes:
    def test_precheck_failure_settles_failed_pre_effect(self):
        prechecks = _clean_prechecks()
        prechecks["mounted"] = True
        with _mock_approval(), _mock_qga(prechecks=prechecks):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["outcome"] == "failed_pre_effect"
        # The grant was minted in-process and settled: no live grant remains.
        assert dg.list_grants() == []

    def test_toctou_failure_settles_failed_pre_effect(self):
        recheck = _clean_prechecks()
        recheck["major_minor"] = "8:18"
        with _mock_approval(), _mock_qga(recheck=recheck):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["outcome"] == "failed_pre_effect"

    def test_exec_nonzero_settles_indeterminate(self):
        exec_result = _exec_ok()
        exec_result["exit_code"] = 1
        exec_result["err_data"] = "mkfs.ext4: Device or resource busy"
        with _mock_approval(), _mock_qga(exec_result=exec_result):
            result = _call()
        assert result["decision"] == "INDETERMINATE"
        assert result["outcome"] == "indeterminate"
        # Never returns to the pool: replay is impossible.
        assert dg.list_grants() == []

    def test_postcheck_loss_settles_indeterminate(self):
        from tools.qga_structured import QgaError

        def _boom(vm_id, device, fs_type, label):
            raise QgaError("postcheck transport lost")

        with _mock_approval(), _mock_qga(postcheck=_boom):
            result = _call()
        assert result["decision"] == "INDETERMINATE"
        assert result["outcome"] == "indeterminate"

    def test_postcheck_mismatch_settles_indeterminate(self):
        with _mock_approval(), _mock_qga(postcheck=_postcheck_ok(fs_type="xfs")):
            result = _call()
        assert result["decision"] == "INDETERMINATE"
        assert result["outcome"] == "indeterminate"

    def test_success_settles_completed(self):
        with _mock_approval(), _mock_qga():
            result = _call()
        assert result["decision"] == "ALLOW"
        assert result["outcome"] == "completed"
        assert dg.list_grants() == []

    def test_settlement_failure_blocks_blind_replay(self):
        """Settlement persistence failure after a possible mutation: the
        grant stays CLAIMED (never returns to LIVE), so a blind replay is
        impossible even though the settlement rename failed."""
        from tools import governed_mkfs_tool as gmt

        real_settle = gmt.settle_grant

        def failing_settle(grant_id, execution_id, outcome):
            if outcome == "completed":
                raise dg.GrantError("simulated settlement persistence failure")
            return real_settle(grant_id, execution_id, outcome)

        with _mock_approval(), _mock_qga():
            with patch.object(gmt, "settle_grant", side_effect=failing_settle):
                result = _call()
        assert result["decision"] == "INDETERMINATE"
        assert result["outcome"] == "indeterminate"
        # The grant is still claimed (in-flight), NOT live: no blind replay.
        grant_id = result["grant_id"]
        assert dg.get_grant_state(grant_id) == "claimed"
        with pytest.raises(dg.GrantInFlightError):
            dg.claim_grant(grant_id, "exec-replay")
        with pytest.raises(dg.GrantInFlightError):
            dg.verify_grant(
                grant_id,
                operation="CREATE_FILESYSTEM", vm_id="101", device="/dev/sdb1",
                fs_type="ext4", label="DATA", session_id="sess-1",
            )


# ---------------------------------------------------------------------------
# B3 — durable generation fencing
# ---------------------------------------------------------------------------


class TestB3GenerationFencing:
    def test_incarnation_mismatch_denied_at_sink(self):
        """The live guest changed generation between approval and sink
        re-read: denied BEFORE any mutation."""
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
        assert result["outcome"] == "failed_pre_effect"

    def test_incarnation_match_allows(self):
        with _mock_approval(), _mock_qga(
            identity={"hostname": "storage-guest", "boot_id": "boot-A",
                      "product_uuid": "uuid-A"},
        ):
            result = _call()
        assert result["decision"] == "ALLOW"

    def test_aba_witness_generation_b_untouched(self):
        """A issued for gen A; VM replaced by gen B; A resumes -> B untouched."""
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
        assert exec_calls == []

    def test_vm_replacement_denied_at_sink(self):
        """Same vm_id + same device + same hostname, NEW product_uuid:
        the grant for the old incarnation must not mutate the new VM."""
        exec_calls = []
        with _mock_approval(), _mock_qga(
            identity={"hostname": "storage-guest", "boot_id": "boot-A",
                      "product_uuid": "uuid-A"},
            identity_after={"hostname": "storage-guest", "boot_id": "boot-A",
                            "product_uuid": "uuid-B"},
            exec_side_effect=lambda: exec_calls.append(1),
        ):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["reason"] == "incarnation_changed"
        assert exec_calls == []
        assert result["outcome"] == "failed_pre_effect"

    def test_boot_id_in_toctou_snapshot(self):
        recheck = _clean_prechecks()
        recheck["boot_id"] = "boot-B"
        with _mock_approval(), _mock_qga(
            identity={"hostname": "storage-guest", "boot_id": "boot-A",
                      "product_uuid": "uuid-A"},
            recheck=recheck,
        ):
            result = _call()
        assert result["decision"] == "DENY"
        assert result["reason"] == "toctou_identity_changed"
