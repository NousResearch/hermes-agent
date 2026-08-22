from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest


def binding(**overrides):
    value = {
        "submission_id": "submit_exact_00000001",
        "connection_id": "connection-a",
        "profile": "worker",
        "runtime_session_id": "runtime-1",
        "stored_session_id": "stored-1",
        "lineage_root_id": "root-1",
        "payload_digest": "a" * 64,
        "source_digest": "b" * 64,
        "context_digest": "c" * 64,
        "attachment_manifest_digest": "d" * 64,
        "attachment_count": 2,
    }
    value.update(overrides)
    return value


def test_one_atomic_record_contains_crash_marker_and_complete_receipt(tmp_path, monkeypatch):
    from tui_gateway import exact_admission

    replacements = []
    real_replace = exact_admission.os.replace
    monkeypatch.setattr(
        exact_admission.os,
        "replace",
        lambda source, destination: replacements.append((source, destination)) or real_replace(source, destination),
    )

    receipt, created = exact_admission.record_exact_admission(
        tmp_path,
        binding=binding(),
        prompt="attachment context\n\n  exact source  ",
        persist_user_text="  exact source  ",
    )

    assert created is True
    assert len(replacements) == 1
    assert receipt == exact_admission.get_exact_receipt(tmp_path, binding()["submission_id"])
    marker = exact_admission.read_exact_turn_marker(tmp_path, "stored-1")
    assert marker == {
        "submission_id": "submit_exact_00000001",
        "prompt": "attachment context\n\n  exact source  ",
        "persist_user_text": "  exact source  ",
        "attempts": 0,
        "started_at": pytest.approx(marker["started_at"], abs=0),
    }
    assert set(receipt) == {
        "version",
        "submission_id",
        "connection_id",
        "profile",
        "runtime_session_id",
        "stored_session_id",
        "lineage_root_id",
        "payload_digest",
        "source_digest",
        "context_digest",
        "attachment_manifest_digest",
        "attachment_count",
        "state",
        "accepted_at",
    }
    assert receipt["state"] == "durably_accepted"
    assert "path" not in str(receipt).lower()
    assert "token" not in str(receipt).lower()


def test_clearing_crash_marker_preserves_receipt(tmp_path):
    from tui_gateway.exact_admission import (
        clear_exact_turn_marker,
        get_exact_receipt,
        read_exact_turn_marker,
        record_exact_admission,
    )

    record_exact_admission(tmp_path, binding=binding(), prompt="context", persist_user_text="source")
    clear_exact_turn_marker(tmp_path, "stored-1", "submit_exact_00000001")

    assert read_exact_turn_marker(tmp_path, "stored-1") is None
    assert get_exact_receipt(tmp_path, "submit_exact_00000001")["state"] == "durably_accepted"


def test_clearing_matching_marker_preserves_conflicting_active_admission(tmp_path):
    from tui_gateway import exact_admission

    exact_admission.record_exact_admission(
        tmp_path,
        binding=binding(submission_id="submit_exact_00000001"),
        prompt="first",
        persist_user_text="first",
    )
    exact_admission.record_exact_admission(
        tmp_path,
        binding=binding(submission_id="submit_exact_00000002", payload_digest="e" * 64),
        prompt="second",
        persist_user_text="second",
    )

    exact_admission.clear_exact_turn_marker(tmp_path, "stored-1", "submit_exact_00000001")

    first = exact_admission._read_record(
        exact_admission._record_path(tmp_path, "submit_exact_00000001"), "submit_exact_00000001"
    )
    second = exact_admission._read_record(
        exact_admission._record_path(tmp_path, "submit_exact_00000002"), "submit_exact_00000002"
    )
    assert first["turn"] is None
    assert second["turn"]["submission_id"] == "submit_exact_00000002"
    assert exact_admission.get_exact_receipt(tmp_path, "submit_exact_00000001")["state"] == "durably_accepted"


@pytest.mark.parametrize(
    "field,value",
    [
        ("connection_id", "connection-b"),
        ("profile", "other-worker"),
        ("runtime_session_id", "runtime-2"),
        ("stored_session_id", "stored-2"),
        ("lineage_root_id", "root-2"),
        ("payload_digest", "0" * 64),
        ("source_digest", "1" * 64),
        ("context_digest", "2" * 64),
        ("attachment_manifest_digest", "3" * 64),
        ("attachment_count", 3),
    ],
)
def test_every_receipt_binding_field_conflicts_without_overwrite(tmp_path, field, value):
    from tui_gateway.exact_admission import ExactAdmissionConflict, record_exact_admission

    first, _ = record_exact_admission(tmp_path, binding=binding(), prompt="context", persist_user_text="source")

    with pytest.raises(ExactAdmissionConflict):
        record_exact_admission(
            tmp_path,
            binding=binding(**{field: value}),
            prompt="context",
            persist_user_text="source",
        )

    replay, created = record_exact_admission(tmp_path, binding=binding(), prompt="context", persist_user_text="source")
    assert created is False
    assert replay == first


def test_concurrent_replay_admits_once(tmp_path):
    from tui_gateway.exact_admission import record_exact_admission

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(
            pool.map(
                lambda _index: record_exact_admission(
                    tmp_path,
                    binding=binding(),
                    prompt="context",
                    persist_user_text="source",
                ),
                range(24),
            )
        )

    assert sum(created for _receipt, created in results) == 1
    assert len({receipt["accepted_at"] for receipt, _created in results}) == 1


def test_rejected_receipt_is_bound_and_has_no_active_turn(tmp_path):
    from tui_gateway.exact_admission import get_exact_receipt, read_exact_turn_marker, record_exact_rejection

    receipt, created = record_exact_rejection(tmp_path, binding=binding(), reason="busy-target")

    assert created is True
    assert receipt["state"] == "rejected"
    assert receipt["reason"] == "busy-target"
    assert get_exact_receipt(tmp_path, binding()["submission_id"]) == receipt
    assert read_exact_turn_marker(tmp_path, "stored-1") is None


def test_invalid_ids_corrupt_records_and_capacity_fail_closed(tmp_path, monkeypatch):
    from tui_gateway import exact_admission

    with pytest.raises(exact_admission.ExactAdmissionInvalid):
        exact_admission.record_exact_admission(
            tmp_path,
            binding=binding(submission_id="../escape"),
            prompt="context",
            persist_user_text="source",
        )

    monkeypatch.setattr(exact_admission, "_MAX_RECORDS", 1)
    exact_admission.record_exact_admission(tmp_path, binding=binding(), prompt="context", persist_user_text="source")
    with pytest.raises(exact_admission.ExactAdmissionError, match="capacity"):
        exact_admission.record_exact_admission(
            tmp_path,
            binding=binding(submission_id="submit_exact_00000002"),
            prompt="context",
            persist_user_text="source",
        )
