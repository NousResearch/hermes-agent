"""Contract tests for Hermes Evidence Transport Receipt v2."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pytest

from tools.evidence_receipt import (
    MAX_RECEIPT_BYTES,
    RECEIPT_FIELDS,
    RECEIPT_SCHEMA,
    VERDICT_INTEGRITY_FAILURE,
    VERDICT_UNAVAILABLE,
    VERDICT_VERIFIED,
    ArtifactStateError,
    ReceiptFormatError,
    encode_receipt,
    parse_receipt,
    produce_receipt,
    verify_artifact,
    verify_receipt,
    write_artifact_exclusive,
)

BINDING = {
    "profile": "engineer-sol",
    "session_id": "fixture-session-001",
    "execution_kind": "standalone-single-query",
    "execution_id": "fixture-session-001",
    "grant_id": "etr-v2b-20260914",
    "cell_id": "C01",
}
PROVENANCE_ENV = {
    "HERMES_PROFILE": BINDING["profile"],
    "HERMES_SESSION_ID": BINDING["session_id"],
    "HERMES_EVIDENCE_GRANT_ID": BINDING["grant_id"],
    "HERMES_EVIDENCE_CELL_ID": BINDING["cell_id"],
}
SENTINEL = "SYNTHETIC-ETR-V2B-SENTINEL-0123456789abcdef"


def _set_provenance(monkeypatch: pytest.MonkeyPatch, *, missing: str | None = None) -> None:
    for name, value in PROVENANCE_ENV.items():
        if name == missing:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _produce(tmp_path, monkeypatch, data: bytes = b"fixture bytes", name: str = "artifact.bin") -> str:
    _set_provenance(monkeypatch)
    return produce_receipt(tmp_path / name, data)


def test_c01_producer_emits_complete_canonical_v2_receipt(tmp_path, monkeypatch):
    text = _produce(tmp_path, monkeypatch)
    receipt = parse_receipt(text)

    assert set(receipt) == RECEIPT_FIELDS
    assert receipt["schema"] == RECEIPT_SCHEMA == "hermes-evidence-receipt-v2"
    assert receipt["status"] == "ready"
    assert receipt["execution_id"] == receipt["session_id"]
    assert encode_receipt(receipt) == text
    assert len(text.encode("utf-8")) <= MAX_RECEIPT_BYTES
    timestamp = datetime.fromisoformat(receipt["created_utc"].replace("Z", "+00:00"))
    assert timestamp.utcoffset() is not None
    assert timestamp.utcoffset().total_seconds() == 0


def test_c04_receipt_contains_identity_but_not_artifact_body(tmp_path, monkeypatch):
    payload = f"prefix:{SENTINEL}:suffix".encode()
    text = _produce(tmp_path, monkeypatch, payload)

    assert SENTINEL not in text
    result = verify_receipt(text, expected_binding=BINDING)
    assert result["verdict"] == VERDICT_VERIFIED
    assert result["observed_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["observed_byte_length"] == len(payload)


@pytest.mark.parametrize("missing", tuple(PROVENANCE_ENV))
def test_c06_missing_provenance_prevents_artifact_and_ready_receipt(
    tmp_path, monkeypatch, missing
):
    _set_provenance(monkeypatch, missing=missing)
    artifact = tmp_path / f"missing-{missing}.bin"

    with pytest.raises(ReceiptFormatError):
        produce_receipt(artifact, b"must not be written")

    assert not artifact.exists()


def test_c07_complete_expected_binding_is_required(tmp_path, monkeypatch):
    text = _produce(tmp_path, monkeypatch)

    assert verify_receipt(text)["verdict"] == VERDICT_UNAVAILABLE
    for omitted in BINDING:
        incomplete = {key: value for key, value in BINDING.items() if key != omitted}
        assert verify_receipt(text, expected_binding=incomplete)["verdict"] == VERDICT_UNAVAILABLE


def test_receipt_identity_matches_finalized_on_disk_bytes(tmp_path, monkeypatch):
    payload = b"raw\x00bytes\r\nwith-final-lf\n"
    text = _produce(tmp_path, monkeypatch, payload)
    receipt = json.loads(text)

    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["byte_length"] == len(payload)
    assert (tmp_path / "artifact.bin").read_bytes() == payload


def test_c02_unknown_fields_are_unavailable_without_echo(tmp_path, monkeypatch):
    receipt = json.loads(_produce(tmp_path, monkeypatch))
    for field in ("meta", "artifact_body", "free_form"):
        hostile = {**receipt, field: SENTINEL}
        text = json.dumps(hostile, separators=(",", ":"), sort_keys=True)
        result = verify_receipt(text, expected_binding=BINDING)
        assert result["verdict"] == VERDICT_UNAVAILABLE
        assert SENTINEL not in json.dumps(result)


def test_c03_extra_metadata_apis_and_v1_are_rejected(tmp_path, monkeypatch):
    _set_provenance(monkeypatch)
    with pytest.raises(TypeError):
        produce_receipt(tmp_path / "extra.bin", b"x", meta={"note": "forbidden"})

    receipt = json.loads(produce_receipt(tmp_path / "v2.bin", b"x"))
    with pytest.raises(ReceiptFormatError):
        encode_receipt({**receipt, "meta": {"note": "forbidden"}})
    receipt["schema"] = "hermes-evidence-receipt-v1"
    v1_text = json.dumps(receipt, separators=(",", ":"), sort_keys=True)
    assert verify_receipt(v1_text, expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE


def test_c05_oversized_receipts_and_fields_are_rejected_not_truncated(
    tmp_path, monkeypatch
):
    receipt = json.loads(_produce(tmp_path, monkeypatch))
    overlong_path = {**receipt, "artifact_path": "/" + ("x" * 1024)}
    with pytest.raises(ReceiptFormatError):
        encode_receipt(overlong_path)
    overlong_text = json.dumps(overlong_path, separators=(",", ":"), sort_keys=True)
    assert verify_receipt(overlong_text, expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE

    oversized = json.dumps(receipt, separators=(",", ":"), sort_keys=True) + (
        " " * MAX_RECEIPT_BYTES
    )
    assert len(oversized.encode()) > MAX_RECEIPT_BYTES
    assert verify_receipt(oversized, expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE


def test_c08_binding_mismatch_precedes_file_read_and_preserves_digest(
    tmp_path, monkeypatch
):
    text = _produce(tmp_path, monkeypatch)
    receipt = json.loads(text)
    (tmp_path / "artifact.bin").unlink()

    for field in BINDING:
        wrong = dict(BINDING)
        wrong[field] = "wrong-kind" if field == "execution_kind" else "wrong-value"
        result = verify_receipt(text, expected_binding=wrong)
        assert result["verdict"] == VERDICT_INTEGRITY_FAILURE
        assert result["expected_sha256"] == receipt["sha256"]


def test_c09_receipt_can_be_the_only_stdout_content(tmp_path, monkeypatch, capsys):
    text = _produce(tmp_path, monkeypatch)
    print(text, end="")

    captured = capsys.readouterr()
    assert captured.out == text
    assert captured.err == ""
    assert verify_receipt(captured.out, expected_binding=BINDING)["verdict"] == VERDICT_VERIFIED


@pytest.mark.parametrize(
    "hostile",
    [
        None,
        b"{}",
        "",
        "[]",
        "null",
        "true",
        "1",
        "not-json",
        '{"schema":null}',
        '{"schema":[]}',
        '{"byte_length":true}',
        '{"byte_length":NaN}',
        '{"schema":"a","schema":"b"}',
        "\ufeff{}",
        "\ud800",
    ],
)
def test_c10_hostile_receipt_text_never_raises(hostile):
    result = verify_receipt(hostile, expected_binding=BINDING)
    assert isinstance(result, dict)
    assert result["verdict"] == VERDICT_UNAVAILABLE


def test_c10_field_types_ranges_and_canonical_form_are_strict(tmp_path, monkeypatch):
    receipt = json.loads(_produce(tmp_path, monkeypatch))
    invalid_overrides = (
        {"byte_length": True},
        {"byte_length": -1},
        {"byte_length": 1 << 63},
        {"sha256": "A" * 64},
        {"artifact_path": "relative/path"},
        {"artifact_path": "/bad\x01path"},
        {"profile": "bad profile"},
        {"execution_id": "different-session"},
        {"created_utc": "2026-09-14T01:02:03+01:00"},
    )
    for override in invalid_overrides:
        text = json.dumps({**receipt, **override}, separators=(",", ":"), sort_keys=True)
        assert verify_receipt(text, expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE

    noncanonical = json.dumps(receipt, sort_keys=True)
    assert verify_receipt(noncanonical, expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE
    assert verify_receipt(encode_receipt(receipt) + "\n", expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE


def test_c11_hash_and_length_mutations_fail_and_missing_is_unavailable(
    tmp_path, monkeypatch
):
    original = b"0123456789" * 500
    mutations = {
        "truncated.bin": original[:-1],
        "appended.bin": original + b"x",
        "flipped.bin": original[:50] + bytes([original[50] ^ 1]) + original[51:],
    }
    for name, mutated in mutations.items():
        text = _produce(tmp_path, monkeypatch, original, name=name)
        expected = json.loads(text)["sha256"]
        (tmp_path / name).write_bytes(mutated)
        result = verify_receipt(text, expected_binding=BINDING)
        assert result["verdict"] == VERDICT_INTEGRITY_FAILURE
        assert result["expected_sha256"] == expected

    missing_text = _produce(tmp_path, monkeypatch, original, name="missing.bin")
    (tmp_path / "missing.bin").unlink()
    assert verify_receipt(missing_text, expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("size-4096.bin", b"x" * 4096),
        (
            "long-61453.jsonl",
            b'{"payload":"' + b"x" * (61453 - len(b'{"payload":"') - len(b'"}\n')) + b'"}\n',
        ),
        ("utf8.txt", "héllo — 你好 — 🚀\n".encode()),
        ("crlf.txt", b"one\r\ntwo\r\n"),
        ("lf.txt", b"one\ntwo\n"),
        ("bom.txt", b"\xef\xbb\xbfBOM\n"),
        ("final-lf.txt", b"final\n"),
        ("no-final-lf.txt", b"final"),
        ("binary.bin", bytes(range(256)) * 4),
    ],
)
def test_c12_exact_byte_regressions(tmp_path, monkeypatch, name, payload):
    text = _produce(tmp_path, monkeypatch, payload, name=name)
    result = verify_receipt(text, expected_binding=BINDING)

    assert result["verdict"] == VERDICT_VERIFIED
    assert result["observed_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["observed_byte_length"] == len(payload)
    assert (tmp_path / name).read_bytes() == payload


def test_c12_presentation_reconstruction_is_not_a_receipt(tmp_path, monkeypatch):
    text = _produce(tmp_path, monkeypatch, b"authoritative bytes")
    for presented in (f"1|{text}\n", f"```json\n{text}\n```", text + "\n"):
        assert verify_receipt(presented, expected_binding=BINDING)["verdict"] == VERDICT_UNAVAILABLE


def test_exclusive_create_and_non_regular_paths_fail_closed(tmp_path, monkeypatch):
    target = tmp_path / "exclusive.bin"
    write_artifact_exclusive(target, b"first")
    with pytest.raises(FileExistsError):
        write_artifact_exclusive(target, b"second")
    assert target.read_bytes() == b"first"

    _set_provenance(monkeypatch)
    with pytest.raises(FileExistsError):
        produce_receipt(target, b"replacement")

    directory = tmp_path / "directory"
    directory.mkdir()
    assert verify_artifact(directory, "0" * 64, 0)["verdict"] == VERDICT_UNAVAILABLE

    link = tmp_path / "link.bin"
    link.symlink_to(target)
    result = verify_artifact(link, hashlib.sha256(b"first").hexdigest(), 5)
    assert result["verdict"] == VERDICT_UNAVAILABLE
    with pytest.raises(FileExistsError):
        write_artifact_exclusive(link, b"diverted")
    assert target.read_bytes() == b"first"


def test_write_helper_requires_bytes(tmp_path):
    with pytest.raises(ArtifactStateError):
        write_artifact_exclusive(tmp_path / "text.txt", "not bytes")
