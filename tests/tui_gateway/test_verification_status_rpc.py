"""Tests for the verification.status JSON-RPC method."""

from __future__ import annotations

from pathlib import Path

import pytest

import tui_gateway.server as server


LEDGER_NAME = "verification_evidence.db"


def _call_verification_status(params: dict) -> dict:
    response = server.handle_request(
        {"jsonrpc": "2.0", "id": "rpc-1", "method": "verification.status", "params": params}
    )
    assert response is not None
    assert "error" not in response, response.get("error")
    return response["result"]["verification"]


@pytest.fixture
def applicable_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = str(workspace.resolve())

    from agent import coding_context

    monkeypatch.setattr(coding_context, "project_facts_for", lambda cwd: {"root": root})
    return workspace


@pytest.fixture
def profile_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "profiles" / "readonly"
    monkeypatch.setattr(server, "_profile_home", lambda profile: home if profile == "readonly" else None)
    return home


def _sidecars(db_path: Path) -> list[Path]:
    return [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
        db_path.with_name(f"{db_path.name}-journal"),
    ]


def test_verification_status_missing_profile_ledger_is_readonly(
    applicable_workspace: Path,
    profile_home: Path,
) -> None:
    db_path = profile_home / LEDGER_NAME
    assert not profile_home.exists()

    verification = _call_verification_status(
        {
            "profile": "readonly",
            "session_id": "session-a",
            "cwd": str(applicable_workspace),
        }
    )

    assert verification["status"] == "unverified"
    assert verification["evidence"] is None
    assert verification["root"] == str(applicable_workspace.resolve())
    assert verification["session_id"] == "session-a"
    assert verification["changed_paths"] == []
    assert not profile_home.exists()
    assert not any(path.exists() for path in _sidecars(db_path))


def test_verification_status_corrupt_profile_ledger_is_unverified_without_mutation(
    applicable_workspace: Path,
    profile_home: Path,
) -> None:
    profile_home.mkdir(parents=True)
    db_path = profile_home / LEDGER_NAME
    original = b"not a sqlite ledger\n"
    db_path.write_bytes(original)

    verification = _call_verification_status(
        {
            "profile": "readonly",
            "session_key": "session-b",
            "cwd": str(applicable_workspace),
        }
    )

    assert verification["status"] == "unverified"
    assert verification["evidence"] is None
    assert verification["root"] == str(applicable_workspace.resolve())
    assert verification["session_id"] == "session-b"
    assert db_path.read_bytes() == original
    assert not any(path.exists() for path in _sidecars(db_path)[1:])


def test_verification_status_not_applicable_does_not_create_ledger(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import coding_context

    monkeypatch.setattr(coding_context, "project_facts_for", lambda cwd: None)
    cwd = tmp_path / "not-a-project"
    cwd.mkdir()
    db_path = profile_home / LEDGER_NAME

    verification = _call_verification_status(
        {"profile": "readonly", "session_id": "session-c", "cwd": str(cwd)}
    )

    assert verification == {"status": "not_applicable", "evidence": None}
    assert not profile_home.exists()
    assert not any(path.exists() for path in _sidecars(db_path))


def test_verification_status_rpc_does_not_use_mutable_status_path(
    applicable_workspace: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import verification_evidence

    def fail_mutable(*_args, **_kwargs):
        raise AssertionError("mutable verification status path was called")

    monkeypatch.setattr(verification_evidence, "verification_status", fail_mutable)
    monkeypatch.setattr(verification_evidence, "_connect", fail_mutable)

    verification = _call_verification_status(
        {
            "profile": "readonly",
            "session_id": "session-d",
            "cwd": str(applicable_workspace),
        }
    )

    assert verification["status"] == "unverified"
    assert not profile_home.exists()
