"""Destructive-action gate for artifacts the session did not create (#84718).

The reported trace ends with the agent working toward deleting a checkout
route that pre-dated the task entirely: the investigation todo that PROVED it
pre-dated the task had already completed, compaction re-injected the action
item without that finding, and nothing checked provenance before acting.
"""

import subprocess
from pathlib import Path

import pytest

from tools.delete_provenance import check_delete_provenance, reset_acknowledgements


@pytest.fixture(autouse=True)
def clean_acknowledgements(monkeypatch):
    monkeypatch.delenv("HERMES_DELETE_PROVENANCE_GATE", raising=False)
    reset_acknowledgements()
    yield
    reset_acknowledgements()


@pytest.fixture
def git_repo(tmp_path):
    def _run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    _run("init", "-q")
    _run("config", "user.email", "tester@example.com")
    _run("config", "user.name", "Tester")
    tracked = tmp_path / "checkout.tsx"
    tracked.write_text("export const Checkout = () => null;\n", encoding="utf-8")
    _run("add", ".")
    _run("commit", "-qm", "feat: add checkout route")
    return tmp_path


def test_pre_existing_file_is_refused_once_then_allowed(git_repo):
    target = git_repo / "checkout.tsx"

    first = check_delete_provenance(str(target))
    assert first is not None
    assert "checkout.tsx" in first
    assert "feat: add checkout route" in first
    # The message must carry the provenance the compaction destroyed.
    assert "pre-dates this session" in first

    # An identical retry is the confirmation — the model is never wedged.
    assert check_delete_provenance(str(target)) is None


def test_untracked_file_is_allowed(git_repo):
    fresh = git_repo / "fresh.tsx"
    fresh.write_text("new\n", encoding="utf-8")
    assert check_delete_provenance(str(fresh)) is None


def test_outside_a_repo_is_allowed(tmp_path):
    loose = tmp_path / "loose.txt"
    loose.write_text("x\n", encoding="utf-8")
    assert check_delete_provenance(str(loose)) is None


def test_gate_can_be_disabled(git_repo, monkeypatch):
    monkeypatch.setenv("HERMES_DELETE_PROVENANCE_GATE", "0")
    assert check_delete_provenance(str(git_repo / "checkout.tsx")) is None


def test_git_failure_fails_open(git_repo, monkeypatch):
    monkeypatch.setattr("tools.delete_provenance._run_git", lambda *a, **k: None)
    assert check_delete_provenance(str(git_repo / "checkout.tsx")) is None


class _FileOps:
    """Minimal read/write surface the V4A applier needs."""

    class _Result:
        def __init__(self, content=None, error=None):
            self.content = content
            self.error = error

    def __init__(self, root: Path):
        self.root = root
        self.deleted = []

    def read_file_raw(self, path):
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        if not p.exists():
            return self._Result(error="file not found")
        return self._Result(content=p.read_text(encoding="utf-8"))

    def delete_file(self, path):
        self.deleted.append(path)
        return self._Result(content="")


def test_patch_delete_blocks_before_touching_the_filesystem(git_repo):
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    target = git_repo / "checkout.tsx"
    parsed = parse_v4a_patch(
        "*** Begin Patch\n"
        f"*** Delete File: {target}\n"
        "*** End Patch\n"
    )
    operations = parsed[0] if isinstance(parsed, tuple) else parsed
    file_ops = _FileOps(git_repo)

    result = apply_v4a_operations(operations, file_ops)
    assert result.success is False
    assert "pre-dates this session" in (result.error or "")
    assert file_ops.deleted == []
    assert target.exists()

    # Re-issuing the identical patch confirms and proceeds to the apply phase.
    second = apply_v4a_operations(operations, file_ops)
    assert second.success is True
    assert file_ops.deleted
