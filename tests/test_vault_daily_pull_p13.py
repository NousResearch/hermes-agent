"""P13 isolation proof for scripts/vault_daily_pull.sh.

Verifies the dry-run flag:
- VAULT_PULL_DRY_RUN=1 prints the planned commands, exits 0, never runs git
- VAULT_PULL_DIR can be redirected so the production vault is untouched
- a missing VAULT_PULL_DIR (in live mode) is reported with exit 1
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "vault_daily_pull.sh"


def _run(env_extra: dict | None = None):
    env = dict(os.environ)
    env.pop("VAULT_PULL_DRY_RUN", None)
    env.pop("VAULT_PULL_DIR", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=15,
    )


def test_dry_run_exits_zero_without_git(tmp_path):
    """VAULT_PULL_DRY_RUN=1 prints the plan and never touches git."""
    fake_vault = tmp_path / "vault"
    fake_vault.mkdir()
    r = _run({"VAULT_PULL_DRY_RUN": "1", "VAULT_PULL_DIR": str(fake_vault)})
    assert r.returncode == 0, r.stderr
    assert "dry-run:" in r.stdout
    assert "git pull" in r.stdout
    assert not (fake_vault / ".git").exists()


def test_dry_run_works_even_when_dir_missing(tmp_path):
    """Dry-run short-circuits before the cd guard."""
    missing = tmp_path / "does-not-exist"
    r = _run({"VAULT_PULL_DRY_RUN": "1", "VAULT_PULL_DIR": str(missing)})
    assert r.returncode == 0, r.stderr
    assert "dry-run:" in r.stdout


def test_missing_dir_without_dry_run_exits_nonzero(tmp_path):
    """Without dry-run, a missing vault dir must report an error and exit 1."""
    missing = tmp_path / "does-not-exist"
    r = _run({"VAULT_PULL_DIR": str(missing)})
    assert r.returncode == 1
    assert "cannot cd" in r.stderr or "cannot cd" in r.stdout
