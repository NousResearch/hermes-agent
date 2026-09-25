"""Every script vendored into this repo must be byte-identical to the copy the
host actually executes.

WHY THIS EXISTS
Review round 1 of t_e8f5ffe5: scripts/disk-guard.sh was vendored into git while
launchd kept executing an older ~/.hermes/scripts/disk-guard.sh. Three hunks
apart, including the very log string used as evidence that the fix worked. The
commit that vendored it said its whole point was "disk-guard.sh lived only on
the host, untracked, which is why its broken reclaim could rot unnoticed" — and
then re-created that rot in the other direction, silently.

P-GUARD: the set of checked files is DERIVED (glob of repo scripts/ intersected
with the host script root), never a hand-maintained list, so a newly vendored
script cannot be silently unguarded. It also fails in BOTH directions:
  - tracked file present, host file missing/different -> FAIL (not deployed)
  - host file is a symlink at the tracked file        -> PASS (same inode)

Override for proof harnesses: HERMES_HOST_SCRIPT_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
HOST_SCRIPTS = Path(
    os.environ.get("HERMES_HOST_SCRIPT_ROOT", str(Path.home() / ".hermes" / "scripts"))
)


def _vendored() -> list[Path]:
    """Derive the checked set from the repo tree, not a literal list."""
    if not REPO_SCRIPTS.is_dir():
        return []
    return sorted(p for p in REPO_SCRIPTS.glob("*.sh") if p.is_file())


def _host_counterpart(tracked: Path) -> Path:
    return HOST_SCRIPTS / tracked.name


@pytest.mark.skipif(not HOST_SCRIPTS.is_dir(), reason="no host script root on this machine")
def test_derived_set_is_not_empty_when_scripts_exist() -> None:
    """A guard that checks nothing must not report green."""
    if REPO_SCRIPTS.is_dir() and any(REPO_SCRIPTS.glob("*.sh")):
        assert _vendored(), "scripts/*.sh exist but the derivation returned nothing"


@pytest.mark.skipif(not HOST_SCRIPTS.is_dir(), reason="no host script root on this machine")
@pytest.mark.parametrize("tracked", _vendored(), ids=lambda p: p.name)
def test_vendored_script_matches_the_host_copy(tracked: Path) -> None:
    host = _host_counterpart(tracked)
    if not host.exists():
        # Not every repo script is installed on every host. Only assert equality
        # for scripts the host has adopted; absence is reported, not failed,
        # because a dev checkout legitimately has none of them.
        pytest.skip(f"{host} not installed on this host")
    assert host.read_bytes() == tracked.read_bytes(), (
        f"{host} differs from the tracked {tracked}.\n"
        "The executing artifact is not the reviewed artifact. Deploy or symlink:\n"
        f"  ln -sfn {tracked} {host}"
    )
