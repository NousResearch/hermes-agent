"""
Regression test for issue #60685 - `hermes update` downgrades
CVE-pinned packages to stale exact versions, overwriting newer
compatible installs.

The fix: change `cryptography==46.0.7` to `cryptography>=46.0.7`
(floor pin) so `hermes update` doesn't force-downgrade newer
compatible installs that also fix the same CVEs.

This test pins the pin in pyproject.toml so a future PR cannot
revert to the exact `==` pin.
"""

from pathlib import Path
import re


def test_cryptography_pin_is_floor_not_exact():
    """The cryptography pin must be a floor (>=) not an exact (==) pin.

    Fails on unfixed code (which has cryptography==46.0.7),
    passes on fixed code (which has cryptography>=46.0.7).
    """
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    pyproject = (worktree / "pyproject.toml").read_text()
    assert "cryptography==46.0.7" not in pyproject, (
        "#60685 regression: cryptography is exact-pinned (==46.0.7). "
        "Floor pin (>=46.0.7) instead."
    )
    assert "cryptography>=46.0.7" in pyproject, (
        "#60685: cryptography should be floor-pinned (>=46.0.7)."
    )


def test_cryptography_pin_is_floor_or_equal():
    """Static check on the constraint: pin must be >= (or >, but >=
    is the conservative choice)."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    pyproject = (worktree / "pyproject.toml").read_text()
    m = re.search(r"cryptography([=<>]+)([0-9.]+)", pyproject)
    assert m, "cryptography pin not found in pyproject.toml"
    op, version = m.group(1), m.group(2)
    assert op in (">=", ">"), (
        f"#60685: cryptography pin is `{op}{version}` (exact). "
        f"Use a floor pin (>=) to allow newer compatible installs."
    )


def test_cve_floor_pin_comment_present():
    """Sanity check: the comment on the pin must reference the CVEs so
    a future maintainer knows why the floor matters."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    pyproject = (worktree / "pyproject.toml").read_text()
    found = False
    for line in pyproject.split("\n"):
        if "cryptography" in line and ("CVE" in line or "CVE-" in line):
            found = True
            break
    assert found, (
        "#60685: cryptography pin should have a CVE comment so a "
        "future maintainer knows why the floor matters."
    )
