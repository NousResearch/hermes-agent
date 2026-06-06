"""Regression tests for Docker stage2 UID/GID handling on NAS hosts.

Unraid commonly runs appdata as nobody:users (99:100). The stage2 hook must
accept those non-root numeric IDs and keep legacy/new pairing stores writable
after targeted ownership reconciliation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _uid_gid_validator(text: str) -> str:
    marker = "# --- UID/GID remap ---"
    before_marker = text.split(marker, 1)[0]
    start = before_marker.index("validate_uid_gid()")
    return before_marker[start:]


def _validate_uid_gid(text: str, value: str) -> bool:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = _uid_gid_validator(text) + '\nvalidate_uid_gid "$CANDIDATE"\n'
    proc = subprocess.run(
        [bash, "-c", script],
        env={"PATH": os.environ.get("PATH", ""), "CANDIDATE": value},
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


@pytest.mark.parametrize("value", ["1", "99", "100", "1000", "65534"])
def test_uid_gid_validator_accepts_non_root_nas_ids(stage2_text: str, value: str) -> None:
    assert _validate_uid_gid(stage2_text, value), (
        f"stage2 hook must accept NAS UID/GID {value}; Unraid uses 99:100 (#38070)"
    )


@pytest.mark.parametrize("value", ["", "0", "abc", "99x", "65535"])
def test_uid_gid_validator_rejects_root_invalid_and_out_of_range(
    stage2_text: str,
    value: str,
) -> None:
    assert not _validate_uid_gid(stage2_text, value)


def _targeted_chown_subdirs(text: str) -> list[str]:
    m = re.search(
        r"for sub in (?P<items>.*?); do\n\s*if \[ -e \"\$HERMES_HOME/\$sub\" \]",
        text,
        re.DOTALL,
    )
    assert m, "stage2-hook.sh must contain the targeted subdir chown loop"
    return m.group("items").split()


def test_targeted_chown_covers_legacy_and_new_pairing_dirs(stage2_text: str) -> None:
    subdirs = _targeted_chown_subdirs(stage2_text)
    assert "pairing" in subdirs
    assert "platforms/pairing" in subdirs


def test_seeded_directory_list_covers_legacy_and_new_pairing_dirs(stage2_text: str) -> None:
    seed_block = stage2_text.split("as_hermes mkdir -p \\", 1)[1].split(
        "# --- Install-method stamp",
        1,
    )[0]
    assert '"$HERMES_HOME/pairing"' in seed_block
    assert '"$HERMES_HOME/platforms/pairing"' in seed_block
