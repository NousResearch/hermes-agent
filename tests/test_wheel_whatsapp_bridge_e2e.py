"""End-to-end: a built wheel/sdist, installed without a source tree, must let
get_bundled_whatsapp_bridge_dir() resolve the Node.js bridge — not point at a
nonexistent path.

This is the test that proves the gateway WhatsApp adapter stops failing with
`whatsapp_bridge_missing` on packaged installs. Metadata unit tests
(test_packaging_metadata.py) prove the data-files glob and MANIFEST graft are
declared; the unit tests (test_whatsapp_bridge_resolution.py) pin the resolution
order with mocks; this proves the runtime actually finds bridge.js after a real
build + install, the way the locales e2e (test_wheel_locales_e2e.py) does for
i18n catalogs.

Assumption: ``import hermes_constants`` needs only the stdlib (it imports os,
sys, sysconfig, contextvars, pathlib — nothing third-party), so the wheel can be
installed --no-deps. If hermes_constants ever gains a top-level non-stdlib
import, this test must install that dep too.

Marked `integration` because it shells out to `uv build` + `venv` + `pip` and
takes ~15-30s. Run with: pytest -m integration tests/test_wheel_whatsapp_bridge_e2e.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tarfile
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    """Cross-platform path to a venv's interpreter (Scripts on Windows, bin elsewhere)."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@pytest.mark.integration
@pytest.mark.timeout(300)  # overrides the global --timeout=30; cold-CI wheel build + venv + pip can exceed it
def test_installed_wheel_resolves_whatsapp_bridge(tmp_path):
    # 1. Build the wheel from the current tree.
    wheel_dir = tmp_path / "wheel"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir), "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, f"uv build failed:\n{build.stderr}"
    wheels = glob.glob(str(wheel_dir / "*.whl"))
    assert wheels, "no wheel produced"
    wheel = wheels[0]

    # 2. Fresh venv, install the wheel WITHOUT deps (the resolver needs only
    #    stdlib). --force-reinstall guards against pip's same-version no-op.
    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    vpy = _venv_python(venv_dir)
    subprocess.run(
        [str(vpy), "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", wheel],
        check=True,
        timeout=300,
    )

    # 3. From a directory that is NOT the source tree, with a clean env (no
    #    PYTHONPATH leaking the repo, no HERMES_WHATSAPP_BRIDGE_DIR override),
    #    resolve the bridge dir and confirm the runtime files are really there.
    probe = (
        "import hermes_constants as hc;"
        "import sys;"
        "d = hc.get_bundled_whatsapp_bridge_dir();"
        "print(repr(str(d)));"
        "ok = (d / 'bridge.js').is_file() and (d / 'allowlist.js').is_file();"
        "sys.exit(0 if ok else 1)"
    )
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "HERMES_WHATSAPP_BRIDGE_DIR")
    }
    env["PATH"] = f"{venv_dir / ('Scripts' if sys.platform == 'win32' else 'bin')}{os.pathsep}{env['PATH']}"
    env["VIRTUAL_ENV"] = str(venv_dir)
    run = subprocess.run(
        [str(vpy), "-c", probe],
        cwd=str(tmp_path),  # NOT the repo root
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert run.returncode == 0, (
        "installed wheel could not resolve the WhatsApp bridge (bridge.js missing "
        "at the resolved path):\n"
        f"stdout: {run.stdout}\nstderr: {run.stderr}"
    )


@pytest.mark.integration
@pytest.mark.timeout(300)  # overrides the global --timeout=30; cold-CI sdist build can exceed it
def test_built_sdist_ships_whatsapp_bridge(tmp_path):
    """The sdist must carry scripts/whatsapp-bridge/ too.

    The wheel is covered above; the sdist is the form Homebrew and other distro
    packagers build the wheel from. MANIFEST.in `graft scripts/whatsapp-bridge`
    is what puts the bridge in the tarball — a stale graft would pass the
    metadata unit test (which only inspects the declaration) while the artifact
    regresses. This inspects the real tarball so that path can't rot silently.
    """
    sdist_dir = tmp_path / "sdist"
    build = subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(sdist_dir), "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, f"uv build --sdist failed:\n{build.stderr}"
    tarballs = glob.glob(str(sdist_dir / "*.tar.gz"))
    assert tarballs, "no sdist produced"

    with tarfile.open(tarballs[0]) as tf:
        names = tf.getnames()
        bridge = [m for m in names if "/scripts/whatsapp-bridge/" in m]

    # The runtime-critical files must be in the tarball; node_modules must not.
    for required in ("bridge.js", "allowlist.js", "package.json", "package-lock.json"):
        assert any(m.endswith(f"/scripts/whatsapp-bridge/{required}") for m in bridge), (
            f"sdist missing scripts/whatsapp-bridge/{required}; shipped: {bridge[:8]}"
        )
    assert not any("/node_modules/" in m for m in bridge), (
        f"sdist must not ship the bridge's node_modules: {[m for m in bridge if '/node_modules/' in m][:5]}"
    )
