"""End-to-end: a built wheel, installed without a source tree, must resolve
i18n catalogs and render human strings — not raw key paths.

This is the test that would have caught #27632 / #35374 / #23943. Metadata
unit tests (test_packaging_metadata.py) prove the glob is declared; this proves
the runtime actually finds the catalogs after a real pip install.

This lives in tests/ (NOT tests/e2e/) so it is collected by the dedicated CI
step in Task 9, not by the existing `python -m pytest tests/e2e/` runner.

Assumption: `from agent import i18n` must import with only stdlib + pyyaml
available (the test installs the wheel --no-deps + pyyaml). agent/__init__.py's
jiter preload swallows ImportError, and i18n.py imports yaml lazily inside
_load_catalog, so this holds today. If i18n.py ever gains a top-level non-stdlib
import, add it to the pip install line below.

Marked `integration` because it shells out to `uv build` + `venv` + `pip` and
takes ~15-30s. Run with: pytest -m integration tests/test_wheel_locales_e2e.py
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


@pytest.mark.integration
@pytest.mark.timeout(300)  # overrides the global --timeout=30; cold-CI wheel build + venv + pip can exceed it
def test_installed_wheel_renders_i18n_strings(tmp_path):
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

    # 2. Fresh venv, install the wheel WITHOUT deps (we only exercise i18n,
    #    which needs pyyaml). --force-reinstall guards against pip's
    #    same-version no-op.
    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    vpy = venv_dir / "bin" / "python"
    subprocess.run([str(vpy), "-m", "pip", "install", "-q", "pyyaml"], check=True, timeout=300)
    subprocess.run(
        [str(vpy), "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", wheel],
        check=True,
        timeout=300,
    )

    # 3. Run from a directory that is NOT the source tree, with a clean env
    #    (no PYTHONPATH leaking the repo, no HERMES_BUNDLED_LOCALES).
    probe = (
        "from agent import i18n;"
        "import sys;"
        "r = i18n.t('gateway.reset.header_default', lang='en');"
        "s = i18n.t('gateway.status.header', lang='en');"
        "print(repr(r)); print(repr(s));"
        "sys.exit(0 if (r != 'gateway.reset.header_default' "
        "and s != 'gateway.status.header') else 1)"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "HERMES_BUNDLED_LOCALES")}
    env["PATH"] = f"{venv_dir / 'bin'}:{env['PATH']}"
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
        "installed wheel returned raw i18n keys instead of human strings:\n"
        f"stdout: {run.stdout}\nstderr: {run.stderr}"
    )


@pytest.mark.integration
@pytest.mark.timeout(300)  # overrides the global --timeout=30; cold-CI sdist build can exceed it
def test_built_sdist_ships_locale_catalogs(tmp_path):
    """The sdist must carry locales/ too.

    The wheel is covered above; the sdist is a separately shipped artifact
    (PyPI, and the form distro/Homebrew packagers build from). MANIFEST.in
    `graft locales` is what puts the catalogs in the tarball — a stale graft or
    a setuptools change would pass the metadata unit test (which only inspects
    the declaration) while the actual artifact regresses. This inspects the
    real tarball so that path can't rot silently. Closes the sdist half of
    #27632 / #35374 / #23943.
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
        # Members are prefixed with the sdist root dir, e.g.
        # hermes_agent-0.15.1/locales/en.yaml — match on the suffix.
        catalogs = [m for m in tf.getnames() if "/locales/" in m and m.endswith(".yaml")]

    # Compare against the canonical language list rather than a hardcoded floor
    # so adding/removing a catalog updates the guard automatically and a dropped
    # catalog (not just a fully-empty graft) trips it.
    from agent.i18n import SUPPORTED_LANGUAGES

    expected = len(SUPPORTED_LANGUAGES)
    assert len(catalogs) == expected, (
        f"sdist shipped {len(catalogs)} locale catalogs, expected {expected} "
        f"({len(SUPPORTED_LANGUAGES)} supported languages) — check `graft "
        "locales` in MANIFEST.in"
    )
    assert any(m.endswith("/locales/en.yaml") for m in catalogs), (
        f"sdist missing locales/en.yaml; shipped: {catalogs[:5]}"
    )
