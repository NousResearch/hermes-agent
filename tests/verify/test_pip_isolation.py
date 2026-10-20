"""Regression: ``hermes verify`` pip installs must not clobber the host generation's
site-packages.

Mechanism of the two real incidents (2026-10-05 DNB-MCP, 2026-10-12
deal-predictions-deterministic): the verify recipe ran a bare ``pip install`` using the
running generation's interpreter/``PYTHONPATH``, so a project that pinned ``mcp<2`` made
pip treat the generation's committed ``mcp`` 2.0.0 as "inside the environment" and
*uninstall* it on the downgrade -- breaking every MCP tool until a gateway reinstall.

The fix (agent/verify/runner.py: ``_clean_child_env`` + ``_isolate_pip``)
runs pip inside the *project's own* ``.venv`` with ``--isolated`` and a
``PYTHONPATH``/``PYTHONHOME``-stripped environment, so the generation's site-packages is
neither importable nor writable from the pip subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from agent.verify.recipes import Recipe
from agent.verify.runner import run_verify


def _write_fake_mcp(site: Path, version: str) -> Path:
    """Write a well-formed (uninstallable) fake ``mcp`` distribution into ``site``."""
    site.mkdir(parents=True, exist_ok=True)
    (site / "mcp").mkdir(exist_ok=True)
    (site / "mcp" / "__init__.py").write_text(f"__version__ = {version!r}\n")
    dist_info = site / f"mcp-{version}.dist-info"
    dist_info.mkdir(exist_ok=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: mcp\nVersion: {version}\n")
    (dist_info / "INSTALLER").write_text("pip\n")
    (dist_info / "top_level.txt").write_text("mcp\n")
    names = [
        "mcp/__init__.py",
        f"mcp-{version}.dist-info/METADATA",
        f"mcp-{version}.dist-info/INSTALLER",
        f"mcp-{version}.dist-info/top_level.txt",
        f"mcp-{version}.dist-info/RECORD",
    ]
    (dist_info / "RECORD").write_text("".join(f"{n},,\n" for n in names))
    return dist_info


def _build_wheel(dest_dir: Path, version: str) -> Path:
    """Build a minimal valid pure-python ``mcp`` wheel so the downgrade is satisfiable
    offline via ``--no-index --find-links`` (keeps the test network-free)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    wheel = dest_dir / f"mcp-{version}-py3-none-any.whl"
    files = {
        "mcp/__init__.py": f"__version__ = {version!r}\n",
        f"mcp-{version}.dist-info/METADATA": f"Metadata-Version: 2.1\nName: mcp\nVersion: {version}\n",
        f"mcp-{version}.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"mcp-{version}.dist-info/top_level.txt": "mcp\n",
    }
    record = "".join(f"{n},,\n" for n in files) + f"mcp-{version}.dist-info/RECORD,,\n"
    files[f"mcp-{version}.dist-info/RECORD"] = record
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return wheel


def _make_generation_venv(path: Path) -> Path:
    """A venv standing in for the committed generation, carrying ``mcp`` 2.0.0 in its
    OWN site-packages -- so its pip treats mcp as inside the environment and will
    uninstall it on a downgrade (the real clobber). Returns the site-packages dir."""
    subprocess.run(
        [sys.executable, "-m", "venv", str(path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    site = Path(
        subprocess.run(
            [str(path / "bin" / "python"), "-c", "import site;print(site.getsitepackages()[0])"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    )
    _write_fake_mcp(site, "2.0.0")
    return site


def test_verify_pip_install_does_not_clobber_generation_site(tmp_path, monkeypatch):
    generation = tmp_path / "generation"
    gen_site = _make_generation_venv(generation)
    gen_dist_info = gen_site / "mcp-2.0.0.dist-info"
    assert gen_dist_info.is_dir()  # the generation ships mcp 2.0.0

    wheels = tmp_path / "wheels"
    _build_wheel(wheels, "1.9.0")

    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text("mcp<2\n")
    recipe = Recipe(
        name="x",
        kind="python",
        bootstrap=[f"pip install --no-index --find-links {wheels} -r requirements.txt"],
    )

    # Mirror how the gateway launches verify from inside the generation: the
    # generation's bin on PATH (so a bare `pip` resolves there) and its site on
    # PYTHONPATH. With the pre-fix code this is exactly what clobbered mcp.
    monkeypatch.setenv("PATH", f"{generation / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("PYTHONPATH", str(gen_site))

    result = run_verify(project, recipe, phases=("bootstrap",), skip_start=True)

    # 1. The generation's committed mcp 2.0.0 is untouched.
    assert gen_dist_info.is_dir(), "verify pip clobbered the generation's mcp 2.0.0"
    assert "2.0.0" in (gen_site / "mcp" / "__init__.py").read_text()

    # 2. pip ran inside the project's own venv, isolated from the host env.
    command = result.phases[0].command
    assert "--isolated" in command
    assert str(project / ".venv") in command
    venv_python = project / ".venv" / "bin" / "python"
    assert venv_python.exists()

    # 3. The downgrade landed in the project venv, not the generation. Run the probe
    # with -I so it ignores the inherited PYTHONPATH and reads only the venv's site.
    installed = subprocess.run(
        [str(venv_python), "-I", "-c", "import mcp;print(mcp.__version__)"],
        capture_output=True, text=True, check=False,
    )
    assert installed.stdout.strip() == "1.9.0"
    assert result.ok
