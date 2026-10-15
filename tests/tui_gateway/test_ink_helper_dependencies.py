"""Ink's Python helpers boot PM's committed dependency generation themselves (regression for #138878).

Ink runs ``ui-tui/scripts/gateway_bootstrap.py`` (``gatewayClient.ts``) and
``ui-tui/scripts/clipboard_image.py`` (``imageAttachments.ts``) on ``HERMES_PYTHON`` with
``PYTHONPATH=<root>`` only. On a source install ``HERMES_PYTHON`` is PM's store Python, which owns no
packages, so the launcher's own activation never reaches these children. Each child here runs with
``-S`` -- the test interpreter's site-packages are out of reach, exactly like the store Python's --
and its dependencies are importable only through a generation committed in the temp HERMES_HOME.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "ui-tui" / "scripts"


def _commit_generation() -> None:
    """Commit a PM generation for this checkout whose .pth hands back the test interpreter's packages."""
    import pm.environments

    version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    venv = pm.environments.install_state_dir(_ROOT) / "environments" / "gen1" / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"version = {version}\n", encoding="utf-8")
    (venv.parent / ".lease-managed").write_text("", encoding="utf-8")
    selected = pm.environments.site_packages(venv)
    selected.mkdir(parents=True)
    (selected / "test_deps.pth").write_text(
        "\n".join(p for p in sys.path if Path(p).name in ("site-packages", "dist-packages")) + "\n",
        encoding="utf-8")
    pm.environments.runtime_facts_path(_ROOT).write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}, "schema": 1}), encoding="utf-8")


def _ink_child(script: str, *args: str, **env: str) -> subprocess.CompletedProcess:
    child_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    child_env.update(PYTHONPATH=str(_ROOT), **env)
    return subprocess.run([sys.executable, "-S", str(_SCRIPTS / script), *args], cwd=_ROOT,
                          env=child_env, capture_output=True, text=True, timeout=60, check=False)


def test_gateway_bootstrap_reaches_a_verdict_on_the_store_python():
    """With no owner running, discovery's fallback imports a dependency (gateway.status -> utils ->
    hermes_yaml -> ruamel). The bootstrap must answer with its bounded gateway verdict, not die on
    the import -- which left a fresh `hermes --tui` unable to start or recover a gateway at all."""
    _commit_generation()

    child = _ink_child("gateway_bootstrap.py")

    assert child.returncode == 1
    assert "No module named" not in child.stderr
    assert child.stderr.strip().splitlines()[0].startswith("gateway "), child.stderr


@pytest.mark.platforms("linux")
def test_clipboard_image_converts_wayland_jpeg_with_the_generations_pillow(tmp_path):
    """A non-PNG Wayland clipboard image is converted with Pillow, a core dependency; ImageMagick
    is only the fallback for interpreters that lack it, so a host without ImageMagick still pastes."""
    from PIL import Image

    _commit_generation()
    jpeg = tmp_path / "clip.jpg"
    Image.new("RGB", (4, 4), "red").save(jpeg, "JPEG")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cat").symlink_to(shutil.which("cat"))
    wl_paste = bin_dir / "wl-paste"
    wl_paste.write_text(
        '#!/bin/sh\n[ "$1" = --list-types ] && { echo image/jpeg; exit 0; }\n'
        f'[ "$1" = --type ] && [ "$2" = image/jpeg ] && exec cat {jpeg}\nexit 1\n', encoding="utf-8")
    wl_paste.chmod(0o755)
    dest = tmp_path / "out" / "paste.png"

    child = _ink_child("clipboard_image.py", str(dest), PATH=str(bin_dir), WAYLAND_DISPLAY="wayland-test")

    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) is True
    assert Image.open(dest).format == "PNG"
