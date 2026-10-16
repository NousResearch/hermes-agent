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


def _ink_child(script: str, *args: str, scripts: Path = _SCRIPTS, **env: str) -> subprocess.CompletedProcess:
    """Exactly what gatewayClient.ts runs: PYTHONPATH and HERMES_PYTHON_SRC_ROOT name the Python root."""
    child_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    child_env.update(PYTHONPATH=str(_ROOT), HERMES_PYTHON_SRC_ROOT=str(_ROOT), **env)
    return subprocess.run([sys.executable, "-S", str(scripts / script), *args], cwd=_ROOT,
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


def test_packaged_gateway_bootstrap_boots_the_root_it_is_given_or_names_why_it_cannot(tmp_path):
    """The TUI product ships the helper in its own dist/scripts/ (a separate Nix store path, #139208),
    where parents[2] is not the Python root: the helper must boot the generation of the root Ink
    passes, and an interpreter that still cannot load Hermes is a named, terminal verdict the client
    stops retrying on, never a bare ImportError retried every 30 s (#138878)."""
    packaged = tmp_path / "hermes-tui" / "dist" / "scripts"
    packaged.mkdir(parents=True)
    shutil.copy2(_SCRIPTS / "gateway_bootstrap.py", packaged / "gateway_bootstrap.py")

    bare = _ink_child("gateway_bootstrap.py", "--start", scripts=packaged)
    assert bare.returncode == 1
    assert bare.stderr.splitlines()[0] == "gateway unavailable: bootstrap_dependencies", bare.stderr

    _commit_generation()
    booted = _ink_child("gateway_bootstrap.py", scripts=packaged)
    assert booted.returncode == 1
    assert booted.stderr.strip().splitlines()[0].startswith("gateway "), booted.stderr
    assert "unavailable" not in booted.stderr


def _store_python_child(script: str, *args: str) -> subprocess.CompletedProcess:
    """The helper on PM's store Python: a non-venv interpreter living under the runtime dir."""
    child_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    child_env.update(PYTHONPATH=str(_ROOT), HERMES_PYTHON_SRC_ROOT=str(_ROOT),
                     HERMES_RUNTIME_DIR=str(Path(sys.base_prefix).resolve().parent))
    python = getattr(sys, "_base_executable", sys.executable)
    return subprocess.run([python, "-S", str(_SCRIPTS / script), *args], cwd=_ROOT, env=child_env,
                          capture_output=True, text=True, timeout=60, check=False)


def test_store_python_without_a_generation_names_the_remedy():
    """No committed generation is the one state no retry fixes: the client stops retrying and
    shows the verdict once, so it must end with the user's next step, not leave a dead TUI."""
    bootstrap = _store_python_child("gateway_bootstrap.py", "--start")
    lines = bootstrap.stderr.strip().splitlines()
    assert bootstrap.returncode == 1
    assert lines[0] == "gateway unavailable: bootstrap_dependencies", bootstrap.stderr
    assert lines[-1] == "gateway unavailable: run `hermes pm repair`, then restart the TUI", bootstrap.stderr


def test_clipboard_helper_without_a_generation_answers_false(tmp_path):
    """The clipboard helper activates before importing Hermes, and a missing generation is `false`
    on stdout (no image), not a traceback before Ink can parse anything."""
    clipboard = _store_python_child("clipboard_image.py", str(tmp_path / "paste.png"))
    assert clipboard.returncode == 0, clipboard.stderr
    assert json.loads(clipboard.stdout) is False
    assert "Traceback" not in clipboard.stderr


def test_unreadable_install_state_is_retried_not_terminal():
    """An I/O error on the install state can be one-off: it takes the ordinary failure path the
    client retries, never the terminal verdict that ends the TUI's gateway for its lifetime."""
    import pm.environments

    state = pm.environments.install_state_dir(_ROOT)
    (state / ".install.lock").mkdir(parents=True)  # os.open(O_RDWR) on a directory: OSError

    child = _ink_child("gateway_bootstrap.py", "--start")

    assert child.returncode == 1
    assert "gateway unavailable" not in child.stderr, child.stderr
    assert "Traceback" not in child.stderr, child.stderr


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
