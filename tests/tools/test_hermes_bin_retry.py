"""A transient missing launcher must not poison future child PATHs (#56634)."""

import json
import os
import subprocess
import sys

import pytest

from tools.environments import local


@pytest.mark.parametrize("source", ["argv", "interpreter", "path"])
def test_child_path_recovers_after_launcher_appears(tmp_path, monkeypatch, source):
    install = tmp_path / "install"
    install.mkdir()
    minimal = tmp_path / "minimal"
    minimal.mkdir()
    shim = install / ("hermes.exe" if os.name == "nt" else "hermes")
    python = sys.executable
    monkeypatch.setattr(local, "_HERMES_BIN_DIR", local._SENTINEL)
    monkeypatch.setenv("PATH", str(install if source == "path" else minimal))
    base = {"PATH": str(minimal)}
    with monkeypatch.context() as launch:
        launch.setattr(sys, "argv", [str(shim) if source == "argv" else "worker"])
        launch.setattr(sys, "executable", str((install if source == "interpreter" else minimal) / "python"))
        for _ in range(2):
            env = local.build_subprocess_env(base)
            assert str(install) not in env["PATH"].split(os.pathsep)
        shim.write_text("fixture launcher", encoding="utf-8")
        shim.chmod(0o755)
        env = local.build_subprocess_env(base)
        assert env["PATH"].split(os.pathsep)[0] == str(install)
        # Both sibling callers use the same resolver; no duplicate prepend.
        assert local._sanitize_subprocess_env(env)["PATH"].split(os.pathsep).count(str(install)) == 1
        assert local._make_run_env(base)["PATH"].split(os.pathsep).count(str(install)) == 1
    child = subprocess.run(
        [python, "-c", "import json,shutil; print(json.dumps(shutil.which('hermes')))"],
        env=env, capture_output=True, text=True, timeout=20, check=True,
    )
    assert os.path.normcase(json.loads(child.stdout)) == os.path.normcase(str(shim))


def test_successful_resolution_stays_cached_and_errors_can_retry(tmp_path, monkeypatch):
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setattr(local, "_HERMES_BIN_DIR", local._SENTINEL)
    monkeypatch.setattr(local.shutil, "which", lambda _: (_ for _ in ()).throw(OSError("probe failed")))
    with pytest.raises(OSError, match="probe failed"):
        local._resolve_hermes_bin_dir()
    monkeypatch.setattr(local.shutil, "which", lambda _: str(install / "hermes"))
    assert local._resolve_hermes_bin_dir() == str(install)
    monkeypatch.setattr(local.shutil, "which", lambda _: pytest.fail("successful cache must avoid re-probing"))
    assert local._resolve_hermes_bin_dir() == str(install)
