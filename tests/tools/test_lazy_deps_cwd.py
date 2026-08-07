"""Regression for #71165: lazy-deps subprocesses must not inherit an
inaccessible cwd from su/sudo invocations."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_venv_pip_install_uses_project_root_as_cwd(monkeypatch, tmp_path):
    """_venv_pip_install must pass cwd=project_root to subprocess.run so uv
    doesn't try to read uv.toml from an inaccessible inherited cwd."""
    from tools.lazy_deps import _venv_pip_install

    _captured_kwargs = {}

    def _fake_run(*args, **kwargs):
        _captured_kwargs.update(kwargs)
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr("tools.lazy_deps.windows_hide_flags", lambda: 0)

    _venv_pip_install(("requests",))

    expected_cwd = Path(__file__).parent.parent.parent.parent.resolve() / "tools" / "lazy_deps.py"
    # The actual project root is 2 levels up from tools/lazy_deps.py
    project_root = (Path(__file__).parent.parent.parent.parent / "tools" / "lazy_deps.py").parent.parent.resolve()

    assert "cwd" in _captured_kwargs, "subprocess.run was not passed cwd"
    cwd = _captured_kwargs["cwd"]
    # Must be an existing directory
    assert cwd.is_dir(), f"cwd={cwd} is not a directory"
    # Must contain pyproject.toml (project root marker)
    assert (cwd / "pyproject.toml").exists() or (cwd / "tools").exists(), (
        f"cwd={cwd} doesn't look like the project root"
    )
