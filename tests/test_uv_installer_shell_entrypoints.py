"""Executable integrity checks for the POSIX uv bootstrap entry points."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_URL = "https://astral.sh/uv/0.11.6/install.sh"
_EXPECTED_DIGEST = "02f6fdf8077f97f7bbd901de06054a65e7aefbd54432c8a83784d42a3e360a45"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_shell_installer(
    tmp_path: Path,
    script: Path,
    *,
    digest_matches: bool,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    bash = shutil.which("bash")
    assert bash, "bash is required for POSIX installer coverage"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    url_log = tmp_path / "urls.txt"
    execution_log = tmp_path / "executions.txt"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(
        "command() {\n"
        "  if [ \"$1\" = -v ] && [ \"${2:-}\" = uv ]; then return 1; fi\n"
        "  builtin command \"$@\"\n"
        "}\n",
        encoding="utf-8",
    )
    _write_executable(
        fake_bin / "curl",
        "#!/bin/bash\n"
        "output=\n"
        "url=\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -o) output=$2; shift 2 ;;\n"
        "    http*) url=$1; shift ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "printf '%s\\n' \"$url\" >> \"$UV_TEST_URL_LOG\"\n"
        "printf '%s\\n' '# fake uv installer' > \"$output\"\n",
    )
    reported_digest = _EXPECTED_DIGEST if digest_matches else "0" * 64
    _write_executable(
        fake_bin / "sha256sum",
        f"#!/bin/bash\nprintf '%s  -\\n' '{reported_digest}'\n",
    )
    _write_executable(
        fake_bin / "sh",
        "#!/bin/bash\n"
        "printf '%s\\n' \"$1\" >> \"$UV_TEST_EXECUTION_LOG\"\n"
        "exit 42\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "BASH_ENV": str(bash_env),
            "HOME": str(tmp_path / "home"),
            "HERMES_HOME": str(tmp_path / "hermes-home"),
            "PATH": os.pathsep.join([str(fake_bin), str(Path(bash).parent)]),
            "UV_TEST_EXECUTION_LOG": str(execution_log),
            "UV_TEST_URL_LOG": str(url_log),
        }
    )
    args = [bash, str(script)]
    if script.name == "install.sh":
        args.extend(["--stage", "prerequisites", "--non-interactive"])
    completed = subprocess.run(
        args,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    urls = url_log.read_text(encoding="utf-8").splitlines()
    executions = (
        execution_log.read_text(encoding="utf-8").splitlines()
        if execution_log.exists()
        else []
    )
    return completed, urls, executions


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "script",
    [_REPO_ROOT / "setup-hermes.sh", _REPO_ROOT / "scripts" / "install.sh"],
    ids=["setup-hermes", "install"],
)
def test_shell_installers_reject_mismatch_before_execution(tmp_path, script):
    completed, urls, executions = _run_shell_installer(
        tmp_path,
        script,
        digest_matches=False,
    )

    assert completed.returncode != 0
    assert urls == [_EXPECTED_URL]
    assert executions == []
    assert "checksum mismatch" in (completed.stdout + completed.stderr)


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "script",
    [_REPO_ROOT / "setup-hermes.sh", _REPO_ROOT / "scripts" / "install.sh"],
    ids=["setup-hermes", "install"],
)
def test_shell_installers_execute_once_after_matching_digest(tmp_path, script):
    completed, urls, executions = _run_shell_installer(
        tmp_path,
        script,
        digest_matches=True,
    )

    assert completed.returncode != 0
    assert urls == [_EXPECTED_URL]
    assert len(executions) == 1
