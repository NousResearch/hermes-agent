import os
import shutil
from pathlib import Path

from tools.environments.android_linux import AndroidLinuxEnvironment


def test_android_linux_environment_builds_prefix_first_runtime_env(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    bin_dir = prefix / "bin"
    lib_dir = prefix / "lib"
    home_dir = prefix / "home"
    tmp_dir = prefix / "tmp"
    share_terminfo = prefix / "share" / "terminfo"
    git_exec = prefix / "libexec" / "git-core"
    for directory in [bin_dir, lib_dir, home_dir, tmp_dir, share_terminfo, git_exec]:
        directory.mkdir(parents=True, exist_ok=True)

    bash_path = shutil.which("bash")
    assert bash_path is not None

    monkeypatch.setenv("HERMES_ANDROID_LINUX_PREFIX", str(prefix))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_BASH", bash_path)
    monkeypatch.setenv("HERMES_ANDROID_LINUX_BIN", str(bin_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_LIB", str(lib_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_HOME", str(home_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_TMP", str(tmp_dir))

    env = AndroidLinuxEnvironment(cwd=str(home_dir), timeout=30)
    run_env = env._build_run_env()

    assert run_env["PREFIX"] == str(prefix)
    assert run_env["TERMUX_PREFIX"] == str(prefix)
    assert run_env["HOME"] == str(home_dir)
    assert run_env["TMPDIR"] == str(tmp_dir)
    assert run_env["PATH"].split(":")[0] == str(bin_dir)
    assert run_env["LD_LIBRARY_PATH"].split(":")[0] == str(lib_dir)
    assert run_env["TERMINFO"] == str(share_terminfo)
    assert run_env["GIT_EXEC_PATH"] == str(git_exec)

    env.cleanup()
