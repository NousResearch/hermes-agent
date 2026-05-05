import shutil

from tools.environments.android_linux import AndroidLinuxEnvironment


def test_android_linux_environment_builds_system_shell_runtime_env(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    bin_dir = prefix / "bin"
    lib_dir = prefix / "lib"
    home_dir = prefix / "home"
    tmp_dir = prefix / "tmp"
    for directory in [bin_dir, lib_dir, home_dir, tmp_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    bash_path = "/system/bin/sh"

    monkeypatch.setenv("HERMES_ANDROID_LINUX_PREFIX", str(prefix))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_BASH", bash_path)
    monkeypatch.setenv("HERMES_ANDROID_LINUX_BIN", str(bin_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_LIB", str(lib_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_HOME", str(home_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_TMP", str(tmp_dir))

    env = AndroidLinuxEnvironment(cwd=str(home_dir), timeout=30)
    run_env = env._build_run_env()

    assert env.process_shell_path == "/system/bin/sh"
    assert run_env["PREFIX"] == str(prefix)
    assert run_env["HOME"] == str(home_dir)
    assert run_env["TMPDIR"] == str(tmp_dir)
    assert run_env["HERMES_ANDROID_SHELL"] == bash_path
    assert run_env["HERMES_ANDROID_EXECUTION_MODE"] == "android_system_shell"
    assert run_env["PATH"].startswith("/system/bin:/system/xbin:/vendor/bin:/odm/bin")
    assert str(bin_dir) not in run_env["PATH"]
    assert run_env["LD_LIBRARY_PATH"].startswith(str(lib_dir))

    env.cleanup()


def test_android_linux_environment_derives_native_library_dir_from_shell_path(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    home_dir = prefix / "home"
    tmp_dir = prefix / "tmp"
    native_dir = tmp_path / "apk-lib"
    shell_path = native_dir / "libhermes_android_bash.so"
    for directory in [home_dir, tmp_dir, native_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    shell_path.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_ANDROID_LINUX_PREFIX", str(prefix))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_BASH", "/system/bin/sh")
    monkeypatch.setenv("HERMES_ANDROID_LINUX_NATIVE_BASH", str(shell_path))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_HOME", str(home_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_TMP", str(tmp_dir))

    env = AndroidLinuxEnvironment(cwd=str(home_dir), timeout=30)
    run_env = env._build_run_env()

    assert env.process_shell_path == "/system/bin/sh"
    assert run_env["HERMES_ANDROID_SHELL"] == "/system/bin/sh"
    assert run_env["HERMES_ANDROID_NATIVE_SHELL"] == str(shell_path)
    assert run_env["LD_LIBRARY_PATH"].startswith(str(native_dir))

    env.cleanup()


def test_android_linux_environment_allows_prefix_bin_when_opted_in(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    bin_dir = prefix / "bin"
    home_dir = prefix / "home"
    tmp_dir = prefix / "tmp"
    for directory in [bin_dir, home_dir, tmp_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    bash_path = shutil.which("bash")
    assert bash_path is not None

    monkeypatch.setenv("HERMES_ANDROID_LINUX_PREFIX", str(prefix))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_BASH", bash_path)
    monkeypatch.setenv("HERMES_ANDROID_LINUX_BIN", str(bin_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_HOME", str(home_dir))
    monkeypatch.setenv("HERMES_ANDROID_LINUX_TMP", str(tmp_dir))
    monkeypatch.setenv("HERMES_ANDROID_ALLOW_PREFIX_BIN", "1")

    env = AndroidLinuxEnvironment(cwd=str(home_dir), timeout=30)
    run_env = env._build_run_env()

    assert run_env["PATH"].startswith("/system/bin:/system/xbin:/vendor/bin:/odm/bin")
    assert str(bin_dir) in run_env["PATH"]

    env.cleanup()
