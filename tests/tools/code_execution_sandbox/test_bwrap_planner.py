"""The bubblewrap planner: config parsing and the argv it builds.

Pure assertions, no processes beyond the socket file the planner expects to exist.
"""

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.code_execution_sandbox import (
    SandboxConfig,
    build_sandbox_argv,
    interpreter_roots,
    load_sandbox_config,
    plan_kernel_sandbox,
    sandbox_enabled,
)


def _fake_bwrap(tmp_path: Path) -> str:
    path = tmp_path / "bwrap"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


def _argv_for(tmp_path: Path, **overrides):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(exist_ok=True)
    sock = tmp_path / "rpc.sock"
    if not sock.exists():
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(sock))
        s.listen(1)
        s.close()
    cfg = SandboxConfig(enabled=True, bwrap_path=_fake_bwrap(tmp_path), **overrides)
    plan = plan_kernel_sandbox(
        [sys.executable, str(kernel_dir / "runner.py")],
        kernel_dir=str(kernel_dir), rpc_socket=str(sock), child_cwd=str(kernel_dir), config=cfg,
    )
    return plan, str(kernel_dir), str(sock)


def test_disabled_returns_argv_unchanged():
    argv = ["/usr/bin/python3", "/tmp/kernel/runner.py"]
    plan = plan_kernel_sandbox(argv, kernel_dir="/tmp", rpc_socket="", child_cwd="/tmp",
                               config=SandboxConfig(enabled=False))
    assert plan.argv == tuple(argv)
    assert plan.enabled is False
    assert plan.reason == "disabled"


def test_missing_bwrap_fails_open(tmp_path):
    kernel_dir = tmp_path / "k"
    kernel_dir.mkdir()
    plan = plan_kernel_sandbox(
        [sys.executable, "runner.py"], kernel_dir=str(kernel_dir), rpc_socket="",
        child_cwd=str(kernel_dir), config=SandboxConfig(enabled=True, bwrap_path=str(tmp_path / "nope")),
    )
    assert plan.enabled is False
    assert plan.argv == (sys.executable, "runner.py")
    assert "bwrap" in plan.reason


def test_confines_network_and_filesystem(short_dir):
    plan, kernel_dir, sock = _argv_for(short_dir)
    argv = list(plan.argv)
    assert plan.enabled is True
    assert "--unshare-net" in argv
    assert argv[-2:] == [sys.executable, str(Path(kernel_dir) / "runner.py")]
    assert ["--bind", kernel_dir, kernel_dir] == argv[argv.index("--bind"):argv.index("--bind") + 3]
    assert ["--bind", sock, sock] == argv[argv.index(sock) - 1:argv.index(sock) + 2]


def test_home_and_profile_are_never_bound(short_dir):
    """An interpreter under $HOME gets its own prefix bound; $HOME itself never is.

    This is the assertion that matters: binding ``~`` (or HERMES_HOME) would put
    ``config.yaml``, ``.ssh`` and every token back inside reach.
    """
    plan, _, _ = _argv_for(short_dir)
    home = os.path.abspath(os.path.expanduser("~"))
    hermes_home = os.path.abspath(os.path.expanduser(os.environ.get("HERMES_HOME") or "~/.hermes"))
    assert home not in plan.mounts
    assert hermes_home not in plan.mounts
    # Any remaining /home mount must be an interpreter root, never the home dir itself.
    allowed = set(interpreter_roots(sys.executable))
    for mount in plan.mounts:
        if mount.startswith(home + os.sep):
            assert home not in mount or mount != home
            assert any(mount == root or mount.startswith(root + os.sep) or root.startswith(mount + os.sep)
                       for root in allowed), mount


def test_allow_network_keeps_net_namespace(short_dir):
    plan, _, _ = _argv_for(short_dir, allow_network=True)
    assert "--unshare-net" not in plan.argv


def test_project_cwd_bound_only_when_different(tmp_path, short_dir):
    plan, kernel_dir, _ = _argv_for(short_dir)
    assert kernel_dir in plan.mounts
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = SandboxConfig(enabled=True, bwrap_path=_fake_bwrap(tmp_path))
    plan2 = plan_kernel_sandbox(
        [sys.executable, "runner.py"], kernel_dir=kernel_dir,
        rpc_socket=str(Path(short_dir, "rpc.sock")), child_cwd=str(repo), config=cfg,
    )
    assert str(repo) in plan2.mounts
    assert ["--chdir", str(repo)] == list(plan2.argv[plan2.argv.index("--chdir"):plan2.argv.index("--chdir") + 2])


def test_extra_paths_keep_their_mode(tmp_path, short_dir):
    ro = tmp_path / "ro"
    rw = tmp_path / "rw"
    ro.mkdir()
    rw.mkdir()
    plan, _, _ = _argv_for(short_dir, extra_ro_paths=(str(ro),), extra_rw_paths=(str(rw),))
    argv = list(plan.argv)
    i = argv.index(str(ro))
    assert argv[i - 1] == "--ro-bind"
    j = argv.index(str(rw))
    assert argv[j - 1] == "--bind"


def test_interpreter_roots_cover_venv_and_base():
    roots = interpreter_roots(sys.executable)
    assert roots
    assert any(os.path.isdir(r) for r in roots)
    assert all(os.path.isabs(r) for r in roots)


def test_config_parsing_accepts_strings_and_rejects_junk(tmp_path):
    cfg = load_sandbox_config({"sandbox": {"enabled": "yes", "allow_network": "0",
                                           "extra_ro_paths": str(tmp_path)}})
    assert cfg.enabled is True
    assert cfg.allow_network is False
    assert cfg.extra_ro_paths == (str(tmp_path),)
    assert load_sandbox_config({"sandbox": "nope"}).enabled is False
    assert sandbox_enabled(SandboxConfig(enabled=False)) is False


BWRAP = os.environ.get("BWRAP") or ("/usr/bin/bwrap" if os.path.exists("/usr/bin/bwrap") else "")