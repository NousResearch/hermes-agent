"""The sandbox, exercised for real: bwrap argv, then the execute_code tool itself.

Two layers here. ``build_sandbox_argv`` is run and probed from inside, which proves the
mount list does what it claims. Then the whole ``execute_code`` tool is driven twice with
the same cell, once confined and once not, which proves the wiring in ``_spawn`` and that
the control case really does reach the host.
"""

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.code_execution_sandbox import SandboxConfig, build_sandbox_argv

from tests.tools._child_env_fixtures import run_code

BWRAP = os.environ.get("BWRAP") or ("/usr/bin/bwrap" if os.path.exists("/usr/bin/bwrap") else "")

PROBE = textwrap.dedent(
    """
    import json, os, socket, sys

    sock_path, secret_path = sys.argv[1], sys.argv[2]
    out = {}
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        out["net"] = "open"
    except Exception as exc:
        out["net"] = type(exc).__name__
    try:
        s = socket.socket(socket.AF_UNIX)
        s.connect(sock_path)
        s.close()
        out["rpc"] = "ok"
    except Exception as exc:
        out["rpc"] = type(exc).__name__
    try:
        out["secret"] = open(secret_path).read()
    except Exception as exc:
        out["secret"] = type(exc).__name__
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "written.txt")
        open(p, "w").write("x")
        out["kernel_dir_writable"] = True
    except Exception as exc:
        out["kernel_dir_writable"] = type(exc).__name__
    try:
        open("/etc/hostname").read()
        out["etc_readable"] = True
    except Exception:
        out["etc_readable"] = False
    print(json.dumps(out))
    """
)


@pytest.mark.skipif(not BWRAP or not sys.platform.startswith("linux"), reason="needs bwrap on Linux")
def test_real_sandbox_blocks_home_and_network(short_dir):
    kernel_dir = short_dir / "kernel"
    kernel_dir.mkdir()
    probe = kernel_dir / "probe.py"
    probe.write_text(PROBE)

    sock_path = str(kernel_dir / "rpc.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    argv = build_sandbox_argv(
        bwrap=BWRAP, real_python=sys.executable, kernel_dir=str(kernel_dir),
        rpc_socket=sock_path, child_cwd=str(kernel_dir),
        config=SandboxConfig(enabled=True, bwrap_path=BWRAP),
    )
    secret = short_dir / "secret.txt"
    secret.write_text("host-only")
    try:
        proc = subprocess.run(argv + [str(probe), sock_path, str(secret)], capture_output=True,
                              text=True, timeout=60, check=False)
    finally:
        server.close()

    assert proc.returncode == 0, proc.stderr
    import json

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["secret"] in ("FileNotFoundError", "PermissionError")
    assert result["net"] != "open"
    assert result["rpc"] == "ok"
    assert result["kernel_dir_writable"] is True
    assert result["etc_readable"] is True


@pytest.mark.skipif(not BWRAP or not sys.platform.startswith("linux"), reason="needs bwrap on Linux")
def test_real_sandbox_allow_network_still_blocks_home(short_dir):
    kernel_dir = short_dir / "kernel"
    kernel_dir.mkdir()
    probe = kernel_dir / "probe.py"
    probe.write_text(PROBE)
    argv = build_sandbox_argv(
        bwrap=BWRAP, real_python=sys.executable, kernel_dir=str(kernel_dir),
        rpc_socket="", child_cwd=str(kernel_dir),
        config=SandboxConfig(enabled=True, allow_network=True, bwrap_path=BWRAP),
    )
    secret = short_dir / "secret.txt"
    secret.write_text("host-only")
    proc = subprocess.run(argv + [str(probe), "/nonexistent.sock", str(secret)],
                          capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    import json

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["secret"] in ("FileNotFoundError", "PermissionError")
    assert result["net"] == "open"

def test_bind_rpc_socket_uses_preferred_dir():
    """A sandboxed kernel's socket has to live inside the one exposed directory.

    Uses a short path on purpose: the AF_UNIX limit is real, and a deep tmpdir falls back
    to the socket-safe root by design (the planner then binds the socket file explicitly).
    """
    import tempfile

    from tools.code_kernel import SessionKernel, _bind_rpc_socket

    short_dir = tempfile.mkdtemp(prefix="k_", dir="/tmp")
    tmp_path = Path(short_dir)
    kernel = SessionKernel(("owner", "session", sys.executable, str(tmp_path), "tools"))
    endpoint = _bind_rpc_socket(kernel, prefer_dir=str(tmp_path))
    try:
        sock_path = kernel.sock_path
        assert sock_path and kernel.server_sock
        assert sock_path.startswith(str(tmp_path))
        assert endpoint == sock_path
        assert os.stat(sock_path).st_mode & 0o777 == 0o600
    finally:
        server = kernel.server_sock
        if server:
            server.close()
        if kernel.sock_path:
            os.unlink(kernel.sock_path)


def test_bind_rpc_socket_unchanged_without_preference(tmp_path):
    from tools.code_kernel import SessionKernel, _bind_rpc_socket

    kernel = SessionKernel(("owner", "session", sys.executable, str(tmp_path), "tools"))
    endpoint = _bind_rpc_socket(kernel)
    try:
        sock_path = kernel.sock_path
        assert sock_path and kernel.server_sock
        assert sock_path.startswith(os.path.dirname(endpoint))
        assert str(tmp_path) not in endpoint
    finally:
        server = kernel.server_sock
        if server:
            server.close()
        if kernel.sock_path:
            os.unlink(kernel.sock_path)


CONFINEMENT_CELL = """
import json, os, socket
from hermes_tools import read_file

def probe(fn):
    try:
        return fn()
    except Exception as exc:
        return type(exc).__name__

out = {
    "rpc": read_file(WITNESS).get("content", ""),
    "home_secret": probe(lambda: open(os.path.join(os.path.expanduser("~"), "home-secret.txt")).read()),
    "config_read": probe(lambda: open(os.path.join(os.environ["HERMES_HOME"], "config.yaml")).read(20)),
    "net": probe(lambda: socket.create_connection(("1.1.1.1", 443), timeout=3).close() or "open"),
    "staging_writable": probe(lambda: (open("written.txt", "w").write("x"), True)[1]),
}
print(json.dumps(out))
"""


def _run_confinement_cell(child_dir, mode="strict"):
    witness = child_dir / "witness.txt"
    witness.write_text("rpc -> ok\n", encoding="utf-8")
    # child_env sets HOME=child_dir, so this file is inside $HOME and nowhere else.
    (child_dir / "home-secret.txt").write_text("host-only", encoding="utf-8")
    return run_code(CONFINEMENT_CELL.replace("WITNESS", repr(str(witness))), mode)


@pytest.mark.skipif(not BWRAP or not sys.platform.startswith("linux"), reason="needs bwrap on Linux")
def test_real_kernel_is_confined(child_env, monkeypatch):
    """Through the real tool: a confined kernel keeps tool RPC and loses the host."""
    from tools import code_execution_sandbox as ces
    from tools.code_kernel import shutdown_all_kernels

    shutdown_all_kernels()
    monkeypatch.setattr(
        ces, "load_sandbox_config",
        lambda config=None: SandboxConfig(enabled=True, bwrap_path=BWRAP),
    )
    result = _run_confinement_cell(child_env)

    assert "rpc -> ok" in result["rpc"]
    assert result["home_secret"] in ("FileNotFoundError", "PermissionError", "NotADirectoryError")
    assert result["config_read"] in ("FileNotFoundError", "PermissionError", "NotADirectoryError")
    assert result["net"] in ("OSError", "FileNotFoundError", "NetworkUnreachable") or result["net"] != "open"
    assert result["staging_writable"] is True


@pytest.mark.skipif(not BWRAP or not sys.platform.startswith("linux"), reason="needs bwrap on Linux")
def test_unconfined_kernel_keeps_host_access(child_env):
    """Control for the test above: same cell, no sandbox, sees the host. Fails if the
    planner silently confines when told not to."""
    from tools.code_kernel import shutdown_all_kernels

    shutdown_all_kernels()
    result = _run_confinement_cell(child_env)

    assert "rpc -> ok" in result["rpc"]
    assert result["home_secret"] == "host-only"
    assert result["net"] == "open"