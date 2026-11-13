"""Tests for Modal bulk download: the remote tar must survive a text-only stdout stream.

Modal's sandbox streams are UTF-8 text, so a raw tar piped to stdout dies on the first
non-UTF-8 byte. _modal_bulk_download therefore base64-encodes the archive in the sandbox and
decodes it strictly on the host; the archive itself stays uncompressed so the caller's
``sync_back_max_bytes`` cap keeps measuring extracted size.
"""

import base64
import io
import tarfile
import textwrap

import pytest

from tools.environments import modal as modal_env


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_env(exec_return):
    """ModalEnvironment without __init__ (no SDK/network); only _exec is exercised."""
    env = object.__new__(modal_env.ModalEnvironment)
    calls: dict = {}

    def fake_exec(cmd, *, timeout, stdin=None, fail_label=None, capture=False):
        calls.update(cmd=cmd, timeout=timeout, capture=capture, fail_label=fail_label)
        return exec_return

    env._exec = fake_exec
    return env, calls


def test_binary_payload_roundtrips_through_text_stream(tmp_path):
    """0xff/0x00 bytes in the archive must reach disk intact (the 1.6.x regression)."""
    original = _tar({"root/.hermes/state.db": b"\x00\xff\xfe binary sqlite"})
    env, calls = _make_env(base64.b64encode(original).decode("ascii"))

    dest = tmp_path / "pull.tar"
    env._modal_bulk_download(dest)

    assert dest.read_bytes() == original
    with tarfile.open(dest) as tar:
        assert tar.extractfile("root/.hermes/state.db").read() == b"\x00\xff\xfe binary sqlite"
    assert calls["capture"] is True
    assert calls["fail_label"] == "bulk download"
    # Uncompressed tar keeps sync_back_max_bytes a bound on extracted size, not on a gzipped
    # stream that could expand far beyond it.
    assert "tar cf -" in calls["cmd"]
    assert "tar czf" not in calls["cmd"]
    assert "base64" in calls["cmd"]
    assert "set -o pipefail" in calls["cmd"]
    assert calls["timeout"] == 120


def test_accepts_line_wrapped_base64(tmp_path):
    """Sandbox base64 wraps at 76 cols by default; the wraps must not corrupt the archive."""
    original = _tar({"root/.hermes/a.bin": b"\x80\x81\x82"})
    wrapped = "\n".join(textwrap.wrap(base64.b64encode(original).decode("ascii"), 76)) + "\n"

    env, _ = _make_env(wrapped)
    dest = tmp_path / "pull.tar"
    env._modal_bulk_download(dest)

    assert dest.read_bytes() == original


def test_accepts_bytes_return(tmp_path):
    """Guards the type contract: command-router readers hand back bytes, server readers str."""
    original = _tar({"root/.hermes/b.bin": b"\xff\x00"})
    env, _ = _make_env(base64.b64encode(original))

    dest = tmp_path / "pull.tar"
    env._modal_bulk_download(dest)

    assert dest.read_bytes() == original


def test_truncated_base64_fails_loudly(tmp_path):
    """A cut-off payload must raise at the transport boundary, not surface later as ReadError."""
    original = _tar({"root/.hermes/c.bin": b"\x01\x02\x03"})
    encoded = base64.b64encode(original).decode("ascii")

    env, _ = _make_env(encoded[:-6])  # drop past a 4-char boundary: lenient decode would "succeed"
    with pytest.raises(RuntimeError, match="corrupt base64"):
        env._modal_bulk_download(tmp_path / "pull.tar")


def test_non_ascii_transport_corruption_fails_loudly(tmp_path):
    """A stream that is not ASCII base64 must raise the same labelled RuntimeError."""
    original = _tar({"root/.hermes/d.bin": b"\x04\x05"})
    encoded = base64.b64encode(original).decode("ascii")
    corrupted = encoded[:20] + "é" + encoded[21:]  # text-mode reader injected a non-ASCII char

    env, _ = _make_env(corrupted)
    with pytest.raises(RuntimeError, match="corrupt base64"):
        env._modal_bulk_download(tmp_path / "pull.tar")
