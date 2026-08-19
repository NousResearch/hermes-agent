"""Test: WhatsApp session directory and bridge.log have restrictive permissions (#80090).

On multi-user hosts, creds.json + signal keys + bridge.log (which may
contain QR pairing payloads) must not be world-readable.
"""
import os
import stat
import tempfile
from pathlib import Path


def test_session_dir_perms():
    """Session directory should be chmod 0o700 after creation."""
    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp) / "whatsapp" / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_dir.chmod(0o700)
        except OSError:
            return  # Windows — chmod not supported

        mode = stat.S_IMODE(session_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"


def test_bridge_log_perms():
    """bridge.log should be chmod 0o600 after creation."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "bridge.log"
        log_path.write_text("test log content")
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            return  # Windows

        mode = stat.S_IMODE(log_path.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


def test_session_dir_not_world_readable():
    """Session directory must not be world-readable."""
    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp) / "whatsapp" / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_dir.chmod(0o700)
        except OSError:
            return

        mode = stat.S_IMODE(session_dir.stat().st_mode)
        assert not (mode & 0o077), f"Session dir is world/group accessible: {oct(mode)}"
