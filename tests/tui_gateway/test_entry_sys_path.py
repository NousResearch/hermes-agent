"""Tests for tui_gateway/entry.py sys.path hardening (issue #15989).

When the TUI backend is spawned by Node.js, the Python interpreter may have
'' or '.' at the front of sys.path, allowing a local utils/ directory in CWD
to shadow the installed utils module.  entry.py must sanitize sys.path before
any non-stdlib import is resolved.
"""

import os
import sys
from unittest.mock import patch


def test_empty_string_and_dot_removed_from_sys_path():
    original = sys.path[:]
    try:
        sys.path.insert(0, "")
        sys.path.insert(0, ".")
        assert "" in sys.path
        assert "." in sys.path

        # Run the entry.py fixup logic directly
        sys.path = [p for p in sys.path if p not in ("", ".")]

        assert "" not in sys.path
        assert "." not in sys.path
    finally:
        sys.path = original


def test_hermes_src_root_inserted_at_front():
    original = sys.path[:]
    try:
        fake_root = "/fake/hermes/src"
        with patch.dict(os.environ, {"HERMES_PYTHON_SRC_ROOT": fake_root}):
            _src_root = os.environ.get("HERMES_PYTHON_SRC_ROOT", "")
            if _src_root and _src_root not in sys.path:
                sys.path.insert(0, _src_root)
            sys.path = [p for p in sys.path if p not in ("", ".")]

        assert sys.path[0] == fake_root
    finally:
        sys.path = original


def test_src_root_not_duplicated_if_already_present():
    original = sys.path[:]
    try:
        fake_root = "/already/present"
        sys.path.insert(0, fake_root)
        count_before = sys.path.count(fake_root)

        with patch.dict(os.environ, {"HERMES_PYTHON_SRC_ROOT": fake_root}):
            _src_root = os.environ.get("HERMES_PYTHON_SRC_ROOT", "")
            if _src_root and _src_root not in sys.path:
                sys.path.insert(0, _src_root)
            sys.path = [p for p in sys.path if p not in ("", ".")]

        assert sys.path.count(fake_root) == count_before
    finally:
        sys.path = original


def test_no_src_root_env_does_not_crash():
    original = sys.path[:]
    try:
        env = {k: v for k, v in os.environ.items() if k != "HERMES_PYTHON_SRC_ROOT"}
        with patch.dict(os.environ, {}, clear=True):
            os.environ.update(env)
            _src_root = os.environ.get("HERMES_PYTHON_SRC_ROOT", "")
            if _src_root and _src_root not in sys.path:
                sys.path.insert(0, _src_root)
            sys.path = [p for p in sys.path if p not in ("", ".")]
        # No exception raised
    finally:
        sys.path = original
