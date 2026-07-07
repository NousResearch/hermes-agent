"""Tests for owner-only permissions on JSON snapshot stores (#60259)."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from tools.environments.base import _save_json_store, _load_json_store


class TestSaveJsonStorePermissions:
    def test_writes_with_owner_only_perms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_store.json"
            _save_json_store(path, {"key": "value"})
            assert path.exists()
            st = path.stat()
            if os.name != "nt":
                assert (st.st_mode & 0o777) == 0o600

    def test_umask_is_restored_after_write(self):
        old_umask = os.umask(0o022)
        os.umask(old_umask)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_store.json"
            _save_json_store(path, {"x": 1})
        restored = os.umask(0o022)
        os.umask(restored)
        assert restored == old_umask

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "nested" / "store.json"
            _save_json_store(path, {})
            assert path.exists()
