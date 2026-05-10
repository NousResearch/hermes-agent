"""
Regression tests for memory file permission preservation (issue #22889).

Hermes' MemoryStore._write_file() previously used tempfile.mkstemp() which
creates files with 0o600 permissions. After atomic_replace(), the target
file inherited these restrictive permissions, breaking group-shared deployments.
"""

import os
import stat
import tempfile
from pathlib import Path


class TestMemoryPermissionPreservation:
    """Tests that MemoryStore preserves existing file permissions."""

    def test_write_file_preserves_group_permissions(self):
        """Existing file with 0o660 should keep 0o660 after update."""
        from tools.memory_tool import MemoryStore

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MEMORY.md"
            p.write_text("old content", encoding="utf-8")
            os.chmod(p, 0o660)

            before = stat.S_IMODE(p.stat().st_mode)
            MemoryStore._write_file(p, ["new entry"])
            after = stat.S_IMODE(p.stat().st_mode)

            assert before == 0o660, f"Setup failed: before={oct(before)}"
            assert after == 0o660, f"Permissions changed: {oct(before)} -> {oct(after)}"

    def test_write_file_preserves_owner_only(self):
        """Existing file with 0o600 should keep 0o600 after update."""
        from tools.memory_tool import MemoryStore

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MEMORY.md"
            p.write_text("old content", encoding="utf-8")
            os.chmod(p, 0o600)

            before = stat.S_IMODE(p.stat().st_mode)
            MemoryStore._write_file(p, ["new entry"])
            after = stat.S_IMODE(p.stat().st_mode)

            assert before == 0o600
            assert after == 0o600

    def test_write_file_preserves_world_readable(self):
        """Existing file with 0o644 should keep 0o644 after update."""
        from tools.memory_tool import MemoryStore

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MEMORY.md"
            p.write_text("old content", encoding="utf-8")
            os.chmod(p, 0o644)

            MemoryStore._write_file(p, ["new entry"])
            after = stat.S_IMODE(p.stat().st_mode)

            assert after == 0o644

    def test_new_file_gets_default_permissions(self):
        """New file (doesn't exist) should get default permissions from mkstemp."""
        from tools.memory_tool import MemoryStore

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "NEW_MEMORY.md"
            # File does not exist yet

            MemoryStore._write_file(p, ["first entry"])
            after = stat.S_IMODE(p.stat().st_mode)

            # mkstemp creates with 0o600, and there's no original to preserve
            assert after == 0o600

    def test_write_file_content_is_correct(self):
        """Permission preservation shouldn't affect content correctness."""
        from tools.memory_tool import MemoryStore

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MEMORY.md"
            p.write_text("old", encoding="utf-8")
            os.chmod(p, 0o660)

            MemoryStore._write_file(p, ["entry 1", "entry 2"])
            content = p.read_text(encoding="utf-8")

            assert "entry 1" in content
            assert "entry 2" in content
            assert "§" in content  # delimiter present

    def test_multiple_writes_preserve_permissions(self):
        """Multiple consecutive writes should all preserve the original mode."""
        from tools.memory_tool import MemoryStore

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MEMORY.md"
            p.write_text("initial", encoding="utf-8")
            os.chmod(p, 0o664)

            for i in range(3):
                MemoryStore._write_file(p, [f"entry {i}"])
                after = stat.S_IMODE(p.stat().st_mode)
                assert after == 0o664, f"Write {i+1} changed permissions: {oct(after)}"


if __name__ == "__main__":
    import sys

    test_class = TestMemoryPermissionPreservation()
    methods = [m for m in dir(test_class) if m.startswith("test_")]
    passed = 0
    failed = 0

    for method_name in methods:
        try:
            getattr(test_class, method_name)()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
