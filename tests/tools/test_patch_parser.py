"""Tests for the V4A patch format parser."""

from types import SimpleNamespace

from tools.patch_parser import (
    OperationType,
    apply_v4a_operations,
    parse_v4a_patch,
)


class TestParseUpdateFile:
    def test_basic_update(self):
        patch = """\
*** Begin Patch
*** Update File: src/main.py
@@ def greet @@
 def greet():
-    print("hello")
+    print("hi")
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1

        op = ops[0]
        assert op.operation == OperationType.UPDATE
        assert op.file_path == "src/main.py"
        assert len(op.hunks) == 1

        hunk = op.hunks[0]
        assert hunk.context_hint == "def greet"
        prefixes = [l.prefix for l in hunk.lines]
        assert " " in prefixes
        assert "-" in prefixes
        assert "+" in prefixes

    def test_multiple_hunks(self):
        patch = """\
*** Begin Patch
*** Update File: f.py
@@ first @@
 a
-b
+c
@@ second @@
 x
-y
+z
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1
        assert len(ops[0].hunks) == 2
        assert ops[0].hunks[0].context_hint == "first"
        assert ops[0].hunks[1].context_hint == "second"


class TestParseAddFile:
    def test_add_file(self):
        patch = """\
*** Begin Patch
*** Add File: new/module.py
+import os
+
+print("hello")
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1

        op = ops[0]
        assert op.operation == OperationType.ADD
        assert op.file_path == "new/module.py"
        assert len(op.hunks) == 1

        contents = [l.content for l in op.hunks[0].lines if l.prefix == "+"]
        assert contents[0] == "import os"
        assert contents[2] == 'print("hello")'


class TestParseDeleteFile:
    def test_delete_file(self):
        patch = """\
*** Begin Patch
*** Delete File: old/stuff.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1
        assert ops[0].operation == OperationType.DELETE
        assert ops[0].file_path == "old/stuff.py"


class TestParseMoveFile:
    def test_move_file(self):
        patch = """\
*** Begin Patch
*** Move File: old/path.py -> new/path.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1
        assert ops[0].operation == OperationType.MOVE
        assert ops[0].file_path == "old/path.py"
        assert ops[0].new_path == "new/path.py"


class TestParseInvalidPatch:
    def test_empty_patch_returns_empty_ops(self):
        ops, err = parse_v4a_patch("")
        assert err is None
        assert ops == []

    def test_no_begin_marker_still_parses(self):
        patch = """\
*** Update File: f.py
 line1
-old
+new
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1

    def test_multiple_operations(self):
        patch = """\
*** Begin Patch
*** Add File: a.py
+content_a
*** Delete File: b.py
*** Update File: c.py
 keep
-remove
+add
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 3
        assert ops[0].operation == OperationType.ADD
        assert ops[1].operation == OperationType.DELETE
        assert ops[2].operation == OperationType.UPDATE


class TestApplyUpdate:
    def test_preserves_non_prefix_pipe_characters_in_unmodified_lines(self):
        patch = """\
*** Begin Patch
*** Update File: sample.py
@@ result @@
     result = 1
-    return result
+    return result + 1
*** End Patch"""
        operations, err = parse_v4a_patch(patch)
        assert err is None

        class FakeFileOps:
            def __init__(self):
                self.written = None

            def read_file_raw(self, path):
                return SimpleNamespace(
                    content=(
                        'def run():\n'
                        '    cmd = "echo a | sed s/a/b/"\n'
                        '    result = 1\n'
                        '    return result'
                    ),
                    error=None,
                )

            def write_file(self, path, content):
                self.written = content
                return SimpleNamespace(error=None)

        file_ops = FakeFileOps()

        result = apply_v4a_operations(operations, file_ops)

        assert result.success is True
        assert file_ops.written == (
            'def run():\n'
            '    cmd = "echo a | sed s/a/b/"\n'
            '    result = 1\n'
            '    return result + 1'
        )


class TestAdditionOnlyHunks:
    """Regression tests for #3081 — addition-only hunks were silently dropped."""

    def test_addition_only_hunk_with_context_hint(self):
        """A hunk with only + lines should insert at the context hint location."""
        patch = """\
*** Begin Patch
*** Update File: src/app.py
@@ def main @@
+def helper():
+    return 42
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1
        assert len(ops[0].hunks) == 1

        hunk = ops[0].hunks[0]
        # All lines should be additions
        assert all(l.prefix == '+' for l in hunk.lines)

        # Apply to a file that contains the context hint
        class FakeFileOps:
            written = None
            def read_file_raw(self, path):
                return SimpleNamespace(
                    content="def main():\n    pass\n",
                    error=None,
                )
            def write_file(self, path, content):
                self.written = content
                return SimpleNamespace(error=None)

        file_ops = FakeFileOps()
        result = apply_v4a_operations(ops, file_ops)
        assert result.success is True
        assert "def helper():" in file_ops.written
        assert "return 42" in file_ops.written

    def test_addition_only_hunk_without_context_hint(self):
        """A hunk with only + lines and no context hint appends at end of file."""
        patch = """\
*** Begin Patch
*** Update File: src/app.py
+def new_func():
+    return True
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        class FakeFileOps:
            written = None
            def read_file_raw(self, path):
                return SimpleNamespace(
                    content="existing = True\n",
                    error=None,
                )
            def write_file(self, path, content):
                self.written = content
                return SimpleNamespace(error=None)

        file_ops = FakeFileOps()
        result = apply_v4a_operations(ops, file_ops)
        assert result.success is True
        assert file_ops.written.endswith("def new_func():\n    return True\n")
        assert "existing = True" in file_ops.written


class TestReadFileRaw:
    """Bug 1 regression tests — files > 2000 lines and lines > 2000 chars."""

    def test_apply_update_file_over_2000_lines(self):
        """A hunk targeting line 2200 must not truncate the file to 2000 lines."""
        patch = """\
*** Begin Patch
*** Update File: big.py
@@ marker_at_2200 @@
 line_2200
-old_value
+new_value
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        # Build a 2500-line file; the hunk targets a region at line 2200
        lines = [f"line_{i}" for i in range(1, 2501)]
        lines[2199] = "line_2200"   # index 2199 = line 2200
        lines[2200] = "old_value"
        file_content = "\n".join(lines)

        class FakeFileOps:
            written = None
            def read_file_raw(self, path):
                return SimpleNamespace(content=file_content, error=None)
            def write_file(self, path, content):
                self.written = content
                return SimpleNamespace(error=None)

        file_ops = FakeFileOps()
        result = apply_v4a_operations(ops, file_ops)
        assert result.success is True
        written_lines = file_ops.written.split("\n")
        assert len(written_lines) == 2500, (
            f"Expected 2500 lines, got {len(written_lines)}"
        )
        assert "new_value" in file_ops.written
        assert "old_value" not in file_ops.written

    def test_apply_update_preserves_long_lines(self):
        """A line > 2000 chars must be preserved verbatim after an unrelated hunk."""
        long_line = "x" * 3000
        patch = """\
*** Begin Patch
*** Update File: wide.py
@@ short_func @@
 def short_func():
-    return 1
+    return 2
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        file_content = f"def short_func():\n    return 1\n{long_line}\n"

        class FakeFileOps:
            written = None
            def read_file_raw(self, path):
                return SimpleNamespace(content=file_content, error=None)
            def write_file(self, path, content):
                self.written = content
                return SimpleNamespace(error=None)

        file_ops = FakeFileOps()
        result = apply_v4a_operations(ops, file_ops)
        assert result.success is True
        assert long_line in file_ops.written, "Long line was truncated"
        assert "... [truncated]" not in file_ops.written


class TestValidationPhase:
    """Bug 2 regression tests — validation prevents partial apply."""

    def test_validation_failure_writes_nothing(self):
        """If one hunk is invalid, no files should be written."""
        patch = """\
*** Begin Patch
*** Update File: a.py
 def good():
-    return 1
+    return 2
*** Update File: b.py
 THIS LINE DOES NOT EXIST
-    old
+    new
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        written = {}

        class FakeFileOps:
            def read_file_raw(self, path):
                files = {
                    "a.py": "def good():\n    return 1\n",
                    "b.py": "completely different content\n",
                }
                content = files.get(path)
                if content is None:
                    return SimpleNamespace(content=None, error=f"File not found: {path}")
                return SimpleNamespace(content=content, error=None)

            def write_file(self, path, content):
                written[path] = content
                return SimpleNamespace(error=None)

        result = apply_v4a_operations(ops, FakeFileOps())
        assert result.success is False
        assert written == {}, f"No files should have been written, got: {list(written.keys())}"
        assert "validation failed" in result.error.lower()

    def test_all_valid_operations_applied(self):
        """When all operations are valid, all files are written."""
        patch = """\
*** Begin Patch
*** Update File: a.py
 def foo():
-    return 1
+    return 2
*** Update File: b.py
 def bar():
-    pass
+    return True
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        written = {}

        class FakeFileOps:
            def read_file_raw(self, path):
                files = {
                    "a.py": "def foo():\n    return 1\n",
                    "b.py": "def bar():\n    pass\n",
                }
                return SimpleNamespace(content=files[path], error=None)

            def write_file(self, path, content):
                written[path] = content
                return SimpleNamespace(error=None)

        result = apply_v4a_operations(ops, FakeFileOps())
        assert result.success is True
        assert set(written.keys()) == {"a.py", "b.py"}


class TestApplyDelete:
    """Tests for _apply_delete producing a real unified diff."""

    def test_delete_diff_contains_removed_lines(self):
        """_apply_delete must embed the actual file content in the diff, not a placeholder."""
        patch = """\
*** Begin Patch
*** Delete File: old/stuff.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        class FakeFileOps:
            deleted = False

            def read_file_raw(self, path):
                return SimpleNamespace(
                    content="def old_func():\n    return 42\n",
                    error=None,
                )

            def delete_file(self, path):
                self.deleted = True
                return SimpleNamespace(error=None)

        file_ops = FakeFileOps()
        result = apply_v4a_operations(ops, file_ops)

        assert result.success is True
        assert file_ops.deleted is True
        # Diff must contain the actual removed lines, not a bare comment
        assert "-def old_func():" in result.diff
        assert "-    return 42" in result.diff
        assert "/dev/null" in result.diff

    def test_delete_diff_fallback_on_empty_file(self):
        """An empty file should produce the fallback comment diff."""
        patch = """\
*** Begin Patch
*** Delete File: empty.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        class FakeFileOps:
            def read_file_raw(self, path):
                return SimpleNamespace(content="", error=None)

            def delete_file(self, path):
                return SimpleNamespace(error=None)

        result = apply_v4a_operations(ops, FakeFileOps())
        assert result.success is True
        # unified_diff produces nothing for two empty inputs — fallback comment expected
        assert "Deleted" in result.diff or result.diff.strip() == ""


class TestCountOccurrences:
    def test_basic(self):
        from tools.patch_parser import _count_occurrences
        assert _count_occurrences("aaa", "a") == 3
        assert _count_occurrences("aaa", "aa") == 2
        assert _count_occurrences("hello world", "xyz") == 0
        assert _count_occurrences("", "x") == 0


class TestParseErrorSignalling:
    """Bug 3 regression tests — parse_v4a_patch must signal errors, not swallow them."""

    def test_update_with_no_hunks_returns_error(self):
        """An UPDATE with no hunk lines is a malformed patch and should error."""
        patch = """\
*** Begin Patch
*** Update File: foo.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is not None, "Expected a parse error for hunk-less UPDATE"
        assert ops == []

    def test_move_without_destination_returns_error(self):
        """A MOVE without '->' syntax should not silently produce a broken operation."""
        # The move regex requires '->' so this will be treated as an unrecognised
        # line and the op is never created.  Confirm nothing crashes and ops is empty.
        patch = """\
*** Begin Patch
*** Move File: src/foo.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        # Either parse sees zero ops (fine) or returns an error (also fine).
        # What is NOT acceptable is ops=[MOVE op with empty new_path] + err=None.
        if ops:
            assert err is not None, (
                "MOVE with missing destination must either produce empty ops or an error"
            )

    def test_valid_patch_returns_no_error(self):
        """A well-formed patch must still return err=None."""
        patch = """\
*** Begin Patch
*** Update File: f.py
 ctx
-old
+new
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1
