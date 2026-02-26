"""Tests for the V4A patch format parser."""

from tools.patch_parser import (
    OperationType,
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
