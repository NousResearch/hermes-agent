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


class TestBoundaryMarkersInContent:
    """Patch boundary markers inside content lines must not be treated as
    real boundaries (docs about the patch format, nested patch text, etc.)."""

    def test_end_patch_marker_in_add_content_does_not_truncate(self):
        patch = """\
*** Begin Patch
*** Add File: notes.md
+doc line one
+*** End Patch
+content after the marker mention
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1
        contents = [l.content for l in ops[0].hunks[0].lines if l.prefix == "+"]
        assert contents == [
            "doc line one",
            "*** End Patch",
            "content after the marker mention",
        ]

    def test_begin_patch_marker_in_content_does_not_discard_operations(self):
        patch = """\
*** Begin Patch
*** Add File: first.md
+first file content
*** Add File: second.md
+*** Begin Patch
+second file content
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert [(op.operation, op.file_path) for op in ops] == [
            (OperationType.ADD, "first.md"),
            (OperationType.ADD, "second.md"),
        ]

    def test_context_line_end_patch_marker_does_not_truncate(self):
        patch = """\
*** Begin Patch
*** Update File: guide.md
@@ section @@
 old line
 *** End Patch
+new line
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None
        assert len(ops) == 1
        hunk_lines = [(l.prefix, l.content) for l in ops[0].hunks[0].lines]
        assert (" ", "*** End Patch") in hunk_lines
        assert ("+", "new line") in hunk_lines


class TestParseInvalidPatch:
    def test_empty_patch_returns_empty_ops(self):
        ops, err = parse_v4a_patch("")
        assert err is None
        assert ops == []


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

            def write_file(self, path, content, pre_content=None):
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
            def write_file(self, path, content, pre_content=None):
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
            def write_file(self, path, content, pre_content=None):
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
            def write_file(self, path, content, pre_content=None):
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
            def write_file(self, path, content, pre_content=None):
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

            def write_file(self, path, content, pre_content=None):
                written[path] = content
                return SimpleNamespace(error=None)

        result = apply_v4a_operations(ops, FakeFileOps())
        assert result.success is False
        assert written == {}, f"No files should have been written, got: {list(written.keys())}"
        assert "validation failed" in result.error.lower()


    def test_validation_error_identifies_hunk_number(self):
        patch = """\
*** Begin Patch
*** Update File: a.py
@@ first @@
-first = 1
+first = 2
@@ missing @@
-does_not_exist = 1
+does_not_exist = 2
*** End Patch"""
        ops, err = parse_v4a_patch(patch)
        assert err is None

        class FakeFileOps:
            def read_file_raw(self, path):
                return SimpleNamespace(content="first = 1\n", error=None)

        result = apply_v4a_operations(ops, FakeFileOps())
        assert result.success is False
        assert "hunk 2" in result.error.lower()


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


class TestV4ALspDiagnosticsPropagation:
    """V4A patches must surface ``WriteResult.lsp_diagnostics`` from the
    underlying ``write_file`` calls on ``PatchResult.lsp_diagnostics``.

    Without explicit propagation the LSP tier's output gets silently
    dropped on the V4A code path — see Copilot review #3271017295 on
    PR #29054.  The shell-linter LSP skip introduced by that PR makes
    this gap visible: a ``.ts`` / ``.go`` / ``.rs`` V4A patch with LSP
    active would otherwise return ``lint = {f: {skipped: True, ...}}``
    and zero diagnostics from any channel.
    """

    def _build_ops_writing(self, path: str, content: str):
        """Build a single ADD operation that writes ``content`` to ``path``."""
        # Use the V4A parser so we don't have to construct PatchOperation
        # / Hunk / Line objects by hand.
        lines = "\n".join(f"+{line}" for line in content.splitlines())
        patch_text = (
            "*** Begin Patch\n"
            f"*** Add File: {path}\n"
            f"{lines}\n"
            "*** End Patch"
        )
        ops, err = parse_v4a_patch(patch_text)
        assert err is None, err
        return ops

    def test_lsp_diagnostics_propagated_from_write_file_on_add(self):
        """ADD op: ``WriteResult.lsp_diagnostics`` flows through to
        ``PatchResult.lsp_diagnostics``."""
        ops = self._build_ops_writing("foo.ts", "const x: number = 1\n")

        diag_block = (
            "<diagnostics file=\"foo.ts\">\n"
            "ERROR [1:7] some diagnostic\n"
            "</diagnostics>"
        )

        class FakeFileOps:
            def write_file(self, path, content, pre_content=None):
                return SimpleNamespace(error=None, lsp_diagnostics=diag_block)

            def _check_lint(self, path):
                return SimpleNamespace(to_dict=lambda: {"skipped": True})

        result = apply_v4a_operations(ops, FakeFileOps())

        assert result.success is True
        assert result.lsp_diagnostics == diag_block

    def test_lsp_diagnostics_propagated_from_write_file_on_update(self):
        """UPDATE op: ``WriteResult.lsp_diagnostics`` flows through to
        ``PatchResult.lsp_diagnostics``."""
        patch_text = (
            "*** Begin Patch\n"
            "*** Update File: bar.ts\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )
        ops, err = parse_v4a_patch(patch_text)
        assert err is None

        diag_block = (
            "<diagnostics file=\"bar.ts\">\n"
            "ERROR [3:1] something\n"
            "</diagnostics>"
        )

        class FakeFileOps:
            def read_file_raw(self, path):
                return SimpleNamespace(content="ctx\nold\nctx\n", error=None)

            def write_file(self, path, content, pre_content=None):
                return SimpleNamespace(error=None, lsp_diagnostics=diag_block)

            def _check_lint(self, path):
                return SimpleNamespace(to_dict=lambda: {"skipped": True})

        result = apply_v4a_operations(ops, FakeFileOps())

        assert result.success is True
        assert result.lsp_diagnostics == diag_block

    def test_lsp_diagnostics_none_when_no_blocks_emitted(self):
        """When no underlying ``write_file`` produced diagnostics, the
        aggregated field stays ``None`` (so it doesn't get serialized
        as an empty string in ``PatchResult.to_dict``)."""
        ops = self._build_ops_writing("foo.py", "x = 1\n")

        class FakeFileOps:
            def write_file(self, path, content, pre_content=None):
                # lsp_diagnostics omitted entirely (older WriteResult shape).
                return SimpleNamespace(error=None)

            def _check_lint(self, path):
                return SimpleNamespace(to_dict=lambda: {"success": True})

        result = apply_v4a_operations(ops, FakeFileOps())

        assert result.success is True
        assert result.lsp_diagnostics is None

    def test_lsp_diagnostics_combined_across_multiple_files(self):
        """When several files in one V4A patch produce diagnostics,
        each block appears in the combined output so per-file attribution
        is preserved."""
        patch_text = (
            "*** Begin Patch\n"
            "*** Add File: a.ts\n"
            "+const a = 1\n"
            "*** Add File: b.ts\n"
            "+const b = 2\n"
            "*** End Patch"
        )
        ops, err = parse_v4a_patch(patch_text)
        assert err is None

        per_file = {
            "a.ts": "<diagnostics file=\"a.ts\">\nERR a\n</diagnostics>",
            "b.ts": "<diagnostics file=\"b.ts\">\nERR b\n</diagnostics>",
        }

        class FakeFileOps:
            def write_file(self, path, content, pre_content=None):
                return SimpleNamespace(error=None, lsp_diagnostics=per_file[path])

            def _check_lint(self, path):
                return SimpleNamespace(to_dict=lambda: {"skipped": True})

        result = apply_v4a_operations(ops, FakeFileOps())

        assert result.success is True
        assert result.lsp_diagnostics is not None
        assert per_file["a.ts"] in result.lsp_diagnostics
        assert per_file["b.ts"] in result.lsp_diagnostics


class _DictFileOps:
    """In-memory file_ops backing store supporting update/move/delete/add."""

    def __init__(self, files):
        self.files = dict(files)

    def read_file_raw(self, path):
        if path in self.files:
            return SimpleNamespace(content=self.files[path], error=None)
        return SimpleNamespace(content="", error="file not found")

    def write_file(self, path, content, pre_content=None):
        self.files[path] = content
        return SimpleNamespace(error=None)

    def move_file(self, src, dst):
        self.files[dst] = self.files.pop(src)
        return SimpleNamespace(error=None)

    def delete_file(self, path):
        self.files.pop(path, None)
        return SimpleNamespace(error=None)


class TestMoveThenUpdateSameFile:
    """A rename-then-edit patch must validate and apply (was rejected).

    Regression: _validate_operations read the UPDATE's target from disk before
    the MOVE ran, so `Move a->b` + `Update b` failed with 'b: file not found'.
    """

    def test_move_then_update_destination(self):
        patch = (
            "*** Begin Patch\n"
            "*** Move File: a.py -> b.py\n"
            "*** Update File: b.py\n"
            "@@\n"
            "-x = 1\n"
            "+x = 42\n"
            "*** End Patch\n"
        )
        ops, err = parse_v4a_patch(patch)
        assert err is None
        fo = _DictFileOps({"a.py": "x = 1\nkeep = 2\n"})
        result = apply_v4a_operations(ops, fo)
        assert result.success is True, getattr(result, "error", None)
        assert "a.py" not in fo.files
        assert fo.files["b.py"] == "x = 42\nkeep = 2\n"

    def test_move_onto_existing_destination_still_rejected(self):
        """The overlay must not mask a genuine 'destination exists' conflict."""
        patch = (
            "*** Begin Patch\n"
            "*** Move File: a.py -> b.py\n"
            "*** End Patch\n"
        )
        ops, err = parse_v4a_patch(patch)
        assert err is None
        fo = _DictFileOps({"a.py": "1\n", "b.py": "2\n"})
        result = apply_v4a_operations(ops, fo)
        assert result.success is False
        assert "already exists" in (result.error or "")


class TestCrlfPatchBody:
    """A CRLF-encoded patch body must not inject stray carriage returns."""

    def test_crlf_body_applied_to_lf_file(self):
        patch = (
            "*** Begin Patch\r\n"
            "*** Update File: f.py\r\n"
            "@@\r\n"
            "-    x = 1\r\n"
            "+    x = 2\r\n"
            "*** End Patch\r\n"
        )
        ops, err = parse_v4a_patch(patch)
        assert err is None
        fo = _DictFileOps({"f.py": "def f():\n    x = 1\n    return x\n"})
        result = apply_v4a_operations(ops, fo)
        assert result.success is True, getattr(result, "error", None)
        assert "\r" not in fo.files["f.py"]
        assert fo.files["f.py"] == "def f():\n    x = 2\n    return x\n"


class TestV4ABomRoundTrip:
    """V4A patches must not silently strip a UTF-8 BOM on UPDATE.

    ``read_file_raw`` deliberately strips the BOM (the agent should
    never see U+FEFF), but the underlying ``write_file`` must restore
    it on rewrite — otherwise a V4A patch turns an existing BOM-bearing
    file into a plain UTF-8 file.  Regression for teknium1 review on
    PR #55661.
    """

    BOM = "\ufeff"

    def _file_ops_for_update(self, file_path: str, original_bytes: bytes):
        """Build a FakeFileOps whose ``write_file`` writes real bytes to
        ``file_path``, simulating BOM-preserving behaviour like the real
        ``FileOperations.write_file`` (which probes disk for the marker)."""
        from pathlib import Path
        from tools.file_operations import _has_bom, _UTF8_BOM

        target = Path(file_path)
        _bom = self.BOM  # capture for inner class

        class FakeFileOps:
            def read_file_raw(self, path):
                # Simulate BOM-stripped read — same as the real
                # read_file_raw which strips the marker before returning.
                decoded = original_bytes.decode("utf-8")
                if decoded.startswith(_bom):
                    decoded = decoded[1:]
                return SimpleNamespace(content=decoded, error=None)

            def write_file(self, path, content, pre_content=None):
                # Simulate real write_file: probe the target for a BOM
                # (the real impl calls _file_has_bom → head -c 3) and
                # prepend if the original had one.
                had_bom = target.exists() and target.read_bytes().startswith(
                    _bom.encode("utf-8")
                )
                if had_bom and not _has_bom(content):
                    content = _UTF8_BOM + content
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return SimpleNamespace(error=None)

        return FakeFileOps()

    def test_update_preserves_bom(self, tmp_path):
        """A V4A UPDATE on a BOM-bearing file keeps the BOM."""
        from tools.patch_parser import parse_v4a_patch, apply_v4a_operations

        target = tmp_path / "bom_config.py"
        original = self.BOM + "setting = 'old'\n"
        target.write_text(original, encoding="utf-8")

        patch = """\
*** Begin Patch
*** Update File: bom_config.py
@@ setting @@
-setting = 'old'
+setting = 'new'
*** End Patch"""

        ops, err = parse_v4a_patch(patch)
        assert err is None

        file_ops = self._file_ops_for_update(str(target), original.encode("utf-8"))
        result = apply_v4a_operations(ops, file_ops)

        assert result.success is True
        raw = target.read_bytes()
        assert raw.startswith(
            self.BOM.encode("utf-8")
        ), "BOM was stripped by V4A round-trip"
        assert b"setting = 'new'" in raw
        assert b"setting = 'old'" not in raw

    def test_update_no_bom_when_original_had_none(self, tmp_path):
        """A V4A UPDATE on a plain file must NOT inject a BOM."""
        from tools.patch_parser import parse_v4a_patch, apply_v4a_operations

        target = tmp_path / "plain.py"
        original = "print('hello')\n"
        target.write_text(original, encoding="utf-8")

        patch = """\
*** Begin Patch
*** Update File: plain.py
@@ print @@
-print('hello')
+print('world')
*** End Patch"""

        ops, err = parse_v4a_patch(patch)
        assert err is None

        file_ops = self._file_ops_for_update(str(target), original.encode("utf-8"))
        result = apply_v4a_operations(ops, file_ops)

        assert result.success is True
        raw = target.read_bytes()
        assert not raw.startswith(
            self.BOM.encode("utf-8")
        ), "BOM was injected on a plain file"
        assert b"print('world')" in raw
