"""Tests for code_index tool."""

import json
import os
import tempfile

import pytest


class TestCodeIndex:
    @pytest.fixture
    def temp_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "src")
            os.makedirs(src)
            py_file = os.path.join(src, "main.py")
            with open(py_file, "w") as f:
                f.write("class User:\n    def __init__(self, name):\n        self.name = name\n")
                f.write("\ndef hello():\n    return 'world'\n")
                f.write("\nimport os\nfrom pathlib import Path\n")
            ts_file = os.path.join(src, "types.ts")
            with open(ts_file, "w") as f:
                f.write("export interface User {\n  name: string;\n  age: number;\n}\n")
                f.write("\nexport function greet(name: string): string {\n  return `Hello ${name}`;\n}\n")
            yield tmpdir

    def test_check_requirements(self):
        from tools.code_index import check_code_index_requirements
        assert check_code_index_requirements() is True

    def test_build_index(self, temp_project):
        from tools.code_index import code_index
        output = code_index(temp_project, operation="build")
        data = json.loads(output)
        assert data["success"] is True
        assert data["operation"] == "build"
        assert data["stats"]["new_files"] >= 2
        assert data["stats"]["symbols_found"] >= 4

    def test_clear_index(self, temp_project):
        from tools.code_index import code_index
        code_index(temp_project, operation="build")
        output = code_index(temp_project, operation="clear")
        data = json.loads(output)
        assert data["success"] is True
        assert data["operation"] == "clear"

    def test_status_after_build(self, temp_project):
        from tools.code_index import code_index
        code_index(temp_project, operation="build")
        output = code_index(temp_project, operation="status")
        data = json.loads(output)
        assert data["success"] is True
        assert data["indexed"] is True
        assert data["symbols"] >= 4

    def test_status_no_index(self, temp_project):
        from tools.code_index import code_index
        output = code_index(temp_project, operation="status")
        data = json.loads(output)
        assert data["success"] is True
        assert data["indexed"] is False

    def test_project_root_not_found(self):
        from tools.code_index import code_index
        output = code_index("/nonexistent/path", operation="build")
        data = json.loads(output)
        assert data["success"] is False

    def test_build_with_file_pattern(self, temp_project):
        from tools.code_index import code_index
        output = code_index(temp_project, operation="build", file_pattern="*.ts")
        data = json.loads(output)
        assert data["success"] is True
        assert data["stats"]["new_files"] >= 1

    def test_build_targeted_path(self, temp_project):
        from tools.code_index import code_index
        src = os.path.join(temp_project, "src")
        output = code_index(temp_project, operation="build", path=src)
        data = json.loads(output)
        assert data["success"] is True
        assert data["stats"]["new_files"] >= 2


class TestCodeIndexSchema:
    def test_schema_has_required_fields(self):
        from tools.code_index import CODE_INDEX_SCHEMA
        assert CODE_INDEX_SCHEMA["name"] == "code_index"
        props = CODE_INDEX_SCHEMA["parameters"]["properties"]
        assert "project_root" in props
        assert "operation" in props
        assert props["operation"]["enum"] == ["build", "status", "clear"]


class TestExtractSymbols:
    def test_python_symbols(self):
        from tools.code_index import _extract_symbols
        content = "class Foo:\n    def bar(self):\n        pass\ndef baz():\n    pass"
        symbols = _extract_symbols(content, "python")
        names = [s["name"] for s in symbols if s["type"] in ("class", "function")]
        assert "Foo" in names
        assert "baz" in names

    def test_ts_symbols(self):
        from tools.code_index import _extract_symbols
        content = "interface User {}\nfunction greet() {}\nconst x = 1"
        symbols = _extract_symbols(content, "typescript")
        names = [s["name"] for s in symbols]
        assert "User" in names
        assert "greet" in names

    def test_go_symbols(self):
        from tools.code_index import _extract_symbols
        content = "func Hello() {}\ntype Config struct {}\nconst MaxValue = 100"
        symbols = _extract_symbols(content, "go")
        names = [s["name"] for s in symbols]
        assert "Hello" in names

    def test_rust_symbols(self):
        from tools.code_index import _extract_symbols
        content = "fn hello() {}\nstruct Config {}"
        symbols = _extract_symbols(content, "rust")
        names = [s["name"] for s in symbols]
        assert "hello" in names

    def test_detect_language(self):
        from tools.code_index import _detect_language
        assert _detect_language("test.py") == "python"
        assert _detect_language("test.ts") == "typescript"
        assert _detect_language("test.js") == "javascript"
        assert _detect_language("test.go") == "go"
        assert _detect_language("test.rs") == "rust"
        assert _detect_language("test.unknown") == "unknown"


# ---------------------------------------------------------------------------
# Regression: Rust impl capture group
# ---------------------------------------------------------------------------


class TestRustImplCapture:
    """Rust impl lines must be captured with a valid symbol name, not skipped."""

    def test_impl_named_type(self):
        from tools.code_index import _extract_symbols
        content = "impl Foo {\n    fn bar(&self) {}\n}"
        symbols = _extract_symbols(content, "rust")
        impl_names = [s["name"] for s in symbols if s["type"] == "impl"]
        assert "Foo" in impl_names

    def test_impl_for_type(self):
        from tools.code_index import _extract_symbols
        content = "impl Display for MyStruct {\n    fn fmt(&self) {}\n}"
        symbols = _extract_symbols(content, "rust")
        impl_names = [s["name"] for s in symbols if s["type"] == "impl"]
        assert "Display for MyStruct" in impl_names

    def test_impl_no_crash_on_bare_impl(self):
        """Bare 'impl {' without a name must not crash."""
        from tools.code_index import _extract_symbols
        content = "impl {\n    fn new() {}\n}"
        # Should not raise IndexError
        symbols = _extract_symbols(content, "rust")
        # No impl symbol should be captured (no name to capture)
        impl_syms = [s for s in symbols if s["type"] == "impl"]
        assert len(impl_syms) == 0


# ---------------------------------------------------------------------------
# Regression: DB key collision (basename → hash)
# ---------------------------------------------------------------------------


class TestIndexDbCollision:
    """Two projects with the same basename must get separate indexes."""

    def test_same_basename_different_roots(self):
        from tools.code_index import code_index, _index_db_path
        with tempfile.TemporaryDirectory() as tmpdir:
            root_a = os.path.join(tmpdir, "work", "a", "api")
            root_b = os.path.join(tmpdir, "work", "b", "api")
            os.makedirs(root_a)
            os.makedirs(root_b)

            with open(os.path.join(root_a, "a.py"), "w") as f:
                f.write("def alpha(): pass\n")
            with open(os.path.join(root_b, "b.py"), "w") as f:
                f.write("def beta(): pass\n")

            code_index(root_a, operation="build")
            code_index(root_b, operation="build")

            db_a = _index_db_path(os.path.abspath(root_a))
            db_b = _index_db_path(os.path.abspath(root_b))
            assert db_a != db_b

            # Verify each index only contains its own file
            import sqlite3
            conn_a = sqlite3.connect(db_a)
            rows_a = conn_a.execute("SELECT path FROM files").fetchall()
            conn_a.close()
            assert any("a.py" in r[0] for r in rows_a)
            assert not any("b.py" in r[0] for r in rows_a)

            conn_b = sqlite3.connect(db_b)
            rows_b = conn_b.execute("SELECT path FROM files").fetchall()
            conn_b.close()
            assert any("b.py" in r[0] for r in rows_b)
            assert not any("a.py" in r[0] for r in rows_b)


# ---------------------------------------------------------------------------
# Regression: stale rows on rebuild (deleted files pruned)
# ---------------------------------------------------------------------------


class TestStaleRowPruning:
    """Rebuild must remove rows for files that no longer exist on disk."""

    def test_deleted_file_removed_after_rebuild(self):
        from tools.code_index import code_index
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "keep.py")
            f2 = os.path.join(tmpdir, "delete_me.py")
            with open(f1, "w") as f:
                f.write("def keep(): pass\n")
            with open(f2, "w") as f:
                f.write("def doomed(): pass\n")

            code_index(tmpdir, operation="build")

            # Verify both are indexed
            status1 = json.loads(code_index(tmpdir, operation="status"))
            assert status1["files"] == 2

            # Delete one file and rebuild
            os.unlink(f2)
            code_index(tmpdir, operation="build")

            status2 = json.loads(code_index(tmpdir, operation="status"))
            assert status2["files"] == 1

            # Verify the deleted file's symbols are also gone
            import sqlite3, hashlib
            abs_root = os.path.abspath(tmpdir)
            root_hash = hashlib.sha256(abs_root.encode()).hexdigest()[:16]
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(abs_root)),  # walk up from tmpdir
            )
            # Use the helper to find the DB
            from tools.code_index import _index_db_path
            db_path = _index_db_path(abs_root)
            conn = sqlite3.connect(db_path)
            sym_files = conn.execute("SELECT DISTINCT file_path FROM symbols").fetchall()
            conn.close()
            assert not any("delete_me.py" in r[0] for r in sym_files)
