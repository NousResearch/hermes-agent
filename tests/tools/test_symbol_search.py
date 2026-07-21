"""Tests for symbol_search tool."""

import json
import os
import tempfile

import pytest


class TestSymbolSearch:
    @pytest.fixture
    def indexed_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "src")
            os.makedirs(src)
            with open(os.path.join(src, "main.py"), "w") as f:
                f.write("class User:\n    def __init__(self, name):\n        self.name = name\n")
                f.write("\ndef hello():\n    return 'world'\n")
                f.write("\nimport os\nfrom pathlib import Path\n")
            with open(os.path.join(src, "utils.py"), "w") as f:
                f.write("def helper():\n    pass\n")
                f.write("\nclass Config:\n    pass\n")
            from tools.code_index import code_index
            code_index(tmpdir, operation="build")
            yield tmpdir

    def test_check_requirements(self):
        from tools.symbol_search import check_symbol_search_requirements
        assert check_symbol_search_requirements() is True

    def test_search_by_name(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("User", indexed_project)
        data = json.loads(output)
        assert data["success"] is True
        assert data["total"] >= 1
        assert any(r["name"] == "User" for r in data["results"])

    def test_search_partial(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("help", indexed_project)
        data = json.loads(output)
        assert data["success"] is True
        assert data["total"] >= 1

    def test_search_exact(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("nonexistent_symbol_xyz", indexed_project, exact=True)
        data = json.loads(output)
        assert data["success"] is True
        assert data["total"] == 0

    def test_search_by_type(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("", indexed_project, symbol_type="class")
        data = json.loads(output)
        assert data["success"] is True
        assert data["total"] >= 2
        types = [r["type"] for r in data["results"]]
        assert all(t == "class" for t in types)

    def test_search_by_language(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("", indexed_project, language="python")
        data = json.loads(output)
        assert data["success"] is True
        assert data["total"] >= 4

    def test_search_find_usages(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("User", indexed_project, find_usages=True)
        data = json.loads(output)
        assert data["success"] is True
        if data["total"] > 0:
            assert "usages" in data["results"][0]
            assert "usage_count" in data["results"][0]

    def test_project_root_not_found(self):
        from tools.symbol_search import symbol_search
        output = symbol_search("test", "/nonexistent/path")
        data = json.loads(output)
        assert data["success"] is False

    def test_no_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from tools.symbol_search import symbol_search
            output = symbol_search("test", tmpdir)
            data = json.loads(output)
            assert data["success"] is False
            assert "Run 'code_index'" in data["error"]

    def test_limit(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("", indexed_project, limit=2)
        data = json.loads(output)
        assert data["success"] is True
        assert len(data["results"]) <= 2

    def test_results_have_by_file(self, indexed_project):
        from tools.symbol_search import symbol_search
        output = symbol_search("", indexed_project)
        data = json.loads(output)
        assert data["success"] is True
        assert "by_file" in data


class TestSymbolSearchSchema:
    def test_schema_has_required_fields(self):
        from tools.symbol_search import SYMBOL_SEARCH_SCHEMA
        assert SYMBOL_SEARCH_SCHEMA["name"] == "symbol_search"
        props = SYMBOL_SEARCH_SCHEMA["parameters"]["properties"]
        assert "query" in props
        assert "project_root" in props
        assert "symbol_type" in props
        assert "language" in props
        assert "find_usages" in props


# ---------------------------------------------------------------------------
# Regression: cross-file find_usages
# ---------------------------------------------------------------------------


class TestCrossFileUsages:
    """find_usages must search all indexed files, not just the definition's file."""

    def test_usage_in_different_file(self):
        from tools.symbol_search import symbol_search
        from tools.code_index import code_index
        with tempfile.TemporaryDirectory() as tmpdir:
            # File A defines the symbol
            with open(os.path.join(tmpdir, "defs.py"), "w") as f:
                f.write("class Calculator:\n    pass\n")
            # File B uses the symbol
            with open(os.path.join(tmpdir, "main.py"), "w") as f:
                f.write("from defs import Calculator\ncalc = Calculator()\n")

            code_index(tmpdir, operation="build")
            output = symbol_search("Calculator", tmpdir, find_usages=True)
            data = json.loads(output)
            assert data["success"] is True
            assert data["total"] >= 1
            result = data["results"][0]
            assert result["usage_count"] >= 2  # def + import + usage
            # Usages should span both files
            usage_files = {u["file"] for u in result["usages"]}
            assert len(usage_files) >= 2
