from __future__ import annotations

import argparse
import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools import ezra_graph


def _refresh(root: Path, db: Path) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        ezra_graph.refresh(argparse.Namespace(root=[str(root)], db=str(db), alias_debug=False))


def _callees(db: Path) -> list[tuple[str, str]]:
    con = sqlite3.connect(db)
    return con.execute("SELECT callee, raw_callee FROM calls ORDER BY line").fetchall()


class EzraGraphDottedCallTests(unittest.TestCase):
    def test_bare_attribute_call_records_full_dotted_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            (root / "sample.py").write_text(
                "import json\n\n"
                "def emit(payload):\n"
                "    return json.dumps(payload)\n"
            )

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)

            self.assertIn(("json.dumps", "json.dumps"), _callees(db))

    def test_aliased_import_call_resolves_to_import_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            (root / "sample.py").write_text(
                "import json as j\n\n"
                "def emit(payload):\n"
                "    return j.dumps(payload)\n"
            )

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)

            self.assertIn(("json.dumps", "j.dumps"), _callees(db))

    def test_method_chain_call_records_full_dotted_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            (root / "sample.py").write_text(
                "class Worker:\n"
                "    def run(self):\n"
                "        return self.client.messages.create(model='x')\n"
            )

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)

            self.assertIn(("self.client.messages.create", "self.client.messages.create"), _callees(db))

    def test_callers_matches_suffix_for_bare_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            (root / "sample.py").write_text(
                "def register(ctx):\n"
                "    ctx.register_tool('x')\n"
            )

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ezra_graph.callers(argparse.Namespace(db=str(db), symbol="register_tool", limit=20, no_rank=False))

            out = buf.getvalue()
            self.assertIn("ctx.register_tool", out)
            self.assertIn("called_by=register", out)


    def test_dotted_import_without_alias_keeps_top_level_binding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            (root / "sample.py").write_text(
                "import json.tool\n\n"
                "def emit(payload):\n"
                "    return json.dumps(payload)\n"
            )

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)

            # `json` is bound to the top-level package; `json.dumps` must not
            # be rewritten to `json.tool.dumps`.
            self.assertIn(("json.dumps", "json.dumps"), _callees(db))
            self.assertNotIn(("json.tool.dumps", "json.dumps"), _callees(db))

    def test_query_commands_fail_when_database_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "missing.sqlite"
            with self.assertRaises(SystemExit):
                ezra_graph.callers(argparse.Namespace(db=str(db), symbol="x", limit=10, no_rank=False))

    def test_js_function_declaration_is_not_self_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            (root / "sample.js").write_text("function unused() {}\n")

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)

            self.assertEqual(_callees(db), [])

    def test_js_one_liner_call_is_recorded_but_declaration_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            (root / "sample.js").write_text("function foo() { bar(); }\n")

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)

            self.assertIn(("bar", "bar"), _callees(db))

    def test_blast_radius_finds_from_package_import_reverse_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir(parents=True)
            (root / "pkg").mkdir(parents=True)
            (root / "pkg" / "__init__.py").write_text("")
            (root / "pkg" / "target.py").write_text("def helper(): pass\n")
            (root / "client.py").write_text("from pkg import target\n")

            db = Path(td) / "graph.sqlite"
            _refresh(root, db)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ezra_graph.blast_radius(
                    argparse.Namespace(db=str(db), file=str(root / "pkg" / "target.py"), limit=20)
                )

            out = buf.getvalue()
            self.assertIn("client.py", out)
            self.assertIn("imports pkg.target", out)


if __name__ == "__main__":
    unittest.main()
