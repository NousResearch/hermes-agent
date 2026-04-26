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


if __name__ == "__main__":
    unittest.main()
