#!/usr/bin/env python3
"""ezra-graph: lightweight static code graph for Ezra/Hermes repos.

Builds a SQLite artifact with module imports, function definitions, dotted call
edges, and reverse dependencies across Hermes + OpenClaw + Mission Control.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 2
DEFAULT_DB = Path(os.environ.get("EZRA_GRAPH_DB", "/Users/Prime/.ezra/graph/ezra-graph.sqlite"))
DEFAULT_ROOTS = [
    Path("/Users/Prime/.hermes/hermes-agent"),
    Path("/Users/Prime/.openclaw/openclaw"),
    Path("/Users/Prime/.openclaw/mission-control"),
]
EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", ".cache", "coverage"}

SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY, repo TEXT, path TEXT UNIQUE, relpath TEXT, lang TEXT,
  sha256 TEXT, mtime REAL, size INTEGER
);
CREATE TABLE IF NOT EXISTS symbols(
  id INTEGER PRIMARY KEY, file_id INTEGER, name TEXT, qualname TEXT, kind TEXT, line INTEGER,
  UNIQUE(file_id, qualname, kind, line)
);
CREATE TABLE IF NOT EXISTS imports(
  id INTEGER PRIMARY KEY, file_id INTEGER, module TEXT, imported TEXT, alias TEXT, line INTEGER
);
CREATE TABLE IF NOT EXISTS calls(
  id INTEGER PRIMARY KEY, caller_symbol_id INTEGER, file_id INTEGER, callee TEXT, raw_callee TEXT, line INTEGER
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_qual ON symbols(qualname);
CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(module);
CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee);
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
"""

JS_IMPORT_RE = re.compile(r"^\s*(?:import\s+(?:(?P<what>.*?)\s+from\s+)?[\"'](?P<mod>[^\"']+)[\"']|(?:const|let|var)\s+(?P<reqwhat>[\w{}*,\s]+)\s*=\s*require\([\"'](?P<reqmod>[^\"']+)[\"']\))")
JS_FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(|^\s*(?:export\s+)?(?:const|let|var)\s+(?P<const>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?[^=]*?\)?\s*=>|^\s*(?P<method>[A-Za-z_$][\w$]*)\s*\([^)]*\)\s*{")
CALL_RE = re.compile(r"(?<![\w$.])(?P<name>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(")
PY_KW = {"if", "for", "while", "return", "with", "print", "len", "str", "int", "float", "list", "dict", "set", "tuple", "open", "range", "isinstance", "hasattr", "getattr", "setattr", "super"}
JS_KW = {"if", "for", "while", "switch", "catch", "function", "return", "console", "require"}


def roots_from_args(values: list[str] | None) -> list[Path]:
    roots = [Path(v).expanduser() for v in values] if values else DEFAULT_ROOTS
    return [p.resolve() for p in roots if p.exists() and p.is_dir()]


def repo_name(root: Path) -> str:
    return root.name or str(root)


def iter_files(roots: list[Path]) -> Iterable[tuple[Path, Path]]:
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            dp = Path(dirpath)
            for fn in filenames:
                p = dp / fn
                if p.suffix in EXTS and p.is_file():
                    yield root, p


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SQL)
    ensure_schema(con)
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    call_cols = {row[1] for row in con.execute("PRAGMA table_info(calls)")}
    if "raw_callee" not in call_cols:
        con.execute("ALTER TABLE calls ADD COLUMN raw_callee TEXT")
    import_cols = {row[1] for row in con.execute("PRAGMA table_info(imports)")}
    if "alias" not in import_cols:
        con.execute("ALTER TABLE imports ADD COLUMN alias TEXT")
    con.execute("CREATE INDEX IF NOT EXISTS idx_calls_raw_callee ON calls(raw_callee)")
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))


def reset(con: sqlite3.Connection) -> None:
    for table in ["calls", "imports", "symbols", "files"]:
        con.execute(f"DELETE FROM {table}")


def insert_file(con: sqlite3.Connection, root: Path, p: Path) -> int:
    st = p.stat()
    rel = str(p.relative_to(root))
    lang = p.suffix.lstrip(".")
    cur = con.execute(
        "INSERT INTO files(repo,path,relpath,lang,sha256,mtime,size) VALUES(?,?,?,?,?,?,?)",
        (repo_name(root), str(p), rel, lang, sha(p), st.st_mtime, st.st_size),
    )
    return int(cur.lastrowid)


def insert_symbol(con: sqlite3.Connection, file_id: int, name: str, qual: str, kind: str, line: int) -> int:
    cur = con.execute(
        "INSERT OR IGNORE INTO symbols(file_id,name,qualname,kind,line) VALUES(?,?,?,?,?)",
        (file_id, name, qual, kind, line),
    )
    if cur.lastrowid:
        return int(cur.lastrowid)
    return int(
        con.execute(
            "SELECT id FROM symbols WHERE file_id=? AND qualname=? AND kind=? AND line=?",
            (file_id, qual, kind, line),
        ).fetchone()[0]
    )


def py_dotted_name(node: ast.AST) -> str | None:
    """Return a full dotted name for Name/Attribute chains.

    Examples: ``json.dumps`` and ``self.client.messages.create``. Calls on
    arbitrary expressions are intentionally skipped because static resolution
    would become speculative.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return None


def collect_top_level_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                bound = item.asname or item.name.split(".", 1)[0]
                aliases[bound] = item.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative imports depend on package context; keep the imported
                # binding visible but do not invent an absolute module path.
                base = "." * node.level + (node.module or "")
            else:
                base = node.module or ""
            for item in node.names:
                if item.name == "*":
                    continue
                bound = item.asname or item.name
                aliases[bound] = f"{base}.{item.name}" if base else item.name
    return aliases


def resolve_alias(dotted: str, aliases: dict[str, str]) -> str:
    first, sep, rest = dotted.partition(".")
    target = aliases.get(first)
    if not target:
        return dotted
    return f"{target}.{rest}" if sep else target


class PyVisitor(ast.NodeVisitor):
    def __init__(self, con: sqlite3.Connection, file_id: int, aliases: dict[str, str]):
        self.con = con
        self.file_id = file_id
        self.aliases = aliases
        self.stack: list[str] = []
        self.current: int | None = None

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            self.con.execute(
                "INSERT INTO imports(file_id,module,imported,alias,line) VALUES(?,?,?,?,?)",
                (self.file_id, item.name, "", item.asname or "", node.lineno),
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = "." * node.level + (node.module or "")
        for item in node.names:
            self.con.execute(
                "INSERT INTO imports(file_id,module,imported,alias,line) VALUES(?,?,?,?,?)",
                (self.file_id, mod, item.name, item.asname or "", node.lineno),
            )

    def _func(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        qual = ".".join(self.stack + [node.name])
        sid = insert_symbol(self.con, self.file_id, node.name, qual, kind, node.lineno)
        self.stack.append(node.name)
        old = self.current
        self.current = sid
        self.generic_visit(node)
        self.current = old
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func(node, "function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual = ".".join(self.stack + [node.name])
        insert_symbol(self.con, self.file_id, node.name, qual, "class", node.lineno)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        raw = py_dotted_name(node.func)
        if raw and raw.split(".", 1)[0] not in PY_KW:
            callee = resolve_alias(raw, self.aliases)
            self.con.execute(
                "INSERT INTO calls(caller_symbol_id,file_id,callee,raw_callee,line) VALUES(?,?,?,?,?)",
                (self.current, self.file_id, callee, raw, getattr(node, "lineno", 0)),
            )
        self.generic_visit(node)


def scan_python(con: sqlite3.Connection, file_id: int, text: str, alias_debug: bool = False) -> dict[str, str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    aliases = collect_top_level_aliases(tree)
    PyVisitor(con, file_id, aliases).visit(tree)
    if alias_debug:
        return aliases
    return {}


def scan_js(con: sqlite3.Connection, file_id: int, text: str) -> None:
    current = None
    for i, line in enumerate(text.splitlines(), 1):
        m = JS_IMPORT_RE.search(line)
        if m:
            mod = m.group("mod") or m.group("reqmod")
            what = m.group("what") or m.group("reqwhat") or ""
            con.execute(
                "INSERT INTO imports(file_id,module,imported,alias,line) VALUES(?,?,?,?,?)",
                (file_id, mod, what.strip(), "", i),
            )
        fm = JS_FUNC_RE.search(line)
        if fm:
            name = fm.group("name") or fm.group("const") or fm.group("method")
            if name and name not in JS_KW:
                current = insert_symbol(con, file_id, name, name, "function", i)
        for cm in CALL_RE.finditer(line):
            name = cm.group("name")
            if name.split(".", 1)[0] not in JS_KW:
                con.execute(
                    "INSERT INTO calls(caller_symbol_id,file_id,callee,raw_callee,line) VALUES(?,?,?,?,?)",
                    (current, file_id, name, name, i),
                )


def refresh(args: argparse.Namespace) -> None:
    roots = roots_from_args(args.root)
    if not roots:
        raise SystemExit("No scan roots found; pass --root PATH")
    t0 = time.time()
    con = connect(Path(args.db))
    reset(con)
    files = 0
    alias_samples: dict[str, dict[str, str]] = {}
    for root, p in iter_files(roots):
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        fid = insert_file(con, root, p)
        files += 1
        if p.suffix == ".py":
            aliases = scan_python(con, fid, text, alias_debug=bool(getattr(args, "alias_debug", False)))
            if aliases and len(alias_samples) < 20:
                alias_samples[str(p)] = dict(sorted(aliases.items())[:20])
        else:
            scan_js(con, fid, text)
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('roots',?)", (json.dumps([str(r) for r in roots]),))
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('refreshed_at',datetime('now'))")
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
    con.commit()
    counts = {table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ["files", "symbols", "imports", "calls"]}
    payload = {"db": str(Path(args.db)), "seconds": round(time.time() - t0, 3), "roots": [str(r) for r in roots], **counts}
    if getattr(args, "alias_debug", False):
        payload["alias_samples"] = alias_samples
    print(json.dumps(payload, indent=2))


@dataclass(frozen=True)
class CallerRow:
    caller: str
    path: str
    relpath: str
    line: int
    callee: str
    raw_callee: str
    caller_count: int = 0
    caller_file_diversity: int = 0
    locality: int = 0


def _matches_where(symbol: str) -> tuple[str, tuple[str, ...]]:
    if "." in symbol:
        return "(c.callee=? OR c.raw_callee=?)", (symbol, symbol)
    return "(c.callee=? OR c.raw_callee=? OR c.callee LIKE ? OR c.raw_callee LIKE ?)", (symbol, symbol, f"%.{symbol}", f"%.{symbol}")


def _locality(symbol: str, relpath: str) -> int:
    if "." not in symbol:
        return 0
    first = symbol.split(".", 1)[0]
    parts = Path(relpath).parts
    return 1 if first in parts else 0


def callers(args: argparse.Namespace) -> None:
    con = connect(Path(args.db))
    where, params = _matches_where(args.symbol)
    rows_raw = con.execute(
        f"""
      SELECT COALESCE(s.qualname,'<module>') caller, f.path, f.relpath, c.line, c.callee, COALESCE(c.raw_callee,c.callee)
      FROM calls c JOIN files f ON f.id=c.file_id LEFT JOIN symbols s ON s.id=c.caller_symbol_id
      WHERE {where}
    """,
        params,
    ).fetchall()
    counts: dict[str, int] = {}
    files_by_caller: dict[str, set[str]] = {}
    for caller, path, _relpath, _line, _callee, _raw in rows_raw:
        counts[caller] = counts.get(caller, 0) + 1
        files_by_caller.setdefault(caller, set()).add(path)
    rows = [
        CallerRow(
            caller=caller,
            path=path,
            relpath=relpath,
            line=line,
            callee=callee,
            raw_callee=raw,
            caller_count=counts[caller],
            caller_file_diversity=len(files_by_caller[caller]),
            locality=_locality(args.symbol, relpath),
        )
        for caller, path, relpath, line, callee, raw in rows_raw
    ]
    if not getattr(args, "no_rank", False):
        rows.sort(key=lambda r: (-r.caller_count, -r.caller_file_diversity, -r.locality, r.path, r.line, r.callee))
    else:
        rows.sort(key=lambda r: (r.path, r.line, r.callee))
    rows = rows[: args.limit]
    for row in rows:
        raw_note = f" raw={row.raw_callee}" if row.raw_callee != row.callee else ""
        rank_note = "" if getattr(args, "no_rank", False) else f" rank=count:{row.caller_count},files:{row.caller_file_diversity},local:{row.locality}"
        print(f"{row.callee}\tcalled_by={row.caller}\t{row.path}:{row.line}{raw_note}{rank_note}")
    if not rows:
        print(f"No callers found for {args.symbol}")


def blast_radius(args: argparse.Namespace) -> None:
    con = connect(Path(args.db))
    target = str(Path(args.file).expanduser())
    rows = con.execute(
        """
      SELECT DISTINCT f2.path, i.module, i.line
      FROM files f1 JOIN files f2 JOIN imports i ON i.file_id=f2.id
      WHERE f1.path=? AND (i.module LIKE '%' || replace(replace(f1.relpath,'/','.'),'.py','') || '%' OR i.module LIKE '%' || replace(f1.relpath,'.py','') || '%')
      ORDER BY f2.path LIMIT ?
    """,
        (target, args.limit),
    ).fetchall()
    syms = con.execute("SELECT qualname,name FROM symbols JOIN files ON files.id=symbols.file_id WHERE files.path=?", (target,)).fetchall()
    called = []
    for qual, name in syms[:50]:
        called += con.execute(
            "SELECT DISTINCT f.path, c.line, c.callee FROM calls c JOIN files f ON f.id=c.file_id WHERE c.callee IN (?,?) LIMIT ?",
            (name, qual, args.limit),
        ).fetchall()
    print(f"Blast radius for {target}")
    print("Import reverse-deps:")
    for path, mod, line in rows:
        print(f"  {path}:{line} imports {mod}")
    print("Call references:")
    seen = set()
    for path, line, callee in called:
        key = (path, line, callee)
        if key not in seen:
            seen.add(key)
            print(f"  {path}:{line} calls {callee}")
    if not rows and not seen:
        print("  No reverse dependencies found")


def orphans(args: argparse.Namespace) -> None:
    con = connect(Path(args.db))
    rows = con.execute(
        """
      SELECT s.qualname,s.name,f.path,s.line FROM symbols s JOIN files f ON f.id=s.file_id
      WHERE s.kind='function' AND NOT EXISTS (
        SELECT 1 FROM calls c WHERE c.callee=s.name OR c.callee=s.qualname OR c.callee LIKE '%.' || s.name
      ) ORDER BY f.path,s.line LIMIT ?
    """,
        (args.limit,),
    ).fetchall()
    for qual, _name, path, line in rows:
        print(f"{qual}\t{path}:{line}")
    if not rows:
        print("No orphan functions found")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ezra static code graph")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh")
    r.add_argument("--root", action="append")
    r.add_argument("--alias-debug", action="store_true", help="include per-file import alias samples in refresh JSON")
    r.set_defaults(fn=refresh)
    c = sub.add_parser("callers")
    c.add_argument("symbol")
    c.add_argument("--limit", type=int, default=50)
    c.add_argument("--no-rank", action="store_true", help="sort caller rows by path/line instead of frequency/diversity/locality")
    c.set_defaults(fn=callers)
    b = sub.add_parser("blast-radius")
    b.add_argument("file")
    b.add_argument("--limit", type=int, default=50)
    b.set_defaults(fn=blast_radius)
    o = sub.add_parser("orphans")
    o.add_argument("--limit", type=int, default=50)
    o.set_defaults(fn=orphans)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
