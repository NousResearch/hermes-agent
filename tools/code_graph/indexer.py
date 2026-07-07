from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

from tools.code_graph.storage import CodeGraphStore, cache_path_for_root


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    "coverage",
}
LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
}
DEFAULT_MAX_FILE_SIZE_BYTES = 512_000
CHUNK_LINES = 80
MAX_REFERENCE_EDGES_PER_FILE = 500


def _iter_source_files(root: Path) -> Iterable[Path]:
    root = Path(root).resolve()
    for current_dir, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRS]
        for filename in filenames:
            path = Path(current_dir) / filename
            if path.suffix.lower() in LANGUAGE_BY_SUFFIX:
                yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _language_for_path(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX[path.suffix.lower()]


def _module_name(rel_path: str) -> str:
    if rel_path.endswith(".pyi"):
        module = rel_path[:-4]
    elif rel_path.endswith(".py"):
        module = rel_path[:-3]
    else:
        module = rel_path
    module = module.replace("/", ".")
    if module == "__init__":
        return ""
    if module.endswith(".__init__"):
        return module[: -len(".__init__")]
    return module


def _signature(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args: list[str] = []
        args.extend(arg.arg for arg in node.args.posonlyargs)
        args.extend(arg.arg for arg in node.args.args)
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        args.extend(arg.arg for arg in node.args.kwonlyargs)
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{prefix}{node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    return None


def _extract_python_symbols(tree: ast.AST, rel_path: str) -> list[dict]:
    module = _module_name(rel_path)
    symbols: list[dict] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def _qualname(self, name: str) -> str:
            local = ".".join([*self.scope, name]) if self.scope else name
            return f"{module}.{local}" if module else local

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            symbols.append(
                {
                    "name": node.name,
                    "qualname": self._qualname(node.name),
                    "kind": "class",
                    "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "signature": _signature(node),
                    "docstring": ast.get_docstring(node),
                }
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            symbols.append(
                {
                    "name": node.name,
                    "qualname": self._qualname(node.name),
                    "kind": "function",
                    "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "signature": _signature(node),
                    "docstring": ast.get_docstring(node),
                }
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return symbols


def _extract_python_imports(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            module = node.module or ""
            modules.append(f"{prefix}{module}" or ".")
    return sorted(dict.fromkeys(modules))


def _reference_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _extract_python_references(tree: ast.AST) -> list[dict]:
    refs: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _reference_name(node.func)
            if name:
                refs.append({"name": name, "line": node.lineno, "kind": "call"})
    deduped: dict[tuple[str, int, str], dict] = {}
    for ref in refs:
        deduped[(ref["name"], ref["line"], ref["kind"])] = ref
    return list(deduped.values())[:MAX_REFERENCE_EDGES_PER_FILE]


def _line_chunks(source: str) -> Iterable[tuple[int, int, str]]:
    lines = source.splitlines()
    if not lines:
        return
    for start in range(0, len(lines), CHUNK_LINES):
        chunk = lines[start : start + CHUNK_LINES]
        yield start + 1, start + len(chunk), "\n".join(chunk)


def _delete_file_owned_rows(conn, repo_id: int, file_id: int) -> None:
    conn.execute("DELETE FROM symbols WHERE repo_id = ? AND file_id = ?", (repo_id, file_id))
    conn.execute("DELETE FROM chunks WHERE repo_id = ? AND file_id = ?", (repo_id, file_id))
    conn.execute(
        """
        DELETE FROM edges
        WHERE repo_id = ?
          AND (
            (source_kind = 'file' AND source_id = ?)
            OR (target_kind = 'file' AND target_id = ?)
          )
        """,
        (repo_id, file_id, file_id),
    )


def _delete_files_by_path(conn, repo_id: int, paths: set[str]) -> int:
    deleted = 0
    for rel_path in sorted(paths):
        row = conn.execute(
            "SELECT id FROM files WHERE repo_id = ? AND path = ?",
            (repo_id, rel_path),
        ).fetchone()
        if not row:
            continue
        file_id = int(row["id"])
        _delete_file_owned_rows(conn, repo_id, file_id)
        conn.execute("DELETE FROM files WHERE repo_id = ? AND id = ?", (repo_id, file_id))
        deleted += 1
    return deleted


def index_repo(
    root: Path,
    *,
    store: CodeGraphStore | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    force: bool = False,
) -> dict:
    root = Path(root).resolve()
    if not root.exists():
        raise ValueError(f"root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"root is not a directory: {root}")

    store = store or CodeGraphStore(cache_path_for_root(root))
    store.initialize()
    repo_id = store.ensure_repo(root)
    if force:
        store.clear_repo_index(repo_id)

    started_at = time.time()
    files_seen = 0
    files_indexed = 0
    skipped_files = 0
    deleted_files = 0
    parse_errors: list[dict] = []

    with store.connect() as conn:
        run = conn.execute(
            "INSERT INTO index_runs(repo_id, started_at, status) VALUES (?, ?, ?)",
            (repo_id, started_at, "running"),
        )
        run_id = int(run.lastrowid)
        try:
            existing_rows = conn.execute(
                "SELECT id, path, sha256 FROM files WHERE repo_id = ?",
                (repo_id,),
            ).fetchall()
            existing_by_path = {
                row["path"]: {"id": int(row["id"]), "sha256": row["sha256"]}
                for row in existing_rows
            }
            current_paths: set[str] = set()

            for path in _iter_source_files(root):
                rel_path = path.relative_to(root).as_posix()
                stat = path.stat()
                if stat.st_size > max_file_size_bytes:
                    skipped_files += 1
                    continue

                current_paths.add(rel_path)
                files_seen += 1
                digest = _sha256(path)
                existing = existing_by_path.get(rel_path)
                if existing and existing["sha256"] == digest and not force:
                    continue

                source_text = path.read_text(encoding="utf-8", errors="replace")
                language = _language_for_path(path)
                indexed_at = time.time()
                conn.execute(
                    """
                    INSERT INTO files(repo_id, path, language, size, mtime_ns, sha256, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(repo_id, path) DO UPDATE SET
                        language=excluded.language,
                        size=excluded.size,
                        mtime_ns=excluded.mtime_ns,
                        sha256=excluded.sha256,
                        indexed_at=excluded.indexed_at
                    """,
                    (
                        repo_id,
                        rel_path,
                        language,
                        stat.st_size,
                        stat.st_mtime_ns,
                        digest,
                        indexed_at,
                    ),
                )
                file_id = int(
                    conn.execute(
                        "SELECT id FROM files WHERE repo_id = ? AND path = ?",
                        (repo_id, rel_path),
                    ).fetchone()["id"]
                )
                _delete_file_owned_rows(conn, repo_id, file_id)

                for start_line, end_line, text in _line_chunks(source_text):
                    conn.execute(
                        """
                        INSERT INTO chunks(repo_id, file_id, start_line, end_line, text)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (repo_id, file_id, start_line, end_line, text),
                    )

                symbols: list[dict] = []
                imports: list[str] = []
                references: list[dict] = []
                if language == "python":
                    try:
                        tree = ast.parse(source_text, filename=rel_path)
                        symbols = _extract_python_symbols(tree, rel_path)
                        imports = _extract_python_imports(tree)
                        references = _extract_python_references(tree)
                    except SyntaxError as exc:
                        parse_errors.append(
                            {"path": rel_path, "line": exc.lineno, "error": exc.msg}
                        )

                for symbol in symbols:
                    conn.execute(
                        """
                        INSERT INTO symbols(
                            repo_id, file_id, name, qualname, kind, start_line,
                            end_line, signature, docstring
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            repo_id,
                            file_id,
                            symbol["name"],
                            symbol["qualname"],
                            symbol["kind"],
                            symbol["start_line"],
                            symbol["end_line"],
                            symbol["signature"],
                            symbol["docstring"],
                        ),
                    )

                for module in imports:
                    conn.execute(
                        """
                        INSERT INTO edges(
                            repo_id, source_kind, source_id, target_kind,
                            target_id, edge_type, metadata_json
                        )
                        VALUES (?, 'file', ?, 'module', 0, 'imports', ?)
                        """,
                        (
                            repo_id,
                            file_id,
                            json.dumps({"module": module}, sort_keys=True),
                        ),
                    )

                for ref in references:
                    edge_type = "calls" if ref["kind"] == "call" else "references"
                    conn.execute(
                        """
                        INSERT INTO edges(
                            repo_id, source_kind, source_id, target_kind,
                            target_id, edge_type, metadata_json
                        )
                        VALUES (?, 'file', ?, 'symbol_name', 0, ?, ?)
                        """,
                        (
                            repo_id,
                            file_id,
                            edge_type,
                            json.dumps(ref, sort_keys=True),
                        ),
                    )
                files_indexed += 1

            deleted_files = _delete_files_by_path(
                conn,
                repo_id,
                set(existing_by_path) - current_paths,
            )
            conn.execute(
                """
                UPDATE index_runs
                SET finished_at = ?, files_seen = ?, files_indexed = ?, status = ?, message = ?
                WHERE id = ?
                """,
                (
                    time.time(),
                    files_seen,
                    files_indexed,
                    "ok",
                    None,
                    run_id,
                ),
            )
        except Exception as exc:
            conn.execute(
                """
                UPDATE index_runs
                SET finished_at = ?, files_seen = ?, files_indexed = ?, status = ?, message = ?
                WHERE id = ?
                """,
                (time.time(), files_seen, files_indexed, "error", str(exc), run_id),
            )
            raise

    return {
        "success": True,
        "repo_root": str(root),
        "cache_path": str(store.db_path),
        "files_seen": files_seen,
        "files_indexed": files_indexed,
        "skipped_files": skipped_files,
        "deleted_files": deleted_files,
        "parse_errors": parse_errors[:20],
        "force": bool(force),
    }
