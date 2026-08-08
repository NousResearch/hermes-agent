#!/usr/bin/env python3
"""Code Index Tool - Build and maintain a persistent code index.

Provides codebase indexing: symbol extraction, dependency graph,
and persistent storage for fast code search and navigation.
"""

import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

INDEX_VERSION = 2
MAX_FILE_SIZE = 5 * 1024 * 1024


def _index_db_path(abs_root: str) -> str:
    """Stable DB name from a hash of the canonical project root."""
    root_hash = hashlib.sha256(abs_root.encode()).hexdigest()[:16]
    index_dir = get_hermes_home() / "code-index"
    return str(index_dir / f"{root_hash}.db")


def _detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".py": "python", ".pyw": "python", ".pyx": "python",
        ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java", ".kt": "kotlin",
        ".rb": "ruby",
        ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
        ".cs": "csharp",
        ".swift": "swift",
    }
    return mapping.get(ext, "unknown")


def _extract_symbols(content: str, language: str) -> List[Dict[str, Any]]:
    symbols = []

    if language == "python":
        patterns = [
            (r"^class\s+(\w+)", "class"),
            (r"^def\s+(\w+)", "function"),
            (r"^async\s+def\s+(\w+)", "function"),
            (r"^\s+class\s+(\w+)", "class"),
            (r"^\s+def\s+(\w+)", "method"),
            (r"^\s+async\s+def\s+(\w+)", "method"),
        ]
    elif language in ("javascript", "typescript"):
        patterns = [
            (r"(?:export\s+)?(?:class|interface)\s+(\w+)", "class"),
            (r"(?:export\s+)?function\s+(\w+)", "function"),
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[=:]\s*(?:async\s+)?(?:\(|function)", "function"),
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)", "variable"),
            (r"(?:export\s+)?type\s+(\w+)", "type"),
        ]
    elif language == "go":
        patterns = [
            (r"^func\s+(\w+)", "function"),
            (r"^type\s+(\w+)\s+struct", "struct"),
            (r"^type\s+(\w+)\s+interface", "interface"),
            (r"^type\s+(\w+)\s*=", "type"),
            (r"^const\s+(\w+)", "constant"),
            (r"^var\s+(\w+)", "variable"),
        ]
    elif language == "rust":
        patterns = [
            (r"^fn\s+(\w+)", "function"),
            (r"^struct\s+(\w+)", "struct"),
            (r"^enum\s+(\w+)", "enum"),
            (r"^trait\s+(\w+)", "trait"),
            (r"^impl\s+(\w+(?:\s+for\s+\w+)?)", "impl"),
            (r"^(?:pub\s+)?(?:const|let|static)\s+(?:mut\s+)?(\w+)", "variable"),
        ]
    else:
        patterns = [
            (r"(?:class|struct|enum|interface)\s+(\w+)", "type"),
            (r"(?:def|function|fn|func)\s+(\w+)", "function"),
        ]

    for pattern, sym_type in patterns:
        for match in re.finditer(pattern, content, re.MULTILINE):
            if match.lastindex:
                name = match.group(1)
            else:
                continue
            line_num = content[:match.start()].count("\n") + 1
            symbols.append({
                "name": name,
                "type": sym_type,
                "line": line_num,
            })

    return symbols


def _extract_dependencies(content: str, language: str) -> List[str]:
    deps = []

    if language == "python":
        matches = re.findall(
            r"^(?:from\s+([^\s]+)\s+import|import\s+([^\s]+))",
            content, re.MULTILINE,
        )
        for m in matches:
            deps.append(m[0] or m[1])
    elif language in ("javascript", "typescript"):
        matches = re.findall(
            r"(?:import\s+(?:\{[^}]*\}\s+from\s+)?['\"]([^'\"]+)['\"]"
            r"|require\(['\"]([^'\"]+)['\"]\))",
            content,
        )
        for m in matches:
            deps.append(m[0] or m[1])
    elif language == "go":
        matches = re.findall(r'^\s+"([^"]+)"', content, re.MULTILINE)
        deps.extend(matches)

    return list(set(deps))


def code_index(
    project_root: str,
    operation: str = "build",
    file_pattern: Optional[str] = None,
    path: Optional[str] = None,
    include_unsafe: bool = False,
    task_id: Optional[str] = None,
) -> str:
    """Build and maintain a persistent code index."""
    if not os.path.isdir(project_root):
        return json.dumps({
            "success": False,
            "error": f"Project root not found: {project_root}",
        })

    abs_root = os.path.abspath(project_root)
    db_path = _index_db_path(abs_root)

    if operation == "clear":
        if os.path.exists(db_path):
            os.unlink(db_path)
            return json.dumps({"success": True, "operation": "clear", "message": "Index cleared"})
        return json.dumps({"success": True, "operation": "clear", "message": "No index to clear"})

    if operation == "status":
        if not os.path.exists(db_path):
            return json.dumps({"success": True, "operation": "status", "indexed": False, "message": "No index found"})
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM symbols")
            symbol_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT file_path) FROM symbols")
            indexed_files = cursor.fetchone()[0]
            return json.dumps({
                "success": True,
                "operation": "status",
                "indexed": True,
                "files": file_count,
                "files_with_symbols": indexed_files,
                "symbols": symbol_count,
                "index_path": db_path,
            })
        except sqlite3.Error as e:
            return json.dumps({"success": False, "error": f"Index read error: {e}"})
        finally:
            if conn:
                conn.close()

    if operation == "build":
        conn = None
        try:
            index_dir = os.path.dirname(db_path)
            os.makedirs(index_dir, exist_ok=True)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    language TEXT,
                    last_modified REAL,
                    size INTEGER,
                    line_count INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    name TEXT,
                    type TEXT,
                    line INTEGER,
                    FOREIGN KEY (file_path) REFERENCES files(path)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    dependency TEXT,
                    FOREIGN KEY (file_path) REFERENCES files(path)
                )
            """)
            cursor.execute("PRAGMA synchronous = OFF")
            cursor.execute("PRAGMA journal_mode = MEMORY")

            cursor.execute(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                ("version", str(INDEX_VERSION)),
            )
            cursor.execute(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                ("project_root", abs_root),
            )

            exclude_dirs = {
                ".git", "__pycache__", "node_modules", ".venv", "venv", ".env",
                ".tox", ".eggs", "build", "dist", ".mypy_cache", ".pytest_cache",
                ".ruff_cache", ".hermes",
            }

            stats = {"files_scanned": 0, "new_files": 0, "symbols_found": 0}
            walk_root = path if path else abs_root
            seen_files: set = set()

            for root, dirs, files in os.walk(walk_root):
                if not include_unsafe:
                    dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, abs_root)

                    if file_pattern:
                        pat = file_pattern.replace("*", ".*").replace("?", ".")
                        if not re.search(pat, file_path):
                            continue

                    try:
                        stat_info = os.stat(file_path)
                        if stat_info.st_size > MAX_FILE_SIZE:
                            continue
                    except OSError:
                        continue

                    seen_files.add(rel_path)

                    cursor.execute(
                        "SELECT last_modified FROM files WHERE path = ?",
                        (rel_path,),
                    )
                    existing = cursor.fetchone()
                    if existing and existing[0] >= stat_info.st_mtime:
                        continue

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue

                    language = _detect_language(file_path)
                    line_count = content.count("\n") + 1

                    cursor.execute(
                        "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
                        (rel_path, language, stat_info.st_mtime, stat_info.st_size, line_count),
                    )
                    stats["new_files"] += 1

                    cursor.execute("DELETE FROM symbols WHERE file_path = ?", (rel_path,))
                    symbols = _extract_symbols(content, language)
                    for sym in symbols:
                        cursor.execute(
                            "INSERT INTO symbols (file_path, name, type, line) VALUES (?, ?, ?, ?)",
                            (rel_path, sym["name"], sym["type"], sym["line"]),
                        )
                    stats["symbols_found"] += len(symbols)

                    cursor.execute("DELETE FROM dependencies WHERE file_path = ?", (rel_path,))
                    deps = _extract_dependencies(content, language)
                    for dep in deps:
                        cursor.execute(
                            "INSERT INTO dependencies (file_path, dependency) VALUES (?, ?)",
                            (rel_path, dep),
                        )

                    stats["files_scanned"] += 1

            # Prune stale rows for files that no longer exist on disk.
            cursor.execute("SELECT path FROM files")
            for (existing_path,) in cursor.fetchall():
                if existing_path not in seen_files:
                    cursor.execute("DELETE FROM symbols WHERE file_path = ?", (existing_path,))
                    cursor.execute("DELETE FROM dependencies WHERE file_path = ?", (existing_path,))
                    cursor.execute("DELETE FROM files WHERE path = ?", (existing_path,))

            conn.commit()
            return json.dumps({
                "success": True,
                "operation": "build",
                "project": abs_root,
                "stats": stats,
                "message": (
                    f"Scanned {stats['files_scanned']} files, "
                    f"indexed {stats['new_files']} new, "
                    f"found {stats['symbols_found']} symbols"
                ),
            })

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Index build failed: {e}",
            })
        finally:
            if conn:
                conn.close()

    return json.dumps({
        "success": False,
        "error": f"Unknown operation: {operation}",
    })


def check_code_index_requirements() -> bool:
    """Code index uses Python standard library."""
    return True


CODE_INDEX_SCHEMA = {
    "name": "code_index",
    "description": (
        "Build and maintain a persistent code index for fast code search and navigation.\n\n"
        "Operations:\n"
        "- build: Scan project, extract symbols (classes, functions), track dependencies\n"
        "- status: Show index statistics (files, symbols, indexed paths)\n"
        "- clear: Remove the index\n\n"
        "Extracts: classes, functions, methods, interfaces, structs, enums, traits\n"
        "Supported: Python, TypeScript, JavaScript, Go, Rust"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project_root": {
                "type": "string",
                "description": "Root directory of the project to index",
            },
            "operation": {
                "type": "string",
                "description": "Operation: build, status, clear",
                "enum": ["build", "status", "clear"],
                "default": "build",
            },
            "file_pattern": {
                "type": "string",
                "description": "File pattern to filter (e.g., *.py, src/**/*.ts)",
            },
            "path": {
                "type": "string",
                "description": "Specific file or directory to index (for targeted updates)",
            },
            "include_unsafe": {
                "type": "boolean",
                "description": "Include hidden/node_modules directories",
                "default": False,
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
        "required": ["project_root"],
    },
}


from tools.registry import registry

registry.register(
    name="code_index",
    toolset="code_search",
    schema=CODE_INDEX_SCHEMA,
    handler=lambda args, **kw: code_index(
        project_root=args.get("project_root", ""),
        operation=args.get("operation", "build"),
        file_pattern=args.get("file_pattern"),
        path=args.get("path"),
        include_unsafe=args.get("include_unsafe", False),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_code_index_requirements,
    emoji="📑",
)
