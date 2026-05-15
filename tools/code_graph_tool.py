"""Hermes-native clean-room code graph tools.

This module intentionally uses only Python standard-library parsing for the MVP.
It does not import, shell out to, or reuse GitNexus.  Graph output is read-only
navigation / impact evidence; it never authorizes edits by itself.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from agent.redact import redact_sensitive_text
from tools.registry import registry

_CODE_GRAPH_TOOLSET = "code_graph"
CODE_GRAPH_INDEX_TOOL_NAME = "code_graph_index"
CODE_GRAPH_CONTEXT_TOOL_NAME = "code_graph_context"
CODE_GRAPH_IMPACT_TOOL_NAME = "code_graph_impact"
CODE_GRAPH_TOOL_NAMES = (
    CODE_GRAPH_INDEX_TOOL_NAME,
    CODE_GRAPH_CONTEXT_TOOL_NAME,
    CODE_GRAPH_IMPACT_TOOL_NAME,
)
_MAX_FILE_BYTES = 512 * 1024
_MAX_SKIPPED_SAMPLES = 25
_MAX_SKIPPED_REASON_KEYS = 20
_MAX_MEMBER_SYMBOLS = 100
_MAX_RELATED_EDGES = 200
_MAX_IMPACTED = 200
_MAX_IMPACT_EDGES = 200
_MAX_INDEX_NODES = 10_000
_MAX_INDEX_EDGES = 20_000
_HIGH_CONFIDENCE_CALLS = {"import-resolved", "same-file"}

_PROJECT_MARKERS = {
    ".git",
    ".hermes.md",
    "AGENTS.md",
    "HERMES.md",
    "CLAUDE.md",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
}

_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
}

_EXCLUDED_SUFFIXES = {
    ".env",
    ".p8",
    ".p12",
    ".pem",
    ".key",
    ".mobileprovision",
    ".der",
    ".cer",
    ".crt",
}

_SENSITIVE_NAME_MARKERS = (
    "secret",
    "token",
    "credential",
    "cookie",
    "session",
    "auth.json",
)

_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|secret|token|password|private[_-]?key|credential)\s*[:=]\s*['\"]?[^'\"\s]{6,}"),
    re.compile(r"(?i)\b[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PRIVATE[_-]?KEY|ACCESS[_-]?KEY|CREDENTIAL)[A-Z0-9_]*\s*[:=]\s*['\"]?[^'\"\s,)]+"),
    re.compile(r"(?i)\b(?:AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|PRIVATE_KEY)\s*[:=]\s*['\"]?[^'\"\s,)]+"),
    re.compile(r"(?i)['\"](?:api[_-]?key|secret|token|password|private[_-]?key|credential)['\"]\s*:\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
    # Mirror high-confidence token value shapes used by shell-hook log hygiene.
    # CodeGraph can emit source excerpts from non-secret filenames; redact known
    # standalone token formats even when no nearby variable name says "secret".
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"TEST_FAKE_VALUE_[A-Za-z0-9_-]{8,}"),
)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _redact(text: str) -> str:
    redacted = redact_sensitive_text(text, force=True, code_file=True)
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _redact_data(value: Any) -> Any:
    """Recursively redact all string fields before CodeGraph leaves the tool."""
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact_data(item) for item in value]
    if isinstance(value, dict):
        return {_redact(str(key)): _redact_data(item) for key, item in value.items()}
    return value


def _is_excluded_relative(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    for part in parts:
        if part in {".", ""}:
            continue
        if part in _EXCLUDED_DIRS:
            return True
        if part.startswith("."):
            return True
        if any(marker in part for marker in _SENSITIVE_NAME_MARKERS):
            return True
    name = path.name.lower()
    if name.startswith(".env"):
        return True
    return any(name.endswith(suffix) for suffix in _EXCLUDED_SUFFIXES)


def _is_sensitive_path_component(part: str) -> bool:
    name = part.lower()
    if name.startswith(".env"):
        return True
    if any(marker in name for marker in _SENSITIVE_NAME_MARKERS):
        return True
    return any(name.endswith(suffix) for suffix in _EXCLUDED_SUFFIXES)


def _redact_relative_path(path: Path) -> str:
    parts = [
        "[REDACTED]" if _is_sensitive_path_component(part) else _redact(part)
        for part in path.parts
        if part not in {"", "."}
    ]
    return "/".join(parts) or "."


class _SkipRecorder:
    def __init__(self) -> None:
        self.reasons: Counter[str] = Counter()
        self.samples: list[dict[str, str]] = []

    def add(self, rel_path: Path, reason: str) -> None:
        self.reasons[reason] += 1
        if len(self.samples) >= _MAX_SKIPPED_SAMPLES:
            return
        self.samples.append({"path": _redact_relative_path(rel_path), "reason": reason})

    def reason_counts(self) -> dict[str, int]:
        return dict(sorted(self.reasons.most_common(_MAX_SKIPPED_REASON_KEYS)))

    def total(self) -> int:
        return sum(self.reasons.values())


def _has_project_marker(root: Path) -> bool:
    home = Path.home().resolve()
    current = root
    while True:
        if any((current / marker).exists() for marker in _PROJECT_MARKERS):
            return True
        if current == current.parent or current == home or current.parent == Path("/Volumes").resolve():
            return False
        current = current.parent


def _validate_repo_path(root: Path) -> None:
    home = Path.home().resolve()
    broad_roots = {
        Path("/").resolve(),
        home,
        Path("/Users").resolve(),
        Path("/Volumes").resolve(),
        Path("/tmp").resolve(),
        Path("/private").resolve(),
        Path("/var").resolve(),
        Path("/opt").resolve(),
        Path("/Applications").resolve(),
    }
    if root in broad_roots or root.parent == Path("/Volumes").resolve():
        raise ValueError("repo_path is too broad for code graph indexing")
    if _is_excluded_relative(Path(root.name)):
        raise ValueError("repo_path appears sensitive or excluded")
    if not _has_project_marker(root):
        raise ValueError(
            "repo_path must be an explicit project/corpus root with a known marker "
            f"({', '.join(sorted(_PROJECT_MARKERS))})"
        )


def _clean_repo_path(repo_path: str) -> str:
    if not isinstance(repo_path, str) or not repo_path.strip():
        raise ValueError("repo_path must be a non-empty directory path")
    return repo_path.strip()


def _clean_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    return symbol.strip()


def _iter_python_files(root: Path, max_files: int) -> tuple[list[Path], _SkipRecorder]:
    indexed: list[Path] = []
    skipped = _SkipRecorder()
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        keep_dirs: list[str] = []
        for name in sorted(dirnames):
            dir_path = current / name
            rel_dir = dir_path.relative_to(root)
            if _is_excluded_relative(rel_dir):
                skipped.add(rel_dir, "excluded_directory")
                continue
            if dir_path.is_symlink():
                skipped.add(rel_dir, "symlink")
                continue
            keep_dirs.append(name)
        dirnames[:] = keep_dirs

        for filename in sorted(filenames):
            path = current / filename
            rel_file = path.relative_to(root)
            if _is_excluded_relative(rel_file):
                skipped.add(rel_file, "excluded_file")
                continue
            if path.is_symlink():
                skipped.add(rel_file, "symlink")
                continue
            if path.suffix != ".py":
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    skipped.add(rel_file, "oversized_file")
                    continue
            except OSError:
                skipped.add(rel_file, "stat_error")
                continue
            if len(indexed) >= max_files:
                skipped.add(Path("[TRUNCATED]"), "max_files_reached")
                return indexed, skipped
            indexed.append(path)
    return indexed, skipped


def _call_name(node: ast.AST) -> str | None:
    """Return high-confidence bare call names only.

    Attribute calls such as ``client.save()`` are deliberately not resolved in
    the MVP because receiver type is unknown. Emitting no edge is safer than
    emitting a false high-confidence impact edge.
    """
    if isinstance(node, ast.Name):
        return node.id
    return None


def _source_excerpt(lines: list[str], lineno: int | None) -> str:
    if not lineno or lineno < 1 or lineno > len(lines):
        return ""
    return _redact(lines[lineno - 1].strip()[:240])


def _store_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _store_target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_store_target_names(item))
        return names
    return set()


def _collect_function_shadowed_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect local names that make bare-call resolution unsafe.

    The CodeGraph MVP only emits source-backed bare-call edges. If a function
    parameter, local assignment, local import, or nested definition binds the
    same name as a module-level symbol/import, a later ``name()`` call may target
    that local value instead. We therefore mark it shadowed and leave the call
    unresolved rather than producing a false high-confidence edge.
    """
    shadowed: set[str] = set()
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg:
        args.append(node.args.vararg)
    if node.args.kwarg:
        args.append(node.args.kwarg)
    shadowed.update(arg.arg for arg in args)

    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(child.name)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            shadowed.add(child.id)
        elif isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                shadowed.update(_store_target_names(target))
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            shadowed.update(_store_target_names(child.target))
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            for item in child.items:
                if item.optional_vars:
                    shadowed.update(_store_target_names(item.optional_vars))
        elif isinstance(child, ast.ExceptHandler) and child.name:
            shadowed.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                shadowed.add(alias.asname or alias.name.split(".")[0])
    return shadowed


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str, lines: list[str]):
        self.rel_path = rel_path
        self.lines = lines
        self.class_stack: list[str] = []
        self.symbols: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str, str, int | None]] = []
        self.current_symbol: str | None = None
        self.shadowed_stack: list[set[str]] = []

    def _symbol_id(self, name: str) -> str:
        if self.class_stack:
            return f"{self.rel_path}::{'.'.join(self.class_stack)}.{name}"
        return f"{self.rel_path}::{name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        symbol_id = self._symbol_id(node.name)
        self.symbols.append(
            {
                "id": symbol_id,
                "kind": "class",
                "name": node.name,
                "file": self.rel_path,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "source_excerpt": _source_excerpt(self.lines, node.lineno),
            }
        )
        previous = self.current_symbol
        self.current_symbol = symbol_id
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
        self.current_symbol = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        symbol_id = self._symbol_id(node.name)
        self.symbols.append(
            {
                "id": symbol_id,
                "kind": "method" if self.class_stack else "function",
                "name": node.name,
                "file": self.rel_path,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "source_excerpt": _source_excerpt(self.lines, node.lineno),
            }
        )
        previous = self.current_symbol
        self.current_symbol = symbol_id
        self.shadowed_stack.append(_collect_function_shadowed_names(node))
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            self.visit(stmt)
        self.shadowed_stack.pop()
        self.current_symbol = previous

    def visit_Call(self, node: ast.Call) -> Any:
        if self.current_symbol:
            name = _call_name(node.func)
            if name and not any(name in shadowed for shadowed in self.shadowed_stack):
                self.calls.append((self.rel_path, self.current_symbol, name, getattr(node, "lineno", None)))
        self.generic_visit(node)


def _module_to_rel_path(module: str) -> str | None:
    if not module or module.startswith("."):
        return None
    return module.replace(".", "/") + ".py"


def _relative_module_to_rel_path(module: str, level: int, rel_path: str) -> str | None:
    if level < 1:
        return _module_to_rel_path(module)
    package_parts = list(Path(rel_path).parent.parts)
    keep_count = len(package_parts) - level + 1
    if keep_count < 0:
        return None
    base_parts = package_parts[:keep_count]
    module_parts = module.split(".") if module else []
    target_parts = base_parts + module_parts
    if not target_parts:
        return None
    return "/".join(target_parts) + ".py"


def _collect_import_edges_and_aliases(tree: ast.AST, rel_path: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    edges: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Import):
            for alias in node.names:
                asname = alias.asname or alias.name.split(".")[0]
                aliases[asname] = alias.name
                edges.append({"kind": "imports", "source": rel_path, "target": alias.name, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            prefix = "." * node.level
            display_module = f"{prefix}{module}" if module else prefix
            for alias in node.names:
                target = f"{display_module}.{alias.name}" if display_module else alias.name
                edges.append(
                    {
                        "kind": "imports",
                        "source": rel_path,
                        "target": target,
                        "line": node.lineno,
                        "relative": bool(node.level),
                    }
                )
                asname = alias.asname or alias.name
                module_file = _relative_module_to_rel_path(module, node.level, rel_path)
                if module_file:
                    aliases[asname] = f"{module_file}::{alias.name}"
                elif not node.level:
                    aliases[asname] = alias.name
    return edges, aliases


def _add_module_bindings(stmt: ast.stmt, bindings: Counter[str]) -> None:
    """Collect names bound at module scope.

    Bare call edges are only high-confidence when the called name has a single
    module-level meaning. If a module imports/defines a name and later rebinds
    or deletes it, Python runtime lookup may no longer target the original
    import/function/class. In that case we suppress the call edge instead of
    producing inspect evidence that looks more certain than it is.
    """
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bindings[stmt.name] += 1
        return
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            bindings[alias.asname or alias.name.split(".")[0]] += 1
        return
    if isinstance(stmt, ast.ImportFrom):
        for alias in stmt.names:
            if alias.name == "*":
                bindings["*"] += 2
            else:
                bindings[alias.asname or alias.name] += 1
        return
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            for name in _store_target_names(target):
                bindings[name] += 1
        return
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign, ast.Delete)):
        targets = stmt.targets if isinstance(stmt, ast.Delete) else [stmt.target]
        for target in targets:
            for name in _store_target_names(target):
                bindings[name] += 1
        return
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        for name in _store_target_names(stmt.target):
            bindings[name] += 1
        for child in [*stmt.body, *stmt.orelse]:
            _add_module_bindings(child, bindings)
        return
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            if item.optional_vars:
                for name in _store_target_names(item.optional_vars):
                    bindings[name] += 1
        for child in stmt.body:
            _add_module_bindings(child, bindings)
        return
    if isinstance(stmt, ast.If):
        for child in [*stmt.body, *stmt.orelse]:
            _add_module_bindings(child, bindings)
        return
    if isinstance(stmt, ast.While):
        for child in [*stmt.body, *stmt.orelse]:
            _add_module_bindings(child, bindings)
        return
    if isinstance(stmt, ast.Try):
        for child in [*stmt.body, *stmt.orelse, *stmt.finalbody]:
            _add_module_bindings(child, bindings)
        for handler in stmt.handlers:
            if handler.name:
                bindings[handler.name] += 1
            for child in handler.body:
                _add_module_bindings(child, bindings)
        return
    if isinstance(stmt, ast.Match):
        for case in stmt.cases:
            for name in _pattern_binding_names(case.pattern):
                bindings[name] += 1
            for child in case.body:
                _add_module_bindings(child, bindings)


def _pattern_binding_names(pattern: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(pattern, ast.MatchAs):
        if pattern.name:
            names.add(pattern.name)
        if pattern.pattern:
            names.update(_pattern_binding_names(pattern.pattern))
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name:
            names.add(pattern.name)
    elif isinstance(pattern, ast.MatchMapping):
        if pattern.rest:
            names.add(pattern.rest)
        for child in pattern.patterns:
            names.update(_pattern_binding_names(child))
    elif isinstance(pattern, ast.MatchClass):
        for child in [*pattern.patterns, *pattern.kwd_patterns]:
            names.update(_pattern_binding_names(child))
    elif isinstance(pattern, (ast.MatchSequence, ast.MatchOr)):
        for child in pattern.patterns:
            names.update(_pattern_binding_names(child))
    return names


def _collect_module_ambiguous_names(tree: ast.AST) -> set[str]:
    bindings: Counter[str] = Counter()
    for stmt in getattr(tree, "body", []):
        _add_module_bindings(stmt, bindings)
    return {name for name, count in bindings.items() if count > 1}


def _resolve_call_targets(
    call_name: str,
    symbols: Iterable[dict[str, Any]],
    source_file: str,
    import_aliases: dict[str, str] | None = None,
    module_ambiguous_names: set[str] | None = None,
) -> list[tuple[str, str]]:
    ambiguous_names = module_ambiguous_names or set()
    if "*" in ambiguous_names or call_name in ambiguous_names:
        return []
    symbol_ids = {sym["id"] for sym in symbols}
    import_target = (import_aliases or {}).get(call_name)
    if import_target in symbol_ids:
        return [(import_target, "import-resolved")]

    matches = [
        sym["id"]
        for sym in symbols
        if sym.get("name") == call_name
        and sym.get("file") == source_file
        and sym.get("kind") in {"function", "class"}
    ]
    if len(matches) == 1:
        return [(matches[0], "same-file")]
    return []


def build_code_graph(repo_path: str, max_files: int = 500) -> dict[str, Any]:
    """Build a bounded, read-only Python code graph for *repo_path*."""
    repo_path = _clean_repo_path(repo_path)
    root = Path(repo_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError("repo_path must be an existing directory")
    _validate_repo_path(root)
    max_files = max(1, min(int(max_files), 2000))

    files, skipped = _iter_python_files(root, max_files=max_files)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    pending_calls: list[tuple[str, str, str, int | None]] = []
    imports_by_file: dict[str, dict[str, str]] = {}
    module_ambiguous_by_file: dict[str, set[str]] = {}
    parse_errors: list[dict[str, Any]] = []

    for path in files:
        rel_path = _rel(path, root)
        try:
            with tokenize.open(path) as source_file:
                text = source_file.read()
            tree = ast.parse(text, filename=rel_path)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            parse_errors.append({"file": rel_path, "error": _redact(f"{type(exc).__name__}: {exc}")})
            continue

        lines = text.splitlines()
        nodes.append({"id": rel_path, "kind": "file", "name": Path(rel_path).name, "file": rel_path, "line": 1})
        visitor = _SymbolVisitor(rel_path, lines)
        visitor.visit(tree)
        nodes.extend(visitor.symbols)
        pending_calls.extend(visitor.calls)
        import_edges, import_aliases = _collect_import_edges_and_aliases(tree, rel_path)
        edges.extend(import_edges)
        imports_by_file[rel_path] = import_aliases
        module_ambiguous_by_file[rel_path] = _collect_module_ambiguous_names(tree)
        for sym in visitor.symbols:
            edges.append({"kind": "contains", "source": rel_path, "target": sym["id"], "line": sym["line"]})

    symbol_nodes = [node for node in nodes if node.get("kind") in {"function", "method", "class"}]
    unresolved_calls = 0
    for rel_path, source, call_name, line in pending_calls:
        targets = _resolve_call_targets(
            call_name,
            symbol_nodes,
            rel_path,
            imports_by_file.get(rel_path),
            module_ambiguous_by_file.get(rel_path),
        )
        if not targets:
            unresolved_calls += 1
            continue
        for target, confidence in targets[:20]:
            if target != source:
                edges.append(
                    {
                        "kind": "calls",
                        "source": source,
                        "target": target,
                        "line": line,
                        "call": call_name,
                        "confidence": confidence,
                    }
                )

    sorted_nodes = sorted(nodes, key=lambda n: (n.get("file", ""), n.get("line", 0), n.get("id", "")))
    sorted_edges = sorted(edges, key=lambda e: (e.get("kind", ""), e.get("source", ""), e.get("target", ""), e.get("line") or 0))
    nodes_truncated = len(sorted_nodes) > _MAX_INDEX_NODES
    edges_truncated = len(sorted_edges) > _MAX_INDEX_EDGES

    result = {
        "schema_version": "hermes-codegraph-v1",
        "repo_path": "[REDACTED_ABSOLUTE_PATH]",
        "repo_name": _redact(root.name),
        "scope": "read-only graph evidence; inspect-only; not canonical authority",
        "languages": ["python"],
        "nodes": sorted_nodes[:_MAX_INDEX_NODES],
        "edges": sorted_edges[:_MAX_INDEX_EDGES],
        "truncation": {
            "max_files_reached": bool(skipped.reasons.get("max_files_reached")),
            "nodes_truncated": nodes_truncated,
            "edges_truncated": edges_truncated,
            "node_limit": _MAX_INDEX_NODES,
            "edge_limit": _MAX_INDEX_EDGES,
        },
        "parse_errors": parse_errors[:_MAX_SKIPPED_SAMPLES],
        "skipped_reasons": skipped.reason_counts(),
        "skipped_samples": skipped.samples,
        "stats": {
            "files_indexed": len(files) - len(parse_errors),
            "files_seen": len(files),
            "files_skipped": skipped.total(),
            "nodes": min(len(nodes), _MAX_INDEX_NODES),
            "edges": min(len(edges), _MAX_INDEX_EDGES),
            "nodes_discovered": len(nodes),
            "edges_discovered": len(edges),
            "parse_errors": len(parse_errors),
            "unresolved_calls": unresolved_calls,
        },
    }
    return _redact_data(result)


def _matching_symbols(graph: dict[str, Any], symbol: str) -> list[dict[str, Any]]:
    symbol_lower = symbol.lower()
    return [
        node
        for node in graph.get("nodes", [])
        if node.get("kind") in {"function", "method", "class"}
        and (node.get("name", "").lower() == symbol_lower or symbol_lower in node.get("id", "").lower())
    ]


def _find_symbols(graph: dict[str, Any], symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    return _matching_symbols(graph, symbol)[: max(1, min(limit, 100))]


def _match_metadata(graph: dict[str, Any], symbol: str, limit: int) -> dict[str, Any]:
    total = len(_matching_symbols(graph, symbol))
    returned = min(total, max(1, min(limit, 100)))
    return {"total_matches": total, "returned_matches": returned, "matches_truncated": total > returned}


def code_graph_context(repo_path: str, symbol: str, max_files: int = 500, limit: int = 20) -> dict[str, Any]:
    symbol = _clean_symbol(symbol)
    graph = build_code_graph(repo_path, max_files=max_files)
    matches = _find_symbols(graph, symbol, limit=limit)
    match_ids = {node["id"] for node in matches}
    member_symbols = [
        node for node in graph.get("nodes", [])
        if node.get("kind") in {"method", "function", "class"}
        and any(node.get("id", "").startswith(f"{match_id}.") for match_id in match_ids)
    ][:_MAX_MEMBER_SYMBOLS]
    related_edges = [
        edge for edge in graph.get("edges", [])
        if edge.get("source") in match_ids or edge.get("target") in match_ids
    ][:_MAX_RELATED_EDGES]
    result = {
        "query": {"symbol": symbol, **_match_metadata(graph, symbol, limit)},
        "scope": graph["scope"],
        "matches": matches,
        "member_symbols": member_symbols,
        "related_edges": related_edges,
        "truncation": graph["truncation"],
        "parse_errors": graph["parse_errors"],
        "skipped_reasons": graph["skipped_reasons"],
        "skipped_samples": graph["skipped_samples"],
        "stats": graph["stats"],
    }
    return _redact_data(result)


def code_graph_impact(repo_path: str, symbol: str, max_files: int = 500, depth: int = 2) -> dict[str, Any]:
    symbol = _clean_symbol(symbol)
    graph = build_code_graph(repo_path, max_files=max_files)
    matches = _find_symbols(graph, symbol, limit=50)
    target_ids = {node["id"] for node in matches}
    impacted_ids: set[str] = set()
    frontier = set(target_ids)
    max_depth = max(1, min(int(depth), 5))
    traversed_edges: list[dict[str, Any]] = []

    for current_depth in range(1, max_depth + 1):
        next_frontier: set[str] = set()
        for edge in graph.get("edges", []):
            if edge.get("kind") != "calls":
                continue
            if edge.get("confidence") not in _HIGH_CONFIDENCE_CALLS:
                continue
            if edge.get("target") in frontier:
                source = edge.get("source")
                if source and source not in target_ids and source not in impacted_ids:
                    impacted_ids.add(source)
                    next_frontier.add(source)
                    edge_with_depth = dict(edge)
                    edge_with_depth["depth"] = current_depth
                    traversed_edges.append(edge_with_depth)
        frontier = next_frontier
        if not frontier:
            break

    node_by_id = {node["id"]: node for node in graph.get("nodes", [])}
    impacted = [node_by_id[node_id] for node_id in sorted(impacted_ids) if node_id in node_by_id][:_MAX_IMPACTED]
    result = {
        "query": {"symbol": symbol, "depth": max_depth, **_match_metadata(graph, symbol, 50)},
        "scope": "inspect-only evidence; graph does not authorize edits",
        "matches": matches,
        "impacted": impacted,
        "impact_edges": traversed_edges[:_MAX_IMPACT_EDGES],
        "truncation": graph["truncation"],
        "parse_errors": graph["parse_errors"],
        "skipped_reasons": graph["skipped_reasons"],
        "skipped_samples": graph["skipped_samples"],
        "stats": graph["stats"],
    }
    return _redact_data(result)


INDEX_SCHEMA = {
    "name": CODE_GRAPH_INDEX_TOOL_NAME,
    "description": "Build a bounded read-only Python code graph for a repository. Clean-room Hermes-native implementation; graph output is evidence only, not edit authorization.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Repository/directory path to index; broad roots such as / and the user home are refused.", "minLength": 1},
            "max_files": {"type": "integer", "description": "Maximum Python files to index, capped at 2000.", "default": 500, "minimum": 1, "maximum": 2000},
        },
        "required": ["repo_path"],
    },
}

CONTEXT_SCHEMA = {
    "name": CODE_GRAPH_CONTEXT_TOOL_NAME,
    "description": "Find source-backed context and graph edges for a symbol in a bounded read-only Python code graph.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "minLength": 1},
            "symbol": {"type": "string", "minLength": 1},
            "max_files": {"type": "integer", "default": 500, "minimum": 1, "maximum": 2000},
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
        },
        "required": ["repo_path", "symbol"],
    },
}

IMPACT_SCHEMA = {
    "name": CODE_GRAPH_IMPACT_TOOL_NAME,
    "description": "Return callers/impact candidates for a symbol. Output is inspect-only evidence and does not authorize edits.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "minLength": 1},
            "symbol": {"type": "string", "minLength": 1},
            "max_files": {"type": "integer", "default": 500, "minimum": 1, "maximum": 2000},
            "depth": {"type": "integer", "default": 2, "minimum": 1, "maximum": 5},
        },
        "required": ["repo_path", "symbol"],
    },
}


def _handle_index(args: dict[str, Any], **_: Any) -> str:
    return _json(build_code_graph(args["repo_path"], max_files=args.get("max_files", 500)))


def _handle_context(args: dict[str, Any], **_: Any) -> str:
    return _json(
        code_graph_context(
            args["repo_path"],
            args["symbol"],
            max_files=args.get("max_files", 500),
            limit=args.get("limit", 20),
        )
    )


def _handle_impact(args: dict[str, Any], **_: Any) -> str:
    return _json(
        code_graph_impact(
            args["repo_path"],
            args["symbol"],
            max_files=args.get("max_files", 500),
            depth=args.get("depth", 2),
        )
    )


registry.register(
    name=CODE_GRAPH_INDEX_TOOL_NAME,
    toolset=_CODE_GRAPH_TOOLSET,
    schema=INDEX_SCHEMA,
    handler=_handle_index,
    emoji="🕸️",
    max_result_size_chars=100_000,
)
registry.register(
    name=CODE_GRAPH_CONTEXT_TOOL_NAME,
    toolset=_CODE_GRAPH_TOOLSET,
    schema=CONTEXT_SCHEMA,
    handler=_handle_context,
    emoji="🧭",
    max_result_size_chars=60_000,
)
registry.register(
    name=CODE_GRAPH_IMPACT_TOOL_NAME,
    toolset=_CODE_GRAPH_TOOLSET,
    schema=IMPACT_SCHEMA,
    handler=_handle_impact,
    emoji="💥",
    max_result_size_chars=60_000,
)
