"""Gateway-lifecycle guard for execute_code cells (sibling of tools/code_execution_tool.py).

``cron.lifecycle_guard`` scans raw text, so a Python cell that only *mentions* a lifecycle command
(a ``print`` argument, a runbook written with ``open().write``) was refused exactly like one that
runs it (#146582). A cell that cannot start or signal a process cannot restart the gateway, so it is
treated as inert data, the way ``strip_inert_heredoc_bodies`` already treats quoted heredocs.

"Cannot" is judged conservatively from the AST, never by tracing values from literals to sinks:
the cell must parse, import only modules from a small allowlist, touch no private/dunder or
process-shaped attribute, and load no name it does not bind itself apart from a set of plain
builtins. The last rule matters because the session kernel persists: ``os``, ``subprocess`` or a
``def sh(cmd)`` helper from an earlier cell is still reachable by name. Anything else is scanned as
before. This is accident prevention for the gateway SIGTERM-ing itself, not a security boundary.
"""

from __future__ import annotations

import ast
from typing import Optional

# Modules with no way to spawn, signal or exec anything. Everything else (os, sys, subprocess,
# importlib, ctypes, asyncio, multiprocessing, shutil, pty, hermes_tools, ...) keeps the scan.
_INERT_IMPORTS = frozenset({
    "base64", "collections", "csv", "dataclasses", "datetime", "decimal", "difflib", "enum",
    "fractions", "functools", "hashlib", "html", "itertools", "json", "math", "pathlib", "pprint",
    "random", "re", "statistics", "string", "textwrap", "time", "typing", "unicodedata",
})

# Builtins a cell may use without binding them. No exec/eval/compile/__import__/getattr/globals/
# vars/type/breakpoint: those reach code or objects the AST cannot see.
_SAFE_BUILTINS = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "oct", "open", "ord", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "zip", "None", "True", "False", "Exception", "ValueError", "KeyError", "IndexError",
    "TypeError", "OSError", "StopIteration", "RuntimeError",
})

# Attribute names that reach a process, a module with one, or dynamic code (``pathlib.os``,
# ``typing.sys``) even when the base object came from an inert module.
_SINK_ATTRS = frozenset({
    "os", "sys", "subprocess", "system", "popen", "spawn", "exec", "eval", "kill", "killpg",
    "run", "call", "check_call", "check_output", "Popen", "import_module", "builtins",
})
_SINK_ATTR_PREFIXES = ("_", "exec", "spawn", "posix_spawn", "popen", "kill", "fork", "startfile")

_MAX_MATCH_CHARS = 120


_Pos = tuple[int, int]
_ALWAYS = (0, 0)  # comprehension targets and arguments: bound before any load inside their scope


def _end(node: ast.AST) -> _Pos:
    return (getattr(node, "end_lineno", None) or 0, getattr(node, "end_col_offset", None) or 0)


def _binding_positions(tree: ast.AST) -> dict[str, _Pos]:
    """Earliest point at which each name the cell binds holds the cell's own value.

    Position-aware on purpose: in ``sh = sh`` or ``f()`` before ``f = ...`` the load still reads
    whatever an earlier kernel cell left behind, so it must not count as locally bound.
    """
    bound: dict[str, _Pos] = {}

    def bind(name: str, pos: _Pos) -> None:
        bound[name] = min(pos, bound.get(name, pos))

    def bind_targets(target: ast.AST, pos: _Pos) -> None:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                bind(sub.id, pos)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind_targets(target, _end(node))
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            bind_targets(node.target, _end(node))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind_targets(node.target, _end(node.iter))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind_targets(node.optional_vars, _end(node.context_expr))
        elif isinstance(node, ast.comprehension):
            bind_targets(node.target, _ALWAYS)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name, (node.lineno, node.col_offset))
        elif isinstance(node, ast.arg):
            bind(node.arg, _ALWAYS)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bind((alias.asname or alias.name).split(".")[0], _end(node))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind(node.name, (node.body[0].lineno, 0))
    return bound


def _node_may_reach_process(node: ast.AST, bound: dict[str, _Pos]) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] not in _INERT_IMPORTS for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return bool(node.level) or (node.module or "").split(".")[0] not in _INERT_IMPORTS
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load) or node.id in _SAFE_BUILTINS:
            return False
        return node.id not in bound or bound[node.id] > (node.lineno, node.col_offset)
    if isinstance(node, ast.Attribute):
        return node.attr in _SINK_ATTRS or node.attr.startswith(_SINK_ATTR_PREFIXES)
    return isinstance(node, (ast.Global, ast.Nonlocal))


def _is_inert_python(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    bound = _binding_positions(tree)
    return not any(_node_may_reach_process(node, bound) for node in ast.walk(tree))


def lifecycle_guard_rejection(code: str) -> Optional[str]:
    """Rejection text when *code* may run a gateway-lifecycle command, else None."""
    from cron.lifecycle_guard import (
        HOST_INTERPRETER_KILL_REJECTION,
        contains_gateway_lifecycle_command,
        contains_host_interpreter_kill,
        first_gateway_lifecycle_match,
    )

    if not contains_gateway_lifecycle_command(code) or _is_inert_python(code):
        return None
    if contains_host_interpreter_kill(code):
        return HOST_INTERPRETER_KILL_REJECTION
    matched = first_gateway_lifecycle_match(code)
    trigger = f" Matched `{matched[:_MAX_MATCH_CHARS]}` in the script." if matched else ""
    return (
        "Blocked: cannot restart or stop the gateway from inside the "
        "gateway process. The gateway would kill this script before "
        "it could complete (SIGTERM propagates to child processes)."
        f"{trigger} Run the lifecycle command from a shell outside the gateway. "
        "If the script only prints or writes that text as documentation, drop the "
        "process/shell/dynamic-code calls (os, subprocess, exec, terminal, helpers "
        "from earlier cells) so the cell is treated as inert data."
    )
