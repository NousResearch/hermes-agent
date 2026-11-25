"""Gateway-lifecycle guard for execute_code cells (sibling of tools/code_execution_tool.py).

``cron.lifecycle_guard`` scans raw text, so a Python cell that only *mentions* a lifecycle command
(a ``print`` argument, a runbook written with ``open().write``) was refused exactly like one that
runs it (#146582). A cell that cannot start or signal a process cannot restart the gateway, so it is
treated as inert data, the way ``strip_inert_heredoc_bodies`` already treats quoted heredocs.

"Cannot" is judged conservatively from the AST, never by tracing values from literals to sinks:
the cell must parse, be straight-line top-level code (no def/lambda/class/comprehension, branch,
loop or try), import only modules from a small allowlist, touch no private/dunder, process-shaped
or introspection attribute, and load no name it has not bound in an earlier statement apart from a
set of plain builtins. The last rule matters because the session kernel persists: ``os``, ``subprocess`` or a
``def sh(cmd)`` helper from an earlier cell is still reachable by name. Anything else is scanned as
before. This is accident prevention for the gateway SIGTERM-ing itself, not a security boundary:
like a command spliced at runtime, a cell written to get past it is not detected -- an earlier
cell rebinding a plain builtin (``from os import system as print``), or a dynamic lookup reached
through an allowlisted module (``string.Formatter().get_field``, ``inspect.getattr_static``,
``enum.bltns.getattr``). Closing those gadget by gadget is not a goal.
"""

from __future__ import annotations

import ast
from typing import Optional

# Modules with no way to spawn, signal or exec anything. Everything else (os, sys, subprocess,
# importlib, ctypes, asyncio, multiprocessing, shutil, pty, hermes_tools, ...) keeps the scan.
_INERT_IMPORTS = frozenset({
    "base64", "collections", "csv", "dataclasses", "datetime", "decimal", "difflib", "enum",
    "fractions", "functools", "hashlib", "html", "itertools", "json", "math", "pathlib", "pprint",
    "random", "re", "statistics", "string", "textwrap", "time", "unicodedata",
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


# Only straight-line top-level code qualifies: every statement runs, in order, in module scope, so
# "bound earlier in this cell" is exact. Nested scopes (def/lambda/class/comprehension), branches,
# loops, try and walrus bind names a module-level read may never see, and a leftover helper from an
# earlier kernel cell would answer instead -- such a cell keeps the normal scan.
_STRAIGHT_LINE_STMTS = (ast.Assign, ast.AnnAssign, ast.Expr, ast.Import, ast.ImportFrom, ast.Pass, ast.With)
_SCOPE_OR_FLOW_NODES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.NamedExpr, ast.Global, ast.Nonlocal, ast.Await, ast.Yield,
    ast.YieldFrom, ast.AsyncWith,
)
# Frame / generator / coroutine / traceback introspection reaches globals and builtins by name.
_INTROSPECTION_PREFIXES = ("gi_", "f_", "tb_", "cr_", "ag_")
_INTROSPECTION_ATTRS = frozenset({"get_type_hints"})


def _statement_bindings(stmt: ast.stmt) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(stmt, ast.Assign):
        targets = list(stmt.targets)
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets = [stmt.target]
    elif isinstance(stmt, ast.With):
        targets = [item.optional_vars for item in stmt.items if item.optional_vars is not None]
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(alias.asname or alias.name).split(".")[0] for alias in stmt.names}
    return {sub.id for target in targets for sub in ast.walk(target) if isinstance(sub, ast.Name)}


def _node_may_reach_process(node: ast.AST, bound: set[str]) -> bool:
    if isinstance(node, _SCOPE_OR_FLOW_NODES):
        return True
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] not in _INERT_IMPORTS for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return bool(node.level) or (node.module or "").split(".")[0] not in _INERT_IMPORTS
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and node.id not in _SAFE_BUILTINS and node.id not in bound
    if isinstance(node, ast.Attribute):
        return (node.attr in _SINK_ATTRS or node.attr in _INTROSPECTION_ATTRS
                or node.attr.startswith(_SINK_ATTR_PREFIXES + _INTROSPECTION_PREFIXES))
    return False


def _stmt_may_reach_process(stmt: ast.stmt, bound: set[str]) -> bool:
    """Loads in *stmt* see only names bound by EARLIER top-level statements (``sh = sh`` reads the
    leftover); a ``with`` body's statements are checked in order like the top level."""
    if not isinstance(stmt, _STRAIGHT_LINE_STMTS):
        return True
    if isinstance(stmt, ast.With):
        if any(_node_may_reach_process(node, bound) for item in stmt.items for node in ast.walk(item)):
            return True
        inner = bound | _statement_bindings(stmt)
        for child in stmt.body:
            if _stmt_may_reach_process(child, inner):
                return True
            inner |= _statement_bindings(child)
        return False
    return any(_node_may_reach_process(node, bound) for node in ast.walk(stmt))


def _is_inert_python(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    bound: set[str] = set()
    for stmt in tree.body:
        if _stmt_may_reach_process(stmt, bound):
            return False
        bound |= _statement_bindings(stmt)
        if isinstance(stmt, ast.With):
            for child in stmt.body:
                bound |= _statement_bindings(child)
    return True


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
        "If the script only prints or writes that text as documentation, write it as "
        "straight-line top-level code (no def, lambda, class, loop, if, try or "
        "comprehension) that imports only pure stdlib modules (json, pathlib, re, "
        "textwrap, ...) and calls no process/shell/dynamic-code API (os, subprocess, "
        "exec, terminal, helpers from earlier cells); it is then treated as inert data."
    )
