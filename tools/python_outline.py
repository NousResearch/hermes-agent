#!/usr/bin/env python3
"""Python code-outline scanner for ``read_file`` outline mode.

Stdlib only (``ast``). Deterministic: source order, no timestamps, no
absolute paths. Raises ``SyntaxError`` on unparseable source so the caller
can fall back to a normal full-text read.
"""

import ast

#: Hard cap on entries per outline call; the caller marks truncation.
PYTHON_OUTLINE_MAX_ENTRIES = 500

#: First-docstring-line preview is truncated to this many characters.
_DOC_PREVIEW_MAX_CHARS = 200


def _format_args(args: ast.arguments) -> str:
    """Render a compact ``(a, b, *args, k, **kw)`` parameter list."""
    parts = []
    posonly = getattr(args, "posonlyargs", [])
    for a in list(posonly) + list(args.args):
        parts.append(a.arg)
    if posonly:
        parts.append("/")
    if args.vararg is not None:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg is not None:
        parts.append("**" + args.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def _doc_first_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """First line of the docstring, or ``""``; truncated to a fixed width."""
    try:
        doc = ast.get_docstring(node, clean=True)
    except Exception:
        return ""
    if not doc:
        return ""
    first = doc.split("\n", 1)[0].strip()
    if len(first) > _DOC_PREVIEW_MAX_CHARS:
        return first[:_DOC_PREVIEW_MAX_CHARS] + "..."
    return first


def _entries_for(body, depth: int, out: list) -> None:
    """Append one entry per class/function def; nested defs one level in."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    bases.append("...")
            out.append({
                "line": node.lineno,
                "kind": "class",
                "name": node.name,
                "signature": "(" + ", ".join(bases) + ")",
                "doc": _doc_first_line(node),
                "depth": depth,
            })
            if depth == 0:
                _entries_for(node.body, 1, out)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({
                "line": node.lineno,
                "kind": (
                    "async function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                ),
                "name": node.name,
                "signature": _format_args(node.args),
                "doc": _doc_first_line(node),
                "depth": depth,
            })
            if depth == 0:
                _entries_for(node.body, 1, out)


def python_outline(source: str) -> list:
    """Return the structural outline of Python *source* in source order.

    Each entry is ``{"line", "kind", "name", "signature", "doc", "depth"}``.
    Body lines are never included. Raises ``SyntaxError`` when *source*
    does not parse.
    """
    tree = ast.parse(source)
    out: list = []
    _entries_for(tree.body, 0, out)
    return out
