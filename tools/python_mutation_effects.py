"""Static Python child-process effect recovery for MutationEffectGuard."""

from __future__ import annotations

import ast
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

from tools.mutation_effect_guard import MutationEffect, _resolve_path


_SUBPROCESS_ARGV_CALLS = frozenset({
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
    "asyncio.create_subprocess_exec",
})
_SUBPROCESS_SHELL_CALLS = frozenset({
    "os.popen",
    "os.system",
    "asyncio.create_subprocess_shell",
})
_PATH_CONSTRUCTORS = frozenset({"Path", "pathlib.Path"})
_STATIC_CALLABLES = (
    _SUBPROCESS_ARGV_CALLS
    | _SUBPROCESS_SHELL_CALLS
    | _PATH_CONSTRUCTORS
    | frozenset({
        "eval",
        "exec",
        "os.fspath",
        "runpy.run_path",
        "str",
    })
)
_UNKNOWN = object()


def scan_python_effect(
    source: str,
    *,
    cwd: Path,
    script_path: Path | None,
    depth: int,
    origin: str,
    detect_command: Callable[..., MutationEffect | None],
) -> MutationEffect | None:
    """Parse Python source without executing it and return its first mutation."""

    try:
        tree = ast.parse(source, filename=str(script_path or "<python -c>"))
    except (SyntaxError, ValueError):
        # Python will reject the source before any side effect occurs.
        return None
    scanner = _PythonEffectScanner(
        detect_command=detect_command,
        cwd=cwd,
        script_path=script_path,
        depth=depth,
        origin=origin,
    )
    scanner.visit(tree)
    return scanner.effect


class _PythonEffectScanner(ast.NodeVisitor):
    """Conservative, side-effect-free evaluator for process-spawning Python AST."""

    def __init__(
        self,
        *,
        detect_command: Callable[..., MutationEffect | None],
        cwd: Path,
        script_path: Path | None,
        depth: int,
        origin: str,
    ) -> None:
        self.detect_command = detect_command
        self.cwd = cwd
        self.script_path = script_path
        self.depth = depth
        self.origin = origin
        self.effect: MutationEffect | None = None
        self.bindings: dict[str, Any] = {"__name__": "__main__"}
        self.imports: dict[str, str] = {
            "Path": "pathlib.Path",
        }
        self.functions: dict[
            str,
            ast.FunctionDef | ast.AsyncFunctionDef,
        ] = {}
        self.active_functions: set[str] = set()
        if script_path is not None:
            self.bindings["__file__"] = script_path

    def visit(self, node: ast.AST) -> Any:
        if self.effect is not None:
            return None
        return super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions[node.name] = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions[node.name] = node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class bodies can execute arbitrary code, but descending into methods
        # would confuse definitions with effects. Inspect only decorators and
        # base expressions, which Python evaluates while creating the class.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # A lambda body is dormant until invocation. Static local-function
        # expansion below deliberately supports named defs only.
        return None

    def visit_If(self, node: ast.If) -> None:
        truth = self._truth_value(node.test)
        if truth is True:
            statements = node.body
        elif truth is False:
            statements = node.orelse
        else:
            self.visit(node.test)
            statements = [*node.body, *node.orelse]
        for statement in statements:
            self.visit(statement)
            if self.effect is not None:
                break

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for alias in node.names:
                if alias.name != "*":
                    self.imports[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self._resolve(node.value)
        for target in node.targets:
            self._bind_target(target, value)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = self._resolve(node.value) if node.value is not None else _UNKNOWN
        self._bind_target(node.target, value)
        if node.value is not None:
            self.generic_visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        value = self._resolve(node.value)
        self._bind_target(node.target, value)
        self.generic_visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            self._visit_local_function(node.func.id, node)
            return

        qualified = self._qualified_name(node.func)
        if qualified in {"os.chdir", "os.fchdir"}:
            target = self._resolve_call_arg(node, 0, "path")
            path = self._as_path(target, self.cwd)
            if path is not None:
                self.cwd = path
        elif qualified in _SUBPROCESS_ARGV_CALLS:
            if qualified == "asyncio.create_subprocess_exec":
                command = [self._resolve(arg) for arg in node.args]
            else:
                command = self._resolve_call_arg(node, 0, "args")
            call_cwd = self._call_cwd(node)
            command_text = self._command_text(command)
            if command_text:
                self.effect = self.detect_command(
                    command_text,
                    call_cwd,
                    depth=self.depth + 1,
                    origin=f"{self.origin} via {qualified}",
                )
        elif qualified in _SUBPROCESS_SHELL_CALLS:
            command = self._resolve_call_arg(node, 0, "command")
            if command is _UNKNOWN:
                command = self._resolve_call_arg(node, 0, "cmd")
            if isinstance(command, str):
                self.effect = self.detect_command(
                    command,
                    self._call_cwd(node),
                    depth=self.depth + 1,
                    origin=f"{self.origin} via {qualified}",
                )
        elif qualified in {"exec", "eval"}:
            source = self._resolve_call_arg(node, 0, "source")
            if isinstance(source, str):
                command = shlex.join([sys.executable, "-c", source])
                self.effect = self.detect_command(
                    command,
                    self.cwd,
                    depth=self.depth + 1,
                    origin=f"{self.origin} via {qualified}",
                )
        elif qualified == "runpy.run_path":
            path = self._as_path(
                self._resolve_call_arg(node, 0, "path_name"),
                self.cwd,
            )
            if path is not None:
                command = shlex.join([sys.executable, os.fspath(path)])
                self.effect = self.detect_command(
                    command,
                    self.cwd,
                    depth=self.depth + 1,
                    origin=f"{self.origin} via runpy.run_path",
                )
        if self.effect is None:
            self.generic_visit(node)

    def _visit_local_function(self, name: str, call: ast.Call) -> None:
        function = self.functions[name]
        if name in self.active_functions:
            return

        saved_bindings = dict(self.bindings)
        saved_imports = dict(self.imports)
        saved_functions = dict(self.functions)
        parameters = [
            *function.args.posonlyargs,
            *function.args.args,
        ]
        positional = [self._resolve(value) for value in call.args]
        keyword_values = {
            item.arg: self._resolve(item.value)
            for item in call.keywords
            if item.arg is not None
        }
        defaults_offset = len(parameters) - len(function.args.defaults)
        for index, parameter in enumerate(parameters):
            if index < len(positional):
                value = positional[index]
            elif parameter.arg in keyword_values:
                value = keyword_values[parameter.arg]
            elif index >= defaults_offset:
                value = self._resolve(
                    function.args.defaults[index - defaults_offset]
                )
            else:
                value = _UNKNOWN
            self.bindings[parameter.arg] = value

        self.active_functions.add(name)
        try:
            for statement in function.body:
                self.visit(statement)
                if self.effect is not None:
                    break
        finally:
            self.active_functions.discard(name)
            self.bindings = saved_bindings
            self.imports = saved_imports
            self.functions = saved_functions

    def _truth_value(self, node: ast.AST) -> bool | None:
        value = self._resolve(node)
        if isinstance(value, bool):
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            nested = self._truth_value(node.operand)
            return None if nested is None else not nested
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
        ):
            left = self._resolve(node.left)
            right = self._resolve(node.comparators[0])
            if left is not _UNKNOWN and right is not _UNKNOWN:
                if isinstance(node.ops[0], (ast.Eq, ast.Is)):
                    return left == right
                if isinstance(node.ops[0], (ast.NotEq, ast.IsNot)):
                    return left != right
        return None

    def _bind_target(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.bindings[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value,
            (list, tuple),
        ):
            for item, item_value in zip(target.elts, value):
                self._bind_target(item, item_value)

    def _qualified_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            imported = self.imports.get(node.id)
            if imported is not None:
                return imported
            bound = self.bindings.get(node.id)
            if isinstance(bound, str) and bound in _STATIC_CALLABLES:
                return bound
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._qualified_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else None
        return None

    def _resolve_call_arg(self, node: ast.Call, index: int, keyword: str) -> Any:
        if index < len(node.args):
            return self._resolve(node.args[index])
        for item in node.keywords:
            if item.arg == keyword:
                return self._resolve(item.value)
        return _UNKNOWN

    def _call_cwd(self, node: ast.Call) -> Path:
        for item in node.keywords:
            if item.arg == "cwd":
                path = self._as_path(self._resolve(item.value), self.cwd)
                if path is not None:
                    return path
        return self.cwd

    @staticmethod
    def _command_text(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value and all(
            isinstance(item, (str, os.PathLike)) for item in value
        ):
            return shlex.join(os.fspath(item) for item in value)
        return None

    @staticmethod
    def _as_path(value: Any, base: Path) -> Path | None:
        if isinstance(value, (str, os.PathLike)):
            return _resolve_path(value, base)
        return None

    def _resolve(self, node: ast.AST | None) -> Any:
        if node is None:
            return _UNKNOWN
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id, _UNKNOWN)
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self._resolve(item) for item in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                    continue
                if isinstance(value, ast.FormattedValue):
                    resolved = self._resolve(value.value)
                    if resolved is _UNKNOWN:
                        return _UNKNOWN
                    parts.append(
                        os.fspath(resolved)
                        if isinstance(resolved, os.PathLike)
                        else str(resolved)
                    )
                    continue
                return _UNKNOWN
            return "".join(parts)
        if isinstance(node, ast.BinOp):
            left = self._resolve(node.left)
            right = self._resolve(node.right)
            if isinstance(node.op, ast.Add):
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                if isinstance(left, list) and isinstance(right, list):
                    return left + right
                if isinstance(left, tuple) and isinstance(right, tuple):
                    return left + right
            if (
                isinstance(node.op, ast.Div)
                and isinstance(left, (str, os.PathLike))
                and isinstance(right, (str, os.PathLike))
            ):
                return Path(left) / Path(right)
            return _UNKNOWN
        if isinstance(node, ast.Attribute):
            value = self._resolve(node.value)
            if node.attr == "parent" and isinstance(value, Path):
                return value.parent
            qualified = self._qualified_name(node)
            if qualified == "sys.executable":
                return sys.executable
            return qualified if qualified else _UNKNOWN
        if isinstance(node, ast.Call):
            qualified = self._qualified_name(node.func)
            if qualified in _PATH_CONSTRUCTORS:
                parts = [self._resolve(arg) for arg in node.args]
                if parts and all(
                    isinstance(item, (str, os.PathLike)) for item in parts
                ):
                    return Path(parts[0]).joinpath(
                        *(os.fspath(item) for item in parts[1:])
                    )
                return _UNKNOWN
            if qualified in {"str", "os.fspath"} and node.args:
                value = self._resolve(node.args[0])
                if isinstance(value, (str, os.PathLike)):
                    return os.fspath(value)
                return _UNKNOWN
            if qualified in {"os.getcwd", "pathlib.Path.cwd", "Path.cwd"}:
                return self.cwd
            if qualified == "os.path.join":
                parts = [self._resolve(arg) for arg in node.args]
                if parts and all(
                    isinstance(item, (str, os.PathLike)) for item in parts
                ):
                    return os.path.join(*(os.fspath(item) for item in parts))
                return _UNKNOWN
            if qualified == "os.path.dirname" and node.args:
                value = self._resolve(node.args[0])
                if isinstance(value, (str, os.PathLike)):
                    return os.path.dirname(os.fspath(value))
                return _UNKNOWN
            if isinstance(node.func, ast.Attribute) and node.func.attr == "split":
                value = self._resolve(node.func.value)
                if isinstance(value, str) and not node.args and not node.keywords:
                    return value.split()
                return _UNKNOWN
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "absolute", "resolve",
            }:
                value = self._resolve(node.func.value)
                path = self._as_path(value, self.cwd)
                return path.resolve() if path is not None else _UNKNOWN
            return _UNKNOWN
        return _UNKNOWN
