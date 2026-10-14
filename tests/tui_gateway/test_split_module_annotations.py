"""Split-module annotations resolve at import time on Python < 3.14.

``bind_module`` rebinds a split module's function BODIES onto the server's namespace, so a body
may use a name the module never imported. A signature annotation is not rebound: ``def f() ->
tuple[Path, ...]`` is evaluated at def time (before PEP 649, Python < 3.14), so an unimported
name there raises NameError while ``tui_gateway.server`` imports and ``hermes gateway run``
never starts. The suite runs on 3.14, where annotations are lazy, so it must be checked statically.
"""

import ast
import builtins
from pathlib import Path

import pytest

_MODULES = sorted(Path(__file__).resolve().parents[2].glob("tui_gateway/methods_*.py"))


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


@pytest.mark.parametrize("path", _MODULES, ids=lambda p: p.name)
def test_signature_annotations_only_use_names_the_module_defines(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if any(isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names) for n in tree.body):
        pytest.skip("annotations are strings here")
    known = _module_level_names(tree) | set(dir(builtins))
    missing = set()
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        args = fn.args
        annotated = [a.annotation for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg] if a and a.annotation]
        for ann in [*annotated, fn.returns]:
            if ann is not None:
                missing |= {n.id for n in ast.walk(ann) if isinstance(n, ast.Name) and n.id not in known}
    assert not missing, f"{path.name}: annotations use names it never imports: {sorted(missing)}"
