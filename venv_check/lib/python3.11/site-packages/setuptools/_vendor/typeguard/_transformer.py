from __future__ import annotations

import ast
import builtins
import sys
import typing
from ast import (
    AST,
    Add,
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    Attribute,
    AugAssign,
    BinOp,
    BitAnd,
    BitOr,
    BitXor,
    Call,
    ClassDef,
    Constant,
    Dict,
    Div,
    Expr,
    Expression,
    FloorDiv,
    FunctionDef,
    If,
    Import,
    ImportFrom,
    Index,
    List,
    Load,
    LShift,
    MatMult,
    Mod,
    Module,
    Mult,
    Name,
    NamedExpr,
    NodeTransformer,
    NodeVisitor,
    Pass,
    Pow,
    Return,
    RShift,
    Starred,
    Store,
    Sub,
    Subscript,
    Tuple,
    Yield,
    YieldFrom,
    alias,
    copy_location,
    expr,
    fix_missing_locations,
    keyword,
    walk,
)
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload

generator_names = (
    "typing.Generator",
    "collections.abc.Generator",
    "typing.Iterator",
    "collections.abc.Iterator",
    "typing.Iterable",
    "collections.abc.Iterable",
    "typing.AsyncIterator",
    "collections.abc.AsyncIterator",
    "typing.AsyncIterable",
    "collections.abc.AsyncIterable",
    "typing.AsyncGenerator",
    "collections.abc.AsyncGenerator",
)
anytype_names = (
    "typing.Any",
    "typing_extensions.Any",
)
literal_names = (
    "typing.Literal",
    "typing_extensions.Literal",
)
annotated_names = (
    "typing.Annotated",
    "typing_extensions.Annotated",
)
ignore_decorators = (
    "typing.no_type_check",
    "typeguard.typeguard_ignore",
)
aug_assign_functions = {
    Add: "iadd",
    Sub: "isub",
    Mult: "imul",
    MatMult: "imatmul",
    Div: "itruediv",
    FloorDiv: "ifloordiv",
    Mod: "imod",
    Pow: "ipow",
    LShift: "ilshift",
    RShift: "irshift",
    BitAnd: "iand",
    BitXor: "ixor",
    BitOr: "ior",
}


@dataclass
class TransformMemo:
    node: Module | ClassDef | FunctionDef | AsyncFunctionDef | None
    parent: TransformMemo | None
    path: tuple[str, ...]
    joined_path: Constant = field(init=False)
    return_annotation: expr | None = None
    yield_annotation: expr | None = None
    send_annotation: expr | None = None
    is_async: bool = False
    local_names: set[str] = field(init=False, default_factory=set)
    imported_names: dict[str, str] = field(init=False, default_factory=dict)
    ignored_names: set[str] = field(init=False, default_factory=set)
    load_names: defaultdict[str, dict[str, Name]] = field(
        init=False, default_factory=lambda: defaultdict(dict)
    )
    has_yield_expressions: bool = field(init=False, default=False)
    has_return_expressions: bool = field(init=False, default=False)
    memo_var_name: Name | None = field(init=False, default=None)
    should_instrument: bool = field(init=False, default=True)
    variable_annotations: dict[str, expr] = field(init=False, default_factory=dict)
    configuration_overrides: dict[str, Any] = field(init=False, default_factory=dict)
    code_inject_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        elements: list[str] = []
        memo = self
        while isinstance(memo.node, (ClassDef, FunctionDef, AsyncFunctionDef)):
            elements.insert(0, memo.node.name)
            if not memo.parent:
                break

            memo = memo.parent
            if isinstance(memo.node, (FunctionDef, AsyncFunctionDef)):
                elements.insert(0, "<locals>")

        self.joined_path = Constant(".".join(elements))

        # Figure out where to insert instrumentation code
        if self.node:
            for index, child in enumerate(self.node.body):
                if isinstance(child, ImportFrom) and child.module == "__future__":
                    # (module only) __future__ imports must come first
                    continue
                elif (
                    isinstance(child, Expr)
                    and isinstance(child.value, Constant)
                    and isinstance(child.value.value, str)
                ):
                    continue  # docstring

                self.code_inject_index = index
                break

    def get_unused_name(self, name: str) -> str:
        memo: TransformMemo | None = self
        while memo is not None:
            if name in memo.local_names:
                memo = self
                name += "_"
            else:
                memo = memo.parent

        self.local_names.add(name)
        return name

    def is_ignored_name(self, expression: expr | Expr | None) -> bool:
        top_expression = (
            expression.value if isinstance(expression, Expr) else expression
        )

        if isinstance(top_expression, Attribute) and isinstance(
            top_expression.value, Name
        ):
            name = top_expression.value.id
        elif isinstance(top_expression, Name):
            name = top_expression.id
        else:
            return False

        memo: TransformMemo | None = self
        while memo is not None:
            if name in memo.ignored_names:
                return True

            memo = memo.parent

        return False

    def get_memo_name(self) -> Name:
        if not self.memo_var_name:
            self.memo_var_name = Name(id="memo", ctx=Load())

        return self.memo_var_name

    def get_import(self, module: str, name: str) -> Name:
        if module in self.load_names and name in self.load_names[module]:
            return self.load_names[module][name]

        qualified_name = f"{module}.{name}"
        if name in self.imported_names and self.imported_names[name] == qualified_name:
            return Name(id=name, ctx=Load())

        alias = self.get_unused_name(name)
        node = self.load_names[module][name] = Name(id=alias, ctx=Load())
        self.imported_names[name] = qualified_name
        return node

    def insert_imports(self, node: Module | FunctionDef | AsyncFunctionDef) -> None:
        """Insert imports needed by injected code."""
        if not self.load_names:
            return

        # Insert imports after any "from __future__ ..." imports and any docstring
        for modulename, names in self.load_names.items():
            aliases = [
                alias(orig_name, new_name.id if orig_name != new_name.id else None)
                for orig_name, new_name in sorted(names.items())
            ]
            node.body.insert(self.code_inject_index, ImportFrom(modulename, aliases, 0))

    def name_matches(self, expression: expr | Expr | None, *names: str) -> bool:
        if expression is None:
            return False

        path: list[str] = []
        top_expression = (
            expression.value if isinstance(expression, Expr) else expression
        )

        if isinstance(top_expression, Subscript):
            top_expression = top_expression.value
        elif isinstance(top_expression, Call):
            top_expression = top_expression.func

        while isinstance(top_expression, Attribute):
            path.insert(0, top_expression.attr)
            top_expression = top_expression.value

        if not isinstance(top_expression, Name):
            return False

        if top_expression.id in self.imported_names:
            translated = self.imported_names[top_expression.id]
        elif hasattr(builtins, top_expression.id):
            translated = "builtins." + top_expression.id
        else:
            translated = top_expression.id

        path.insert(0, translated)
        joined_path = ".".join(path)
        if joined_path in names:
            return True
        elif self.parent:
            return self.parent.name_matches(expression, *names)
        else:
            return False

    def get_config_keywords(self) -> list[keyword]:
        if self.parent and isinstance(self.parent.node, ClassDef):
            overrides = self.parent.configuration_overrides.copy()
        else:
            overrides = {}

        overrides.update(self.configuration_overrides)
        return [keyword(key, value) for key, value in overrides.items()]


class NameCollector(NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Import(self, node: Import) -> None:
        for name in node.names:
            self.names.add(name.asname or name.name)

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        for name in node.names:
            self.names.add(name.asname or name.name)

    def visit_Assign(self, node: Assign) -> None:
        for target in node.targets:
            if isinstance(target, Name):
                self.names.add(target.id)

    def visit_NamedExpr(self, node: NamedExpr) -> Any:
        if isinstance(node.target, Name):
            self.names.add(node.target.id)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ClassDef) -> None:
        pass


class GeneratorDetector(NodeVisitor):
    """Detects if a function node is a generator function."""

    contains_yields: bool = False
    in_root_function: bool = False

    def visit_Yield(self, node: Yield) -> Any:
        self.contains_yields = True

    def visit_YieldFrom(self, node: YieldFrom) -> Any:
        self.contains_yields = True

    def visit_ClassDef(self, node: ClassDef) -> Any:
        pass

    def visit_FunctionDef(self, node: FunctionDef | AsyncFunctionDef) -> Any:
        if not self.in_root_function:
            self.in_root_function = True
            self.generic_visit(node)
            self.in_root_function = False

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)


class AnnotationTransformer(NodeTransformer):
    type_substitutions: ClassVar[dict[str, tuple[str, str]]] = {
        "builtins.dict": ("typing", "Dict"),
        "builtins.list": ("typing", "List"),
        "builtins.tuple": ("typing", "Tuple"),
        "builtins.set": ("typing", "Set"),
        "builtins.frozenset": ("typing", "FrozenSet"),
    }

    def __init__(self, transformer: TypeguardTransformer):
        self.transformer = transformer
        self._memo = transformer._memo
        self._level = 0

    def visit(self, node: AST) -> Any:
        # Don't process Literals
        if isinstance(node, expr) and self._memo.name_matches(node, *literal_names):
            return node

        self._level += 1
        new_node = super().visit(node)
        self._level -= 1

        if isinstance(new_node, Expression) and not hasattr(new_node, "body"):
            return None

        # Return None if this new node matches a variation of typing.Any
        if (
            self._level == 0
            and isinstance(new_node, expr)
            and self._memo.name_matches(new_node, *anytype_names)
        ):
            return None

        return new_node

    def visit_BinOp(self, node: BinOp) -> Any:
        self.generic_visit(node)

        if isinstance(node.op, BitOr):
            # If either branch of the BinOp has been transformed to `None`, it means
            # that a type in the union was ignored, so the entire annotation should e
            # ignored
            if not hasattr(node, "left") or not hasattr(node, "right"):
                return None

            # Return Any if either side is Any
            if self._memo.name_matches(node.left, *anytype_names):
                return node.left
            elif self._memo.name_matches(node.right, *anytype_names):
                return node.right

            if sys.version_info < (3, 10):
                union_name = self.transformer._get_import("typing", "Union")
                return Subscript(
                    value=union_name,
                    slice=Index(
                        Tuple(elts=[node.left, node.right], ctx=Load()), ctx=Load()
                    ),
                    ctx=Load(),
                )

        return node

    def visit_Attribute(self, node: Attribute) -> Any:
        if self._memo.is_ignored_name(node):
            return None

        return node

    def visit_Subscript(self, node: Subscript) -> Any:
        if self._memo.is_ignored_name(node.value):
            return None

        # The subscript of typing(_extensions).Literal can be any arbitrary string, so
        # don't try to evaluate it as code
        if node.slice:
            if isinstance(node.slice, Index):
                # Python 3.8
                slice_value = node.slice.value  # type: ignore[attr-defined]
            else:
                slice_value = node.slice

            if isinstance(slice_value, Tuple):
                if self._memo.name_matches(node.value, *annotated_names):
                    # Only treat the first argument to typing.Annotated as a potential
                    # forward reference
                    items = cast(
                        typing.List[expr],
                        [self.visit(slice_value.elts[0])] + slice_value.elts[1:],
                    )
                else:
                    items = cast(
                        typing.List[expr],
                        [self.visit(item) for item in slice_value.elts],
                    )

                # If this is a Union and any of the items is Any, erase the entire
                # annotation
                if self._memo.name_matches(node.value, "typing.Union") and any(
                    item is None
                    or (
                        isinstance(item, expr)
                        and self._memo.name_matches(item, *anytype_names)
                    )
                    for item in items
                ):
                    return None

                # If all items in the subscript were Any, erase the subscript entirely
                if all(item is None for item in items):
                    return node.value

                for index, item in enumerate(items):
                    if item is None:
                        items[index] = self.transformer._get_import("typing", "Any")

                slice_value.elts = items
            else:
                self.generic_visit(node)

                # If the transformer erased the slice entirely, just return the node
                # value without the subscript (unless it's Optional, in which case erase
                # the node entirely
                if self._memo.name_matches(
                    node.value, "typing.Optional"
                ) and not hasattr(node, "slice"):
                    return None
                if sys.version_info >= (3, 9) and not hasattr(node, "slice"):
                    return node.value
                elif sys.version_info < (3, 9) and not hasattr(node.slice, "value"):
                    return node.value

        return node

    def visit_Name(self, node: Name) -> Any:
        if self._memo.is_ignored_name(node):
            return None

        if sys.version_info < (3, 9):
            for typename, substitute in self.type_substitutions.items():
                if self._memo.name_matches(node, typename):
                    new_node = self.transformer._get_import(*substitute)
                    return copy_location(new_node, node)

        return node

    def visit_Call(self, node: Call) -> Any:
        # Don't recurse into calls
        return node

    def visit_Constant(self, node: Constant) -> Any:
        if isinstance(node.value, str):
            expression = ast.parse(node.value, mode="eval")
            new_node = self.visit(expression)
            if new_node:
                return copy_location(new_node.body, node)
            else:
                return None

        return node


class TypeguardTransformer(NodeTransformer):
    def __init__(
        self, target_path: Sequence[str] | None = None, target_lineno: int | None = None
    ) -> None:
        self._target_path = tuple(target_path) if target_path else None
        self._memo = self._module_memo = TransformMemo(None, None, ())
        self.names_used_in_annotations: set[str] = set()
        self.target_node: FunctionDef | AsyncFunctionDef | None = None
        self.target_lineno = target_lineno

    def generic_visit(self, node: AST) -> AST:
        has_non_empty_body_initially = bool(getattr(node, "body", None))
        initial_type = type(node)

        node = super().generic_visit(node)

        if (
            type(node) is initial_type
            and has_non_empty_body_initially
            and hasattr(node, "body")
            and not node.body
        ):
            # If we have still the same node type after transformation
            # but we've optimised it's body away, we add a `pass` statement.
            node.body = [Pass()]

        return node

    @contextmanager
    def _use_memo(
        self, node: ClassDef | FunctionDef | AsyncFunctionDef
    ) -> Generator[None, Any, None]:
        new_memo = TransformMemo(node, self._memo, self._memo.path + (node.name,))
        old_memo = self._memo
        self._memo = new_memo

        if isinstance(node, (FunctionDef, AsyncFunctionDef)):
            new_memo.should_instrument = (
                self._target_path is None or new_memo.path == self._target_path
            )
            if new_memo.should_instrument:
                # Check if the function is a generator function
                detector = GeneratorDetector()
                detector.visit(node)

                # Extract yield, send and return types where possible from a subscripted
                # annotation like Generator[int, str, bool]
                return_annotation = deepcopy(node.returns)
                if detector.contains_yields and new_memo.name_matches(
                    return_annotation, *generator_names
                ):
                    if isinstance(return_annotation, Subscript):
                        annotation_slice = return_annotation.slice

                        # Python < 3.9
                        if isinstance(annotation_slice, Index):
                            annotation_slice = (
                                annotation_slice.value  # type: ignore[attr-defined]
                            )

                        if isinstance(annotation_slice, Tuple):
                            items = annotation_slice.elts
                        else:
                            items = [annotation_slice]

                        if len(items) > 0:
                            new_memo.yield_annotation = self._convert_annotation(
                                items[0]
                            )

                        if len(items) > 1:
                            new_memo.send_annotation = self._convert_annotation(
                                items[1]
                            )

                        if len(items) > 2:
                            new_memo.return_annotation = self._convert_annotation(
                                items[2]
                            )
                else:
                    new_memo.return_annotation = self._convert_annotation(
                        return_annotation
                    )

        if isinstance(node, AsyncFunctionDef):
            new_memo.is_async = True

        yield
        self._memo = old_memo

    def _get_import(self, module: str, name: str) -> Name:
        memo = self._memo if self._target_path else self._module_memo
        return memo.get_import(module, name)

    @overload
    def _convert_annotation(self, annotation: None) -> None: ...

    @overload
    def _convert_annotation(self, annotation: expr) -> expr: ...

    def _convert_annotation(self, annotation: expr | None) -> expr | None:
        if annotation is None:
            return None

        # Convert PEP 604 unions (x | y) and generic built-in collections where
        # necessary, and undo forward references
        new_annotation = cast(expr, AnnotationTransformer(self).visit(annotation))
        if isinstance(new_annotation, expr):
            new_annotation = ast.copy_location(new_annotation, annotation)

            # Store names used in the annotation
            names = {node.id for node in walk(new_annotation) if isinstance(node, Name)}
            self.names_used_in_annotations.update(names)

        return new_annotation

    def visit_Name(self, node: Name) -> Name:
        self._memo.local_names.add(node.id)
        return node

    def visit_Module(self, node: Module) -> Module:
        self._module_memo = self._memo = TransformMemo(node, None, ())
        self.generic_visit(node)
        self._module_memo.insert_imports(node)

        fix_missing_locations(node)
        return node

    def visit_Import(self, node: Import) -> Import:
        for name in node.names:
            self._memo.local_names.add(name.asname or name.name)
            self._memo.imported_names[name.asname or name.name] = name.name

        return node

    def visit_ImportFrom(self, node: ImportFrom) -> ImportFrom:
        for name in node.names:
            if name.name != "*":
                alias = name.asname or name.name
                self._memo.local_names.add(alias)
                self._memo.imported_names[alias] = f"{node.module}.{name.name}"

        return node

    def visit_ClassDef(self, node: ClassDef) -> ClassDef | None:
        self._memo.local_names.add(node.name)

        # Eliminate top level classes not belonging to the target path
        if (
            self._target_path is not None
            and not self._memo.path
            and node.name != self._target_path[0]
        ):
            return None

        with self._use_memo(node):
            for decorator in node.decorator_list.copy():
                if self._memo.name_matches(decorator, "typeguard.typechecked"):
                    # Remove the decorator to prevent duplicate instrumentation
                    node.decorator_list.remove(decorator)

                    # Store any configuration overrides
                    if isinstance(decorator, Call) and decorator.keywords:
                        self._memo.configuration_overrides.update(
                            {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
                        )

            self.generic_visit(node)
            return node

    def visit_FunctionDef(
        self, node: FunctionDef | AsyncFunctionDef
    ) -> FunctionDef | AsyncFunctionDef | None:
        """
        Injects type checks for function arguments, and for a return of None if the
        function is annotated to return something else than Any or None, and the body
        ends without an explicit "return".

        """
        self._memo.local_names.add(node.name)

        # Eliminate top level functions not belonging to the target path
        if (
            self._target_path is not None
            and not self._memo.path
            and node.name != self._target_path[0]
        ):
            return None

        # Skip instrumentation if we're instrumenting the whole module and the function
        # contains either @no_type_check or @typeguard_ignore
        if self._target_path is None:
            for decorator in node.decorator_list:
                if self._memo.name_matches(decorator, *ignore_decorators):
                    return node

        with self._use_memo(node):
            arg_annotations: dict[str, Any] = {}
            if self._target_path is None or self._memo.path == self._target_path:
                # Find line number we're supposed to match against
                if node.decorator_list:
                    first_lineno = node.decorator_list[0].lineno
                else:
                    first_lineno = node.lineno

                for decorator in node.decorator_list.copy():
                    if self._memo.name_matches(decorator, "typing.overload"):
                        # Remove overloads entirely
                        return None
                    elif self._memo.name_matches(decorator, "typeguard.typechecked"):
                        # Remove the decorator to prevent duplicate instrumentation
                        node.decorator_list.remove(decorator)

                        # Store any configuration overrides
                        if isinstance(decorator, Call) and decorator.keywords:
                            self._memo.configuration_overrides = {
                                kw.arg: kw.value for kw in decorator.keywords if kw.arg
                            }

                if self.target_lineno == first_lineno:
                    assert self.target_node is None
                    self.target_node = node
                    if node.decorator_list:
                        self.target_lineno = node.decorator_list[0].lineno
                    else:
                        self.target_lineno = node.lineno

                all_args = node.args.args + node.args.kwonlyargs + node.args.posonlyargs

                # Ensure that any type shadowed by the positional or keyword-only
                # argument names are ignored in this function
                for arg in all_args:
                    self._memo.ignored_names.add(arg.arg)

                # Ensure that any type shadowed by the variable positional argument name
                # (e.g. "args" in *args) is ignored this function
                if node.args.vararg:
                    self._memo.ignored_names.add(node.args.vararg.arg)

                # Ensure that any type shadowed by the variable keywrod argument name
                # (e.g. "kwargs" in *kwargs) is ignored this function
                if node.args.kwarg:
                    self._memo.ignored_names.add(node.args.kwarg.arg)

                for arg in all_args:
                    annotation = self._convert_annotation(deepcopy(arg.annotation))
                    if annotation:
                        arg_annotations[arg.arg] = annotation

                if node.args.vararg:
                    annotation_ = self._convert_annotation(node.args.vararg.annotation)
                    if annotation_:
                        if sys.version_info >= (3, 9):
                            container = Name("tuple", ctx=Load())
                        else:
                            container = self._get_import("typing", "Tuple")

                        subscript_slice: Tuple | Index = Tuple(
                            [
                                annotation_,
                                Constant(Ellipsis),
                            ],
                            ctx=Load(),
                        )
                        if sys.version_info < (3, 9):
                            subscript_slice = Index(subscript_slice, ctx=Load())

                        arg_annotations[node.args.vararg.arg] = Subscript(
                            container, subscript_slice, ctx=Load()
                        )

                if node.args.kwarg:
                    annotation_ = self._convert_annotation(node.args.kwarg.annotation)
                    if annotation_:
                        if sys.version_info >= (3, 9):
                            container = Name("dict", ctx=Load())
                        else:
                            container = self._get_import("typing", "Dict")

                        subscript_slice = Tuple(
                            [
                                Name("str", ctx=Load()),
                                annotation_,
                            ],
                            ctx=Load(),
                        )
                        if sys.version_info < (3, 9):
                            subscript_slice = Index(subscript_slice, ctx=Load())

                        arg_annotations[node.args.kwarg.arg] = Subscript(
                            container, subscript_slice, ctx=Load()
                        )

                if arg_annotations:
                    self._memo.variable_annotations.update(arg_annotations)

            self.generic_visit(node)

            if arg_annotations:
                annotations_dict = Dict(
                    keys=[Constant(key) for key in arg_annotations.keys()],
                    values=[
                        Tuple([Name(key, ctx=Load()), annotation], ctx=Load())
                        for key, annotation in arg_annotations.items()
                    ],
                )
                func_name = self._get_import(
                    "typeguard._functions", "check_argument_types"
                )
                args = [
                    self._memo.joined_path,
                    annotations_dict,
                    self._memo.get_memo_name(),
                ]
                node.body.insert(
                    self._memo.code_inject_index, Expr(Call(func_name, args, []))
                )

            # Add a checked "return None" to the end if there's no explicit return
            # Skip if the return annotation is None or Any
            if (
                self._memo.return_annotation
                and (not self._memo.is_async or not self._memo.has_yield_expressions)
                and not isinstance(node.body[-1], Return)
                and (
                    not isinstance(self._memo.return_annotation, Constant)
                    or self._memo.return_annotation.value is not None
                )
            ):
                func_name = self._get_import(
                    "typeguard._functions", "check_return_type"
                )
                return_node = Return(
                    Call(
                        func_name,
                        [
                            self._memo.joined_path,
                            Constant(None),
                            self._memo.return_annotation,
                            self._memo.get_memo_name(),
                        ],
                        [],
                    )
                )

                # Replace a placeholder "pass" at the end
                if isinstance(node.body[-1], Pass):
                    copy_location(return_node, node.body[-1])
                    del node.body[-1]

                node.body.append(return_node)

            # Insert code to create the call memo, if it was ever needed for this
            # function
            if self._memo.memo_var_name:
                memo_kwargs: dict[str, Any] = {}
                if self._memo.parent and isinstance(self._memo.parent.node, ClassDef):
                    for decorator in node.decorator_list:
                        if (
                            isinstance(decorator, Name)
                            and decorator.id == "staticmethod"
                        ):
                            break
                        elif (
                            isinstance(decorator, Name)
                            and decorator.id == "classmethod"
                        ):
                            memo_kwargs["self_type"] = Name(
                                id=node.args.args[0].arg, ctx=Load()
                            )
                            break
                    else:
                        if node.args.args:
                            if node.name == "__new__":
                                memo_kwargs["self_type"] = Name(
                                    id=node.args.args[0].arg, ctx=Load()
                                )
                            else:
                                memo_kwargs["self_type"] = Attribute(
                                    Name(id=node.args.args[0].arg, ctx=Load()),
                                    "__class__",
                                    ctx=Load(),
                                )

                # Construct the function reference
                # Nested functions get special treatment: the function name is added
                # to free variables (and the closure of the resulting function)
                names: list[str] = [node.name]
                memo = self._memo.parent
                while memo:
                    if isinstance(memo.node, (FunctionDef, AsyncFunctionDef)):
                        # This is a nested function. Use the function name as-is.
                        del names[:-1]
                        break
                    elif not isinstance(memo.node, ClassDef):
                        break

                    names.insert(0, memo.node.name)
                    memo = memo.parent

                config_keywords = self._memo.get_config_keywords()
                if config_keywords:
                    memo_kwargs["config"] = Call(
                        self._get_import("dataclasses", "replace"),
                        [self._get_import("typeguard._config", "global_config")],
                        config_keywords,
                    )

                self._memo.memo_var_name.id = self._memo.get_unused_name("memo")
                memo_store_name = Name(id=self._memo.memo_var_name.id, ctx=Store())
                globals_call = Call(Name(id="globals", ctx=Load()), [], [])
                locals_call = Call(Name(id="locals", ctx=Load()), [], [])
                memo_expr = Call(
                    self._get_import("typeguard", "TypeCheckMemo"),
                    [globals_call, locals_call],
                    [keyword(key, value) for key, value in memo_kwargs.items()],
                )
                node.body.insert(
                    self._memo.code_inject_index,
                    Assign([memo_store_name], memo_expr),
                )

                self._memo.insert_imports(node)

                # Special case the __new__() method to create a local alias from the
                # class name to the first argument (usually "cls")
                if (
                    isinstance(node, FunctionDef)
                    and node.args
                    and self._memo.parent is not None
                    and isinstance(self._memo.parent.node, ClassDef)
                    and node.name == "__new__"
                ):
                    first_args_expr = Name(node.args.args[0].arg, ctx=Load())
                    cls_name = Name(self._memo.parent.node.name, ctx=Store())
                    node.body.insert(
                        self._memo.code_inject_index,
                        Assign([cls_name], first_args_expr),
                    )

                # Rmove any placeholder "pass" at the end
                if isinstance(node.body[-1], Pass):
                    del node.body[-1]

        return node

    def visit_AsyncFunctionDef(
        self, node: AsyncFunctionDef
    ) -> FunctionDef | AsyncFunctionDef | None:
        return self.visit_FunctionDef(node)

    def visit_Return(self, node: Return) -> Return:
        """This injects type checks into "return" statements."""
        self.generic_visit(node)
        if (
            self._memo.return_annotation
            and self._memo.should_instrument
            and not self._memo.is_ignored_name(self._memo.return_annotation)
        ):
            func_name = self._get_import("typeguard._functions", "check_return_type")
            old_node = node
            retval = old_node.value or Constant(None)
            node = Return(
                Call(
                    func_name,
                    [
                        self._memo.joined_path,
                        retval,
                        self._memo.return_annotation,
                        self._memo.get_memo_name(),
                    ],
                    [],
                )
            )
            copy_location(node, old_node)

        return node

    def visit_Yield(self, node: Yield) -> Yield | Call:
        """
        This injects type checks into "yield" expressions, checking both the yielded
        value and the value sent back to the generator, when appropriate.

        """
        self._memo.has_yield_expressions = True
        self.generic_visit(node)

        if (
            self._memo.yield_annotation
            and self._memo.should_instrument
            and not self._memo.is_ignored_name(self._memo.yield_annotation)
        ):
            func_name = self._get_import("typeguard._functions", "check_yield_type")
            yieldval = node.value or Constant(None)
            node.value = Call(
                func_name,
                [
                    self._memo.joined_path,
                    yieldval,
                    self._memo.yield_annotation,
                    self._memo.get_memo_name(),
                ],
                [],
            )

        if (
            self._memo.send_annotation
            and self._memo.should_instrument
            and not self._memo.is_ignored_name(self._memo.send_annotation)
        ):
            func_name = self._get_import("typeguard._functions", "check_send_type")
            old_node = node
            call_node = Call(
                func_name,
                [
                    self._memo.joined_path,
                    old_node,
                    self._memo.send_annotation,
                    self._memo.get_memo_name(),
                ],
                [],
            )
            copy_location(call_node, old_node)
            return call_node

        return node

    def visit_AnnAssign(self, node: AnnAssign) -> Any:
        """
        This injects a type check into a local variable annotation-assignment within a
        function body.

        """
        self.generic_visit(node)

        if (
            isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef))
            and node.annotation
            and isinstance(node.target, Name)
        ):
            self._memo.ignored_names.add(node.target.id)
            annotation = self._convert_annotation(deepcopy(node.annotation))
            if annotation:
                self._memo.variable_annotations[node.target.id] = annotation
                if node.value:
                    func_name = self._get_import(
                        "typeguard._functions", "check_variable_assignment"
                    )
                    node.value = Call(
                        func_name,
                        [
                            node.value,
                            Constant(node.target.id),
                            annotation,
                            self._memo.get_memo_name(),
                        ],
                        [],
                    )

        return node

    def visit_Assign(self, node: Assign) -> Any:
        """
        This injects a type check into a local variable assignment within a function
        body. The variable must have been annotated earlier in the function body.

        """
        self.generic_visit(node)

        # Only instrument function-local assignments
        if isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef)):
            targets: list[dict[Constant, expr | None]] = []
            check_required = False
            for target in node.targets:
                elts: Sequence[expr]
                if isinstance(target, Name):
                    elts = [target]
                elif isinstance(target, Tuple):
                    elts = target.elts
                else:
                    continue

                annotations_: dict[Constant, expr | None] = {}
                for exp in elts:
                    prefix = ""
                    if isinstance(exp, Starred):
                        exp = exp.value
                        prefix = "*"

                    if isinstance(exp, Name):
                        self._memo.ignored_names.add(exp.id)
                        name = prefix + exp.id
                        annotation = self._memo.variable_annotations.get(exp.id)
                        if annotation:
                            annotations_[Constant(name)] = annotation
                            check_required = True
                        else:
                            annotations_[Constant(name)] = None

                targets.append(annotations_)

            if check_required:
                # Replace missing annotations with typing.Any
                for item in targets:
                    for key, expression in item.items():
                        if expression is None:
                            item[key] = self._get_import("typing", "Any")

                if len(targets) == 1 and len(targets[0]) == 1:
                    func_name = self._get_import(
                        "typeguard._functions", "check_variable_assignment"
                    )
                    target_varname = next(iter(targets[0]))
                    node.value = Call(
                        func_name,
                        [
                            node.value,
                            target_varname,
                            targets[0][target_varname],
                            self._memo.get_memo_name(),
                        ],
                        [],
                    )
                elif targets:
                    func_name = self._get_import(
                        "typeguard._functions", "check_multi_variable_assignment"
                    )
                    targets_arg = List(
                        [
                            Dict(keys=list(target), values=list(target.values()))
                            for target in targets
                        ],
                        ctx=Load(),
                    )
                    node.value = Call(
                        func_name,
                        [node.value, targets_arg, self._memo.get_memo_name()],
                        [],
                    )

        return node

    def visit_NamedExpr(self, node: NamedExpr) -> Any:
        """This injects a type check into an assignment expression (a := foo())."""
        self.generic_visit(node)

        # Only instrument function-local assignments
        if isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef)) and isinstance(
            node.target, Name
        ):
            self._memo.ignored_names.add(node.target.id)

            # Bail out if no matching annotation is found
            annotation = self._memo.variable_annotations.get(node.target.id)
            if annotation is None:
                return node

            func_name = self._get_import(
                "typeguard._functions", "check_variable_assignment"
            )
            node.value = Call(
                func_name,
                [
                    node.value,
                    Constant(node.target.id),
                    annotation,
                    self._memo.get_memo_name(),
                ],
                [],
            )

        return node

    def visit_AugAssign(self, node: AugAssign) -> Any:
        """
        This injects a type check into an augmented assignment expression (a += 1).

        """
        self.generic_visit(node)

        # Only instrument function-local assignments
        if isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef)) and isinstance(
            node.target, Name
        ):
            # Bail out if no matching annotation is found
            annotation = self._memo.variable_annotations.get(node.target.id)
            if annotation is None:
                return node

            # Bail out if the operator is not found (newer Python version?)
            try:
                operator_func_name = aug_assign_functions[node.op.__class__]
            except KeyError:
                return node

            operator_func = self._get_import("operator", operator_func_name)
            operator_call = Call(
                operator_func, [Name(node.target.id, ctx=Load()), node.value], []
            )
            check_call = Call(
                self._get_import("typeguard._functions", "check_variable_assignment"),
                [
                    operator_call,
                    Constant(node.target.id),
                    annotation,
                    self._memo.get_memo_name(),
                ],
                [],
            )
            return Assign(targets=[node.target], value=check_call)

        return node

    def visit_If(self, node: If) -> Any:
        """
        This blocks names from being collected from a module-level
        "if typing.TYPE_CHECKING:" block, so that they won't be type checked.

        """
        self.generic_visit(node)

        if (
            self._memo is self._module_memo
            and isinstance(node.test, Name)
            and self._memo.name_matches(node.test, "typing.TYPE_CHECKING")
        ):
            collector = NameCollector()
            collector.visit(node)
            self._memo.ignored_names.update(collector.names)

        return node
