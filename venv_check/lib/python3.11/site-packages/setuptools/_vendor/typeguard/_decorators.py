from __future__ import annotations

import ast
import inspect
import sys
from collections.abc import Sequence
from functools import partial
from inspect import isclass, isfunction
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, ForwardRef, TypeVar, cast, overload
from warnings import warn

from ._config import CollectionCheckStrategy, ForwardRefPolicy, global_config
from ._exceptions import InstrumentationWarning
from ._functions import TypeCheckFailCallback
from ._transformer import TypeguardTransformer
from ._utils import Unset, function_name, get_stacklevel, is_method_of, unset

if TYPE_CHECKING:
    from typeshed.stdlib.types import _Cell

    _F = TypeVar("_F")

    def typeguard_ignore(f: _F) -> _F:
        """This decorator is a noop during static type-checking."""
        return f

else:
    from typing import no_type_check as typeguard_ignore  # noqa: F401

T_CallableOrType = TypeVar("T_CallableOrType", bound=Callable[..., Any])


def make_cell(value: object) -> _Cell:
    return (lambda: value).__closure__[0]  # type: ignore[index]


def find_target_function(
    new_code: CodeType, target_path: Sequence[str], firstlineno: int
) -> CodeType | None:
    target_name = target_path[0]
    for const in new_code.co_consts:
        if isinstance(const, CodeType):
            if const.co_name == target_name:
                if const.co_firstlineno == firstlineno:
                    return const
                elif len(target_path) > 1:
                    target_code = find_target_function(
                        const, target_path[1:], firstlineno
                    )
                    if target_code:
                        return target_code

    return None


def instrument(f: T_CallableOrType) -> FunctionType | str:
    if not getattr(f, "__code__", None):
        return "no code associated"
    elif not getattr(f, "__module__", None):
        return "__module__ attribute is not set"
    elif f.__code__.co_filename == "<stdin>":
        return "cannot instrument functions defined in a REPL"
    elif hasattr(f, "__wrapped__"):
        return (
            "@typechecked only supports instrumenting functions wrapped with "
            "@classmethod, @staticmethod or @property"
        )

    target_path = [item for item in f.__qualname__.split(".") if item != "<locals>"]
    module_source = inspect.getsource(sys.modules[f.__module__])
    module_ast = ast.parse(module_source)
    instrumentor = TypeguardTransformer(target_path, f.__code__.co_firstlineno)
    instrumentor.visit(module_ast)

    if not instrumentor.target_node or instrumentor.target_lineno is None:
        return "instrumentor did not find the target function"

    module_code = compile(module_ast, f.__code__.co_filename, "exec", dont_inherit=True)
    new_code = find_target_function(
        module_code, target_path, instrumentor.target_lineno
    )
    if not new_code:
        return "cannot find the target function in the AST"

    if global_config.debug_instrumentation and sys.version_info >= (3, 9):
        # Find the matching AST node, then unparse it to source and print to stdout
        print(
            f"Source code of {f.__qualname__}() after instrumentation:"
            "\n----------------------------------------------",
            file=sys.stderr,
        )
        print(ast.unparse(instrumentor.target_node), file=sys.stderr)
        print(
            "----------------------------------------------",
            file=sys.stderr,
        )

    closure = f.__closure__
    if new_code.co_freevars != f.__code__.co_freevars:
        # Create a new closure and find values for the new free variables
        frame = cast(FrameType, inspect.currentframe())
        frame = cast(FrameType, frame.f_back)
        frame_locals = cast(FrameType, frame.f_back).f_locals
        cells: list[_Cell] = []
        for key in new_code.co_freevars:
            if key in instrumentor.names_used_in_annotations:
                # Find the value and make a new cell from it
                value = frame_locals.get(key) or ForwardRef(key)
                cells.append(make_cell(value))
            else:
                # Reuse the cell from the existing closure
                assert f.__closure__
                cells.append(f.__closure__[f.__code__.co_freevars.index(key)])

        closure = tuple(cells)

    new_function = FunctionType(new_code, f.__globals__, f.__name__, closure=closure)
    new_function.__module__ = f.__module__
    new_function.__name__ = f.__name__
    new_function.__qualname__ = f.__qualname__
    new_function.__annotations__ = f.__annotations__
    new_function.__doc__ = f.__doc__
    new_function.__defaults__ = f.__defaults__
    new_function.__kwdefaults__ = f.__kwdefaults__
    return new_function


@overload
def typechecked(
    *,
    forward_ref_policy: ForwardRefPolicy | Unset = unset,
    typecheck_fail_callback: TypeCheckFailCallback | Unset = unset,
    collection_check_strategy: CollectionCheckStrategy | Unset = unset,
    debug_instrumentation: bool | Unset = unset,
) -> Callable[[T_CallableOrType], T_CallableOrType]: ...


@overload
def typechecked(target: T_CallableOrType) -> T_CallableOrType: ...


def typechecked(
    target: T_CallableOrType | None = None,
    *,
    forward_ref_policy: ForwardRefPolicy | Unset = unset,
    typecheck_fail_callback: TypeCheckFailCallback | Unset = unset,
    collection_check_strategy: CollectionCheckStrategy | Unset = unset,
    debug_instrumentation: bool | Unset = unset,
) -> Any:
    """
    Instrument the target function to perform run-time type checking.

    This decorator recompiles the target function, injecting code to type check
    arguments, return values, yield values (excluding ``yield from``) and assignments to
    annotated local variables.

    This can also be used as a class decorator. This will instrument all type annotated
    methods, including :func:`@classmethod <classmethod>`,
    :func:`@staticmethod <staticmethod>`,  and :class:`@property <property>` decorated
    methods in the class.

    .. note:: When Python is run in optimized mode (``-O`` or ``-OO``, this decorator
        is a no-op). This is a feature meant for selectively introducing type checking
        into a code base where the checks aren't meant to be run in production.

    :param target: the function or class to enable type checking for
    :param forward_ref_policy: override for
        :attr:`.TypeCheckConfiguration.forward_ref_policy`
    :param typecheck_fail_callback: override for
        :attr:`.TypeCheckConfiguration.typecheck_fail_callback`
    :param collection_check_strategy: override for
        :attr:`.TypeCheckConfiguration.collection_check_strategy`
    :param debug_instrumentation: override for
        :attr:`.TypeCheckConfiguration.debug_instrumentation`

    """
    if target is None:
        return partial(
            typechecked,
            forward_ref_policy=forward_ref_policy,
            typecheck_fail_callback=typecheck_fail_callback,
            collection_check_strategy=collection_check_strategy,
            debug_instrumentation=debug_instrumentation,
        )

    if not __debug__:
        return target

    if isclass(target):
        for key, attr in target.__dict__.items():
            if is_method_of(attr, target):
                retval = instrument(attr)
                if isfunction(retval):
                    setattr(target, key, retval)
            elif isinstance(attr, (classmethod, staticmethod)):
                if is_method_of(attr.__func__, target):
                    retval = instrument(attr.__func__)
                    if isfunction(retval):
                        wrapper = attr.__class__(retval)
                        setattr(target, key, wrapper)
            elif isinstance(attr, property):
                kwargs: dict[str, Any] = dict(doc=attr.__doc__)
                for name in ("fset", "fget", "fdel"):
                    property_func = kwargs[name] = getattr(attr, name)
                    if is_method_of(property_func, target):
                        retval = instrument(property_func)
                        if isfunction(retval):
                            kwargs[name] = retval

                setattr(target, key, attr.__class__(**kwargs))

        return target

    # Find either the first Python wrapper or the actual function
    wrapper_class: (
        type[classmethod[Any, Any, Any]] | type[staticmethod[Any, Any]] | None
    ) = None
    if isinstance(target, (classmethod, staticmethod)):
        wrapper_class = target.__class__
        target = target.__func__

    retval = instrument(target)
    if isinstance(retval, str):
        warn(
            f"{retval} -- not typechecking {function_name(target)}",
            InstrumentationWarning,
            stacklevel=get_stacklevel(),
        )
        return target

    if wrapper_class is None:
        return retval
    else:
        return wrapper_class(retval)
