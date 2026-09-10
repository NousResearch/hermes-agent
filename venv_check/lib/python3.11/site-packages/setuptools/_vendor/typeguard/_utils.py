from __future__ import annotations

import inspect
import sys
from importlib import import_module
from inspect import currentframe
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, ForwardRef, Union, cast, final
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from ._memo import TypeCheckMemo

if sys.version_info >= (3, 13):
    from typing import get_args, get_origin

    def evaluate_forwardref(forwardref: ForwardRef, memo: TypeCheckMemo) -> Any:
        return forwardref._evaluate(
            memo.globals, memo.locals, type_params=(), recursive_guard=frozenset()
        )

elif sys.version_info >= (3, 10):
    from typing import get_args, get_origin

    def evaluate_forwardref(forwardref: ForwardRef, memo: TypeCheckMemo) -> Any:
        return forwardref._evaluate(
            memo.globals, memo.locals, recursive_guard=frozenset()
        )

else:
    from typing_extensions import get_args, get_origin

    evaluate_extra_args: tuple[frozenset[Any], ...] = (
        (frozenset(),) if sys.version_info >= (3, 9) else ()
    )

    def evaluate_forwardref(forwardref: ForwardRef, memo: TypeCheckMemo) -> Any:
        from ._union_transformer import compile_type_hint, type_substitutions

        if not forwardref.__forward_evaluated__:
            forwardref.__forward_code__ = compile_type_hint(forwardref.__forward_arg__)

        try:
            return forwardref._evaluate(memo.globals, memo.locals, *evaluate_extra_args)
        except NameError:
            if sys.version_info < (3, 10):
                # Try again, with the type substitutions (list -> List etc.) in place
                new_globals = memo.globals.copy()
                new_globals.setdefault("Union", Union)
                if sys.version_info < (3, 9):
                    new_globals.update(type_substitutions)

                return forwardref._evaluate(
                    new_globals, memo.locals or new_globals, *evaluate_extra_args
                )

            raise


_functions_map: WeakValueDictionary[CodeType, FunctionType] = WeakValueDictionary()


def get_type_name(type_: Any) -> str:
    name: str
    for attrname in "__name__", "_name", "__forward_arg__":
        candidate = getattr(type_, attrname, None)
        if isinstance(candidate, str):
            name = candidate
            break
    else:
        origin = get_origin(type_)
        candidate = getattr(origin, "_name", None)
        if candidate is None:
            candidate = type_.__class__.__name__.strip("_")

        if isinstance(candidate, str):
            name = candidate
        else:
            return "(unknown)"

    args = get_args(type_)
    if args:
        if name == "Literal":
            formatted_args = ", ".join(repr(arg) for arg in args)
        else:
            formatted_args = ", ".join(get_type_name(arg) for arg in args)

        name += f"[{formatted_args}]"

    module = getattr(type_, "__module__", None)
    if module and module not in (None, "typing", "typing_extensions", "builtins"):
        name = module + "." + name

    return name


def qualified_name(obj: Any, *, add_class_prefix: bool = False) -> str:
    """
    Return the qualified name (e.g. package.module.Type) for the given object.

    Builtins and types from the :mod:`typing` package get special treatment by having
    the module name stripped from the generated name.

    """
    if obj is None:
        return "None"
    elif inspect.isclass(obj):
        prefix = "class " if add_class_prefix else ""
        type_ = obj
    else:
        prefix = ""
        type_ = type(obj)

    module = type_.__module__
    qualname = type_.__qualname__
    name = qualname if module in ("typing", "builtins") else f"{module}.{qualname}"
    return prefix + name


def function_name(func: Callable[..., Any]) -> str:
    """
    Return the qualified name of the given function.

    Builtins and types from the :mod:`typing` package get special treatment by having
    the module name stripped from the generated name.

    """
    # For partial functions and objects with __call__ defined, __qualname__ does not
    # exist
    module = getattr(func, "__module__", "")
    qualname = (module + ".") if module not in ("builtins", "") else ""
    return qualname + getattr(func, "__qualname__", repr(func))


def resolve_reference(reference: str) -> Any:
    modulename, varname = reference.partition(":")[::2]
    if not modulename or not varname:
        raise ValueError(f"{reference!r} is not a module:varname reference")

    obj = import_module(modulename)
    for attr in varname.split("."):
        obj = getattr(obj, attr)

    return obj


def is_method_of(obj: object, cls: type) -> bool:
    return (
        inspect.isfunction(obj)
        and obj.__module__ == cls.__module__
        and obj.__qualname__.startswith(cls.__qualname__ + ".")
    )


def get_stacklevel() -> int:
    level = 1
    frame = cast(FrameType, currentframe()).f_back
    while frame and frame.f_globals.get("__name__", "").startswith("typeguard."):
        level += 1
        frame = frame.f_back

    return level


@final
class Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


unset = Unset()
