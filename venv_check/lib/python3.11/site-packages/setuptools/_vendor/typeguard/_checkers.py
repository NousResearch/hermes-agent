from __future__ import annotations

import collections.abc
import inspect
import sys
import types
import typing
import warnings
from enum import Enum
from inspect import Parameter, isclass, isfunction
from io import BufferedIOBase, IOBase, RawIOBase, TextIOBase
from textwrap import indent
from typing import (
    IO,
    AbstractSet,
    Any,
    BinaryIO,
    Callable,
    Dict,
    ForwardRef,
    List,
    Mapping,
    MutableMapping,
    NewType,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest.mock import Mock
from weakref import WeakKeyDictionary

try:
    import typing_extensions
except ImportError:
    typing_extensions = None  # type: ignore[assignment]

# Must use this because typing.is_typeddict does not recognize
# TypedDict from typing_extensions, and as of version 4.12.0
# typing_extensions.TypedDict is different from typing.TypedDict
# on all versions.
from typing_extensions import is_typeddict

from ._config import ForwardRefPolicy
from ._exceptions import TypeCheckError, TypeHintWarning
from ._memo import TypeCheckMemo
from ._utils import evaluate_forwardref, get_stacklevel, get_type_name, qualified_name

if sys.version_info >= (3, 11):
    from typing import (
        Annotated,
        NotRequired,
        TypeAlias,
        get_args,
        get_origin,
    )

    SubclassableAny = Any
else:
    from typing_extensions import (
        Annotated,
        NotRequired,
        TypeAlias,
        get_args,
        get_origin,
    )
    from typing_extensions import Any as SubclassableAny

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
    from typing import ParamSpec
else:
    from importlib_metadata import entry_points
    from typing_extensions import ParamSpec

TypeCheckerCallable: TypeAlias = Callable[
    [Any, Any, Tuple[Any, ...], TypeCheckMemo], Any
]
TypeCheckLookupCallback: TypeAlias = Callable[
    [Any, Tuple[Any, ...], Tuple[Any, ...]], Optional[TypeCheckerCallable]
]

checker_lookup_functions: list[TypeCheckLookupCallback] = []
generic_alias_types: tuple[type, ...] = (type(List), type(List[Any]))
if sys.version_info >= (3, 9):
    generic_alias_types += (types.GenericAlias,)

protocol_check_cache: WeakKeyDictionary[
    type[Any], dict[type[Any], TypeCheckError | None]
] = WeakKeyDictionary()

# Sentinel
_missing = object()

# Lifted from mypy.sharedparse
BINARY_MAGIC_METHODS = {
    "__add__",
    "__and__",
    "__cmp__",
    "__divmod__",
    "__div__",
    "__eq__",
    "__floordiv__",
    "__ge__",
    "__gt__",
    "__iadd__",
    "__iand__",
    "__idiv__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__or__",
    "__pow__",
    "__radd__",
    "__rand__",
    "__rdiv__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmatmul__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__sub__",
    "__truediv__",
    "__xor__",
}


def check_callable(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not callable(value):
        raise TypeCheckError("is not callable")

    if args:
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return

        argument_types = args[0]
        if isinstance(argument_types, list) and not any(
            type(item) is ParamSpec for item in argument_types
        ):
            # The callable must not have keyword-only arguments without defaults
            unfulfilled_kwonlyargs = [
                param.name
                for param in signature.parameters.values()
                if param.kind == Parameter.KEYWORD_ONLY
                and param.default == Parameter.empty
            ]
            if unfulfilled_kwonlyargs:
                raise TypeCheckError(
                    f"has mandatory keyword-only arguments in its declaration: "
                    f'{", ".join(unfulfilled_kwonlyargs)}'
                )

            num_positional_args = num_mandatory_pos_args = 0
            has_varargs = False
            for param in signature.parameters.values():
                if param.kind in (
                    Parameter.POSITIONAL_ONLY,
                    Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    num_positional_args += 1
                    if param.default is Parameter.empty:
                        num_mandatory_pos_args += 1
                elif param.kind == Parameter.VAR_POSITIONAL:
                    has_varargs = True

            if num_mandatory_pos_args > len(argument_types):
                raise TypeCheckError(
                    f"has too many mandatory positional arguments in its declaration; "
                    f"expected {len(argument_types)} but {num_mandatory_pos_args} "
                    f"mandatory positional argument(s) declared"
                )
            elif not has_varargs and num_positional_args < len(argument_types):
                raise TypeCheckError(
                    f"has too few arguments in its declaration; expected "
                    f"{len(argument_types)} but {num_positional_args} argument(s) "
                    f"declared"
                )


def check_mapping(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if origin_type is Dict or origin_type is dict:
        if not isinstance(value, dict):
            raise TypeCheckError("is not a dict")
    if origin_type is MutableMapping or origin_type is collections.abc.MutableMapping:
        if not isinstance(value, collections.abc.MutableMapping):
            raise TypeCheckError("is not a mutable mapping")
    elif not isinstance(value, collections.abc.Mapping):
        raise TypeCheckError("is not a mapping")

    if args:
        key_type, value_type = args
        if key_type is not Any or value_type is not Any:
            samples = memo.config.collection_check_strategy.iterate_samples(
                value.items()
            )
            for k, v in samples:
                try:
                    check_type_internal(k, key_type, memo)
                except TypeCheckError as exc:
                    exc.append_path_element(f"key {k!r}")
                    raise

                try:
                    check_type_internal(v, value_type, memo)
                except TypeCheckError as exc:
                    exc.append_path_element(f"value of key {k!r}")
                    raise


def check_typed_dict(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not isinstance(value, dict):
        raise TypeCheckError("is not a dict")

    declared_keys = frozenset(origin_type.__annotations__)
    if hasattr(origin_type, "__required_keys__"):
        required_keys = set(origin_type.__required_keys__)
    else:  # py3.8 and lower
        required_keys = set(declared_keys) if origin_type.__total__ else set()

    existing_keys = set(value)
    extra_keys = existing_keys - declared_keys
    if extra_keys:
        keys_formatted = ", ".join(f'"{key}"' for key in sorted(extra_keys, key=repr))
        raise TypeCheckError(f"has unexpected extra key(s): {keys_formatted}")

    # Detect NotRequired fields which are hidden by get_type_hints()
    type_hints: dict[str, type] = {}
    for key, annotation in origin_type.__annotations__.items():
        if isinstance(annotation, ForwardRef):
            annotation = evaluate_forwardref(annotation, memo)
            if get_origin(annotation) is NotRequired:
                required_keys.discard(key)
                annotation = get_args(annotation)[0]

        type_hints[key] = annotation

    missing_keys = required_keys - existing_keys
    if missing_keys:
        keys_formatted = ", ".join(f'"{key}"' for key in sorted(missing_keys, key=repr))
        raise TypeCheckError(f"is missing required key(s): {keys_formatted}")

    for key, argtype in type_hints.items():
        argvalue = value.get(key, _missing)
        if argvalue is not _missing:
            try:
                check_type_internal(argvalue, argtype, memo)
            except TypeCheckError as exc:
                exc.append_path_element(f"value of key {key!r}")
                raise


def check_list(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not isinstance(value, list):
        raise TypeCheckError("is not a list")

    if args and args != (Any,):
        samples = memo.config.collection_check_strategy.iterate_samples(value)
        for i, v in enumerate(samples):
            try:
                check_type_internal(v, args[0], memo)
            except TypeCheckError as exc:
                exc.append_path_element(f"item {i}")
                raise


def check_sequence(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not isinstance(value, collections.abc.Sequence):
        raise TypeCheckError("is not a sequence")

    if args and args != (Any,):
        samples = memo.config.collection_check_strategy.iterate_samples(value)
        for i, v in enumerate(samples):
            try:
                check_type_internal(v, args[0], memo)
            except TypeCheckError as exc:
                exc.append_path_element(f"item {i}")
                raise


def check_set(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if origin_type is frozenset:
        if not isinstance(value, frozenset):
            raise TypeCheckError("is not a frozenset")
    elif not isinstance(value, AbstractSet):
        raise TypeCheckError("is not a set")

    if args and args != (Any,):
        samples = memo.config.collection_check_strategy.iterate_samples(value)
        for v in samples:
            try:
                check_type_internal(v, args[0], memo)
            except TypeCheckError as exc:
                exc.append_path_element(f"[{v}]")
                raise


def check_tuple(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    # Specialized check for NamedTuples
    if field_types := getattr(origin_type, "__annotations__", None):
        if not isinstance(value, origin_type):
            raise TypeCheckError(
                f"is not a named tuple of type {qualified_name(origin_type)}"
            )

        for name, field_type in field_types.items():
            try:
                check_type_internal(getattr(value, name), field_type, memo)
            except TypeCheckError as exc:
                exc.append_path_element(f"attribute {name!r}")
                raise

        return
    elif not isinstance(value, tuple):
        raise TypeCheckError("is not a tuple")

    if args:
        use_ellipsis = args[-1] is Ellipsis
        tuple_params = args[: -1 if use_ellipsis else None]
    else:
        # Unparametrized Tuple or plain tuple
        return

    if use_ellipsis:
        element_type = tuple_params[0]
        samples = memo.config.collection_check_strategy.iterate_samples(value)
        for i, element in enumerate(samples):
            try:
                check_type_internal(element, element_type, memo)
            except TypeCheckError as exc:
                exc.append_path_element(f"item {i}")
                raise
    elif tuple_params == ((),):
        if value != ():
            raise TypeCheckError("is not an empty tuple")
    else:
        if len(value) != len(tuple_params):
            raise TypeCheckError(
                f"has wrong number of elements (expected {len(tuple_params)}, got "
                f"{len(value)} instead)"
            )

        for i, (element, element_type) in enumerate(zip(value, tuple_params)):
            try:
                check_type_internal(element, element_type, memo)
            except TypeCheckError as exc:
                exc.append_path_element(f"item {i}")
                raise


def check_union(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    errors: dict[str, TypeCheckError] = {}
    try:
        for type_ in args:
            try:
                check_type_internal(value, type_, memo)
                return
            except TypeCheckError as exc:
                errors[get_type_name(type_)] = exc

        formatted_errors = indent(
            "\n".join(f"{key}: {error}" for key, error in errors.items()), "  "
        )
    finally:
        del errors  # avoid creating ref cycle
    raise TypeCheckError(f"did not match any element in the union:\n{formatted_errors}")


def check_uniontype(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    errors: dict[str, TypeCheckError] = {}
    for type_ in args:
        try:
            check_type_internal(value, type_, memo)
            return
        except TypeCheckError as exc:
            errors[get_type_name(type_)] = exc

    formatted_errors = indent(
        "\n".join(f"{key}: {error}" for key, error in errors.items()), "  "
    )
    raise TypeCheckError(f"did not match any element in the union:\n{formatted_errors}")


def check_class(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not isclass(value) and not isinstance(value, generic_alias_types):
        raise TypeCheckError("is not a class")

    if not args:
        return

    if isinstance(args[0], ForwardRef):
        expected_class = evaluate_forwardref(args[0], memo)
    else:
        expected_class = args[0]

    if expected_class is Any:
        return
    elif getattr(expected_class, "_is_protocol", False):
        check_protocol(value, expected_class, (), memo)
    elif isinstance(expected_class, TypeVar):
        check_typevar(value, expected_class, (), memo, subclass_check=True)
    elif get_origin(expected_class) is Union:
        errors: dict[str, TypeCheckError] = {}
        for arg in get_args(expected_class):
            if arg is Any:
                return

            try:
                check_class(value, type, (arg,), memo)
                return
            except TypeCheckError as exc:
                errors[get_type_name(arg)] = exc
        else:
            formatted_errors = indent(
                "\n".join(f"{key}: {error}" for key, error in errors.items()), "  "
            )
            raise TypeCheckError(
                f"did not match any element in the union:\n{formatted_errors}"
            )
    elif not issubclass(value, expected_class):  # type: ignore[arg-type]
        raise TypeCheckError(f"is not a subclass of {qualified_name(expected_class)}")


def check_newtype(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    check_type_internal(value, origin_type.__supertype__, memo)


def check_instance(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not isinstance(value, origin_type):
        raise TypeCheckError(f"is not an instance of {qualified_name(origin_type)}")


def check_typevar(
    value: Any,
    origin_type: TypeVar,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
    *,
    subclass_check: bool = False,
) -> None:
    if origin_type.__bound__ is not None:
        annotation = (
            Type[origin_type.__bound__] if subclass_check else origin_type.__bound__
        )
        check_type_internal(value, annotation, memo)
    elif origin_type.__constraints__:
        for constraint in origin_type.__constraints__:
            annotation = Type[constraint] if subclass_check else constraint
            try:
                check_type_internal(value, annotation, memo)
            except TypeCheckError:
                pass
            else:
                break
        else:
            formatted_constraints = ", ".join(
                get_type_name(constraint) for constraint in origin_type.__constraints__
            )
            raise TypeCheckError(
                f"does not match any of the constraints " f"({formatted_constraints})"
            )


if typing_extensions is None:

    def _is_literal_type(typ: object) -> bool:
        return typ is typing.Literal

else:

    def _is_literal_type(typ: object) -> bool:
        return typ is typing.Literal or typ is typing_extensions.Literal


def check_literal(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    def get_literal_args(literal_args: tuple[Any, ...]) -> tuple[Any, ...]:
        retval: list[Any] = []
        for arg in literal_args:
            if _is_literal_type(get_origin(arg)):
                retval.extend(get_literal_args(arg.__args__))
            elif arg is None or isinstance(arg, (int, str, bytes, bool, Enum)):
                retval.append(arg)
            else:
                raise TypeError(
                    f"Illegal literal value: {arg}"
                )  # TypeError here is deliberate

        return tuple(retval)

    final_args = tuple(get_literal_args(args))
    try:
        index = final_args.index(value)
    except ValueError:
        pass
    else:
        if type(final_args[index]) is type(value):
            return

    formatted_args = ", ".join(repr(arg) for arg in final_args)
    raise TypeCheckError(f"is not any of ({formatted_args})") from None


def check_literal_string(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    check_type_internal(value, str, memo)


def check_typeguard(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    check_type_internal(value, bool, memo)


def check_none(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if value is not None:
        raise TypeCheckError("is not None")


def check_number(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if origin_type is complex and not isinstance(value, (complex, float, int)):
        raise TypeCheckError("is neither complex, float or int")
    elif origin_type is float and not isinstance(value, (float, int)):
        raise TypeCheckError("is neither float or int")


def check_io(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if origin_type is TextIO or (origin_type is IO and args == (str,)):
        if not isinstance(value, TextIOBase):
            raise TypeCheckError("is not a text based I/O object")
    elif origin_type is BinaryIO or (origin_type is IO and args == (bytes,)):
        if not isinstance(value, (RawIOBase, BufferedIOBase)):
            raise TypeCheckError("is not a binary I/O object")
    elif not isinstance(value, IOBase):
        raise TypeCheckError("is not an I/O object")


def check_protocol(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    subject: type[Any] = value if isclass(value) else type(value)

    if subject in protocol_check_cache:
        result_map = protocol_check_cache[subject]
        if origin_type in result_map:
            if exc := result_map[origin_type]:
                raise exc
            else:
                return

    # Collect a set of methods and non-method attributes present in the protocol
    ignored_attrs = set(dir(typing.Protocol)) | {
        "__annotations__",
        "__non_callable_proto_members__",
    }
    expected_methods: dict[str, tuple[Any, Any]] = {}
    expected_noncallable_members: dict[str, Any] = {}
    for attrname in dir(origin_type):
        # Skip attributes present in typing.Protocol
        if attrname in ignored_attrs:
            continue

        member = getattr(origin_type, attrname)
        if callable(member):
            signature = inspect.signature(member)
            argtypes = [
                (p.annotation if p.annotation is not Parameter.empty else Any)
                for p in signature.parameters.values()
                if p.kind is not Parameter.KEYWORD_ONLY
            ] or Ellipsis
            return_annotation = (
                signature.return_annotation
                if signature.return_annotation is not Parameter.empty
                else Any
            )
            expected_methods[attrname] = argtypes, return_annotation
        else:
            expected_noncallable_members[attrname] = member

    for attrname, annotation in typing.get_type_hints(origin_type).items():
        expected_noncallable_members[attrname] = annotation

    subject_annotations = typing.get_type_hints(subject)

    # Check that all required methods are present and their signatures are compatible
    result_map = protocol_check_cache.setdefault(subject, {})
    try:
        for attrname, callable_args in expected_methods.items():
            try:
                method = getattr(subject, attrname)
            except AttributeError:
                if attrname in subject_annotations:
                    raise TypeCheckError(
                        f"is not compatible with the {origin_type.__qualname__} protocol "
                        f"because its {attrname!r} attribute is not a method"
                    ) from None
                else:
                    raise TypeCheckError(
                        f"is not compatible with the {origin_type.__qualname__} protocol "
                        f"because it has no method named {attrname!r}"
                    ) from None

            if not callable(method):
                raise TypeCheckError(
                    f"is not compatible with the {origin_type.__qualname__} protocol "
                    f"because its {attrname!r} attribute is not a callable"
                )

            # TODO: raise exception on added keyword-only arguments without defaults
            try:
                check_callable(method, Callable, callable_args, memo)
            except TypeCheckError as exc:
                raise TypeCheckError(
                    f"is not compatible with the {origin_type.__qualname__} protocol "
                    f"because its {attrname!r} method {exc}"
                ) from None

        # Check that all required non-callable members are present
        for attrname in expected_noncallable_members:
            # TODO: implement assignability checks for non-callable members
            if attrname not in subject_annotations and not hasattr(subject, attrname):
                raise TypeCheckError(
                    f"is not compatible with the {origin_type.__qualname__} protocol "
                    f"because it has no attribute named {attrname!r}"
                )
    except TypeCheckError as exc:
        result_map[origin_type] = exc
        raise
    else:
        result_map[origin_type] = None


def check_byteslike(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not isinstance(value, (bytearray, bytes, memoryview)):
        raise TypeCheckError("is not bytes-like")


def check_self(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if memo.self_type is None:
        raise TypeCheckError("cannot be checked against Self outside of a method call")

    if isclass(value):
        if not issubclass(value, memo.self_type):
            raise TypeCheckError(
                f"is not an instance of the self type "
                f"({qualified_name(memo.self_type)})"
            )
    elif not isinstance(value, memo.self_type):
        raise TypeCheckError(
            f"is not an instance of the self type ({qualified_name(memo.self_type)})"
        )


def check_paramspec(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    pass  # No-op for now


def check_instanceof(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> None:
    if not isinstance(value, origin_type):
        raise TypeCheckError(f"is not an instance of {qualified_name(origin_type)}")


def check_type_internal(
    value: Any,
    annotation: Any,
    memo: TypeCheckMemo,
) -> None:
    """
    Check that the given object is compatible with the given type annotation.

    This function should only be used by type checker callables. Applications should use
    :func:`~.check_type` instead.

    :param value: the value to check
    :param annotation: the type annotation to check against
    :param memo: a memo object containing configuration and information necessary for
        looking up forward references
    """

    if isinstance(annotation, ForwardRef):
        try:
            annotation = evaluate_forwardref(annotation, memo)
        except NameError:
            if memo.config.forward_ref_policy is ForwardRefPolicy.ERROR:
                raise
            elif memo.config.forward_ref_policy is ForwardRefPolicy.WARN:
                warnings.warn(
                    f"Cannot resolve forward reference {annotation.__forward_arg__!r}",
                    TypeHintWarning,
                    stacklevel=get_stacklevel(),
                )

            return

    if annotation is Any or annotation is SubclassableAny or isinstance(value, Mock):
        return

    # Skip type checks if value is an instance of a class that inherits from Any
    if not isclass(value) and SubclassableAny in type(value).__bases__:
        return

    extras: tuple[Any, ...]
    origin_type = get_origin(annotation)
    if origin_type is Annotated:
        annotation, *extras_ = get_args(annotation)
        extras = tuple(extras_)
        origin_type = get_origin(annotation)
    else:
        extras = ()

    if origin_type is not None:
        args = get_args(annotation)

        # Compatibility hack to distinguish between unparametrized and empty tuple
        # (tuple[()]), necessary due to https://github.com/python/cpython/issues/91137
        if origin_type in (tuple, Tuple) and annotation is not Tuple and not args:
            args = ((),)
    else:
        origin_type = annotation
        args = ()

    for lookup_func in checker_lookup_functions:
        checker = lookup_func(origin_type, args, extras)
        if checker:
            checker(value, origin_type, args, memo)
            return

    if isclass(origin_type):
        if not isinstance(value, origin_type):
            raise TypeCheckError(f"is not an instance of {qualified_name(origin_type)}")
    elif type(origin_type) is str:  # noqa: E721
        warnings.warn(
            f"Skipping type check against {origin_type!r}; this looks like a "
            f"string-form forward reference imported from another module",
            TypeHintWarning,
            stacklevel=get_stacklevel(),
        )


# Equality checks are applied to these
origin_type_checkers = {
    bytes: check_byteslike,
    AbstractSet: check_set,
    BinaryIO: check_io,
    Callable: check_callable,
    collections.abc.Callable: check_callable,
    complex: check_number,
    dict: check_mapping,
    Dict: check_mapping,
    float: check_number,
    frozenset: check_set,
    IO: check_io,
    list: check_list,
    List: check_list,
    typing.Literal: check_literal,
    Mapping: check_mapping,
    MutableMapping: check_mapping,
    None: check_none,
    collections.abc.Mapping: check_mapping,
    collections.abc.MutableMapping: check_mapping,
    Sequence: check_sequence,
    collections.abc.Sequence: check_sequence,
    collections.abc.Set: check_set,
    set: check_set,
    Set: check_set,
    TextIO: check_io,
    tuple: check_tuple,
    Tuple: check_tuple,
    type: check_class,
    Type: check_class,
    Union: check_union,
}
if sys.version_info >= (3, 10):
    origin_type_checkers[types.UnionType] = check_uniontype
    origin_type_checkers[typing.TypeGuard] = check_typeguard
if sys.version_info >= (3, 11):
    origin_type_checkers.update(
        {typing.LiteralString: check_literal_string, typing.Self: check_self}
    )
if typing_extensions is not None:
    # On some Python versions, these may simply be re-exports from typing,
    # but exactly which Python versions is subject to change,
    # so it's best to err on the safe side
    # and update the dictionary on all Python versions
    # if typing_extensions is installed
    origin_type_checkers[typing_extensions.Literal] = check_literal
    origin_type_checkers[typing_extensions.LiteralString] = check_literal_string
    origin_type_checkers[typing_extensions.Self] = check_self
    origin_type_checkers[typing_extensions.TypeGuard] = check_typeguard


def builtin_checker_lookup(
    origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]
) -> TypeCheckerCallable | None:
    checker = origin_type_checkers.get(origin_type)
    if checker is not None:
        return checker
    elif is_typeddict(origin_type):
        return check_typed_dict
    elif isclass(origin_type) and issubclass(
        origin_type,
        Tuple,  # type: ignore[arg-type]
    ):
        # NamedTuple
        return check_tuple
    elif getattr(origin_type, "_is_protocol", False):
        return check_protocol
    elif isinstance(origin_type, ParamSpec):
        return check_paramspec
    elif isinstance(origin_type, TypeVar):
        return check_typevar
    elif origin_type.__class__ is NewType:
        # typing.NewType on Python 3.10+
        return check_newtype
    elif (
        isfunction(origin_type)
        and getattr(origin_type, "__module__", None) == "typing"
        and getattr(origin_type, "__qualname__", "").startswith("NewType.")
        and hasattr(origin_type, "__supertype__")
    ):
        # typing.NewType on Python 3.9 and below
        return check_newtype

    return None


checker_lookup_functions.append(builtin_checker_lookup)


def load_plugins() -> None:
    """
    Load all type checker lookup functions from entry points.

    All entry points from the ``typeguard.checker_lookup`` group are loaded, and the
    returned lookup functions are added to :data:`typeguard.checker_lookup_functions`.

    .. note:: This function is called implicitly on import, unless the
        ``TYPEGUARD_DISABLE_PLUGIN_AUTOLOAD`` environment variable is present.
    """

    for ep in entry_points(group="typeguard.checker_lookup"):
        try:
            plugin = ep.load()
        except Exception as exc:
            warnings.warn(
                f"Failed to load plugin {ep.name!r}: " f"{qualified_name(exc)}: {exc}",
                stacklevel=2,
            )
            continue

        if not callable(plugin):
            warnings.warn(
                f"Plugin {ep} returned a non-callable object: {plugin!r}", stacklevel=2
            )
            continue

        checker_lookup_functions.insert(0, plugin)
