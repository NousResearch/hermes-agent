from __future__ import annotations

import sys
import warnings
from typing import Any, Callable, NoReturn, TypeVar, Union, overload

from . import _suppression
from ._checkers import BINARY_MAGIC_METHODS, check_type_internal
from ._config import (
    CollectionCheckStrategy,
    ForwardRefPolicy,
    TypeCheckConfiguration,
)
from ._exceptions import TypeCheckError, TypeCheckWarning
from ._memo import TypeCheckMemo
from ._utils import get_stacklevel, qualified_name

if sys.version_info >= (3, 11):
    from typing import Literal, Never, TypeAlias
else:
    from typing_extensions import Literal, Never, TypeAlias

T = TypeVar("T")
TypeCheckFailCallback: TypeAlias = Callable[[TypeCheckError, TypeCheckMemo], Any]


@overload
def check_type(
    value: object,
    expected_type: type[T],
    *,
    forward_ref_policy: ForwardRefPolicy = ...,
    typecheck_fail_callback: TypeCheckFailCallback | None = ...,
    collection_check_strategy: CollectionCheckStrategy = ...,
) -> T: ...


@overload
def check_type(
    value: object,
    expected_type: Any,
    *,
    forward_ref_policy: ForwardRefPolicy = ...,
    typecheck_fail_callback: TypeCheckFailCallback | None = ...,
    collection_check_strategy: CollectionCheckStrategy = ...,
) -> Any: ...


def check_type(
    value: object,
    expected_type: Any,
    *,
    forward_ref_policy: ForwardRefPolicy = TypeCheckConfiguration().forward_ref_policy,
    typecheck_fail_callback: TypeCheckFailCallback | None = (
        TypeCheckConfiguration().typecheck_fail_callback
    ),
    collection_check_strategy: CollectionCheckStrategy = (
        TypeCheckConfiguration().collection_check_strategy
    ),
) -> Any:
    """
    Ensure that ``value`` matches ``expected_type``.

    The types from the :mod:`typing` module do not support :func:`isinstance` or
    :func:`issubclass` so a number of type specific checks are required. This function
    knows which checker to call for which type.

    This function wraps :func:`~.check_type_internal` in the following ways:

    * Respects type checking suppression (:func:`~.suppress_type_checks`)
    * Forms a :class:`~.TypeCheckMemo` from the current stack frame
    * Calls the configured type check fail callback if the check fails

    Note that this function is independent of the globally shared configuration in
    :data:`typeguard.config`. This means that usage within libraries is safe from being
    affected configuration changes made by other libraries or by the integrating
    application. Instead, configuration options have the same default values as their
    corresponding fields in :class:`TypeCheckConfiguration`.

    :param value: value to be checked against ``expected_type``
    :param expected_type: a class or generic type instance, or a tuple of such things
    :param forward_ref_policy: see :attr:`TypeCheckConfiguration.forward_ref_policy`
    :param typecheck_fail_callback:
        see :attr`TypeCheckConfiguration.typecheck_fail_callback`
    :param collection_check_strategy:
        see :attr:`TypeCheckConfiguration.collection_check_strategy`
    :return: ``value``, unmodified
    :raises TypeCheckError: if there is a type mismatch

    """
    if type(expected_type) is tuple:
        expected_type = Union[expected_type]

    config = TypeCheckConfiguration(
        forward_ref_policy=forward_ref_policy,
        typecheck_fail_callback=typecheck_fail_callback,
        collection_check_strategy=collection_check_strategy,
    )

    if _suppression.type_checks_suppressed or expected_type is Any:
        return value

    frame = sys._getframe(1)
    memo = TypeCheckMemo(frame.f_globals, frame.f_locals, config=config)
    try:
        check_type_internal(value, expected_type, memo)
    except TypeCheckError as exc:
        exc.append_path_element(qualified_name(value, add_class_prefix=True))
        if config.typecheck_fail_callback:
            config.typecheck_fail_callback(exc, memo)
        else:
            raise

    return value


def check_argument_types(
    func_name: str,
    arguments: dict[str, tuple[Any, Any]],
    memo: TypeCheckMemo,
) -> Literal[True]:
    if _suppression.type_checks_suppressed:
        return True

    for argname, (value, annotation) in arguments.items():
        if annotation is NoReturn or annotation is Never:
            exc = TypeCheckError(
                f"{func_name}() was declared never to be called but it was"
            )
            if memo.config.typecheck_fail_callback:
                memo.config.typecheck_fail_callback(exc, memo)
            else:
                raise exc

        try:
            check_type_internal(value, annotation, memo)
        except TypeCheckError as exc:
            qualname = qualified_name(value, add_class_prefix=True)
            exc.append_path_element(f'argument "{argname}" ({qualname})')
            if memo.config.typecheck_fail_callback:
                memo.config.typecheck_fail_callback(exc, memo)
            else:
                raise

    return True


def check_return_type(
    func_name: str,
    retval: T,
    annotation: Any,
    memo: TypeCheckMemo,
) -> T:
    if _suppression.type_checks_suppressed:
        return retval

    if annotation is NoReturn or annotation is Never:
        exc = TypeCheckError(f"{func_name}() was declared never to return but it did")
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise exc

    try:
        check_type_internal(retval, annotation, memo)
    except TypeCheckError as exc:
        # Allow NotImplemented if this is a binary magic method (__eq__() et al)
        if retval is NotImplemented and annotation is bool:
            # This does (and cannot) not check if it's actually a method
            func_name = func_name.rsplit(".", 1)[-1]
            if func_name in BINARY_MAGIC_METHODS:
                return retval

        qualname = qualified_name(retval, add_class_prefix=True)
        exc.append_path_element(f"the return value ({qualname})")
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise

    return retval


def check_send_type(
    func_name: str,
    sendval: T,
    annotation: Any,
    memo: TypeCheckMemo,
) -> T:
    if _suppression.type_checks_suppressed:
        return sendval

    if annotation is NoReturn or annotation is Never:
        exc = TypeCheckError(
            f"{func_name}() was declared never to be sent a value to but it was"
        )
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise exc

    try:
        check_type_internal(sendval, annotation, memo)
    except TypeCheckError as exc:
        qualname = qualified_name(sendval, add_class_prefix=True)
        exc.append_path_element(f"the value sent to generator ({qualname})")
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise

    return sendval


def check_yield_type(
    func_name: str,
    yieldval: T,
    annotation: Any,
    memo: TypeCheckMemo,
) -> T:
    if _suppression.type_checks_suppressed:
        return yieldval

    if annotation is NoReturn or annotation is Never:
        exc = TypeCheckError(f"{func_name}() was declared never to yield but it did")
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise exc

    try:
        check_type_internal(yieldval, annotation, memo)
    except TypeCheckError as exc:
        qualname = qualified_name(yieldval, add_class_prefix=True)
        exc.append_path_element(f"the yielded value ({qualname})")
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise

    return yieldval


def check_variable_assignment(
    value: object, varname: str, annotation: Any, memo: TypeCheckMemo
) -> Any:
    if _suppression.type_checks_suppressed:
        return value

    try:
        check_type_internal(value, annotation, memo)
    except TypeCheckError as exc:
        qualname = qualified_name(value, add_class_prefix=True)
        exc.append_path_element(f"value assigned to {varname} ({qualname})")
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise

    return value


def check_multi_variable_assignment(
    value: Any, targets: list[dict[str, Any]], memo: TypeCheckMemo
) -> Any:
    if max(len(target) for target in targets) == 1:
        iterated_values = [value]
    else:
        iterated_values = list(value)

    if not _suppression.type_checks_suppressed:
        for expected_types in targets:
            value_index = 0
            for ann_index, (varname, expected_type) in enumerate(
                expected_types.items()
            ):
                if varname.startswith("*"):
                    varname = varname[1:]
                    keys_left = len(expected_types) - 1 - ann_index
                    next_value_index = len(iterated_values) - keys_left
                    obj: object = iterated_values[value_index:next_value_index]
                    value_index = next_value_index
                else:
                    obj = iterated_values[value_index]
                    value_index += 1

                try:
                    check_type_internal(obj, expected_type, memo)
                except TypeCheckError as exc:
                    qualname = qualified_name(obj, add_class_prefix=True)
                    exc.append_path_element(f"value assigned to {varname} ({qualname})")
                    if memo.config.typecheck_fail_callback:
                        memo.config.typecheck_fail_callback(exc, memo)
                    else:
                        raise

    return iterated_values[0] if len(iterated_values) == 1 else iterated_values


def warn_on_error(exc: TypeCheckError, memo: TypeCheckMemo) -> None:
    """
    Emit a warning on a type mismatch.

    This is intended to be used as an error handler in
    :attr:`TypeCheckConfiguration.typecheck_fail_callback`.

    """
    warnings.warn(TypeCheckWarning(str(exc)), stacklevel=get_stacklevel())
