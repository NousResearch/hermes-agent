from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ._functions import TypeCheckFailCallback

T = TypeVar("T")


class ForwardRefPolicy(Enum):
    """
    Defines how unresolved forward references are handled.

    Members:

    * ``ERROR``: propagate the :exc:`NameError` when the forward reference lookup fails
    * ``WARN``: emit a :class:`~.TypeHintWarning` if the forward reference lookup fails
    * ``IGNORE``: silently skip checks for unresolveable forward references
    """

    ERROR = auto()
    WARN = auto()
    IGNORE = auto()


class CollectionCheckStrategy(Enum):
    """
    Specifies how thoroughly the contents of collections are type checked.

    This has an effect on the following built-in checkers:

    * ``AbstractSet``
    * ``Dict``
    * ``List``
    * ``Mapping``
    * ``Set``
    * ``Tuple[<type>, ...]`` (arbitrarily sized tuples)

    Members:

    * ``FIRST_ITEM``: check only the first item
    * ``ALL_ITEMS``: check all items
    """

    FIRST_ITEM = auto()
    ALL_ITEMS = auto()

    def iterate_samples(self, collection: Iterable[T]) -> Iterable[T]:
        if self is CollectionCheckStrategy.FIRST_ITEM:
            try:
                return [next(iter(collection))]
            except StopIteration:
                return ()
        else:
            return collection


@dataclass
class TypeCheckConfiguration:
    """
     You can change Typeguard's behavior with these settings.

    .. attribute:: typecheck_fail_callback
       :type: Callable[[TypeCheckError, TypeCheckMemo], Any]

         Callable that is called when type checking fails.

         Default: ``None`` (the :exc:`~.TypeCheckError` is raised directly)

    .. attribute:: forward_ref_policy
       :type: ForwardRefPolicy

         Specifies what to do when a forward reference fails to resolve.

         Default: ``WARN``

    .. attribute:: collection_check_strategy
       :type: CollectionCheckStrategy

         Specifies how thoroughly the contents of collections (list, dict, etc.) are
         type checked.

         Default: ``FIRST_ITEM``

    .. attribute:: debug_instrumentation
       :type: bool

         If set to ``True``, the code of modules or functions instrumented by typeguard
         is printed to ``sys.stderr`` after the instrumentation is done

         Requires Python 3.9 or newer.

         Default: ``False``
    """

    forward_ref_policy: ForwardRefPolicy = ForwardRefPolicy.WARN
    typecheck_fail_callback: TypeCheckFailCallback | None = None
    collection_check_strategy: CollectionCheckStrategy = (
        CollectionCheckStrategy.FIRST_ITEM
    )
    debug_instrumentation: bool = False


global_config = TypeCheckConfiguration()
