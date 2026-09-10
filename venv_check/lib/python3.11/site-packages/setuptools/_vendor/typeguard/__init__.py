import os
from typing import Any

from ._checkers import TypeCheckerCallable as TypeCheckerCallable
from ._checkers import TypeCheckLookupCallback as TypeCheckLookupCallback
from ._checkers import check_type_internal as check_type_internal
from ._checkers import checker_lookup_functions as checker_lookup_functions
from ._checkers import load_plugins as load_plugins
from ._config import CollectionCheckStrategy as CollectionCheckStrategy
from ._config import ForwardRefPolicy as ForwardRefPolicy
from ._config import TypeCheckConfiguration as TypeCheckConfiguration
from ._decorators import typechecked as typechecked
from ._decorators import typeguard_ignore as typeguard_ignore
from ._exceptions import InstrumentationWarning as InstrumentationWarning
from ._exceptions import TypeCheckError as TypeCheckError
from ._exceptions import TypeCheckWarning as TypeCheckWarning
from ._exceptions import TypeHintWarning as TypeHintWarning
from ._functions import TypeCheckFailCallback as TypeCheckFailCallback
from ._functions import check_type as check_type
from ._functions import warn_on_error as warn_on_error
from ._importhook import ImportHookManager as ImportHookManager
from ._importhook import TypeguardFinder as TypeguardFinder
from ._importhook import install_import_hook as install_import_hook
from ._memo import TypeCheckMemo as TypeCheckMemo
from ._suppression import suppress_type_checks as suppress_type_checks
from ._utils import Unset as Unset

# Re-export imports so they look like they live directly in this package
for value in list(locals().values()):
    if getattr(value, "__module__", "").startswith(f"{__name__}."):
        value.__module__ = __name__


config: TypeCheckConfiguration


def __getattr__(name: str) -> Any:
    if name == "config":
        from ._config import global_config

        return global_config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Automatically load checker lookup functions unless explicitly disabled
if "TYPEGUARD_DISABLE_PLUGIN_AUTOLOAD" not in os.environ:
    load_plugins()
