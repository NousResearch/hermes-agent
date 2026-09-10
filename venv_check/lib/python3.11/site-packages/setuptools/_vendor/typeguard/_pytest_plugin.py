from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any, Literal

from typeguard._config import CollectionCheckStrategy, ForwardRefPolicy, global_config
from typeguard._exceptions import InstrumentationWarning
from typeguard._importhook import install_import_hook
from typeguard._utils import qualified_name, resolve_reference

if TYPE_CHECKING:
    from pytest import Config, Parser


def pytest_addoption(parser: Parser) -> None:
    def add_ini_option(
        opt_type: (
            Literal["string", "paths", "pathlist", "args", "linelist", "bool"] | None
        ),
    ) -> None:
        parser.addini(
            group.options[-1].names()[0][2:],
            group.options[-1].attrs()["help"],
            opt_type,
        )

    group = parser.getgroup("typeguard")
    group.addoption(
        "--typeguard-packages",
        action="store",
        help="comma separated name list of packages and modules to instrument for "
        "type checking, or :all: to instrument all modules loaded after typeguard",
    )
    add_ini_option("linelist")

    group.addoption(
        "--typeguard-debug-instrumentation",
        action="store_true",
        help="print all instrumented code to stderr",
    )
    add_ini_option("bool")

    group.addoption(
        "--typeguard-typecheck-fail-callback",
        action="store",
        help=(
            "a module:varname (e.g. typeguard:warn_on_error) reference to a function "
            "that is called (with the exception, and memo object as arguments) to "
            "handle a TypeCheckError"
        ),
    )
    add_ini_option("string")

    group.addoption(
        "--typeguard-forward-ref-policy",
        action="store",
        choices=list(ForwardRefPolicy.__members__),
        help=(
            "determines how to deal with unresolveable forward references in type "
            "annotations"
        ),
    )
    add_ini_option("string")

    group.addoption(
        "--typeguard-collection-check-strategy",
        action="store",
        choices=list(CollectionCheckStrategy.__members__),
        help="determines how thoroughly to check collections (list, dict, etc)",
    )
    add_ini_option("string")


def pytest_configure(config: Config) -> None:
    def getoption(name: str) -> Any:
        return config.getoption(name.replace("-", "_")) or config.getini(name)

    packages: list[str] | None = []
    if packages_option := config.getoption("typeguard_packages"):
        packages = [pkg.strip() for pkg in packages_option.split(",")]
    elif packages_ini := config.getini("typeguard-packages"):
        packages = packages_ini

    if packages:
        if packages == [":all:"]:
            packages = None
        else:
            already_imported_packages = sorted(
                package for package in packages if package in sys.modules
            )
            if already_imported_packages:
                warnings.warn(
                    f"typeguard cannot check these packages because they are already "
                    f"imported: {', '.join(already_imported_packages)}",
                    InstrumentationWarning,
                    stacklevel=1,
                )

        install_import_hook(packages=packages)

    debug_option = getoption("typeguard-debug-instrumentation")
    if debug_option:
        global_config.debug_instrumentation = True

    fail_callback_option = getoption("typeguard-typecheck-fail-callback")
    if fail_callback_option:
        callback = resolve_reference(fail_callback_option)
        if not callable(callback):
            raise TypeError(
                f"{fail_callback_option} ({qualified_name(callback.__class__)}) is not "
                f"a callable"
            )

        global_config.typecheck_fail_callback = callback

    forward_ref_policy_option = getoption("typeguard-forward-ref-policy")
    if forward_ref_policy_option:
        forward_ref_policy = ForwardRefPolicy.__members__[forward_ref_policy_option]
        global_config.forward_ref_policy = forward_ref_policy

    collection_check_strategy_option = getoption("typeguard-collection-check-strategy")
    if collection_check_strategy_option:
        collection_check_strategy = CollectionCheckStrategy.__members__[
            collection_check_strategy_option
        ]
        global_config.collection_check_strategy = collection_check_strategy
