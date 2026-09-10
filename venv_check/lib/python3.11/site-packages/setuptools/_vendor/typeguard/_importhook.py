from __future__ import annotations

import ast
import sys
import types
from collections.abc import Callable, Iterable
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from os import PathLike
from types import CodeType, ModuleType, TracebackType
from typing import Sequence, TypeVar
from unittest.mock import patch

from ._config import global_config
from ._transformer import TypeguardTransformer

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

if sys.version_info >= (3, 11):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

if sys.version_info >= (3, 10):
    from importlib.metadata import PackageNotFoundError, version
else:
    from importlib_metadata import PackageNotFoundError, version

try:
    OPTIMIZATION = "typeguard" + "".join(version("typeguard").split(".")[:3])
except PackageNotFoundError:
    OPTIMIZATION = "typeguard"

P = ParamSpec("P")
T = TypeVar("T")


# The name of this function is magical
def _call_with_frames_removed(
    f: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    return f(*args, **kwargs)


def optimized_cache_from_source(path: str, debug_override: bool | None = None) -> str:
    return cache_from_source(path, debug_override, optimization=OPTIMIZATION)


class TypeguardLoader(SourceFileLoader):
    @staticmethod
    def source_to_code(
        data: Buffer | str | ast.Module | ast.Expression | ast.Interactive,
        path: Buffer | str | PathLike[str] = "<string>",
    ) -> CodeType:
        if isinstance(data, (ast.Module, ast.Expression, ast.Interactive)):
            tree = data
        else:
            if isinstance(data, str):
                source = data
            else:
                source = decode_source(data)

            tree = _call_with_frames_removed(
                ast.parse,
                source,
                path,
                "exec",
            )

        tree = TypeguardTransformer().visit(tree)
        ast.fix_missing_locations(tree)

        if global_config.debug_instrumentation and sys.version_info >= (3, 9):
            print(
                f"Source code of {path!r} after instrumentation:\n"
                "----------------------------------------------",
                file=sys.stderr,
            )
            print(ast.unparse(tree), file=sys.stderr)
            print("----------------------------------------------", file=sys.stderr)

        return _call_with_frames_removed(
            compile, tree, path, "exec", 0, dont_inherit=True
        )

    def exec_module(self, module: ModuleType) -> None:
        # Use a custom optimization marker – the import lock should make this monkey
        # patch safe
        with patch(
            "importlib._bootstrap_external.cache_from_source",
            optimized_cache_from_source,
        ):
            super().exec_module(module)


class TypeguardFinder(MetaPathFinder):
    """
    Wraps another path finder and instruments the module with
    :func:`@typechecked <typeguard.typechecked>` if :meth:`should_instrument` returns
    ``True``.

    Should not be used directly, but rather via :func:`~.install_import_hook`.

    .. versionadded:: 2.6
    """

    def __init__(self, packages: list[str] | None, original_pathfinder: MetaPathFinder):
        self.packages = packages
        self._original_pathfinder = original_pathfinder

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        if self.should_instrument(fullname):
            spec = self._original_pathfinder.find_spec(fullname, path, target)
            if spec is not None and isinstance(spec.loader, SourceFileLoader):
                spec.loader = TypeguardLoader(spec.loader.name, spec.loader.path)
                return spec

        return None

    def should_instrument(self, module_name: str) -> bool:
        """
        Determine whether the module with the given name should be instrumented.

        :param module_name: full name of the module that is about to be imported (e.g.
            ``xyz.abc``)

        """
        if self.packages is None:
            return True

        for package in self.packages:
            if module_name == package or module_name.startswith(package + "."):
                return True

        return False


class ImportHookManager:
    """
    A handle that can be used to uninstall the Typeguard import hook.
    """

    def __init__(self, hook: MetaPathFinder):
        self.hook = hook

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        self.uninstall()

    def uninstall(self) -> None:
        """Uninstall the import hook."""
        try:
            sys.meta_path.remove(self.hook)
        except ValueError:
            pass  # already removed


def install_import_hook(
    packages: Iterable[str] | None = None,
    *,
    cls: type[TypeguardFinder] = TypeguardFinder,
) -> ImportHookManager:
    """
    Install an import hook that instruments functions for automatic type checking.

    This only affects modules loaded **after** this hook has been installed.

    :param packages: an iterable of package names to instrument, or ``None`` to
        instrument all packages
    :param cls: a custom meta path finder class
    :return: a context manager that uninstalls the hook on exit (or when you call
        ``.uninstall()``)

    .. versionadded:: 2.6

    """
    if packages is None:
        target_packages: list[str] | None = None
    elif isinstance(packages, str):
        target_packages = [packages]
    else:
        target_packages = list(packages)

    for finder in sys.meta_path:
        if (
            isclass(finder)
            and finder.__name__ == "PathFinder"
            and hasattr(finder, "find_spec")
        ):
            break
    else:
        raise RuntimeError("Cannot find a PathFinder in sys.meta_path")

    hook = cls(target_packages, finder)
    sys.meta_path.insert(0, hook)
    return ImportHookManager(hook)
