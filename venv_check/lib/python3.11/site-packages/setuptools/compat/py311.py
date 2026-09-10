from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from _typeshed import ExcInfo, StrOrBytesPath
    from typing_extensions import TypeAlias

# Same as shutil._OnExcCallback from typeshed
_OnExcCallback: TypeAlias = Callable[[Callable[..., Any], str, BaseException], object]


def shutil_rmtree(
    path: StrOrBytesPath,
    ignore_errors: bool = False,
    onexc: _OnExcCallback | None = None,
) -> None:
    if sys.version_info >= (3, 12):
        return shutil.rmtree(path, ignore_errors, onexc=onexc)

    def _handler(fn: Callable[..., Any], path: str, excinfo: ExcInfo) -> None:
        if onexc:
            onexc(fn, path, excinfo[1])

    return shutil.rmtree(path, ignore_errors, onerror=_handler)
