"""Approval replay requires the same exact patch anchor the operator reviewed."""
from contextvars import ContextVar

approved_patch_replay: ContextVar[bool] = ContextVar('approved_patch_replay', default=False)


def exact_patch_error(strategy: str | None) -> str | None:
    if strategy == 'exact':
        return None
    return (f"Approved patches require an exact old_string anchor; this patch used {strategy}. "
            "Read the current file and re-stage the patch with its verbatim anchor.")
