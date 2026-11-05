"""The one reading of a ``/rollback`` argument every surface shares: a list number or a checkpoint hash.

``/undo`` prints ``/rollback <hash>`` (a position moves as soon as another checkpoint is taken), and a
short hash can be all digits. A digit string without a leading zero is a list number while it is in
range (``1..N``); any other string of 4+ characters (the hash minimum) is a hash. Shorter out-of-range
digits stay main's "Invalid checkpoint number" error. ``/undo`` never prints an all-digit short hash.
"""
from __future__ import annotations

_MIN_HASH = 4


def resolve_checkpoint_ref(checkpoints: list, ref: str) -> str:
    """The commit ``ref`` names in ``checkpoints`` (``list_checkpoints``, newest first). Raises
    ``ValueError`` for an out-of-range list number; any other string is passed through as a hash
    (``restore``/``diff`` validate it)."""
    ref = (ref or "").strip()
    if ref.isdigit() and not ref.startswith("0"):
        if int(ref) <= len(checkpoints):
            return checkpoints[int(ref) - 1].get("hash", ref)
    if ref.isdigit() and len(ref) < _MIN_HASH:
        raise ValueError(f"Invalid checkpoint number. Use 1-{len(checkpoints)}.")
    return ref
