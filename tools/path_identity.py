"""The one authority on how a filesystem path may be SPELLED, and on whether
two spellings denote the same real file.

Why this module exists
----------------------

Nothing owned that question, so each caller re-derived it with whatever
string comparison was nearest to hand — ``os.path.abspath(a) != abspath(b)``,
``normpath(a) == normpath(b)``, ``realpath(a) == realpath(b)`` — and each
spelling answers a different question. On Windows the first two call one file
two files when the spellings differ only in case or separator, which is how
``shutil.copyfile`` in the TTS single-chunk path could be handed the same file
twice and raise ``SameFileError``. And the verification-artifact cleanup
exemption in ``tools/approval_detection.py`` only knew the native spelling of
the temp dir, so the Git-Bash spelling ``/c/Users/.../Temp/x`` of the same
file was refused, and refused loudly: the leading ``/`` also tripped the
"delete in root path" rule.

What this module owns
---------------------

* **Identity** — whether two operands name the same real file after symlink
  resolution and platform case folding (:func:`denotes_same_file`).
* **The one spelling translation a guard may apply** — Git-Bash/MSYS drive
  paths to their Windows spelling (:func:`windows_spelling_of_msys_path`),
  asked for explicitly at the call site.

What this module is not
-----------------------

* **Not policy.** It answers identity; it never decides what is allowed. Which
  roots are sensitive, which basenames are exempt, whether a mismatch should
  fail open or closed — all of that stays in the guard. That separation is why
  a guard can adopt this without its rejection table changing.
* **Not containment.** ``is_within``-style questions already have an authority:
  :func:`tools.path_security.validate_within_dir` (``resolve()`` +
  ``relative_to``). Adding a second one here would recreate the exact
  duplicate-authority defect this module exists to retire.
* **Not the shell hop.** ``tools/environments/local.py`` emits path spellings
  *for a shell to consume* (``_bash_safe_path`` -> ``/c/...`` for bash itself,
  ``_shell_arg_safe_path`` -> ``C:/...`` for native argv). Those are encoders
  with a target consumer, not identity predicates, and the escaping defect
  recorded against that seam (a search PATTERN, not a path, corrupted by
  MSVCRT re-quoting) is a different bug with a different filed fix. Left alone
  on purpose.
* **Not tilde expansion.** ``~`` resolves against the *effective profile home*,
  which is a Hermes policy decision with a config/env dependency
  (``hermes_constants.get_subprocess_home``). Callers expand first; keeping it
  out is what lets this module stay pure and import-light enough for
  ``tools/approval.py`` to depend on it.

Purity contract
---------------

``os`` and ``re`` only. No logging, no config, no exceptions out: every public
function is total for any input, including ``None``-ish and non-string values.
The sole I/O is ``os.path.realpath`` inside :func:`denotes_same_file`, and it is
error-swallowing — resolution failure degrades to the spelling comparison the
callers used before, never to a raised exception and never to an invented
identity.
"""

from __future__ import annotations

import os
import re
from typing import Optional

__all__ = [
    "denotes_same_file",
    "windows_spelling_of_msys_path",
]


#: A Git-Bash/MSYS absolute path: a single-letter drive component at the root,
#: then the rest of the path. ``/c/Users/...`` is MSYS's spelling of
#: ``C:\Users\...``.
_MSYS_DRIVE_PATH = re.compile(r"/([A-Za-z])/(.+)", re.DOTALL)

_IS_WINDOWS = os.name == "nt"


def windows_spelling_of_msys_path(operand: str) -> Optional[str]:
    """Translate a Git-Bash/MSYS drive path to its Windows spelling, or None.

    ``_find_shell()`` returns Git Bash on Windows, so the cleanup command an
    agent actually writes spells its temp artifact the MSYS way
    (``/c/Users/.../Temp/hermes-verify-x.py``) rather than the native way
    (``C:\\Users\\...\\Temp\\hermes-verify-x.py``). The two spellings name the
    same real file, but only the native one could ever satisfy the exemption in
    ``tools/approval.py``, so routine cleanup was refused on Windows — and
    refused loudly, because a leading ``/`` also trips the "delete in root
    path" rule.

    This is a pure SPELLING translation and nothing else. Its output is meant to
    be handed back through exactly the same checks the native spelling goes
    through, so it cannot exempt anything a native path would not be exempted
    for. It is deliberately narrow:

      * Windows only — on POSIX ``/c/...`` is a real path of its own and
        reinterpreting it as a drive would be a lie about the filesystem.
      * A path already containing a backslash is not a clean MSYS spelling;
        leave it alone rather than guess at a mixed one.

    **Known, deliberate gap.** ``tools/environments/local.py``'s
    ``_msys_to_windows_path`` additionally accepts the Cygwin
    (``/cygdrive/c/...``) and WSL-mount (``/mnt/c/...``) spellings and a bare
    drive root. This one does not, and widening it here would widen every guard
    that consumes it — a loosening, which is not a refactor's to make. The two
    functions answer different questions on purpose: that one prepares a cwd for
    ``Popen``/``isdir`` (be generous, a wrong answer fails loudly), this one
    feeds a security exemption (be exact, a wrong answer fails silently).
    """
    if not _IS_WINDOWS:
        return None
    if not isinstance(operand, str):
        return None
    match = _MSYS_DRIVE_PATH.fullmatch(operand)
    if match is None:
        return None
    drive, rest = match.groups()
    if "\\" in rest:
        return None
    return f"{drive.upper()}:\\" + rest.replace("/", "\\")


def _canonical_spelling(path: str) -> Optional[str]:
    """One path reduced to the form identity is decided on, or None.

    ``realpath`` resolves symlinks and, on Windows, recovers the on-disk case of
    an existing path. ``normcase`` finishes the job for paths that do NOT exist
    yet (a write target, a socket dir that has been removed): on Windows it
    folds case and unifies ``/`` with ``\\``, and on POSIX it is the identity.

    Returns None for an empty or unusable operand — an empty path denotes no
    file, and ``realpath("")`` would otherwise silently hand back the process
    cwd and make two empty operands "the same file".
    """
    if not isinstance(path, str):
        try:
            path = os.fspath(path)
        except TypeError:
            return None
    if not path:
        return None
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        try:
            resolved = os.path.normpath(path)
        except (OSError, ValueError):
            resolved = path
    try:
        return os.path.normcase(resolved)
    except (OSError, ValueError, TypeError):
        return resolved


def denotes_same_file(a: str, b: str) -> bool:
    """Whether *a* and *b* name the same real file.

    Resolution-based, not string-prefix-based: symlinks are followed and, on
    Windows, case is folded, so ``C:\\Temp\\x``, ``c:/temp/x`` and a symlink
    pointing at either all answer True. That is strictly what
    ``os.path.normpath(a) == os.path.normpath(b)`` — the form this replaced —
    could not do.

    Deliberately NOT covered, because covering it silently would widen every
    caller at once:

      * **MSYS spellings.** ``/c/Temp/x`` and ``C:\\Temp\\x`` answer False here.
        A guard that wants that equivalence must ask for it explicitly via
        :func:`windows_spelling_of_msys_path`, so the widening is visible at the
        call site and testable on its own.
      * **Relative operands.** ``realpath`` anchors them to the *process* cwd,
        which is rarely the cwd a tool call means. Pass absolute paths.

    Total: never raises. An empty/unusable operand denotes no file and compares
    equal to nothing, itself included.
    """
    left = _canonical_spelling(a)
    if left is None:
        return False
    right = _canonical_spelling(b)
    if right is None:
        return False
    return left == right
