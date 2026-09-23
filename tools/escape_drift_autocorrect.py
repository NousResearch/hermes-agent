"""Auto-correction for a JSON-double-escaping wire artifact on ``patch`` calls.

``tools/fuzzy_match.py::_detect_backslash_doubling`` already diagnoses, with certainty, the
case where every backslash run in a ``patch`` call's ``old_string`` is exactly twice as long
as in the matched region of the file (the tool-call arguments were JSON-escaped one extra
time) -- but it only rejects the call and asks the model to resend, costing a full round trip.

Applied unconditionally, not gated to any particular model: the correction is safe by
construction rather than by knowing which model is talking. It only ever fires when the
halved ``old_string`` is verified to appear verbatim in the file's actual content -- a model
that deliberately sent a genuinely-doubled backslash (e.g. a correctly escaped ``\\`` in a
regex or shell string) would need that halved form to coincidentally already exist at the
same spot in the file, which the check below rules out; a wrong guess simply fails
verification and falls through to the existing reject-and-ask guard unchanged. An earlier
version of this module gated the correction to a model-name substring match; that gate was
removed because the substring check turned out to be unreliable in practice (the resolved
"model" value at the check site did not reliably match the serving model's real name), and
because the verification-against-file-content step already provides the safety property the
gate was meant to add. Per-model gating can be reintroduced later if a model is found where
the verified correction itself proves unsafe -- nothing observed so far suggests that.

Called once from ``tools/file_operations.py::patch_replace``, before fuzzy matching runs, so
a verified correction produces a clean exact match instead of ever reaching the guard.
"""

from __future__ import annotations

import re
from typing import Tuple

_BACKSLASH_RUN_RE = re.compile(r"\\+")


def _halve_backslash_runs(s: str) -> str:
    """Replace every maximal run of backslashes with half its length (floor).

    Correctness does not depend on this being exactly right for every input -- the caller
    only accepts the result after verifying it appears verbatim in the actual file content,
    so a wrong guess is simply discarded, never applied.
    """
    return _BACKSLASH_RUN_RE.sub(lambda m: "\\" * (len(m.group(0)) // 2), s)


def maybe_correct_backslash_doubling(old_string: str, new_string: str, content: str) -> Tuple[str, str]:
    """Return a corrected ``(old_string, new_string)`` if a doubled-backslash wire artifact is
    detected AND verified against the file's actual content; otherwise return them unchanged.

    Verification requirement: the halved ``old_string`` must appear verbatim in ``content``.
    This is what makes the correction safe to apply automatically -- a wrong guess simply
    fails verification and falls through to the existing reject-and-ask guard unchanged.
    """
    if "\\" not in old_string or old_string in content:
        return old_string, new_string

    halved_old = _halve_backslash_runs(old_string)
    if halved_old == old_string or halved_old not in content:
        return old_string, new_string

    return halved_old, _halve_backslash_runs(new_string)
