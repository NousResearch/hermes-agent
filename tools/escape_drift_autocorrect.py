"""Model-specific auto-correction for a JSON-double-escaping wire artifact on ``patch`` calls.

``tools/fuzzy_match.py::_detect_backslash_doubling`` already diagnoses, with certainty, the
case where every backslash run in a ``patch`` call's ``old_string`` is exactly twice as long
as in the matched region of the file (the tool-call arguments were JSON-escaped one extra
time) -- but it only rejects the call and asks the model to resend, costing a full round trip.

This is a known, model-specific wire artifact (observed heavily on one model family,
essentially absent on others), so the correction is gated to that model family rather than
applied to every model's calls: a blanket auto-correct could silently mangle a model's
deliberately-doubled backslash (e.g. a correctly escaped ``\\`` in a regex or shell string)
for a model that doesn't actually have this artifact.

Called once from ``tools/file_operations.py::patch_replace``, before fuzzy matching runs, so
a verified correction produces a clean exact match instead of ever reaching the guard.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# Same substring-match convention used elsewhere in this codebase for model-family gating
# (tools/tool_search.py's _DEFAULT_DEFERRED_TOOLS, agent/prompt_builder.py's
# EXECUTION_GUIDANCE_MODELS/TOOL_USE_ENFORCEMENT_MODELS).
_AFFECTED_MODEL_SUBSTRINGS = ("nemotron",)

_BACKSLASH_RUN_RE = re.compile(r"\\+")


def _current_model_name() -> str:
    """Best-effort current model name, empty string on any failure.

    Reads the same ``model`` key from config.yaml that tools/tool_search.py's
    model-gating reads via hermes_cli.config.
    """
    try:
        import hermes_cli.config as _cfg_mod

        config = _cfg_mod.load_config_readonly() or {}
        return str(config.get("model") or "")
    except Exception as exc:
        logger.debug("escape_drift_autocorrect: could not resolve model name: %s", exc)
        return ""


def _is_affected_model() -> bool:
    name = _current_model_name().lower()
    return any(substr in name for substr in _AFFECTED_MODEL_SUBSTRINGS)


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
    if not _is_affected_model():
        return old_string, new_string

    halved_old = _halve_backslash_runs(old_string)
    if halved_old == old_string or halved_old not in content:
        return old_string, new_string

    return halved_old, _halve_backslash_runs(new_string)
