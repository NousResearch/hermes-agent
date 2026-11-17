"""Per-call approval: the pre-call verdict every tool dispatch shares, what its card shows, and the
decisions an approval tool's ``apply`` runs under.

Modules: ``gate`` (the verdict: plugin directive, ``approvals.scope: all``, grants, decision
records), ``preview`` (the ``write_file`` / ``patch`` card, rendered from the tool's own plan),
``decisions`` (the exact-args, run-once journal for tools that declare ``preview`` + ``apply``).
Only the names below are cross-package surface.
"""

from tools.approval_gate.decisions import apply_approved
from tools.approval_gate.gate import asks_every_call, pop_record, resolve_pre_tool_block
from tools.approval_gate.preview import patch_preview, take_planned_sha256, write_file_preview

__all__ = [
    "apply_approved",
    "asks_every_call",
    "patch_preview",
    "pop_record",
    "resolve_pre_tool_block",
    "take_planned_sha256",
    "write_file_preview",
]
