"""Filing (``filing.*``), organization (``organization.*``) and assets
(``assets.*``) wire contracts.

The Android daily-driver trio: correction-aware Projects filing, batch
pin/archive with durable undo, and the server-authoritative asset index.
Handlers live in ``methods_filing.py`` / ``methods_organization.py`` /
``methods_assets.py``; the open dicts they mirror (contract audit rows,
per-session failure records) travel as ``JsonValue``.
"""

from __future__ import annotations

from .base import JsonValue, Params
from .common import OpenModel, ProfileParams
from .registry import method

# ── filing ─────────────────────────────────────────────────────────────────────


class FilingStatusResult(OpenModel):
    available: bool
    hook_installed: bool
    contract_path: str
    rules: int
    exclusions: int
    audit_entries: int


class FilingRulesResult(OpenModel):
    rules: list[JsonValue]
    exclusions: list[JsonValue]
    audit: list[JsonValue]


class FilingSuggestionRow(OpenModel):
    session_id: str
    title: str
    cwd: str | None = None
    project_id: str
    project_name: str
    reason: str
    confidence: float


class FilingSuggestResult(OpenModel):
    suggestions: list[FilingSuggestionRow]


class FilingApplyParams(Params):
    path: str
    project: str
    source: str | None = None


class FilingApplyResult(OpenModel):
    applied: bool
    note: str
    db: JsonValue = None


class FilingRejectParams(Params):
    path: str
    project: str | None = None
    source: str | None = None


class FilingRejectResult(OpenModel):
    recorded: bool
    note: str
    db: JsonValue = None


method("filing.status", params=ProfileParams, result=FilingStatusResult,
       doc="Is the filing contract installed; hook presence and counts.")
method("filing.rules", params=ProfileParams, result=FilingRulesResult,
       doc="Contract rules / exclusions with the recent audit trail.")
method("filing.suggest", params=ProfileParams, result=FilingSuggestResult,
       doc="Deterministic filing suggestions (contract rule / cwd match).")
method("filing.apply", params=FilingApplyParams, result=FilingApplyResult,
       doc="Accept: record the rule and mirror it into projects.db.")
method("filing.reject", params=FilingRejectParams, result=FilingRejectResult,
       doc="Reject: record an exclusion with retroactive unfile.")

# ── organization ───────────────────────────────────────────────────────────────


class OrganizationBatchParams(Params):
    session_ids: list[str]
    pinned: bool | None = None
    archived: bool | None = None
    profile: str | None = None


class OrganizationFailureRow(OpenModel):
    session_id: str
    error: str


class OrganizationBatchResult(OpenModel):
    batch_id: str
    applied: list[str]
    failed: list[OrganizationFailureRow]
    partial: bool


class OrganizationHistoryRow(OpenModel):
    id: str
    ts: float
    action: str
    value: int
    count: int
    undone: bool


class OrganizationHistoryResult(OpenModel):
    batches: list[OrganizationHistoryRow]


class OrganizationUndoParams(Params):
    batch_id: str | None = None
    profile: str | None = None


class OrganizationUndoResult(OpenModel):
    batch_id: str | None = None
    restored: list[str]
    failed: list[OrganizationFailureRow]
    partial: bool


method("organization.pin", params=OrganizationBatchParams, result=OrganizationBatchResult,
       doc="Batch pin/unpin with per-session outcomes and a durable undo record.")
method("organization.archive", params=OrganizationBatchParams, result=OrganizationBatchResult,
       doc="Batch archive/unarchive with per-session outcomes and a durable undo record.")
method("organization.history", params=ProfileParams, result=OrganizationHistoryResult,
       doc="List recorded batches, newest last.")
method("organization.undo", params=OrganizationUndoParams, result=OrganizationUndoResult,
       doc="Reverse one batch (by id, or the newest not-yet-undone).")

# ── assets ─────────────────────────────────────────────────────────────────────


class AssetsStatusResult(OpenModel):
    available: bool
    attachments_dir: str
    attachments_present: bool


class AssetRow(OpenModel):
    id: str
    kind: str  # artifact | attachment | media
    name: str
    path: str
    mime_type: str
    size_bytes: int
    modified_at: float
    session_id: str
    session_title: str
    project_id: str | None = None
    project_name: str | None = None


class AssetsListParams(Params):
    kind: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    profile: str | None = None


class AssetsListResult(OpenModel):
    assets: list[AssetRow]
    total: int


method("assets.status", params=ProfileParams, result=AssetsStatusResult,
       doc="Is the server-side asset index reachable.")
method("assets.list", params=AssetsListParams, result=AssetsListResult,
       doc="Index of delivered/attached/generated files, newest first, filterable by "
           "kind, project, or session.")
