"""Reliability gate for Hermes self-improvement evidence.

This is the small repo-local reliability floor retained after the Hadto-specific
orchestration stack moved to plugins. It deliberately avoids Linear writeback
and cross-repo ontology orchestration; callers provide paths to evidence files
and get a deterministic scorecard back.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import display_hermes_home, get_hermes_home
from tools.registry import registry
from utils import atomic_json_write

logger = logging.getLogger(__name__)


DEFAULT_JOURNAL_PATH = Path("/home/david/stacks/hermes-journal/src/data/journal.json")
DEFAULT_CODEX_RUNS_PATH = get_hermes_home() / "codex" / "runs.json"
DEFAULT_CTX_BINDINGS_PATH = get_hermes_home() / "ctx" / "session_bindings.json"
DEFAULT_ONTOLOGY_ROOT = Path("/home/david/stacks/smb-ontology-platform")
DEFAULT_BENCHMARK_HISTORY_PATH = get_hermes_home() / "self_improvement" / "benchmark_history.json"
DEFAULT_FRESHNESS_HOURS = 72
DEFAULT_ACTIVE_STALE_HOURS = 12
PROVENANCE_CONTRACT_VERSION = "v1"
BENCHMARK_CONTRACT_VERSION = "v1"
JOURNAL_REPORTING_CONTRACT_VERSION = "hermes_journal_self_improvement.v1"
LEADING_INDICATOR_REPORT_CONTRACT_VERSION = "leading_indicator_harbingers.v1"
_BENCHMARK_HISTORY_LIMIT = 200
_EXECUTION_LOOP_WINDOW_DAYS = 14
_EXECUTION_LOOP_MANY_COMPLETED_THRESHOLD = 3
_EXECUTION_LOOP_MIN_JOURNAL_FOLLOW_THROUGH_RATE = 0.5
_THROUGHPUT_CODEX_COMPLETION_MIN = _EXECUTION_LOOP_MANY_COMPLETED_THRESHOLD
_THROUGHPUT_JOURNAL_RATIO_MIN = _EXECUTION_LOOP_MIN_JOURNAL_FOLLOW_THROUGH_RATE
_CAPACITY_SPARE_KEYS = {
    "available_capacity",
    "available_slots",
    "parallel_slots_available",
    "remaining_capacity",
    "spare_capacity",
    "spare_slots",
}
_CAPACITY_MAX_KEYS = {
    "concurrency_limit",
    "desired_parallelism",
    "max_capacity",
    "max_concurrent",
    "max_parallelism",
    "target_capacity",
}
_CAPACITY_ACTIVE_KEYS = {
    "active_count",
    "active_execution_count",
    "active_worker_count",
    "running_count",
}
_BACKLOG_CANDIDATE_KEYS = {
    "backlog_candidates",
    "candidate_backlog_items",
    "candidate_issues",
    "linear_backlog_candidates",
    "linear_candidate_issues",
    "repo_backed_candidates",
    "safe_repo_backed_candidates",
}
_LINEAR_PLANNING_SURFACE_FIELD_ALIASES = {
    "lane": ("lane", "work_lane", "workLane", "planning_lane", "planningLane"),
    "verification": (
        "verification",
        "verification_expectation",
        "verificationExpectation",
        "verification_plan",
        "verificationPlan",
        "verification_targets",
        "verificationTargets",
    ),
    "active_status_comment": (
        "active_status_comment",
        "activeStatusComment",
        "latest_status_comment",
        "latestStatusComment",
        "status_comment",
        "statusComment",
        "status_comments",
        "statusComments",
        "comments",
    ),
}
_LINEAR_PLANNING_SURFACE_SAMPLE_LIMIT = 5
_SELECTED_WORK_KEYS = {
    "active_selected_work",
    "current_selected_work",
    "selected_backlog_item",
    "selected_issue",
    "selected_work",
}
_REPO_BACKED_KEYS = {
    "repo",
    "repo_name",
    "repo_path",
    "repository",
    "repository_name",
    "repository_path",
    "workspace",
    "workspace_path",
}
_REPO_RESOLVED_KEYS = {
    "repo_resolved",
    "repository_resolved",
    "workspace_resolved",
}
_REPO_UNRESOLVED_KEYS = {
    "repo_unresolved",
    "repository_unresolved",
    "workspace_unresolved",
}
_IGNORED_PROJECT_KEYS = {
    "ignored_project",
    "project_ignored",
}
_OWNER_KEYS = {
    "assignee",
    "owner",
    "owner_kind",
    "owner_type",
    "work_owner",
}
_DUPLICATE_KEYS = {
    "duplicate",
    "is_duplicate",
}
_JOURNAL_REPORTING_FOCUS_FIELD = "selfImprovementFocus"
_JOURNAL_REPORTING_FOCUS_REQUIRED_FIELDS = (
    "title",
    "activeLinearIssueIds",
    "outcomeNote",
)
_JOURNAL_REPORTING_OUTCOME_FIELDS = (
    "entryId",
    "occurredAt",
    "title",
    "activeLinearIssueIds",
    "outcomeNote",
)
_JOURNAL_REPORTING_OUTCOME_LIMIT = 4
_JOURNAL_OPERATOR_SUPPORT_ID_LIMIT = 200
_JOURNAL_OPERATOR_SUPPORT_EXAMPLE_LIMIT = 200
_LEADING_INDICATOR_CHECK_IDS = (
    "reliability_gate",
    "anti_make_work_check",
    "operator_value_alignment",
)
_LEADING_INDICATOR_HARBINGERS = (
    "critical_slowing_down",
    "variance_explosion",
    "flickering",
    "correlation_explosion",
)
_LEADING_INDICATOR_WARN_DELTA = -0.01
_LEADING_INDICATOR_FAIL_DELTA = -0.05
_LEGACY_BENCHMARK_CHECK_FIELD_ALIASES = {
    "reliability_gate": ("reliability_gate",),
    "anti_make_work_check": ("anti_make_work_check", "anti_make_work"),
    "operator_value_alignment": ("operator_value_alignment", "operator_value_score"),
}
_HARBINGER_EVIDENCE_FIELDS = {
    "critical_slowing_down": (
        "sample_count",
        "prior_peak",
        "current_score",
        "recovery_gap",
        "recent_deltas",
        "flat_or_negative_delta_count",
    ),
    "variance_explosion": (
        "sample_count",
        "baseline_stddev",
        "recent_stddev",
        "recent_range",
        "recent_scores",
    ),
    "flickering": (
        "sample_count",
        "recent_statuses",
        "transition_count",
        "pass_boundary_crossings",
    ),
    "correlation_explosion": (
        "dropped_check_count",
        "dropped_checks",
        "check_deltas",
        "correlated_drop_threshold",
    ),
}
_BACKLOG_CANDIDATE_ID_KEYS = (
    "id",
    "identifier",
    "key",
    "issue_id",
    "issueId",
    "linear_id",
    "linearIssueId",
)
_BACKLOG_CANDIDATE_REPO_KEYS = (
    "repo",
    "repos",
    "repository",
    "repositories",
    "repository_name",
    "repositoryName",
    "repo_name",
    "repoName",
    "target_repo",
    "targetRepo",
)
_BACKLOG_CANDIDATE_STATUS_KEYS = (
    "status",
    "state",
    "state_type",
    "stateType",
    "resolution",
    "workflow_state",
    "workflowState",
    "workflow_state_type",
    "workflowStateType",
)
_BACKLOG_CANDIDATE_HUMAN_OWNER_LABELS = {"owner:human", "owner=human"}
_BACKLOG_CANDIDATE_HERMES_DELEGATE_LABELS = {
    "delegate:codex",
    "delegate:hermes",
    "delegate=codex",
    "delegate=hermes",
    "delegated:codex",
    "delegated:hermes",
}
_BACKLOG_CANDIDATE_HERMES_DELEGATE_KEYS = (
    "delegate",
    "delegate_to",
    "delegateTo",
    "delegated_to",
    "delegatedTo",
    "delegation",
    "delegation_owner",
    "delegationOwner",
)
_BACKLOG_CANDIDATE_TERMINAL_STATES = {
    "canceled",
    "cancelled",
    "closed",
    "complete",
    "completed",
    "done",
    "merged",
}
_BACKLOG_CANDIDATE_REPO_UNRESOLVED_LABELS = {
    "repo-unresolved",
    "repo unresolved",
    "repository-unresolved",
    "repository unresolved",
}
_BACKLOG_CANDIDATE_IGNORED_PROJECT_LABELS = {
    "ignored-project",
    "ignored project",
    "project:ignored",
    "project=ignored",
}
_BACKLOG_CANDIDATE_SELECTED_KEYS = (
    "selected",
    "selected_for_review",
    "selectedForReview",
    "active_review",
    "activeReview",
)
_ONTOLOGY_SCAN_SUFFIXES = {".json", ".yaml", ".yml", ".md"}
_ONTOLOGY_SCAN_PRUNED_DIRS = {".git", "__pycache__", ".pytest_cache", "tests"}
_ONTOLOGY_REQUIRED_ARTIFACTS = (
    ("ontology_metrics", Path("evolution/metrics.json")),
    ("ontology_delta_report", Path("evolution/delta_report.json")),
    ("ontology_daily_report", Path("evolution/daily_report.md")),
)
_FUTURE_TIMESTAMP_TOLERANCE_SECONDS = 300
_TEXT_EVIDENCE_EXCLUDED_KEYS = {
    "command",
    "prompt",
    "command_args",
    "ctx_worktree_path",
    "latest_path",
    "last_message_path",
    "record_path",
    "workdir",
    "worktree_path",
}
_CLAIM_TEXT_KEYS = {
    "detail",
    "final_message",
    "last_agent_message",
    "notes",
    "outcome_note",
    "reason",
    "result",
    "summary",
    "title",
}
_CLAIM_CONTAINER_KEYS = {
    "active_agenda",
    "current_strategy",
    "lane_links",
    "self_improvement_focus",
}
_CODEX_ISSUE_ID_KEYS = {
    "active_linear_issue_ids",
    "external_key",
    "issue_id",
    "issue_identifier",
    "linear_issue_id",
    "linear_issues",
}
_OPERATOR_DECISION_SUPPORT_STRUCTURED_KEYS = {
    "blocker",
    "blockers",
    "decision",
    "decisions",
    "manual_step",
    "manual_steps",
    "next_decision",
    "next_operator_decision",
    "operator_decision_support",
    "recommended_next_decision",
    "selected_issue",
    "selected_issue_id",
    "selected_work",
    "trade_off",
    "trade_offs",
    "tradeoff",
    "tradeoffs",
}
_OPERATOR_DECISION_SUPPORT_EVIDENCE_FIELDS = {
    "blocker": "blocker",
    "blockers": "blocker",
    "decision": "decision",
    "decisions": "decision",
    "decision_owner": "owner",
    "manual_step": "manual_step",
    "manual_steps": "manual_step",
    "next_decision": "next_decision",
    "next_operator_decision": "next_decision",
    "operator_decision_support": "operator_decision_support",
    "operator_owner": "owner",
    "owner": "owner",
    "owners": "owner",
    "recommended_next_decision": "next_decision",
    "selected_issue": "selected_work",
    "selected_issue_id": "selected_work",
    "selected_work": "selected_work",
    "trade_off": "tradeoff",
    "trade_offs": "tradeoff",
    "tradeoff": "tradeoff",
    "tradeoffs": "tradeoff",
    "work_owner": "owner",
}
_DURABLE_EVIDENCE_KEYS = {
    "artifact_path",
    "artifact_paths",
    "artifacts",
    "changed_files",
    "changed_paths",
    "checks",
    "ci",
    "commit",
    "commit_sha",
    "commit_shas",
    "commits",
    "control_ownership_preserved",
    "control_preservation",
    "control_preserved",
    "decision",
    "decisions",
    "durable_artifacts",
    "evidence",
    "files_changed",
    "incident_risk_reduced",
    "ownership_preservation",
    "ownership_preserved",
    "operator_decision_support",
    "pr_url",
    "pr_urls",
    "proof_artifact",
    "proof_artifacts",
    "pull_request",
    "pull_request_url",
    "pull_requests",
    "risk_reduction",
    "risk_reduced",
    "system_capability_changed",
    "test_results",
    "tests",
    "verification",
    *_OPERATOR_DECISION_SUPPORT_STRUCTURED_KEYS,
}
_ALLOWED_VALUE_CATEGORIES = (
    ("operator_decision_support", "operator decision support"),
    ("durable_asset_created", "durable asset created"),
    ("control_ownership_preserved", "control/ownership preserved"),
    ("incident_risk_reduced", "incident risk reduced"),
    ("system_capability_changed", "system capability changed"),
)
_ALLOWED_VALUE_CATEGORY_LABELS = {
    category: label for category, label in _ALLOWED_VALUE_CATEGORIES
}
_VALUE_CATEGORY_REMEDIATION = {
    "operator_decision_support": (
        "document the operator decision, blocker, trade-off, or manual choice the work enables"
    ),
    "durable_asset_created": (
        "link a commit, PR, changed file, generated artifact, or verification result"
    ),
    "control_ownership_preserved": (
        "show preserved ownership, authority, rollback state, or protected control boundary"
    ),
    "incident_risk_reduced": (
        "name the incident risk and the mitigation, prevention, or recovery evidence"
    ),
    "system_capability_changed": (
        "identify the behavior, tool, workflow, config, schema, or test capability changed"
    ),
}
_CODEX_BACKFILL_EVIDENCE_FIELDS = (
    "operatorDecisionSupport or nextDecision",
    "changedFiles, tests, commitShas, pullRequests, or artifactPaths",
    "controlOwnershipPreserved, incidentRiskReduced, or systemCapabilityChanged when applicable",
)
_CODEX_RUN_ID_PATTERN = re.compile(r"\bcodex_[A-Za-z0-9]+\b")
_LINEAR_ISSUE_ID_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_CODEX_SKIP_NOTE_ACTION_PATTERNS = (
    ("skip", re.compile(r"\bskip(?:ped|ping)?\b", re.IGNORECASE)),
    ("exempt", re.compile(r"\bexempt(?:ed|ion)?\b", re.IGNORECASE)),
    ("remediation", re.compile(r"\bremediat(?:e|ed|ion)\b", re.IGNORECASE)),
)
_CODEX_NON_DURABLE_RATIONALE_PATTERNS = (
    ("non_durable", re.compile(r"\bnon[- ]durable\b", re.IGNORECASE)),
    ("not_durable", re.compile(r"\bnot\s+durable\b", re.IGNORECASE)),
    (
        "without_durable_proof",
        re.compile(r"\bwithout\s+durable\s+(?:delivery\s+)?proof\b", re.IGNORECASE),
    ),
    (
        "no_delivery_evidence",
        re.compile(
            r"\bno\s+[^.\n]{0,160}\b(?:commit|push|pr|pull request|publish|published|publication)\b"
            r"[^.\n]{0,80}\bevidence\b",
            re.IGNORECASE,
        ),
    ),
    ("untracked", re.compile(r"\buntracked\b", re.IGNORECASE)),
    (
        "provenance_gate_failure",
        re.compile(r"\bprovenance\s+gate\s+failure\b", re.IGNORECASE),
    ),
    (
        "empty_final_message",
        re.compile(r"\bfinal[_ ]message\b[^.\n]{0,140}\bempty\b", re.IGNORECASE),
    ),
    (
        "lacks_delivery_details",
        re.compile(
            r"\black(?:s|ed|ing)?\s+(?:enough\s+)?(?:delivery\s+)?details\b",
            re.IGNORECASE,
        ),
    ),
)
_CODEX_TEXT_DELIVERY_SIGNALS = {
    "artifact",
    "capability_change",
    "changed_files",
    "durable_asset_created",
    "state_transition",
    "verification",
}
_CODEX_SIDECAR_MAX_BYTES = 2_000_000
_CODEX_SIDECAR_MESSAGE_FIELDS = (
    "final_message",
    "last_agent_message",
    "output_tail",
)
_CODEX_SIDECAR_STRUCTURED_FIELDS = (
    "artifact_paths",
    "changed_files",
    "changed_paths",
    "commit_shas",
    "commits",
    "control_ownership_preserved",
    "incident_risk_reduced",
    "operator_decision_support",
    "pull_requests",
    "system_capability_changed",
    "test_results",
    "tests",
    "verification",
    "verification_result",
    "verification_targets",
)
_STATUS_ONLY_PATTERNS = (
    re.compile(r"\bactionable\b", re.IGNORECASE),
    re.compile(r"\bactive work\b", re.IGNORECASE),
    re.compile(r"\bin[- ]progress\b", re.IGNORECASE),
    re.compile(r"\bnext steps?\b", re.IGNORECASE),
    re.compile(r"\bqueued\b", re.IGNORECASE),
    re.compile(r"\bselected\b", re.IGNORECASE),
    re.compile(r"\bstatus(?:\s+update)?\b", re.IGNORECASE),
    re.compile(r"\bsummary\b", re.IGNORECASE),
    re.compile(r"\btriage(?:d|s|)\b", re.IGNORECASE),
    re.compile(r"\bworking on\b", re.IGNORECASE),
)
_CODEX_COMPLETED_VALUE_CLAIM_PATTERNS = (
    re.compile(
        r"\b(?:I|we|codex|agent)\s+(?:have\s+)?"
        r"(?:completed|delivered|implemented|fixed|repaired|resolved|created|updated|"
        r"added|removed|changed|opened|merged|pushed|committed|verified|documented|"
        r"backfilled|generated|published)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\n.!?]\s*|[-*]\s*)"
        r"(?:completed|delivered|implemented|fixed|repaired|resolved|created|updated|"
        r"added|removed|changed|opened|merged|pushed|committed|verified|documented|"
        r"backfilled|generated|published)\b",
        re.IGNORECASE,
    ),
)
_DURABLE_TEXT_PATTERNS = (
    ("commit", re.compile(r"\b(?:commit|committed|sha)\b[^.\n]{0,120}\b[0-9a-f]{7,40}\b", re.IGNORECASE)),
    ("commit", re.compile(r"\b[0-9a-f]{7,40}\b[^.\n]{0,120}\b(?:commit|sha)\b", re.IGNORECASE)),
    ("pull_request", re.compile(r"https://github\.com/[^\s)]+/[^\s)]+/pull/\d+", re.IGNORECASE)),
    ("pull_request", re.compile(r"\b(?:PR|pull request)\s*#?\d+\b", re.IGNORECASE)),
    (
        "verification",
        re.compile(
            r"\b(?:pytest|uv run pytest|ruff|mypy|git diff --check|GitHub Actions|CI|"
            r"npm(?:\s+run)?\s+(?:test|lint|build|type-check|typecheck)|"
            r"pnpm\s+(?:test|lint|build|type-check|typecheck)|"
            r"yarn\s+(?:test|lint|build|type-check|typecheck))\b"
            r"[^.\n]{0,160}\b(?:passed|pass|success|succeeded|green|\d+\s+passed|0\s+failed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "changed_files",
        re.compile(
            r"\bCHANGED_FILES\b|\bchanged files?\b|"
            r"(?:^|[\s(])(?:\[[^\]\n]+\."
            r"(?:py|md|mdx|ts|tsx|js|jsx|json|ya?ml|toml|css|html|sh|sql|txt)"
            r"\]\([^)]+\)|(?:[\w.-]+/)+[\w.-]+\."
            r"(?:py|md|mdx|ts|tsx|js|jsx|json|ya?ml|toml|css|html|sh|sql|txt))",
            re.IGNORECASE,
        ),
    ),
    (
        "artifact",
        re.compile(
            r"\b(?:durable|checked-in|repo-visible)\b[^.\n]{0,120}\b(?:artifact|evidence|record)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "durable_asset_created",
        re.compile(
            r"\b(?:created|wrote|added|published|generated)\b"
            r"[^.\n]{0,160}\b(?:asset|artifact|file|record|report|branch|commit|PR|pull request|test)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "state_transition",
        re.compile(
            r"\b(?:merged|pushed|opened|created|closed|resolved|completed)\b"
            r"[^.\n]{0,120}\b(?:PR|pull request|branch|issue|commit|state|artifact|file|test)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operator_decision_support",
        re.compile(
            r"\b(?:blocked|blocker|missing|unavailable|permission|403|401|unable to)\b"
            r"[^.\n]{0,160}\b(?:operator|token|scope|credential|auth|permission|artifact|secret|manual)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operator_decision_support",
        re.compile(
            r"\b(?:operator|human|user)\b[^.\n]{0,160}"
            r"\b(?:decision|decide|choose|approval|blocker|risk|trade[- ]off|manual step|recommended)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operator_decision_support",
        re.compile(
            r"\b(?:decision|decide|choose|approval|blocker|risk|trade[- ]off|manual step|recommended)\b"
            r"[^.\n]{0,160}\b(?:operator|human|user)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "control_ownership_preserved",
        re.compile(
            r"\b(?:preserved|retained|kept|protected)\b"
            r"[^.\n]{0,160}\b(?:control|ownership|authority|rollback|handoff|state|boundary)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "control_ownership_preserved",
        re.compile(
            r"\b(?:control|ownership|authority|rollback|handoff|state|boundary)\b"
            r"[^.\n]{0,160}\b(?:preserved|retained|kept|protected)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "incident_risk_reduced",
        re.compile(
            r"\b(?:reduced|mitigated|lowered|prevented|removed|closed)\b"
            r"[^.\n]{0,160}\b(?:incident|risk|outage|regression|failure|security|data loss|rollback)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "incident_risk_reduced",
        re.compile(
            r"\b(?:incident|risk|outage|regression|failure|security|data loss|rollback)\b"
            r"[^.\n]{0,160}\b(?:reduced|mitigated|lowered|prevented|removed|closed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "capability_change",
        re.compile(
            r"\b(?:added|implemented|fixed|hardened|repaired|wired|enabled)\b"
            r"[^.\n]{0,160}\b(?:tool|runtime|service|gateway|config|workflow|benchmark|check|test|schema)\b",
            re.IGNORECASE,
        ),
    ),
)
_OPERATOR_DECISION_SUPPORT_SIGNALS = {
    "decision",
    "decisions",
    "operator_decision_support",
    "risk_reduction",
    *_OPERATOR_DECISION_SUPPORT_STRUCTURED_KEYS,
}
_VERIFIED_SYSTEM_CHANGE_SIGNALS = {
    "artifact",
    "artifact_path",
    "artifact_paths",
    "artifacts",
    "capability_change",
    "changed_files",
    "changed_paths",
    "checks",
    "ci",
    "commit",
    "commit_sha",
    "commit_shas",
    "commits",
    "durable_artifacts",
    "evidence",
    "files_changed",
    "pr_url",
    "pr_urls",
    "proof_artifact",
    "proof_artifacts",
    "pull_request",
    "pull_request_url",
    "pull_requests",
    "state_transition",
    "test_results",
    "tests",
    "verification",
}
_VALUE_CATEGORY_SIGNAL_MAP = {
    "operator_decision_support": _OPERATOR_DECISION_SUPPORT_SIGNALS,
    "durable_asset_created": {
        "artifact",
        "artifact_path",
        "artifact_paths",
        "artifacts",
        "changed_files",
        "changed_paths",
        "checks",
        "ci",
        "commit",
        "commit_sha",
        "commit_shas",
        "commits",
        "durable_artifacts",
        "durable_asset_created",
        "evidence",
        "files_changed",
        "pr_url",
        "pr_urls",
        "proof_artifact",
        "proof_artifacts",
        "pull_request",
        "pull_request_url",
        "pull_requests",
        "state_transition",
        "test_results",
        "tests",
        "verification",
    },
    "control_ownership_preserved": {
        "control_ownership_preserved",
        "control_preservation",
        "control_preserved",
        "ownership_preservation",
        "ownership_preserved",
    },
    "incident_risk_reduced": {
        "incident_risk_reduced",
        "risk_reduced",
        "risk_reduction",
    },
    "system_capability_changed": {
        "capability_change",
        "changed_files",
        "changed_paths",
        "files_changed",
        "state_transition",
        "system_capability_changed",
    },
}
_CODEX_SUCCESS_STATUSES = {
    "completed",
    "complete",
    "done",
    "finished",
    "success",
    "succeeded",
}
_CODEX_FAILURE_STATUSES = {
    "aborted",
    "cancelled",
    "canceled",
    "error",
    "errored",
    "failed",
    "failure",
    "killed",
    "stale",
    "timed_out",
    "timeout",
}


SELF_IMPROVEMENT_EVIDENCE_SCHEMA = {
    "name": "self_improvement_evidence_gate",
    "description": (
        "Evaluate freshness and consistency of Hermes self-improvement evidence "
        "without creating or updating backlog work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "journal_path": {"type": "string"},
            "codex_runs_path": {"type": "string"},
            "ctx_bindings_path": {"type": "string"},
            "ontology_root": {"type": "string"},
            "now": {"type": "string"},
            "freshness_hours": {"type": "integer", "minimum": 1},
            "active_stale_hours": {"type": "integer", "minimum": 1},
        },
        "required": [],
    },
}


SELF_IMPROVEMENT_BENCHMARK_SCHEMA = {
    "name": "self_improvement_benchmark",
    "description": (
        "Score the Hermes self-improvement reliability floor and optionally "
        f"persist benchmark history under {display_hermes_home()}/self_improvement."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "journal_path": {"type": "string"},
            "codex_runs_path": {"type": "string"},
            "ctx_bindings_path": {"type": "string"},
            "ontology_root": {"type": "string"},
            "history_path": {"type": "string"},
            "now": {"type": "string"},
            "freshness_hours": {"type": "integer", "minimum": 1},
            "active_stale_hours": {"type": "integer", "minimum": 1},
            "persist": {"type": "boolean"},
        },
        "required": [],
    },
}


SELF_IMPROVEMENT_PIPELINE_SCHEMA = {
    "name": "self_improvement_pipeline",
    "description": (
        "Run the Hermes self-improvement reliability pipeline using the "
        "repo-local benchmark contract without Linear writeback."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "journal_path": {"type": "string"},
            "codex_runs_path": {"type": "string"},
            "ctx_bindings_path": {"type": "string"},
            "ontology_root": {"type": "string"},
            "history_path": {"type": "string"},
            "now": {"type": "string"},
            "freshness_hours": {"type": "integer", "minimum": 1},
            "active_stale_hours": {"type": "integer", "minimum": 1},
            "persist": {"type": "boolean"},
            "candidate_limit": {"type": "integer", "minimum": 1},
            "available_capacity": {"type": "integer", "minimum": 0},
            "selected_candidate_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "backlog_candidates": {
                "type": "array",
                "items": {"type": "object"},
            },
            "auto_repair_linear": {"type": "boolean"},
            "auto_close_resolved": {"type": "boolean"},
        },
        "required": [],
    },
}


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read JSON evidence file %s", path, exc_info=True)
        return None


def _iter_records(payload: Any, key: str) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return
        if all(not isinstance(item, (dict, list)) for item in payload.values()):
            yield payload
            return
        for item in payload.values():
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict):
                        yield child
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _is_codex_sidecar_path(path: Path, *, suffixes: set[str]) -> bool:
    if path.suffix not in suffixes:
        return False
    parts = path.parts
    if len(parts) < 3:
        return False
    return parts[-2] == "hermes-codex" and parts[-3] == ".git"


def _read_codex_sidecar_text(value: Any, *, suffixes: set[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not _is_codex_sidecar_path(path, suffixes=suffixes):
        return None
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > _CODEX_SIDECAR_MAX_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Failed to read Codex sidecar evidence file %s", path, exc_info=True)
        return None


def _merge_codex_sidecar_fields(record: dict[str, Any], sidecar: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    for field in _CODEX_SIDECAR_MESSAGE_FIELDS:
        value = sidecar.get(field)
        if not _value_has_content(value):
            continue
        current_value = record.get(field)
        if not _value_has_content(current_value):
            record[field] = value
            merged.append(field)
            continue
        if (
            isinstance(value, str)
            and isinstance(current_value, str)
            and value != current_value
            and _text_durable_signals(value)
            and not _text_durable_signals(current_value)
        ):
            sidecar_field = f"sidecar_{field}"
            record[sidecar_field] = value
            merged.append(sidecar_field)

    for field in _CODEX_SIDECAR_STRUCTURED_FIELDS:
        value = sidecar.get(field)
        if _value_has_content(value) and not _value_has_content(record.get(field)):
            record[field] = value
            merged.append(field)
    return merged


def _hydrate_codex_sidecar_record(record: dict[str, Any]) -> dict[str, Any]:
    hydrated = dict(record)
    hydrated_fields: list[str] = []

    record_text = "\n".join(_collect_record_text(hydrated))
    if _structured_durable_signals(hydrated) or _text_durable_signals(record_text):
        return hydrated

    record_path_text = _read_codex_sidecar_text(
        hydrated.get("record_path"),
        suffixes={".json"},
    )
    if record_path_text:
        try:
            sidecar = json.loads(record_path_text)
        except json.JSONDecodeError:
            sidecar = None
        if isinstance(sidecar, dict):
            hydrated_fields.extend(_merge_codex_sidecar_fields(hydrated, sidecar))

    last_message = _read_codex_sidecar_text(
        hydrated.get("last_message_path"),
        suffixes={".txt"},
    )
    if last_message and _text_durable_signals(last_message):
        current_final = str(hydrated.get("final_message") or "")
        if not _text_durable_signals(current_final):
            hydrated["sidecar_last_message"] = last_message
            hydrated_fields.append("sidecar_last_message")

    if hydrated_fields:
        hydrated["codex_sidecar_hydrated_fields"] = sorted(set(hydrated_fields))
    return hydrated


def _record_timestamp(record: dict[str, Any], *keys: str) -> Optional[datetime]:
    for key in keys:
        parsed = _parse_time(record.get(key))
        if parsed is not None:
            return parsed
    return None


def _iter_journal_timestamps(payload: Any) -> Iterable[datetime]:
    for record in _iter_records(payload, "entries"):
        parsed = _record_timestamp(
            record,
            "occurredAt",
            "occurred_at",
            "updatedAt",
            "updated_at",
            "createdAt",
            "created_at",
            "timestamp",
            "date",
        )
        if parsed is not None:
            yield parsed


def _journal_reporting_schema() -> dict[str, Any]:
    return {
        "schema_name": "Hermes Journal self-improvement reporting",
        "entry_collection": "entries",
        "entry_id_field": "id",
        "entry_timestamp_field": "occurredAt",
        "focus_field": _JOURNAL_REPORTING_FOCUS_FIELD,
        "focus_item_required_fields": list(_JOURNAL_REPORTING_FOCUS_REQUIRED_FIELDS),
        "recent_outcome_fields": list(_JOURNAL_REPORTING_OUTCOME_FIELDS),
        "recent_outcomes_derive_from": (
            f"entries[].{_JOURNAL_REPORTING_FOCUS_FIELD}[]"
        ),
    }


def _journal_reporting_violation(path: str, issue: str, expected: str) -> dict[str, str]:
    return {
        "path": path,
        "issue": issue,
        "expected": expected,
    }


def _journal_reporting_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _journal_reporting_issue_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    issue_ids: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            issue_ids.append(item.strip())
    return issue_ids


def _journal_reporting_timestamp(record: dict[str, Any]) -> Optional[datetime]:
    return _record_timestamp(
        record,
        "occurredAt",
        "occurred_at",
        "updatedAt",
        "updated_at",
        "createdAt",
        "created_at",
        "timestamp",
        "date",
    )


def _journal_reporting_entry_time_text(record: dict[str, Any]) -> str:
    for key in (
        "occurredAt",
        "occurred_at",
        "updatedAt",
        "updated_at",
        "createdAt",
        "created_at",
        "timestamp",
        "date",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_journal_reporting_contract(
    payload: Any,
    *,
    outcome_limit: int = _JOURNAL_REPORTING_OUTCOME_LIMIT,
) -> dict[str, Any]:
    schema = _journal_reporting_schema()
    if payload is None:
        return {
            "contract_version": JOURNAL_REPORTING_CONTRACT_VERSION,
            "status": "missing",
            "detail": "Journal evidence is unavailable.",
            "schema": schema,
            "active_focus_entry_id": None,
            "active_focus": [],
            "recent_outcomes": [],
            "violations": [
                _journal_reporting_violation(
                    "entries",
                    "missing_journal_payload",
                    "journal JSON with entries[]",
                )
            ],
        }

    entries = list(_iter_records(payload, "entries"))
    violations: list[dict[str, str]] = []
    focus_entries: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []

    for entry_index, entry in enumerate(entries):
        focus_value = entry.get(_JOURNAL_REPORTING_FOCUS_FIELD)
        if focus_value is None:
            continue
        focus_path = f"entries[{entry_index}].{_JOURNAL_REPORTING_FOCUS_FIELD}"
        if not isinstance(focus_value, list):
            violations.append(
                _journal_reporting_violation(
                    focus_path,
                    "invalid_focus_container",
                    "array of focus items",
                )
            )
            continue

        entry_id = _journal_reporting_string(entry.get("id"))
        occurred_at = _journal_reporting_entry_time_text(entry)
        timestamp = _journal_reporting_timestamp(entry)
        if not entry_id:
            violations.append(
                _journal_reporting_violation(
                    f"entries[{entry_index}].id",
                    "missing_required_field",
                    "non-empty string",
                )
            )
        if timestamp is None:
            violations.append(
                _journal_reporting_violation(
                    f"entries[{entry_index}].occurredAt",
                    "missing_or_invalid_timestamp",
                    "ISO-8601 timestamp",
                )
            )

        entry_focus: list[dict[str, Any]] = []
        for focus_index, focus_item in enumerate(focus_value):
            item_path = f"{focus_path}[{focus_index}]"
            if not isinstance(focus_item, dict):
                violations.append(
                    _journal_reporting_violation(
                        item_path,
                        "invalid_focus_item",
                        "object with title, activeLinearIssueIds, and outcomeNote",
                    )
                )
                continue

            title = _journal_reporting_string(focus_item.get("title"))
            issue_value = focus_item.get("activeLinearIssueIds")
            active_issue_ids = _journal_reporting_issue_ids(issue_value)
            outcome_note = _journal_reporting_string(focus_item.get("outcomeNote"))
            if not title:
                violations.append(
                    _journal_reporting_violation(
                        f"{item_path}.title",
                        "missing_required_field",
                        "non-empty string",
                    )
                )
            if not isinstance(issue_value, list) or len(active_issue_ids) != len(issue_value):
                violations.append(
                    _journal_reporting_violation(
                        f"{item_path}.activeLinearIssueIds",
                        "invalid_required_field",
                        "array of non-empty strings",
                    )
                )
            if not outcome_note:
                violations.append(
                    _journal_reporting_violation(
                        f"{item_path}.outcomeNote",
                        "missing_required_field",
                        "non-empty string",
                    )
                )
            if not (entry_id and occurred_at and timestamp and title and outcome_note):
                continue
            if not isinstance(issue_value, list) or len(active_issue_ids) != len(issue_value):
                continue

            focus_contract = {
                "title": title,
                "activeLinearIssueIds": active_issue_ids,
                "outcomeNote": outcome_note,
            }
            entry_focus.append(focus_contract)
            outcome = {
                "entryId": entry_id,
                "occurredAt": occurred_at,
                **focus_contract,
            }
            outcomes.append(
                {
                    "_timestamp": timestamp,
                    **outcome,
                }
            )

        if entry_focus and timestamp is not None:
            focus_entries.append(
                {
                    "entry_id": entry_id,
                    "timestamp": timestamp,
                    "focus": entry_focus,
                }
            )

    focus_entries.sort(key=lambda item: item["timestamp"], reverse=True)
    outcomes.sort(key=lambda item: item["_timestamp"], reverse=True)
    recent_outcomes = [
        {field: item[field] for field in _JOURNAL_REPORTING_OUTCOME_FIELDS}
        for item in outcomes[: max(1, int(outcome_limit or 1))]
    ]
    active_focus_entry = focus_entries[0] if focus_entries else None

    if not entries:
        status = "missing"
        detail = "Journal evidence does not contain entries[]."
    elif not outcomes:
        status = "warn"
        detail = "Journal entries do not expose reusable self-improvement focus outcomes."
    elif violations:
        status = "warn"
        detail = "Journal reporting contract is usable but has schema violations."
    else:
        status = "pass"
        detail = "Journal self-improvement focus items provide reusable recent outcomes."

    return {
        "contract_version": JOURNAL_REPORTING_CONTRACT_VERSION,
        "status": status,
        "detail": detail,
        "schema": schema,
        "active_focus_entry_id": (
            active_focus_entry.get("entry_id") if active_focus_entry else None
        ),
        "active_focus": active_focus_entry.get("focus") if active_focus_entry else [],
        "recent_outcomes": recent_outcomes,
        "violations": violations[:10],
    }


def _iter_codex_records(payload: Any) -> Iterable[dict[str, Any]]:
    for record in _iter_records(payload, "runs"):
        yield _hydrate_codex_sidecar_record(record)


def _iter_codex_timestamps(payload: Any) -> Iterable[datetime]:
    for record in _iter_codex_records(payload):
        parsed = _record_timestamp(
            record,
            "completed_at",
            "updated_at",
            "started_at",
            "process_started_at",
            "created_at",
            "timestamp",
        )
        if parsed is not None:
            yield parsed


def _iter_ctx_records(payload: Any) -> Iterable[dict[str, Any]]:
    yield from _iter_records(payload, "sessions")


def _iter_ctx_timestamps(payload: Any) -> Iterable[datetime]:
    for record in _iter_ctx_records(payload):
        parsed = _record_timestamp(
            record,
            "updated_at",
            "updatedAt",
            "created_at",
            "createdAt",
            "timestamp",
        )
        if parsed is not None:
            yield parsed


def _ctx_record_status(record: dict[str, Any]) -> str:
    return str(record.get("status") or "").strip().lower()


def _ctx_record_is_active(record: dict[str, Any]) -> bool:
    if record.get("active") is True:
        return True
    if record.get("active") is False:
        return False
    return _ctx_record_status(record) in {"active", "running", "in_progress", "queued"}


def _latest_timestamp(values: Iterable[datetime]) -> Optional[datetime]:
    return max(values, default=None)


def _summarize_source(
    name: str,
    latest: Optional[datetime],
    freshness_hours: int,
    now: datetime,
) -> dict[str, Any]:
    if latest is None:
        return {
            "source": name,
            "status": "missing",
            "age_hours": None,
            "latest_timestamp": None,
        }
    age_hours = max(0.0, (now - latest).total_seconds() / 3600)
    return {
        "source": name,
        "status": "fresh" if age_hours <= freshness_hours else "stale",
        "age_hours": round(age_hours, 2),
        "latest_timestamp": latest.isoformat(),
    }


def _summarize_ctx_bindings(
    payload: Any,
    freshness_hours: int,
    now: datetime,
) -> dict[str, Any]:
    if payload is None:
        summary = _summarize_source("ctx_bindings", None, freshness_hours, now)
        summary.update(
            {
                "record_count": None,
                "active_count": None,
                "freshness_required": True,
                "detail": "ctx bindings evidence unavailable.",
            }
        )
        return summary

    records = list(_iter_ctx_records(payload))
    active_records = [record for record in records if _ctx_record_is_active(record)]
    active_latest = _latest_timestamp(
        timestamp
        for record in active_records
        if (
            timestamp := _record_timestamp(
                record,
                "updated_at",
                "updatedAt",
                "created_at",
                "createdAt",
                "timestamp",
            )
        )
        is not None
    )
    latest_record = _latest_timestamp(_iter_ctx_timestamps(payload))
    active_count = len(active_records)
    latest = active_latest if active_count else latest_record
    summary = _summarize_source("ctx_bindings", latest, freshness_hours, now)
    summary.update(
        {
            "record_count": len(records),
            "active_count": active_count,
            "inactive_count": len(records) - active_count,
            "freshness_required": bool(active_count),
            "active_latest_timestamp": active_latest.isoformat() if active_latest else None,
            "latest_record_timestamp": latest_record.isoformat() if latest_record else None,
        }
    )

    if active_count and active_latest is None:
        summary["status"] = "degraded"
        summary["detail"] = "Active ctx bindings do not include freshness timestamps."
    elif active_count == 0:
        summary["status"] = "inactive"
        summary["detail"] = (
            "No active ctx bindings; retired binding timestamps are informational."
            if latest is not None
            else "No ctx bindings recorded; no active ctx sessions require freshness."
        )

    return summary


def _extract_timestamps_from_text(text: str) -> Iterable[datetime]:
    for match in re.finditer(
        r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2})?",
        text,
    ):
        parsed = _parse_time(match.group(0).replace(" ", "T", 1))
        if parsed is not None:
            yield parsed


def _scan_ontology_file(path: Path) -> tuple[list[datetime], list[str]]:
    timestamps: list[datetime] = []
    alerts: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return timestamps, [f"{path.name} unreadable"]

    if path.suffix == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in (
                "generated_at",
                "updated_at",
                "prepared_at",
                "last_evolved",
                "timestamp",
                "created_at",
            ):
                parsed = _parse_time(payload.get(key))
                if parsed is not None:
                    timestamps.append(parsed)
            reliability = payload.get("reliability")
            status = ""
            if isinstance(reliability, dict):
                status = str(reliability.get("status") or "")
            status = status or str(payload.get("status") or "")
            if status.strip().lower() in {"degraded", "error", "failed", "missing", "stale"}:
                alerts.append(f"{path.name} status={status.strip().lower()}")
    else:
        timestamps.extend(_extract_timestamps_from_text(text))

    return timestamps, alerts


def _iter_ontology_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in _ONTOLOGY_SCAN_PRUNED_DIRS
        ]
        current_dir = Path(dirpath)
        for filename in filenames:
            path = current_dir / filename
            if path.suffix.lower() in _ONTOLOGY_SCAN_SUFFIXES:
                yield path


def _split_future_timestamps(
    timestamps: Iterable[datetime],
    now: datetime,
) -> tuple[list[datetime], list[datetime]]:
    valid: list[datetime] = []
    future: list[datetime] = []
    for timestamp in timestamps:
        if (timestamp - now).total_seconds() > _FUTURE_TIMESTAMP_TOLERANCE_SECONDS:
            future.append(timestamp)
        else:
            valid.append(timestamp)
    return valid, future


def _artifact_summary_from_timestamps(
    *,
    name: str,
    path: Path,
    timestamps: Iterable[datetime],
    alerts: Iterable[str],
    freshness_hours: int,
    now: datetime,
) -> dict[str, Any]:
    valid_timestamps, future_timestamps = _split_future_timestamps(timestamps, now)
    latest = _latest_timestamp(valid_timestamps)
    alert_reasons = [str(item).strip() for item in alerts if str(item).strip()]
    reasons = list(alert_reasons)

    if latest is None:
        status = "missing"
        age_hours = None
        latest_timestamp = None
        if future_timestamps:
            reasons.append(f"{name} only has future timestamps")
    else:
        age_hours = round(max(0.0, (now - latest).total_seconds() / 3600), 2)
        latest_timestamp = latest.isoformat()
        status = "fresh" if age_hours <= freshness_hours else "stale"
        if status == "stale":
            reasons.append(f"{name} stale ({age_hours}h)")

    if alert_reasons and status in {"fresh", "stale"}:
        status = "degraded"

    return {
        "source": name,
        "path": str(path),
        "status": status,
        "age_hours": age_hours,
        "latest_timestamp": latest_timestamp,
        "future_timestamp_count": len(future_timestamps),
        "ignored_future_timestamps": [
            timestamp.isoformat() for timestamp in sorted(future_timestamps)[-5:]
        ],
        "reasons": reasons,
    }


def _summarize_required_ontology_artifacts(
    root: Path,
    freshness_hours: int,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for name, relative_path in _ONTOLOGY_REQUIRED_ARTIFACTS:
        path = root / relative_path
        if not path.exists():
            summaries[name] = {
                "source": name,
                "path": str(path),
                "status": "missing",
                "age_hours": None,
                "latest_timestamp": None,
                "future_timestamp_count": 0,
                "ignored_future_timestamps": [],
                "reasons": [f"{name} missing"],
            }
            continue
        timestamps, alerts = _scan_ontology_file(path)
        summaries[name] = _artifact_summary_from_timestamps(
            name=name,
            path=path,
            timestamps=timestamps,
            alerts=alerts,
            freshness_hours=freshness_hours,
            now=now,
        )
    return summaries


def _ontology_external_repair(root: Path, required_artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    invalid_artifacts = [
        summary
        for summary in required_artifacts.values()
        if str(summary.get("status") or "") in {"stale", "missing", "degraded"}
    ]
    return {
        "required": bool(invalid_artifacts),
        "repository": str(root),
        "action": "refresh ontology evolution reporting artifacts",
        "artifacts": invalid_artifacts,
    }


def _summarize_ontology(root: Path, freshness_hours: int, now: datetime) -> dict[str, Any]:
    if not root.exists():
        return {
            "status": "missing",
            "latest_timestamp": None,
            "age_hours": None,
            "reasons": ["ontology root missing"],
            "alerts": [],
            "required_artifacts": {},
        }

    timestamps: list[datetime] = []
    alerts: list[str] = []
    for path in _iter_ontology_files(root):
        file_timestamps, file_alerts = _scan_ontology_file(path)
        timestamps.extend(file_timestamps)
        alerts.extend(file_alerts)

    required_artifacts = _summarize_required_ontology_artifacts(root, freshness_hours, now)
    required_latest = [
        parsed
        for summary in required_artifacts.values()
        if (parsed := _parse_time(summary.get("latest_timestamp"))) is not None
    ]
    invalid_required = [
        summary
        for summary in required_artifacts.values()
        if str(summary.get("status") or "") in {"stale", "missing", "degraded"}
    ]
    external_repair = _ontology_external_repair(root, required_artifacts)

    valid_timestamps, future_timestamps = _split_future_timestamps(timestamps, now)
    latest = _latest_timestamp(required_latest) or _latest_timestamp(valid_timestamps)
    if latest is None:
        reasons = ["ontology intelligence timestamp missing"]
        if future_timestamps:
            reasons.append("ontology intelligence only has future timestamps")
        return {
            "status": "missing",
            "latest_timestamp": None,
            "age_hours": None,
            "reasons": reasons,
            "alerts": alerts,
            "ignored_future_timestamp_count": len(future_timestamps),
            "required_artifacts": required_artifacts,
            "external_repair": external_repair,
        }

    age_hours = max(0.0, (now - latest).total_seconds() / 3600)
    status = "fresh" if age_hours <= freshness_hours else "stale"
    reasons: list[str] = []
    if status == "stale":
        reasons.append(f"ontology_intelligence stale ({round(age_hours, 2)}h)")
    for summary in invalid_required:
        for reason in summary.get("reasons") or [f"{summary.get('source')} {summary.get('status')}"]:
            text = str(reason).strip()
            if text and text not in reasons:
                reasons.append(text)
    if invalid_required:
        required_statuses = {str(summary.get("status") or "") for summary in invalid_required}
        status = "degraded" if required_statuses.intersection({"missing", "degraded"}) else "stale"
    if alerts:
        status = "degraded"
        reasons.extend(alerts)
    return {
        "status": status,
        "latest_timestamp": latest.isoformat(),
        "age_hours": round(age_hours, 2),
        "reasons": reasons,
        "alerts": alerts,
        "ignored_future_timestamp_count": len(future_timestamps),
        "required_artifacts": required_artifacts,
        "external_repair": external_repair,
    }


def _codex_record_status(record: dict[str, Any]) -> str:
    return str(record.get("status") or "").strip().lower()


def _codex_record_is_completed(record: dict[str, Any]) -> bool:
    status = _codex_record_status(record)
    if status in _CODEX_FAILURE_STATUSES:
        return False
    if status in _CODEX_SUCCESS_STATUSES:
        return True
    exit_code = record.get("exit_code")
    if exit_code == 0 or str(exit_code).strip() == "0":
        return True
    return record.get("completed_at") is not None


def _codex_record_is_active(record: dict[str, Any]) -> bool:
    status = _codex_record_status(record)
    if status in _CODEX_FAILURE_STATUSES or status in _CODEX_SUCCESS_STATUSES:
        return False
    if status in {"running", "queued", "in_progress", "active", "unknown"}:
        return True
    if record.get("active") is True:
        return True
    return record.get("completed_at") is None and record.get("exit_code") is None and bool(status)


def _find_stale_active_codex(
    payload: Any,
    now: datetime,
    active_stale_hours: int,
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for record in _iter_codex_records(payload):
        if not _codex_record_is_active(record):
            continue
        started = _record_timestamp(
            record,
            "updated_at",
            "started_at",
            "process_started_at",
            "created_at",
            "timestamp",
        )
        if started is None:
            continue
        age_hours = max(0.0, (now - started).total_seconds() / 3600)
        if age_hours > active_stale_hours:
            stale.append(
                {
                    "run_id": record.get("run_id") or record.get("id"),
                    "status": record.get("status"),
                    "age_hours": round(age_hours, 2),
                    "latest_timestamp": started.isoformat(),
                }
            )
    return stale


def _find_stale_active_ctx(
    payload: Any,
    now: datetime,
    active_stale_hours: int,
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for record in _iter_ctx_records(payload):
        if not _ctx_record_is_active(record):
            continue
        updated = _record_timestamp(record, "updated_at", "updatedAt", "created_at", "createdAt")
        if updated is None:
            continue
        age_hours = max(0.0, (now - updated).total_seconds() / 3600)
        if age_hours > active_stale_hours:
            stale.append(
                {
                    "session_id": record.get("session_id") or record.get("id"),
                    "task_id": record.get("task_id"),
                    "age_hours": round(age_hours, 2),
                    "latest_timestamp": updated.isoformat(),
                }
            )
    return stale


def _find_planning_contradictions(codex_payload: Any, ctx_payload: Any) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for record in _iter_ctx_records(ctx_payload):
        if not _ctx_record_is_active(record):
            continue
        worktree_path = str(record.get("worktree_path") or "").strip()
        reason = str(record.get("reason") or "").strip().lower()
        if "retired" in reason:
            contradictions.append(
                {
                    "type": "ctx_binding_retired_but_active",
                    "session_id": record.get("session_id"),
                    "task_id": record.get("task_id"),
                }
            )
        if not worktree_path:
            contradictions.append(
                {
                    "type": "ctx_binding_missing_worktree_path",
                    "session_id": record.get("session_id"),
                    "task_id": record.get("task_id"),
                }
            )
        elif not Path(worktree_path).exists():
            contradictions.append(
                {
                    "type": "ctx_binding_worktree_missing",
                    "session_id": record.get("session_id"),
                    "task_id": record.get("task_id"),
                    "worktree_path": worktree_path,
                }
            )

    for record in _iter_codex_records(codex_payload):
        if not _codex_record_is_active(record):
            continue
        if record.get("completed_at") is not None or record.get("exit_code") is not None:
            contradictions.append(
                {
                    "type": "codex_running_but_completed",
                    "run_id": record.get("run_id") or record.get("id"),
                }
            )
        worktree_path = str(record.get("ctx_worktree_path") or "").strip()
        if worktree_path and not Path(worktree_path).exists():
            contradictions.append(
                {
                    "type": "codex_worktree_missing",
                    "run_id": record.get("run_id") or record.get("id"),
                    "ctx_worktree_path": worktree_path,
                }
            )
    return contradictions


def _build_ctx_remediation(
    ctx_bindings_path: Path,
    ctx_summary: dict[str, Any],
    stale_active_ctx: list[dict[str, Any]],
    planning_contradictions: list[dict[str, Any]],
) -> dict[str, Any]:
    status = str(ctx_summary.get("status") or "")
    ctx_contradictions = [
        item
        for item in planning_contradictions
        if str(item.get("type") or "").startswith("ctx_")
    ]
    required = status in {"missing", "stale", "degraded"} or bool(stale_active_ctx or ctx_contradictions)

    if not required:
        return {
            "required": False,
            "path": str(ctx_bindings_path),
            "action": "none",
            "reason": str(
                ctx_summary.get("detail")
                or "ctx session-binding evidence is current."
            ),
            "active_count": ctx_summary.get("active_count"),
            "stale_active_count": 0,
            "contradiction_count": 0,
        }

    actions: list[str] = []
    reasons: list[str] = []
    if status == "missing":
        actions.append("restore or regenerate ctx session-binding evidence")
        reasons.append("ctx session-binding evidence is unavailable")
    elif status == "stale":
        actions.append("refresh active ctx session bindings or retire sessions that are no longer live")
        reasons.append(f"ctx session-binding evidence is stale ({ctx_summary.get('age_hours')}h)")
    elif status == "degraded":
        actions.append("repair degraded ctx session-binding evidence before selecting new work")
        reasons.append("ctx session-binding evidence is degraded")

    if stale_active_ctx:
        actions.append("retire stale active ctx bindings or refresh them from the live ctx runtime")
        reasons.append(f"{len(stale_active_ctx)} active ctx binding(s) exceed freshness limits")
    if ctx_contradictions:
        actions.append("repair ctx binding store contradictions")
        reasons.append(f"{len(ctx_contradictions)} ctx binding contradiction(s) detected")

    return {
        "required": True,
        "path": str(ctx_bindings_path),
        "action": "; ".join(dict.fromkeys(actions)),
        "reasons": list(dict.fromkeys(reasons)),
        "active_count": ctx_summary.get("active_count"),
        "stale_active_count": len(stale_active_ctx),
        "stale_active_sessions": stale_active_ctx[:5],
        "contradiction_count": len(ctx_contradictions),
        "contradictions": ctx_contradictions[:5],
    }


def _build_provenance_item(tag: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": tag,
        "path": str(path),
        "status": summary.get("status"),
        "latest_timestamp": summary.get("latest_timestamp"),
        "age_hours": summary.get("age_hours"),
    }


def format_evidence_provenance(items: Iterable[dict[str, Any]]) -> str:
    lines = ["Evidence provenance:"]
    for item in items:
        details: list[str] = []
        for key in ("status", "age_hours", "latest_timestamp", "path", "notes"):
            value = item.get(key)
            if value is not None and value != "":
                label = "latest" if key == "latest_timestamp" else key
                details.append(f"{label}={value}")
        lines.append(f"- [{item.get('tag')}] " + "; ".join(details))
    return "\n".join(lines)


def evaluate_self_improvement_evidence(
    *,
    journal_path: Path = DEFAULT_JOURNAL_PATH,
    codex_runs_path: Path = DEFAULT_CODEX_RUNS_PATH,
    ctx_bindings_path: Path = DEFAULT_CTX_BINDINGS_PATH,
    ontology_root: Path = DEFAULT_ONTOLOGY_ROOT,
    now: Optional[datetime] = None,
    freshness_hours: int = DEFAULT_FRESHNESS_HOURS,
    active_stale_hours: int = DEFAULT_ACTIVE_STALE_HOURS,
) -> dict[str, Any]:
    current = now or datetime.now(tz=timezone.utc)
    journal_payload = _load_json(journal_path)
    codex_payload = _load_json(codex_runs_path)
    ctx_payload = _load_json(ctx_bindings_path)
    ontology_summary = _summarize_ontology(ontology_root, freshness_hours, current)

    journal_latest = _latest_timestamp(_iter_journal_timestamps(journal_payload))
    codex_latest = _latest_timestamp(_iter_codex_timestamps(codex_payload))
    ctx_summary = _summarize_ctx_bindings(ctx_payload, freshness_hours, current)
    ontology_latest = _parse_time(ontology_summary.get("latest_timestamp"))

    sources = {
        "journal_entries": _summarize_source("journal_entries", journal_latest, freshness_hours, current),
        "codex_runs": _summarize_source("codex_runs", codex_latest, freshness_hours, current),
        "ctx_bindings": ctx_summary,
        "ontology_intelligence": {
            "source": "ontology_intelligence",
            "status": ontology_summary.get("status"),
            "age_hours": ontology_summary.get("age_hours"),
            "latest_timestamp": ontology_summary.get("latest_timestamp"),
        },
    }

    stale_active_codex = _find_stale_active_codex(codex_payload, current, active_stale_hours)
    stale_active_ctx = _find_stale_active_ctx(ctx_payload, current, active_stale_hours)
    planning_contradictions = _find_planning_contradictions(codex_payload, ctx_payload)
    ctx_remediation = _build_ctx_remediation(
        ctx_bindings_path,
        ctx_summary,
        stale_active_ctx,
        planning_contradictions,
    )

    ctx_effective_latest = (
        None
        if ctx_summary.get("status") == "inactive"
        else _parse_time(ctx_summary.get("latest_timestamp"))
    )
    latest_timestamps = [
        item
        for item in (
            journal_latest,
            codex_latest,
            ctx_effective_latest,
            ontology_latest,
        )
        if item is not None
    ]
    freshness_spread_hours = None
    if len(latest_timestamps) >= 2:
        spread_seconds = (max(latest_timestamps) - min(latest_timestamps)).total_seconds()
        freshness_spread_hours = round(spread_seconds / 3600, 2)

    reasons: list[str] = []
    contradictions: list[str] = []
    warnings: list[str] = []
    for name, entry in sources.items():
        status = str(entry.get("status") or "")
        if status == "missing":
            reasons.append(f"{name} evidence missing")
            warnings.append(f"{name} evidence missing")
        elif status == "stale":
            reasons.append(f"{name} evidence stale ({entry.get('age_hours')}h)")
            warnings.append(f"{name} evidence stale")
        elif status == "degraded":
            reasons.append(f"{name} evidence degraded")
            warnings.append(f"{name} evidence degraded")

    statuses = {str(entry.get("status") or "") for entry in sources.values()}
    if "fresh" in statuses and statuses.intersection({"stale", "missing", "degraded"}):
        contradictions.append("evidence freshness mismatch across sources")
    if stale_active_codex:
        contradictions.append("stale active Codex runs detected")
        warnings.append("stale active Codex runs detected")
        reasons.append(f"{len(stale_active_codex)} active Codex run(s) exceed {active_stale_hours}h")
    if stale_active_ctx:
        contradictions.append("stale active ctx bindings detected")
        warnings.append("stale active ctx bindings detected")
        reasons.append(f"{len(stale_active_ctx)} active ctx binding(s) exceed {active_stale_hours}h")
    if planning_contradictions:
        contradictions.append("planning contradictions detected")
        warnings.append("planning contradictions detected")
        reasons.append(f"{len(planning_contradictions)} planning contradiction(s) detected")

    ontology_alerts = [
        str(item).strip()
        for item in ontology_summary.get("alerts", [])
        if str(item).strip()
    ]
    for item in ontology_summary.get("reasons", []):
        text = str(item).strip()
        if text and text not in reasons:
            reasons.append(text)
    if str(ontology_summary.get("status") or "") in {"missing", "stale", "degraded"}:
        message = "ontology intelligence artifacts are stale, missing, or degraded"
        if message not in contradictions:
            contradictions.append(message)

    degraded = bool(reasons or contradictions)
    gate_status = "degraded" if degraded else "healthy"

    provenance_items = [
        _build_provenance_item("journal", journal_path, sources["journal_entries"]),
        _build_provenance_item("codex", codex_runs_path, sources["codex_runs"]),
        _build_provenance_item("ctx", ctx_bindings_path, sources["ctx_bindings"]),
        _build_provenance_item("ontology", ontology_root, sources["ontology_intelligence"]),
    ]
    if stale_active_codex:
        provenance_items[1]["notes"] = f"{len(stale_active_codex)} active run(s) exceed {active_stale_hours}h"
    if stale_active_ctx:
        provenance_items[2]["notes"] = f"{len(stale_active_ctx)} active session(s) exceed {active_stale_hours}h"
    elif ctx_summary.get("status") == "inactive":
        provenance_items[2]["notes"] = str(ctx_summary.get("detail") or "ctx inactive")
    if ontology_alerts:
        provenance_items[3]["notes"] = " | ".join(ontology_alerts)

    return {
        "status": gate_status,
        "freshness_hours": freshness_hours,
        "active_stale_hours": active_stale_hours,
        "sources": sources,
        "freshness_spread_hours": freshness_spread_hours,
        "stale_active_codex": stale_active_codex,
        "stale_active_ctx": stale_active_ctx,
        "planning_contradictions": planning_contradictions,
        "warnings": warnings,
        "ontology": ontology_summary,
        "ontology_alerts": ontology_alerts,
        "ctx_remediation": ctx_remediation,
        "contradictions": contradictions,
        "reasons": reasons,
        "suppression": {
            "suppress_non_maintenance": degraded,
            "message": (
                "Reliability floor degraded: non-maintenance work suppressed."
                if degraded
                else "Reliability floor healthy: normal lane selection permitted."
            ),
        },
        "provenance": {
            "contract_version": PROVENANCE_CONTRACT_VERSION,
            "items": provenance_items,
            "summary_markdown": format_evidence_provenance(provenance_items),
        },
    }


def _check_status(score: float) -> str:
    if score >= 0.85:
        return "pass"
    if score >= 0.6:
        return "warn"
    return "fail"


def _build_benchmark_item(
    benchmark_id: str,
    label: str,
    *,
    score: float,
    weight: int,
    detail: str,
    critical: bool,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": benchmark_id,
        "label": label,
        "score": round(max(0.0, min(1.0, score)), 4),
        "weight": weight,
        "status": _check_status(score),
        "detail": detail,
        "critical": critical,
        "metrics": metrics,
    }


def _load_benchmark_history(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        history = payload
    elif isinstance(payload, list):
        history = {"runs": payload}
    else:
        history = {"runs": []}
    if not isinstance(history.get("runs"), list):
        history["runs"] = []
    return history


def _save_benchmark_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, payload)


def _benchmark_history_check_snapshot(check_id: str, check: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "score": check.get("score"),
        "status": check.get("status"),
    }
    if check_id == "operator_value_alignment":
        metrics = check.get("metrics") or {}
        snapshot["detail"] = check.get("detail")
        snapshot["metrics"] = {
            "missing_operator_decision_support_fields": metrics.get("missing_operator_decision_support_fields") or [],
            "missing_operator_decision_support_examples": metrics.get("missing_operator_decision_support_examples") or [],
            "operator_decision_support_fields": metrics.get("operator_decision_support_fields") or [],
            "operator_decision_support_examples": metrics.get("operator_decision_support_examples") or [],
        }
        return snapshot

    if check_id != "leading_indicator_drift":
        return snapshot

    metrics = check.get("metrics") or {}
    snapshot["detail"] = check.get("detail")
    snapshot["triggered_harbingers"] = metrics.get("triggered_harbingers") or []
    snapshot["harbinger_scorecard"] = metrics.get("harbinger_scorecard") or {}
    snapshot["leading_indicator_report"] = check.get("report") or _build_leading_indicator_report(
        check
    )
    snapshot["recommended_mitigations"] = metrics.get("recommended_mitigations") or []
    snapshot["execution_throughput_remediation"] = (
        metrics.get("execution_throughput_remediation") or {}
    )
    return snapshot


def _normalize_evidence_key(key: Any) -> str:
    text = str(key or "").strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _value_has_content(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_value_has_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_value_has_content(item) for item in value)
    return True


def _collect_record_text(value: Any, key: str = "") -> list[str]:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        parts: list[str] = []
        for child_key, child_value in value.items():
            parts.extend(_collect_record_text(child_value, str(child_key)))
        return parts
    if isinstance(value, list):
        parts = []
        for child_value in value:
            parts.extend(_collect_record_text(child_value, key))
        return parts
    return []


def _collect_claim_text(value: Any, key: str = "") -> list[str]:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return []
    if normalized_key in _CLAIM_TEXT_KEYS:
        return _collect_record_text(value, key)
    if isinstance(value, dict):
        parts: list[str] = []
        for child_key, child_value in value.items():
            parts.extend(_collect_claim_text(child_value, str(child_key)))
        return parts
    if isinstance(value, list):
        parts = []
        for child_value in value:
            parts.extend(_collect_claim_text(child_value, key))
        return parts
    return []


def _redact_operator_evidence_value(text: str) -> str:
    redacted = re.sub(
        r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"\b(token|api[_-]?key|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _compact_operator_evidence_value(value: Any, *, limit: int = 240) -> Optional[str]:
    if not _value_has_content(value):
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        except TypeError:
            text = str(value)
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    text = _redact_operator_evidence_value(text)
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _dedupe_operator_evidence(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            str(item.get("field") or ""),
            str(item.get("source_key") or ""),
            str(item.get("value") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _collect_operator_decision_support_evidence(value: Any, key: str = "") -> list[dict[str, str]]:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return []

    field = _OPERATOR_DECISION_SUPPORT_EVIDENCE_FIELDS.get(normalized_key)
    if field:
        compact_value = _compact_operator_evidence_value(value)
        if compact_value is not None:
            return [
                {
                    "field": field,
                    "source_key": normalized_key,
                    "value": compact_value,
                }
            ]
        return []

    if isinstance(value, dict):
        evidence: list[dict[str, str]] = []
        for child_key, child_value in value.items():
            evidence.extend(
                _collect_operator_decision_support_evidence(child_value, str(child_key))
            )
        return _dedupe_operator_evidence(evidence)
    if isinstance(value, list):
        evidence = []
        for child_value in value:
            evidence.extend(_collect_operator_decision_support_evidence(child_value, key))
        return _dedupe_operator_evidence(evidence)
    return []


def _operator_decision_support_text_evidence(text: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for label, pattern in _DURABLE_TEXT_PATTERNS:
        if label != "operator_decision_support":
            continue
        match = pattern.search(text)
        if not match:
            continue
        compact_value = _compact_operator_evidence_value(match.group(0))
        if compact_value is None:
            continue
        evidence.append(
            {
                "field": "operator_decision_support",
                "source_key": "text_match",
                "value": compact_value,
            }
        )
    return _dedupe_operator_evidence(evidence)


def _record_has_claim_field(value: Any, key: str = "") -> bool:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _CLAIM_TEXT_KEYS and _value_has_content(value):
        return True
    if normalized_key in _CLAIM_CONTAINER_KEYS and _value_has_content(value):
        return True
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return False
    if isinstance(value, dict):
        return any(_record_has_claim_field(child_value, str(child_key)) for child_key, child_value in value.items())
    if isinstance(value, list):
        return any(_record_has_claim_field(child_value, key) for child_value in value)
    return False


def _structured_durable_signals(value: Any, key: str = "") -> set[str]:
    normalized_key = _normalize_evidence_key(key)
    signals: set[str] = set()
    if normalized_key in _DURABLE_EVIDENCE_KEYS and _value_has_content(value):
        signals.add(normalized_key)
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            signals.update(_structured_durable_signals(child_value, str(child_key)))
    elif isinstance(value, list):
        for child_value in value:
            signals.update(_structured_durable_signals(child_value, key))
    return signals


def _text_durable_signals(text: str) -> set[str]:
    return {label for label, pattern in _DURABLE_TEXT_PATTERNS if pattern.search(text)}


def _status_only_markers(text: str) -> set[str]:
    return {pattern.pattern for pattern in _STATUS_ONLY_PATTERNS if pattern.search(text)}


def _codex_completed_value_claim_markers(text: str) -> set[str]:
    return {
        pattern.pattern
        for pattern in _CODEX_COMPLETED_VALUE_CLAIM_PATTERNS
        if pattern.search(text)
    }


def _value_categories_from_signals(signals: set[str]) -> list[str]:
    categories: list[str] = []
    for category, _label in _ALLOWED_VALUE_CATEGORIES:
        if signals.intersection(_VALUE_CATEGORY_SIGNAL_MAP.get(category, set())):
            categories.append(category)
    return categories


def _value_category_labels(categories: Iterable[str]) -> list[str]:
    return [
        _ALLOWED_VALUE_CATEGORY_LABELS[category]
        for category in categories
        if category in _ALLOWED_VALUE_CATEGORY_LABELS
    ]


def _allowed_value_category_guidance() -> list[dict[str, str]]:
    return [
        {
            "category": category,
            "label": label,
            "remediation": _VALUE_CATEGORY_REMEDIATION[category],
        }
        for category, label in _ALLOWED_VALUE_CATEGORIES
    ]


def _make_work_remediation(
    categories: list[str],
    *,
    source: Optional[str] = None,
    record_id: Any = None,
) -> Optional[str]:
    if categories:
        return None
    guidance = "; ".join(
        f"{label}: {_VALUE_CATEGORY_REMEDIATION[category]}"
        for category, label in _ALLOWED_VALUE_CATEGORIES
    )
    remediation = f"Add evidence for at least one allowed value category: {guidance}."
    if source == "codex_runs":
        identifier = str(record_id or "unknown").strip() or "unknown"
        fields = "; ".join(_CODEX_BACKFILL_EVIDENCE_FIELDS)
        remediation += (
            f" Backfill completed Codex run {identifier} with structured evidence fields: "
            f"{fields}."
        )
    return remediation


def _matching_codex_skip_note_labels(
    text: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]],
) -> list[str]:
    return sorted({label for label, pattern in patterns if pattern.search(text)})


def _codex_non_durable_rationale_labels(text: str) -> list[str]:
    return _matching_codex_skip_note_labels(text, _CODEX_NON_DURABLE_RATIONALE_PATTERNS)


def _journal_focus_note_segments(title: str, outcome_note: str) -> list[str]:
    outcome_segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", outcome_note)
        if segment.strip()
    ]
    if outcome_segments:
        return [
            " ".join(part for part in (title, segment) if part).strip()
            for segment in outcome_segments
        ]
    return [title] if title else []


def _collect_journal_codex_skip_remediations(journal_payload: Any) -> dict[str, dict[str, Any]]:
    remediations: dict[str, dict[str, Any]] = {}
    for entry in _iter_records(journal_payload, "entries"):
        focus_items = entry.get(_JOURNAL_REPORTING_FOCUS_FIELD)
        if not isinstance(focus_items, list):
            continue
        entry_id = _journal_reporting_string(entry.get("id"))
        occurred_at = _journal_reporting_entry_time_text(entry)
        for focus_index, focus_item in enumerate(focus_items):
            if not isinstance(focus_item, dict):
                continue
            title = _journal_reporting_string(focus_item.get("title"))
            outcome_note = _journal_reporting_string(focus_item.get("outcomeNote"))
            if not title and not outcome_note:
                continue
            for segment in _journal_focus_note_segments(title, outcome_note):
                run_ids = sorted(set(_CODEX_RUN_ID_PATTERN.findall(segment)))
                if not run_ids:
                    continue
                action_labels = _matching_codex_skip_note_labels(
                    segment,
                    _CODEX_SKIP_NOTE_ACTION_PATTERNS,
                )
                rationale_labels = _matching_codex_skip_note_labels(
                    segment,
                    _CODEX_NON_DURABLE_RATIONALE_PATTERNS,
                )
                if not action_labels or not rationale_labels:
                    continue
                evidence = {
                    "entry_id": entry_id or None,
                    "occurredAt": occurred_at or None,
                    "focus_index": focus_index,
                    "title": title,
                    "outcomeNote": _compact_operator_evidence_value(
                        outcome_note,
                        limit=500,
                    ),
                    "matched_note": _compact_operator_evidence_value(segment, limit=500),
                    "action_markers": action_labels,
                    "rationale_markers": rationale_labels,
                }
                for run_id in run_ids:
                    remediations.setdefault(run_id, {**evidence, "run_id": run_id})
    return remediations


def _collect_codex_run_ids_from_value(value: Any, key: str = "") -> set[str]:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return set()
    if isinstance(value, str):
        return set(_CODEX_RUN_ID_PATTERN.findall(value))
    if isinstance(value, dict):
        run_ids: set[str] = set()
        for child_key, child_value in value.items():
            run_ids.update(_collect_codex_run_ids_from_value(child_value, str(child_key)))
        return run_ids
    if isinstance(value, list):
        run_ids = set()
        for child_value in value:
            run_ids.update(_collect_codex_run_ids_from_value(child_value, key))
        return run_ids
    return set()


def _normalize_linear_issue_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower().startswith("linear:"):
        text = text.split(":", 1)[1].strip()
    match = _LINEAR_ISSUE_ID_PATTERN.search(text.upper())
    return match.group(0) if match else None


def _collect_linear_issue_ids_from_issue_fields(value: Any, key: str = "") -> set[str]:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return set()
    if isinstance(value, str):
        if not key or normalized_key in _CODEX_ISSUE_ID_KEYS:
            return {
                match.group(0)
                for match in _LINEAR_ISSUE_ID_PATTERN.finditer(value.upper())
            }
        return set()
    if isinstance(value, dict):
        issue_ids: set[str] = set()
        for child_key, child_value in value.items():
            issue_ids.update(
                _collect_linear_issue_ids_from_issue_fields(child_value, str(child_key))
            )
        return issue_ids
    if isinstance(value, list):
        issue_ids = set()
        for child_value in value:
            issue_ids.update(_collect_linear_issue_ids_from_issue_fields(child_value, key))
        return issue_ids
    return set()


def _codex_run_id(record: dict[str, Any]) -> Optional[str]:
    for key in ("run_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_codex_issue_run_id_index(codex_payload: Any) -> dict[str, set[str]]:
    issue_run_ids: dict[str, set[str]] = {}
    for record in _iter_codex_records(codex_payload):
        if _codex_record_status(record) in _CODEX_FAILURE_STATUSES:
            continue
        run_id = _codex_run_id(record)
        if not run_id:
            continue
        for issue_id in sorted(_collect_linear_issue_ids_from_issue_fields(record)):
            issue_run_ids.setdefault(issue_id, set()).add(run_id)
    return issue_run_ids


_OPERATOR_DECISION_SUPPORT_REFERENCE_FIELD_PATH = (
    "operatorDecisionSupport|nextDecision|decision|selectedWork|blocker|tradeoff"
)


def _journal_reference_path(entry_id: str, focus_index: int) -> str:
    return (
        f"entries[{entry_id or 'unknown'}]."
        f"{_JOURNAL_REPORTING_FOCUS_FIELD}[{focus_index}]"
    )


def _collect_journal_codex_operator_support(
    journal_payload: Any,
    codex_payload: Any = None,
) -> dict[str, list[dict[str, str]]]:
    support_by_run_id: dict[str, list[dict[str, str]]] = {}
    codex_issue_run_ids = _build_codex_issue_run_id_index(codex_payload)
    for entry in _iter_records(journal_payload, "entries"):
        focus_items = entry.get(_JOURNAL_REPORTING_FOCUS_FIELD)
        if not isinstance(focus_items, list):
            continue

        entry_evidence = _collect_operator_decision_support_evidence(entry)
        entry_text = " ".join(
            value
            for key, value in (
                ("summary", entry.get("summary")),
                ("notes", entry.get("notes")),
                ("result", entry.get("result")),
                ("details", entry.get("details")),
                ("outcome", entry.get("outcome")),
            )
            if isinstance(value, str) and value.strip()
        )
        entry_text_evidence = (
            _operator_decision_support_text_evidence(entry_text) if entry_text else []
        )
        entry_context = {
            key: value
            for key, value in entry.items()
            if key != _JOURNAL_REPORTING_FOCUS_FIELD
        }
        entry_context_run_ids = _collect_codex_run_ids_from_value(entry_context)

        entry_id = _journal_reporting_string(entry.get("id"))
        occurred_at = _journal_reporting_entry_time_text(entry)
        for focus_index, focus_item in enumerate(focus_items):
            if not isinstance(focus_item, dict):
                continue

            title = _journal_reporting_string(focus_item.get("title"))
            outcome_note = _journal_reporting_string(focus_item.get("outcomeNote"))
            if not title and not outcome_note:
                continue

            focus_evidence = _collect_operator_decision_support_evidence(focus_item)
            if not focus_evidence:
                focus_evidence = _operator_decision_support_text_evidence(
                    " ".join(part for part in (title, outcome_note) if part)
                )
            if not focus_evidence:
                focus_evidence = entry_evidence
            if not focus_evidence:
                focus_evidence = entry_text_evidence
            if not focus_evidence:
                continue
            focus_evidence = [
                {
                    **evidence,
                    "journal_entry_id": entry_id or "unknown",
                    "journal_occurred_at": occurred_at or "unknown",
                    "journal_focus_path": _journal_reference_path(
                        entry_id or "unknown",
                        focus_index,
                    ),
                    "journal_focus_title": title or "unknown",
                }
                for evidence in focus_evidence
            ]

            focus_run_ids = _collect_codex_run_ids_from_value(focus_item)
            active_issue_ids = _journal_reporting_issue_ids(
                focus_item.get("activeLinearIssueIds")
            )
            for issue_id in active_issue_ids:
                normalized_issue_id = _normalize_linear_issue_id(issue_id)
                if normalized_issue_id:
                    focus_run_ids.update(codex_issue_run_ids.get(normalized_issue_id, set()))
            if not focus_run_ids and len(focus_items) == 1:
                focus_run_ids = set(entry_context_run_ids)
            for segment in _journal_focus_note_segments(title, outcome_note):
                segment_run_ids = set(_CODEX_RUN_ID_PATTERN.findall(segment))
                if not segment_run_ids and focus_run_ids:
                    segment_run_ids = set(focus_run_ids)
                if not segment_run_ids:
                    continue

                action_labels = _matching_codex_skip_note_labels(
                    segment,
                    _CODEX_SKIP_NOTE_ACTION_PATTERNS,
                )
                rationale_labels = _matching_codex_skip_note_labels(
                    segment,
                    _CODEX_NON_DURABLE_RATIONALE_PATTERNS,
                )
                if action_labels and rationale_labels:
                    continue

                for run_id in sorted(segment_run_ids):
                    support_by_run_id.setdefault(run_id, []).extend(focus_evidence)

    return {
        run_id: _dedupe_operator_evidence(evidence)
        for run_id, evidence in support_by_run_id.items()
    }


def _codex_record_by_run_id(codex_payload: Any) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _iter_codex_records(codex_payload):
        run_id = _codex_run_id(record)
        if run_id:
            records.setdefault(run_id, record)
    return records


def _operator_support_reference_diagnostics(
    journal_payload: Any,
    codex_payload: Any,
    run_ids: Iterable[str],
) -> list[dict[str, Any]]:
    codex_records = _codex_record_by_run_id(codex_payload)
    diagnostics: list[dict[str, Any]] = []
    for run_id in sorted({str(value).strip() for value in run_ids if str(value).strip()}):
        record = codex_records.get(run_id, {})
        issue_ids = sorted(_collect_linear_issue_ids_from_issue_fields(record))
        matched_paths: list[str] = []
        for entry in _iter_records(journal_payload, "entries"):
            entry_id = _journal_reporting_string(entry.get("id")) or "unknown"
            for focus_index, focus_item in enumerate(
                entry.get(_JOURNAL_REPORTING_FOCUS_FIELD) or []
            ):
                if not isinstance(focus_item, dict):
                    continue
                focus_text = json.dumps(focus_item, sort_keys=True, ensure_ascii=False)
                focus_issue_ids = {
                    _normalize_linear_issue_id(issue_id)
                    for issue_id in _journal_reporting_issue_ids(
                        focus_item.get("activeLinearIssueIds")
                    )
                }
                focus_issue_ids.discard(None)
                if run_id in focus_text or set(issue_ids).intersection(focus_issue_ids):
                    matched_paths.append(_journal_reference_path(entry_id, focus_index))
        matched_paths = list(dict.fromkeys(matched_paths))
        if matched_paths:
            required_path = f"{matched_paths[0]}.{_OPERATOR_DECISION_SUPPORT_REFERENCE_FIELD_PATH}"
            reason = "journal_focus_lacks_operator_decision_support_field"
        else:
            required_path = (
                "entries[*]."
                f"{_JOURNAL_REPORTING_FOCUS_FIELD}[*]."
                f"{_OPERATOR_DECISION_SUPPORT_REFERENCE_FIELD_PATH}"
            )
            reason = "journal_focus_reference_not_found_for_codex_run"
        diagnostics.append(
            {
                "run_id": run_id,
                "codex_issue_id": issue_ids[0] if issue_ids else None,
                "required_journal_reference_path": required_path,
                "matched_journal_reference_paths": matched_paths,
                "reason": reason,
            }
        )
    return diagnostics


def _record_claimed_timestamp(record: dict[str, Any]) -> Optional[datetime]:
    return _record_timestamp(
        record,
        "completed_at",
        "updated_at",
        "updatedAt",
        "occurredAt",
        "occurred_at",
        "created_at",
        "createdAt",
        "started_at",
        "process_started_at",
        "timestamp",
        "date",
    )


def _record_claims_work(record: dict[str, Any], text: str) -> bool:
    if _record_has_claim_field(record):
        return True
    if record.get("active") is True and _status_only_markers(text):
        return True
    status = str(record.get("status") or "").strip().lower()
    if status in {"active", "in_progress", "running", "queued"} and _status_only_markers(text):
        return True
    return False


def _codex_record_claims_work(record: dict[str, Any], text: str) -> bool:
    if _structured_durable_signals(record) or _text_durable_signals(text):
        return True
    claim_text = "\n".join(_collect_claim_text(record))
    if not claim_text.strip():
        return False
    return bool(_codex_completed_value_claim_markers(claim_text))


def _codex_durable_signals(record: dict[str, Any], text: str) -> set[str]:
    structured_signals = _structured_durable_signals(record)
    text_signals = _text_durable_signals(text)
    if _codex_non_durable_rationale_labels(text):
        text_signals.difference_update(_CODEX_TEXT_DELIVERY_SIGNALS)
    return structured_signals | text_signals


def _iter_recent_claimed_work_items(
    *,
    journal_payload: Any,
    codex_payload: Any,
    ctx_payload: Any,
    now: datetime,
    freshness_hours: int,
) -> Iterable[dict[str, Any]]:
    source_records = (
        ("journal_entries", _iter_records(journal_payload, "entries")),
        ("codex_runs", _iter_codex_records(codex_payload)),
        ("ctx_bindings", _iter_ctx_records(ctx_payload)),
    )
    for source, records in source_records:
        for record in records:
            if (
                source == "codex_runs"
                and _codex_record_status(record) in _CODEX_FAILURE_STATUSES
            ):
                continue
            timestamp = _record_claimed_timestamp(record)
            if timestamp is not None:
                age_hours = max(0.0, (now - timestamp).total_seconds() / 3600)
                if age_hours > freshness_hours:
                    continue
            text = "\n".join(_collect_record_text(record))
            if source == "codex_runs":
                claims_work = _codex_record_claims_work(record, text)
            else:
                claims_work = _record_claims_work(record, text)
            if not claims_work:
                continue
            yield {
                "source": source,
                "id": (
                    record.get("id")
                    or record.get("run_id")
                    or record.get("session_id")
                    or record.get("external_key")
                ),
                "timestamp": timestamp.isoformat() if timestamp is not None else None,
                "record": record,
                "text": text,
            }


def _recent_record_timestamp(
    record: dict[str, Any],
    now: datetime,
    freshness_hours: int,
) -> Optional[datetime]:
    timestamp = _record_claimed_timestamp(record)
    if timestamp is None:
        return None
    age_hours = max(0.0, (now - timestamp).total_seconds() / 3600)
    return timestamp if age_hours <= freshness_hours else None


def _codex_record_is_completed_delivery(record: dict[str, Any]) -> bool:
    status = _codex_record_status(record)
    if status in _CODEX_FAILURE_STATUSES:
        return False
    if status in _CODEX_SUCCESS_STATUSES:
        return True
    if record.get("completed_at") is not None:
        return True
    exit_code = record.get("exit_code")
    if exit_code == 0:
        return True
    if isinstance(exit_code, str) and exit_code.strip() == "0":
        return True
    return False


def _journal_entry_claims_work(record: dict[str, Any]) -> bool:
    text = "\n".join(_collect_record_text(record))
    return _record_claims_work(record, text)


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _iter_normalized_key_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_evidence_key(key)
            yield normalized, child
            yield from _iter_normalized_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_normalized_key_values(child)


def _string_leaf_values(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, dict):
        values: list[str] = []
        preferred = (
            "identifier",
            "key",
            "name",
            "title",
            "label",
            "display_name",
            "full_name",
            "path",
        )
        for key in preferred:
            if key in value:
                values.extend(_string_leaf_values(value.get(key)))
        if values:
            return values
        for child in value.values():
            values.extend(_string_leaf_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(_string_leaf_values(child))
        return values
    return []


def _first_text_for_keys(record: dict[str, Any], keys: set[str]) -> Optional[str]:
    for key, value in _iter_normalized_key_values(record):
        if key not in keys:
            continue
        values = _string_leaf_values(value)
        if values:
            return values[0]
    return None


def _first_bool_for_keys(record: dict[str, Any], keys: set[str]) -> Optional[bool]:
    for key, value in _iter_normalized_key_values(record):
        if key in keys:
            parsed = _coerce_bool(value)
            if parsed is not None:
                return parsed
    return None


def _capacity_values_for_keys(record: dict[str, Any], keys: set[str]) -> list[int]:
    values: list[int] = []
    for key, value in _iter_normalized_key_values(record):
        if key not in keys:
            continue
        parsed = _coerce_non_negative_int(value)
        if parsed is not None:
            values.append(parsed)
    return values


def _extract_issue_identifier(record: dict[str, Any]) -> Optional[str]:
    return _first_text_for_keys(
        record,
        {
            "external_key",
            "id",
            "identifier",
            "issue_id",
            "issue_identifier",
            "key",
            "linear_issue_id",
        },
    )


def _capacity_candidate_title(record: dict[str, Any]) -> Optional[str]:
    return _first_text_for_keys(record, {"name", "summary", "title"})


def _capacity_candidate_labels(record: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key, value in _iter_normalized_key_values(record):
        if key in {"label", "labels"}:
            labels.extend(_string_leaf_values(value))
    return list(dict.fromkeys(label for label in labels if label))


def _capacity_candidate_repo(record: dict[str, Any]) -> Optional[str]:
    repo = _first_text_for_keys(record, _REPO_BACKED_KEYS)
    if repo:
        return repo
    return None


def _capacity_candidate_status_text(record: dict[str, Any]) -> str:
    values: list[str] = []
    for key, value in _iter_normalized_key_values(record):
        if key in {
            "review_state",
            "state",
            "state_type",
            "status",
            "workflow_state",
            "workflow_state_type",
        }:
            values.extend(_string_leaf_values(value))
    return " ".join(values)


def _capacity_candidate_owner_is_human(
    record: dict[str, Any],
    labels: Iterable[str],
) -> bool:
    label_values = {label.strip().lower() for label in labels}
    label_keys = {_normalize_evidence_key(label) for label in labels}
    if "owner:human" in label_values or "owner_human" in label_keys:
        return True
    owner_values: list[str] = []
    for key, value in _iter_normalized_key_values(record):
        if key in _OWNER_KEYS:
            owner_values.extend(_string_leaf_values(value))
    return any(
        value.strip().lower() == "human"
        or _normalize_evidence_key(value) == "owner_human"
        for value in owner_values
    )


def _capacity_candidate_is_duplicate(
    record: dict[str, Any],
    labels: Iterable[str],
) -> bool:
    if _first_bool_for_keys(record, _DUPLICATE_KEYS) is True:
        return True
    label_keys = {_normalize_evidence_key(label) for label in labels}
    if "duplicate" in label_keys:
        return True
    status = _normalize_evidence_key(_capacity_candidate_status_text(record))
    return "duplicate" in status.split("_")


def _capacity_candidate_ignored_project(
    record: dict[str, Any],
    labels: Iterable[str],
) -> bool:
    if _first_bool_for_keys(record, _IGNORED_PROJECT_KEYS) is True:
        return True
    label_keys = {_normalize_evidence_key(label) for label in labels}
    if "ignored_project" in label_keys:
        return True
    project_text = _first_text_for_keys(record, {"project", "project_name"})
    return _normalize_evidence_key(project_text or "") == "ignored_project"


def _capacity_candidate_repo_unresolved(record: dict[str, Any]) -> bool:
    if _first_bool_for_keys(record, _REPO_UNRESOLVED_KEYS) is True:
        return True
    if _first_bool_for_keys(record, _REPO_RESOLVED_KEYS) is False:
        return True
    return _capacity_candidate_repo(record) is None


def _capacity_candidate_not_ready(record: dict[str, Any]) -> bool:
    status = _normalize_evidence_key(_capacity_candidate_status_text(record))
    if not status:
        return False
    blocked_states = {
        "blocked",
        "canceled",
        "cancelled",
        "closed",
        "complete",
        "completed",
        "done",
        "merged",
        "pr_held",
        "pr_review",
        "pull_request_held",
        "review",
        "review_held",
        "waiting_for_review",
    }
    return any(state in status for state in blocked_states)


def _coerce_work_objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    child_dicts = [item for item in value.values() if isinstance(item, dict)]
    if child_dicts and len(child_dicts) == len(value):
        coerced: list[dict[str, Any]] = []
        for key, item in value.items():
            child = dict(item)
            child.setdefault("identifier", str(key))
            coerced.append(child)
        return coerced
    return [value]


def _collect_capacity_work_items(
    value: Any,
    *,
    source: str,
    timestamp: Optional[datetime],
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_evidence_key(key)
            if normalized in _SELECTED_WORK_KEYS:
                for item in _coerce_work_objects(child):
                    selected.append(
                        {
                            "source": source,
                            "timestamp": timestamp,
                            "record": item,
                        }
                    )
                continue
            if normalized in _BACKLOG_CANDIDATE_KEYS:
                for item in _coerce_work_objects(child):
                    candidates.append(
                        {
                            "source": source,
                            "timestamp": timestamp,
                            "record": item,
                        }
                    )
                continue
            _collect_capacity_work_items(
                child,
                source=source,
                timestamp=timestamp,
                selected=selected,
                candidates=candidates,
            )
    elif isinstance(value, list):
        for child in value:
            _collect_capacity_work_items(
                child,
                source=source,
                timestamp=timestamp,
                selected=selected,
                candidates=candidates,
            )


def _work_is_pr_review_held(record: dict[str, Any]) -> bool:
    status = _normalize_evidence_key(_capacity_candidate_status_text(record))
    if status in {
        "in_review",
        "pr_held",
        "pr_review",
        "pull_request_held",
        "review",
        "review_held",
        "waiting_for_review",
    }:
        return True
    text = " ".join(_collect_record_text(record)).lower()
    mentions_pr = "pull request" in text or re.search(r"\bpr\b", text) is not None
    mentions_hold = any(
        token in text
        for token in ("held", "merge", "review", "waiting")
    )
    return mentions_pr and mentions_hold


def _capacity_candidate_assessment(
    item: dict[str, Any],
    selected_ids: set[str],
) -> dict[str, Any]:
    record = item.get("record") or {}
    labels = _capacity_candidate_labels(record)
    reasons: list[str] = []
    if _capacity_candidate_ignored_project(record, labels):
        reasons.append("ignored_project")
    if _capacity_candidate_owner_is_human(record, labels):
        reasons.append("owner_human")
    if _capacity_candidate_is_duplicate(record, labels):
        reasons.append("duplicate")
    if _capacity_candidate_repo_unresolved(record):
        reasons.append("repo_unresolved")
    identifier = _extract_issue_identifier(record)
    if identifier and identifier in selected_ids:
        reasons.append("selected_work")
    if _capacity_candidate_not_ready(record):
        reasons.append("not_ready_for_parallel_execution")

    timestamp = item.get("timestamp")
    return {
        "id": identifier,
        "title": _capacity_candidate_title(record),
        "repo": _capacity_candidate_repo(record),
        "source": item.get("source"),
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
        "labels": labels,
        "safe": not reasons,
        "exclusion_reasons": reasons,
    }


def _dedupe_capacity_assessments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            str(item.get("id") or ""),
            str(item.get("repo") or ""),
            str(item.get("title") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_capacity_saturation_signal(
    *,
    journal_payload: Any,
    codex_payload: Any,
    ctx_payload: Any,
    now: datetime,
    freshness_hours: int,
) -> dict[str, Any]:
    source_records = (
        ("journal_entries", _iter_records(journal_payload, "entries")),
        ("codex_runs", _iter_codex_records(codex_payload)),
        ("ctx_bindings", _iter_ctx_records(ctx_payload)),
    )
    selected_items: list[dict[str, Any]] = []
    candidate_items: list[dict[str, Any]] = []
    spare_values: list[int] = []
    max_values: list[int] = []
    active_values: list[int] = []

    for source, records in source_records:
        for record in records:
            timestamp = _record_claimed_timestamp(record)
            if timestamp is not None:
                age_hours = max(0.0, (now - timestamp).total_seconds() / 3600)
                if age_hours > freshness_hours:
                    continue
            spare_values.extend(_capacity_values_for_keys(record, _CAPACITY_SPARE_KEYS))
            max_values.extend(_capacity_values_for_keys(record, _CAPACITY_MAX_KEYS))
            active_values.extend(_capacity_values_for_keys(record, _CAPACITY_ACTIVE_KEYS))
            _collect_capacity_work_items(
                record,
                source=source,
                timestamp=timestamp,
                selected=selected_items,
                candidates=candidate_items,
            )

    if spare_values:
        spare_capacity = max(spare_values)
    elif max_values:
        spare_capacity = max(0, max(max_values) - max(active_values or [0]))
    else:
        spare_capacity = 0

    selected_ids = {
        identifier
        for item in selected_items
        if (identifier := _extract_issue_identifier(item.get("record") or {}))
    }
    selected_pr_review_held = any(
        _work_is_pr_review_held(item.get("record") or {})
        for item in selected_items
    )
    assessments = _dedupe_capacity_assessments(
        [
            _capacity_candidate_assessment(item, selected_ids)
            for item in candidate_items
        ]
    )
    safe_candidates = [item for item in assessments if item.get("safe")]
    excluded_candidates = [item for item in assessments if not item.get("safe")]
    fillable_spare_capacity = min(spare_capacity, len(safe_candidates))
    safeguard_counts = {
        reason: sum(
            1
            for item in excluded_candidates
            if reason in set(item.get("exclusion_reasons") or [])
        )
        for reason in (
            "ignored_project",
            "owner_human",
            "duplicate",
            "repo_unresolved",
        )
    }

    actions: list[str] = []
    if fillable_spare_capacity > 0:
        actions.append(
            "fill "
            f"{fillable_spare_capacity} spare capacity slot(s) with independent "
            "safe repo-backed candidate(s) while selected PR/review-held work "
            "remains separate"
            if selected_pr_review_held
            else (
                "fill "
                f"{fillable_spare_capacity} spare capacity slot(s) with independent "
                "safe repo-backed candidate(s)"
            )
        )
    elif spare_capacity > 0 and assessments:
        actions.append(
            "spare capacity exists, but no independent repo-backed candidate "
            "passed safety safeguards"
        )

    if fillable_spare_capacity > 0:
        state = "fillable_spare_capacity"
    elif spare_capacity > 0 and assessments:
        state = "spare_capacity_guarded"
    elif spare_capacity > 0:
        state = "spare_capacity_without_candidates"
    else:
        state = "no_spare_capacity"

    return {
        "state": state,
        "spare_capacity": spare_capacity,
        "safe_repo_backed_candidate_count": len(safe_candidates),
        "raw_candidate_count": len(assessments),
        "excluded_candidate_count": len(excluded_candidates),
        "fillable_spare_capacity": fillable_spare_capacity,
        "selected_pr_review_held": selected_pr_review_held,
        "selected_work_ids": sorted(selected_ids),
        "selected_state_separate": bool(
            selected_pr_review_held and fillable_spare_capacity > 0
        ),
        "safeguard_exclusion_counts": safeguard_counts,
        "safe_candidates": safe_candidates[:5],
        "excluded_candidates": excluded_candidates[:8],
        "actions": actions,
    }


def _build_execution_throughput_signal(
    *,
    journal_payload: Any,
    codex_payload: Any,
    ctx_payload: Any,
    now: datetime,
    freshness_hours: int,
) -> dict[str, Any]:
    recent_codex_deliveries = [
        {
            "run_id": record.get("run_id") or record.get("id"),
            "status": record.get("status"),
            "timestamp": timestamp.isoformat(),
        }
        for record in _iter_codex_records(codex_payload)
        if _codex_record_is_completed_delivery(record)
        if (timestamp := _recent_record_timestamp(record, now, freshness_hours)) is not None
    ]
    recent_journal_work = [
        {
            "id": record.get("id") or record.get("external_key"),
            "timestamp": timestamp.isoformat(),
        }
        for record in _iter_records(journal_payload, "entries")
        if _journal_entry_claims_work(record)
        if (timestamp := _recent_record_timestamp(record, now, freshness_hours)) is not None
    ]
    capacity_saturation = _build_capacity_saturation_signal(
        journal_payload=journal_payload,
        codex_payload=codex_payload,
        ctx_payload=ctx_payload,
        now=now,
        freshness_hours=freshness_hours,
    )
    journal_operator_support = _collect_journal_codex_operator_support(
        journal_payload,
        codex_payload,
    )
    pending_journal_follow_through = [
        delivery
        for delivery in recent_codex_deliveries
        if delivery.get("run_id")
        and delivery.get("run_id") not in journal_operator_support
    ]
    ctx_summary = _summarize_ctx_bindings(ctx_payload, freshness_hours, now)
    codex_count = len(recent_codex_deliveries)
    journal_count = len(recent_journal_work)
    journal_ratio = round(journal_count / codex_count, 4) if codex_count else 1.0
    journal_gap = (
        codex_count >= _THROUGHPUT_CODEX_COMPLETION_MIN
        and journal_ratio < _THROUGHPUT_JOURNAL_RATIO_MIN
    )
    ctx_status = str(ctx_summary.get("status") or "")
    ctx_active_count = int(ctx_summary.get("active_count") or 0)
    ctx_inactive_informational = ctx_status == "inactive" and ctx_active_count == 0
    actions: list[str] = []
    capacity_actions = [
        str(action).strip()
        for action in capacity_saturation.get("actions", [])
        if str(action).strip()
    ]
    if journal_gap:
        if capacity_actions:
            actions = [
                *capacity_actions,
                "continue backfilling journal entries for completed Codex deliveries that lack follow-through evidence",
                "record changed files, tests, PR or commit, and operator decision support for the next completed delivery",
            ]
        else:
            actions = [
                "backfill journal entries for completed Codex deliveries that lack follow-through evidence",
                "record changed files, tests, PR or commit, and operator decision support for the next completed delivery",
                "select completed Codex deliveries without journal follow-through before starting more raw issue volume",
            ]
    elif capacity_actions:
        actions = capacity_actions

    return {
        "state": "codex_delivery_journal_gap" if journal_gap else "balanced",
        "remediation_required": journal_gap,
        "blocking_surface": "journal_follow_through" if journal_gap else "none",
        "recent_completed_codex_count": codex_count,
        "recent_journal_work_item_count": journal_count,
        "journal_to_codex_ratio": journal_ratio,
        "journal_ratio_threshold": _THROUGHPUT_JOURNAL_RATIO_MIN,
        "codex_completion_threshold": _THROUGHPUT_CODEX_COMPLETION_MIN,
        "sample_codex_deliveries": recent_codex_deliveries[:5],
        "sample_journal_work_items": recent_journal_work[:5],
        "pending_journal_follow_through_count": len(pending_journal_follow_through),
        "pending_journal_follow_through_codex_runs": pending_journal_follow_through[:8],
        "actions": actions,
        "ctx_status": ctx_status,
        "ctx_active_count": ctx_active_count,
        "ctx_inactivity_informational": ctx_inactive_informational,
        "ctx_inactivity_blocking": False,
        "capacity_saturation": capacity_saturation,
    }


def _codex_follow_through_gap_example(
    record: dict[str, Any],
    timestamp: datetime,
) -> dict[str, Any]:
    issue_ids = sorted(_collect_linear_issue_ids_from_issue_fields(record))
    example = _recent_record_example(record, timestamp)
    example.update(
        {
            "external_key": record.get("external_key"),
            "linear_issue_ids": issue_ids,
            "required_journal_fields": [
                "selfImprovementFocus[].title",
                "selfImprovementFocus[].activeLinearIssueIds",
                "selfImprovementFocus[].outcomeNote",
                "changedFiles or commitShas",
                "tests or verification",
                "operatorDecisionSupport or nextDecision",
            ],
            "operator_decision_support_path": (
                "entries[*].selfImprovementFocus[*]."
                f"{_OPERATOR_DECISION_SUPPORT_REFERENCE_FIELD_PATH}"
            ),
        }
    )
    return {key: value for key, value in example.items() if value not in (None, [], "")}


def _timestamp_in_window(timestamp: Optional[datetime], now: datetime, window_hours: int) -> bool:
    if timestamp is None:
        return False
    age_hours = (now - timestamp).total_seconds() / 3600
    return 0 <= age_hours <= window_hours


def _iter_recent_journal_records(
    payload: Any,
    now: datetime,
    window_hours: int,
) -> Iterable[tuple[dict[str, Any], datetime]]:
    for record in _iter_records(payload, "entries"):
        timestamp = _record_claimed_timestamp(record)
        if _timestamp_in_window(timestamp, now, window_hours):
            yield record, timestamp


def _iter_recent_completed_codex_records(
    payload: Any,
    now: datetime,
    window_hours: int,
) -> Iterable[tuple[dict[str, Any], datetime]]:
    for record in _iter_codex_records(payload):
        if not _codex_record_is_completed(record):
            continue
        timestamp = _record_timestamp(
            record,
            "completed_at",
            "updated_at",
            "started_at",
            "process_started_at",
            "created_at",
            "timestamp",
        )
        if _timestamp_in_window(timestamp, now, window_hours):
            yield record, timestamp


def _recent_record_example(record: dict[str, Any], timestamp: datetime) -> dict[str, Any]:
    return {
        "id": (
            record.get("id")
            or record.get("run_id")
            or record.get("session_id")
            or record.get("external_key")
        ),
        "timestamp": timestamp.isoformat(),
        "status": record.get("status"),
    }


def _execution_loop_next_action(
    *,
    completed_codex_count: int,
    active_ctx_count: int,
    journal_count: int,
    journal_follow_through_rate: float,
) -> str:
    _ = active_ctx_count
    if completed_codex_count == 0:
        return (
            "Complete one scoped self-improvement task with local Codex and add journal "
            "evidence."
        )
    if journal_count == 0:
        return "Add journal evidence for recent completed Codex deliveries before launching more work."
    if (
        completed_codex_count >= _EXECUTION_LOOP_MANY_COMPLETED_THRESHOLD
        and journal_follow_through_rate < _EXECUTION_LOOP_MIN_JOURNAL_FOLLOW_THROUGH_RATE
    ):
        return "Backfill journal evidence for recent completed Codex deliveries before launching more work."
    return "Keep converting self-improvement work into completed local Codex runs and journal each delivery."


def _evaluate_execution_loop_check(
    *,
    journal_path: Path,
    codex_runs_path: Path,
    ctx_bindings_path: Path,
    now: datetime,
) -> dict[str, Any]:
    journal_payload = _load_json(journal_path)
    codex_payload = _load_json(codex_runs_path)
    ctx_payload = _load_json(ctx_bindings_path)
    window_hours = _EXECUTION_LOOP_WINDOW_DAYS * 24

    recent_journal = list(_iter_recent_journal_records(journal_payload, now, window_hours))
    recent_completed_codex = list(
        _iter_recent_completed_codex_records(codex_payload, now, window_hours)
    )
    journal_operator_support = _collect_journal_codex_operator_support(
        journal_payload,
        codex_payload,
    )
    pending_journal_follow_through = [
        (record, timestamp)
        for record, timestamp in recent_completed_codex
        if (run_id := _codex_run_id(record))
        and run_id not in journal_operator_support
    ]
    ctx_records = list(_iter_ctx_records(ctx_payload))
    active_ctx_records = [record for record in ctx_records if _ctx_record_is_active(record)]
    capacity_saturation = _build_capacity_saturation_signal(
        journal_payload=journal_payload,
        codex_payload=codex_payload,
        ctx_payload=ctx_payload,
        now=now,
        freshness_hours=window_hours,
    )

    completed_codex_count = len(recent_completed_codex)
    journal_count = len(recent_journal)
    active_ctx_count = len(active_ctx_records)
    journal_follow_through_rate = (
        round(journal_count / completed_codex_count, 4)
        if completed_codex_count
        else 1.0
    )
    latest_completed_at = max(
        (timestamp for _record, timestamp in recent_completed_codex),
        default=None,
    )
    latest_journal_at = max(
        (timestamp for _record, timestamp in recent_journal),
        default=None,
    )
    latest_journal_lag_hours = None
    if latest_completed_at is not None and latest_journal_at is not None:
        latest_journal_lag_hours = round(
            (latest_journal_at - latest_completed_at).total_seconds() / 3600,
            2,
        )

    sparse_journal_follow_through = (
        completed_codex_count >= _EXECUTION_LOOP_MANY_COMPLETED_THRESHOLD
        and journal_follow_through_rate < _EXECUTION_LOOP_MIN_JOURNAL_FOLLOW_THROUGH_RATE
    )
    missing_journal_follow_through = completed_codex_count > 0 and journal_count == 0
    no_completed_codex = completed_codex_count == 0
    ctx_inactivity_informational = active_ctx_count == 0
    ctx_state = "active" if active_ctx_count else "inactive_informational"
    next_action = _execution_loop_next_action(
        completed_codex_count=completed_codex_count,
        active_ctx_count=active_ctx_count,
        journal_count=journal_count,
        journal_follow_through_rate=journal_follow_through_rate,
    )
    capacity_actions = [
        str(action).strip()
        for action in capacity_saturation.get("actions", [])
        if str(action).strip()
    ]
    if capacity_actions:
        if missing_journal_follow_through or sparse_journal_follow_through:
            next_action = (
                capacity_actions[0]
                + "; continue journal follow-through backfill for completed deliveries."
            )
        else:
            next_action = capacity_actions[0]

    if missing_journal_follow_through:
        score = 0.55
        detail = "Completed Codex deliveries lack journal follow-through."
    elif sparse_journal_follow_through:
        score = 0.7
        detail = "Completed Codex delivery volume has sparse journal follow-through."
    elif no_completed_codex:
        score = 0.6
        detail = "No completed local Codex deliveries were recorded in the throughput window."
    else:
        score = 1.0
        detail = "Execution loop is converting local Codex deliveries with journal evidence."
        if ctx_inactivity_informational:
            detail += " Inactive ctx is informational on this host."
    if int(capacity_saturation.get("fillable_spare_capacity") or 0) > 0:
        detail += (
            " Independent safe repo-backed candidate(s) can fill spare capacity "
            "while selected PR/review-held work remains separate."
        )

    return _build_benchmark_item(
        "execution_loop",
        "Execution loop",
        score=score,
        weight=0,
        detail=detail,
        critical=False,
        metrics={
            "window_days": _EXECUTION_LOOP_WINDOW_DAYS,
            "completed_codex_runs_14d": completed_codex_count,
            "active_ctx_binding_count": active_ctx_count,
            "journal_entries_14d": journal_count,
            "journal_follow_through_rate": journal_follow_through_rate,
            "minimum_journal_follow_through_rate": (
                _EXECUTION_LOOP_MIN_JOURNAL_FOLLOW_THROUGH_RATE
            ),
            "many_completed_codex_threshold": _EXECUTION_LOOP_MANY_COMPLETED_THRESHOLD,
            "ctx_binding_state": ctx_state,
            "ctx_inactivity_informational": ctx_inactivity_informational,
            "ctx_inactivity_blocking": False,
            "latest_completed_codex_at": (
                latest_completed_at.isoformat() if latest_completed_at else None
            ),
            "latest_journal_entry_at": (
                latest_journal_at.isoformat() if latest_journal_at else None
            ),
            "latest_journal_lag_hours": latest_journal_lag_hours,
            "missing_journal_follow_through": missing_journal_follow_through,
            "sparse_journal_follow_through": sparse_journal_follow_through,
            "next_throughput_action": next_action,
            "capacity_saturation": capacity_saturation,
            "pending_journal_follow_through_count": len(pending_journal_follow_through),
            "pending_journal_follow_through_codex_runs": [
                _codex_follow_through_gap_example(record, timestamp)
                for record, timestamp in pending_journal_follow_through[:8]
            ],
            "journal_operator_support_codex_run_count": len(journal_operator_support),
            "completed_codex_examples": [
                _recent_record_example(record, timestamp)
                for record, timestamp in recent_completed_codex[:5]
            ],
            "journal_examples": [
                _recent_record_example(record, timestamp)
                for record, timestamp in recent_journal[:5]
            ],
        },
    )


def _assess_make_work_item(item: dict[str, Any]) -> dict[str, Any]:
    record = item.get("record") or {}
    text = str(item.get("text") or "")
    if item.get("source") == "codex_runs":
        durable_signals = _codex_durable_signals(record, text)
    else:
        durable_signals = _structured_durable_signals(record)
        durable_signals.update(_text_durable_signals(text))
    status_markers = _status_only_markers(text)
    value_categories = _value_categories_from_signals(durable_signals)
    category_labels = _value_category_labels(value_categories)
    durable = bool(value_categories)
    issue = None
    if not durable and status_markers:
        issue = "status_language_without_value_category_evidence"
    elif not durable:
        issue = "claimed_work_without_value_category_evidence"

    return {
        "source": item.get("source"),
        "id": item.get("id"),
        "timestamp": item.get("timestamp"),
        "durable": durable,
        "signals": sorted(durable_signals),
        "value_categories": value_categories,
        "value_category_labels": category_labels,
        "status_language": bool(status_markers),
        "issue": issue,
        "backfill_fields": list(_CODEX_BACKFILL_EVIDENCE_FIELDS)
        if item.get("source") == "codex_runs" and not value_categories
        else [],
        "remediation": _make_work_remediation(
            value_categories,
            source=item.get("source"),
            record_id=item.get("id"),
        ),
    }


def _codex_journal_skip_applies_to_assessment(
    record: dict[str, Any],
    assessment: dict[str, Any],
) -> bool:
    if not assessment.get("durable"):
        return True
    if _structured_durable_signals(record):
        return False
    signals = set(assessment.get("signals") or [])
    return bool(signals) and signals.issubset(_CODEX_TEXT_DELIVERY_SIGNALS)


def _assess_operator_value_item(
    item: dict[str, Any],
    *,
    journal_operator_evidence: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    make_work = _assess_make_work_item(item)
    record = item.get("record") or {}
    text = str(item.get("text") or "")
    durable_signals = set(make_work.get("signals") or [])
    decision_support_signals = durable_signals.intersection(_OPERATOR_DECISION_SUPPORT_SIGNALS)
    verified_change_signals = durable_signals.intersection(_VERIFIED_SYSTEM_CHANGE_SIGNALS)
    operator_decision_support_evidence = _collect_operator_decision_support_evidence(record)

    if journal_operator_evidence:
        operator_decision_support_evidence = _dedupe_operator_evidence(
            operator_decision_support_evidence + list(journal_operator_evidence)
        )
        decision_support_signals.update(
            signal
            for signal in (
                str(evidence.get("field")).strip()
                for evidence in journal_operator_evidence
                if isinstance(evidence, dict)
            )
            if signal in _OPERATOR_DECISION_SUPPORT_SIGNALS
        )

    if decision_support_signals and not operator_decision_support_evidence:
        operator_decision_support_evidence = _operator_decision_support_text_evidence(text)
    operator_decision_support_evidence = _dedupe_operator_evidence(
        operator_decision_support_evidence
    )[:8]

    item_score = 0.0
    issue = make_work.get("issue")
    if make_work["durable"]:
        if decision_support_signals and verified_change_signals:
            item_score = 1.0
        elif decision_support_signals:
            item_score = 0.65
            issue = "decision_support_without_verified_system_change"
        elif verified_change_signals:
            item_score = 0.45
            issue = "verified_change_without_operator_decision_support"
        else:
            item_score = 0.25
            issue = "durable_evidence_without_operator_value_signal"

    missing_operator_decision_support_fields: list[str] = []
    operator_value_remediation = None
    if verified_change_signals and not decision_support_signals:
        missing_operator_decision_support_fields = sorted([
            "operatorDecisionSupport",
            "nextDecision",
            "selectedWork",
            "decision",
            "blocker",
            "tradeoff",
        ])
        operator_value_remediation = (
            "Backfill operator decision-support evidence on the claimed work item: "
            "operatorDecisionSupport or nextDecision, plus selectedWork, decision, "
            "blocker, or tradeoff when applicable."
        )

    return {
        "source": make_work.get("source"),
        "id": make_work.get("id"),
        "timestamp": make_work.get("timestamp"),
        "score": item_score,
        "durable": make_work["durable"],
        "signals": sorted(durable_signals),
        "operator_decision_support": bool(decision_support_signals),
        "operator_decision_support_signals": sorted(decision_support_signals),
        "operator_decision_support_evidence": operator_decision_support_evidence,
        "missing_operator_decision_support_fields": missing_operator_decision_support_fields,
        "operator_value_remediation": operator_value_remediation,
        "verified_system_change": bool(verified_change_signals),
        "verified_system_change_signals": sorted(verified_change_signals),
        "aligned": bool(decision_support_signals and verified_change_signals),
        "issue": issue,
    }


def _evaluate_anti_make_work_check(
    *,
    journal_path: Path,
    codex_runs_path: Path,
    ctx_bindings_path: Path,
    now: datetime,
    freshness_hours: int,
) -> dict[str, Any]:
    journal_payload = _load_json(journal_path)
    codex_payload = _load_json(codex_runs_path)
    ctx_payload = _load_json(ctx_bindings_path)
    journal_skip_remediations = _collect_journal_codex_skip_remediations(journal_payload)
    assessments: list[dict[str, Any]] = []
    journal_remediated_codex_items: list[dict[str, Any]] = []
    ignored_journal_remediations: list[dict[str, Any]] = []
    for item in _iter_recent_claimed_work_items(
        journal_payload=journal_payload,
        codex_payload=codex_payload,
        ctx_payload=ctx_payload,
        now=now,
        freshness_hours=freshness_hours,
    ):
        assessment = _assess_make_work_item(item)
        run_id = str(assessment.get("id") or "").strip()
        journal_remediation = journal_skip_remediations.get(run_id)
        if (
            assessment.get("source") == "codex_runs"
            and journal_remediation
            and _codex_journal_skip_applies_to_assessment(
                item.get("record") or {},
                assessment,
            )
        ):
            if _codex_record_is_active(item.get("record") or {}):
                assessment["journal_remediation_ignored"] = journal_remediation
                assessment["journal_remediation_ignored_reason"] = "codex_run_active"
                ignored_journal_remediations.append(assessment)
            else:
                journal_remediated_codex_items.append(
                    {**assessment, "journal_remediation": journal_remediation}
                )
                continue
        assessments.append(assessment)
    raw_claimed_count = len(assessments) + len(journal_remediated_codex_items)
    assessed_count = len(assessments)
    durable_count = sum(1 for item in assessments if item["durable"])
    shallow_items = [item for item in assessments if not item["durable"]]
    status_only_count = sum(1 for item in shallow_items if item["status_language"])
    shallow_codex_count = sum(1 for item in shallow_items if item.get("source") == "codex_runs")
    value_category_counts = {
        category: sum(
            1
            for item in assessments
            if category in set(item.get("value_categories") or [])
        )
        for category, _label in _ALLOWED_VALUE_CATEGORIES
    }

    if assessed_count == 0:
        score = 1.0
        if journal_remediated_codex_items:
            detail = "No unremediated claimed work items required anti-make-work evidence."
        else:
            detail = "No claimed work items required anti-make-work evidence."
    elif not shallow_items:
        score = 1.0
        passing_labels = [
            label
            for category, label in _ALLOWED_VALUE_CATEGORIES
            if value_category_counts.get(category)
        ]
        detail = "Claimed work includes allowed value-category evidence: " + ", ".join(passing_labels)
    else:
        score = durable_count / assessed_count
        if status_only_count:
            score = min(score, 0.55 if durable_count else 0.0)
        else:
            score = min(score, 0.4)
        examples = [
            f"{item.get('source')}:{item.get('id') or 'unknown'}"
            for item in shallow_items[:3]
        ]
        allowed_labels = ", ".join(label for _category, label in _ALLOWED_VALUE_CATEGORIES)
        detail = (
            "Claimed work lacks allowed value-category evidence "
            f"({allowed_labels}): "
            + ", ".join(examples)
            + ". Remediation: add category evidence to each shallow claimed-work record."
        )
        if shallow_codex_count:
            detail += (
                " For shallow completed Codex runs, backfill structured fields: "
                + "; ".join(_CODEX_BACKFILL_EVIDENCE_FIELDS)
                + "."
            )
    if journal_remediated_codex_items:
        detail += (
            f" Explicit journal skip/remediation notes exempted "
            f"{len(journal_remediated_codex_items)} inactive shallow Codex run(s)."
        )

    return _build_benchmark_item(
        "anti_make_work_check",
        "Anti make-work check",
        score=score,
        weight=25,
        detail=detail,
        critical=True,
        metrics={
            "raw_claimed_work_item_count": raw_claimed_count,
            "assessed_work_item_count": assessed_count,
            "durable_evidence_count": durable_count,
            "shallow_work_item_count": len(shallow_items),
            "shallow_codex_work_item_count": shallow_codex_count,
            "status_language_only_count": status_only_count,
            "journal_remediated_codex_work_item_count": len(journal_remediated_codex_items),
            "journal_remediated_codex_examples": journal_remediated_codex_items[:5],
            "journal_remediation_ignored_codex_count": len(ignored_journal_remediations),
            "journal_remediation_ignored_codex_examples": ignored_journal_remediations[:5],
            "allowed_value_categories": _allowed_value_category_guidance(),
            "value_category_counts": value_category_counts,
            "durable_examples": [item for item in assessments if item["durable"]][:5],
            "shallow_examples": shallow_items[:5],
        },
    )


def _operator_value_example(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": item.get("source"),
        "id": item.get("id"),
        "timestamp": item.get("timestamp"),
        "score": item.get("score"),
        "issue": item.get("issue"),
        "aligned": item.get("aligned"),
        "verified_system_change": item.get("verified_system_change"),
        "verified_system_change_signals": item.get("verified_system_change_signals") or [],
        "operator_decision_support_signals": item.get("operator_decision_support_signals") or [],
        "missing_operator_decision_support_fields": item.get("missing_operator_decision_support_fields") or [],
        "remediation": item.get("operator_value_remediation"),
        "evidence": item.get("operator_decision_support_evidence") or [],
    }


def _evaluate_operator_value_alignment_check(
    *,
    journal_path: Path,
    codex_runs_path: Path,
    ctx_bindings_path: Path,
    now: datetime,
    freshness_hours: int,
) -> dict[str, Any]:
    journal_payload = _load_json(journal_path)
    codex_payload = _load_json(codex_runs_path)
    ctx_payload = _load_json(ctx_bindings_path)
    journal_operator_support = _collect_journal_codex_operator_support(
        journal_payload,
        codex_payload,
    )
    journal_operator_support_examples = [
        {
            "run_id": run_id,
            "evidence": evidence[:3],
        }
        for run_id, evidence in sorted(journal_operator_support.items())[
            :_JOURNAL_OPERATOR_SUPPORT_EXAMPLE_LIMIT
        ]
    ]
    execution_throughput = _build_execution_throughput_signal(
        journal_payload=journal_payload,
        codex_payload=codex_payload,
        ctx_payload=ctx_payload,
        now=now,
        freshness_hours=freshness_hours,
    )
    assessments = []
    for item in _iter_recent_claimed_work_items(
        journal_payload=journal_payload,
        codex_payload=codex_payload,
        ctx_payload=ctx_payload,
        now=now,
        freshness_hours=freshness_hours,
    ):
        journal_support = None
        if item.get("source") == "codex_runs":
            run_id = str(item.get("id") or "").strip()
            journal_support = journal_operator_support.get(run_id)
        assessments.append(
            _assess_operator_value_item(
                item,
                journal_operator_evidence=journal_support,
            )
        )

    assessed_count = len(assessments)
    durable_count = sum(1 for item in assessments if item["durable"])
    decision_support_count = sum(1 for item in assessments if item["operator_decision_support"])
    verified_change_count = sum(1 for item in assessments if item["verified_system_change"])
    aligned_count = sum(1 for item in assessments if item["aligned"])
    issue_items = [item for item in assessments if item.get("issue")]
    decision_support_examples = [
        _operator_value_example(item)
        for item in assessments
        if item["operator_decision_support"]
    ][:5]
    missing_decision_support_examples = [
        _operator_value_example(item)
        for item in assessments
        if item["verified_system_change"] and not item["operator_decision_support"]
    ][:5]
    missing_decision_support_diagnostics = {
        item["run_id"]: item
        for item in _operator_support_reference_diagnostics(
            journal_payload,
            codex_payload,
            [
                str(item.get("id") or "").strip()
                for item in missing_decision_support_examples
                if item.get("source") == "codex_runs"
                if str(item.get("id") or "").strip()
            ],
        )
    }
    for item in missing_decision_support_examples:
        run_id = str(item.get("id") or "").strip()
        diagnostic = missing_decision_support_diagnostics.get(run_id)
        if diagnostic:
            item["journal_reference_diagnostic"] = diagnostic
    missing_decision_support_journal_diagnostics = [
        missing_decision_support_diagnostics[run_id]
        for run_id in sorted(missing_decision_support_diagnostics)
    ]
    missing_decision_support_fields = sorted(
        {
            field
            for item in missing_decision_support_examples
            for field in item.get("missing_operator_decision_support_fields", [])
            if str(field).strip()
        }
    )
    decision_support_fields = sorted(
        {
            evidence.get("field")
            for item in decision_support_examples
            for evidence in item.get("evidence", [])
            if evidence.get("field")
        }
    )
    decision_support_evidence_count = sum(
        len(item.get("evidence") or [])
        for item in decision_support_examples
    )

    if assessed_count == 0:
        score = 1.0
        detail = "No claimed work items required operator-value assessment."
    else:
        score = sum(float(item["score"]) for item in assessments) / assessed_count
        if verified_change_count and not decision_support_count:
            score = min(score, 0.55)
        if assessed_count >= 3 and not aligned_count:
            score = min(score, 0.55)

        if aligned_count == assessed_count:
            detail = "Claimed work pairs operator decision support with verified system change."
        elif not decision_support_count:
            detail = "Claimed work shows throughput, but lacks operator decision support."
        elif not verified_change_count:
            detail = "Claimed work supports operator decisions, but lacks verified system change."
        else:
            detail = "Operator-value evidence is incomplete across claimed work."
        if execution_throughput.get("remediation_required"):
            detail += (
                " Completed Codex deliveries outpace journal follow-through; "
                "treat journal evidence as the execution-loop bottleneck."
            )
        if decision_support_fields:
            detail += (
                " Decision-support evidence fields: "
                + ", ".join(decision_support_fields)
                + "."
            )
        if missing_decision_support_fields:
            detail += (
                " Missing operator decision-support fields for verified changes: "
                + ", ".join(missing_decision_support_fields)
                + "."
            )

    return _build_benchmark_item(
        "operator_value_alignment",
        "Operator-value alignment",
        score=score,
        weight=30,
        detail=detail,
        critical=True,
        metrics={
            "assessed_work_item_count": assessed_count,
            "durable_evidence_count": durable_count,
            "operator_decision_support_count": decision_support_count,
            "operator_decision_support_evidence_count": decision_support_evidence_count,
            "operator_decision_support_fields": decision_support_fields,
            "missing_operator_decision_support_fields": missing_decision_support_fields,
            "verified_system_change_count": verified_change_count,
            "aligned_work_item_count": aligned_count,
            "operator_decision_support_rate": (
                round(decision_support_count / assessed_count, 4)
                if assessed_count
                else 1.0
            ),
            "verified_system_change_rate": (
                round(verified_change_count / assessed_count, 4)
                if assessed_count
                else 1.0
            ),
            "aligned_work_rate": (
                round(aligned_count / assessed_count, 4)
                if assessed_count
                else 1.0
            ),
            "quantity_guardrail_basis": "average_evidence_quality_not_item_count",
            "journal_operator_support_codex_run_count": len(journal_operator_support),
            "journal_operator_support_codex_run_ids": sorted(journal_operator_support)[
                :_JOURNAL_OPERATOR_SUPPORT_ID_LIMIT
            ],
            "journal_operator_support_examples": journal_operator_support_examples,
            "execution_throughput": execution_throughput,
            "issue_examples": issue_items[:5],
            "aligned_examples": [item for item in assessments if item["aligned"]][:5],
            "operator_decision_support_examples": decision_support_examples,
            "missing_operator_decision_support_examples": missing_decision_support_examples,
            "missing_operator_decision_support_journal_diagnostics": (
                missing_decision_support_journal_diagnostics
            ),
        },
    )


def _weighted_project_score(checks: dict[str, dict[str, Any]]) -> float:
    total_weight = sum(max(0, int(check.get("weight") or 0)) for check in checks.values())
    if total_weight <= 0:
        return 0.0
    weighted_score = sum(
        float(check.get("score") or 0.0) * max(0, int(check.get("weight") or 0))
        for check in checks.values()
    )
    return round((weighted_score / total_weight) * 100, 2)


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_benchmark_history_entries(history: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("evaluations", "runs"):
        entries = history.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield entry


def _history_check_scores(history: dict[str, Any], check_id: str) -> list[float]:
    scores: list[float] = []
    for entry in _iter_benchmark_history_entries(history):
        score = _coerce_score(_benchmark_entry_check_value(entry, check_id))
        if score is not None:
            scores.append(score)
    return scores


def _latest_history_project_score(history: dict[str, Any]) -> Optional[float]:
    for entry in reversed(list(_iter_benchmark_history_entries(history))):
        score = _coerce_score(entry.get("project_score"))
        if score is None:
            score = _coerce_score(entry.get("score"))
        if score is not None:
            return score
    return None


def _score_direction(current: float, previous: Optional[float], *, threshold: float = 0.01) -> str:
    if previous is None:
        return "stable"
    delta = current - previous
    if delta > threshold:
        return "positive"
    if delta < -threshold:
        return "negative"
    return "stable"


def _rounded_float(value: Any, digits: int = 4) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _population_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _detect_stabilization_hold(scores: list[float]) -> dict[str, Any]:
    hold_window = 3
    delta_tolerance = 0.001
    range_tolerance = 0.005
    min_recovery_gap = 0.05

    recent_scores = scores[-hold_window:] if len(scores) >= hold_window else list(scores)
    recent_deltas = [
        recent_scores[idx] - recent_scores[idx - 1]
        for idx in range(1, len(recent_scores))
    ]
    prior_scores = scores[:-hold_window]
    prior_peak = max(prior_scores) if prior_scores else None
    current_score = scores[-1] if scores else None
    recovery_gap = (
        prior_peak - current_score
        if prior_peak is not None and current_score is not None
        else 0.0
    )
    recent_range = max(recent_scores) - min(recent_scores) if recent_scores else 0.0
    low_variance_hold = (
        len(recent_scores) == hold_window
        and recent_range <= range_tolerance
        and len(recent_deltas) == hold_window - 1
        and all(abs(delta) <= delta_tolerance for delta in recent_deltas)
    )
    active = (
        len(scores) >= 6
        and current_score is not None
        and _check_status(current_score) != "pass"
        and recovery_gap >= min_recovery_gap
        and low_variance_hold
    )
    recovered = (
        len(scores) >= 6
        and current_score is not None
        and _check_status(current_score) == "pass"
        and low_variance_hold
    )
    state = "none"
    if active:
        state = "stabilization_hold"
    elif recovered:
        state = "recovered_low_variance"

    return {
        "active": active,
        "recovered": recovered,
        "settled": active or recovered,
        "state": state,
        "sample_count": len(scores),
        "hold_window": hold_window,
        "recent_scores": [round(score, 4) for score in recent_scores],
        "recent_deltas": [round(delta, 4) for delta in recent_deltas],
        "prior_peak": _rounded_float(prior_peak),
        "current_score": _rounded_float(current_score),
        "recovery_gap": round(recovery_gap, 4),
        "recent_range": round(recent_range, 4),
        "delta_tolerance": delta_tolerance,
        "range_tolerance": range_tolerance,
        "mitigation": (
            "Hold the operator-value guardrail steady instead of escalating inactive drift."
            if active
            else ""
        ),
        "next_action": (
            "Require recovery in operator-value evidence before treating the plateau as resolved."
            if active
            else ""
        ),
    }


def _coerce_check_status(value: Any) -> str:
    score = _coerce_score(value)
    if isinstance(value, dict):
        status = str(value.get("status") or "").strip().lower()
        if status:
            return status
    if score is not None:
        return _check_status(score)
    return "unknown"


def _normalize_benchmark_check(value: Any) -> Optional[dict[str, Any]]:
    score = _coerce_score(value)
    if score is None:
        return None
    return {"score": round(score, 4), "status": _coerce_check_status(value)}


def _benchmark_entry_check_value(entry: dict[str, Any], check_id: str) -> Any:
    checks = entry.get("checks")
    if isinstance(checks, dict) and check_id in checks:
        return checks.get(check_id)
    for field in _LEGACY_BENCHMARK_CHECK_FIELD_ALIASES.get(check_id, (check_id,)):
        if field in entry:
            return entry.get(field)
    return None


def _benchmark_indicator_series(
    history: dict[str, Any],
    current_checks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for entry in _iter_benchmark_history_entries(history):
        normalized_checks = {
            check_id: normalized
            for check_id in _LEADING_INDICATOR_CHECK_IDS
            if (
                normalized := _normalize_benchmark_check(
                    _benchmark_entry_check_value(entry, check_id)
                )
            )
            is not None
        }
        if not normalized_checks:
            continue
        series.append(
            {
                "source": "history",
                "generated_at": entry.get("generated_at") or entry.get("evaluated_at"),
                "checks": normalized_checks,
            }
        )

    normalized_current = {
        check_id: normalized
        for check_id in _LEADING_INDICATOR_CHECK_IDS
        if (normalized := _normalize_benchmark_check(current_checks.get(check_id))) is not None
    }
    if normalized_current:
        series.append({"source": "current", "generated_at": None, "checks": normalized_current})
    return series


def _check_score_series(series: list[dict[str, Any]], check_id: str) -> list[float]:
    scores: list[float] = []
    for entry in series:
        check = (entry.get("checks") or {}).get(check_id)
        score = _coerce_score(check)
        if score is not None:
            scores.append(score)
    return scores


def _check_status_series(series: list[dict[str, Any]], check_id: str) -> list[str]:
    statuses: list[str] = []
    for entry in series:
        check = (entry.get("checks") or {}).get(check_id)
        if check is None:
            continue
        status = _coerce_check_status(check)
        if status != "unknown":
            statuses.append(status)
    return statuses


def _harbinger_payload(
    *,
    triggered: bool,
    evidence: dict[str, Any],
    mitigation: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "triggered": triggered,
        "severity": "fail" if triggered else "none",
        "evidence": evidence,
        "mitigation": mitigation,
        "next_action": next_action,
    }


def _compact_harbinger_metric(value: Any) -> str:
    if isinstance(value, float):
        return str(round(value, 4))
    if isinstance(value, list):
        return "[" + ", ".join(_compact_harbinger_metric(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = [
            f"{key}: {_compact_harbinger_metric(value[key])}"
            for key in sorted(value)
        ]
        return "{" + ", ".join(parts) + "}"
    if value is None:
        return "None"
    return str(value)


def _harbinger_evidence_summary(harbinger: str, evidence: dict[str, Any]) -> str:
    fields = _HARBINGER_EVIDENCE_FIELDS.get(harbinger, tuple(evidence.keys()))
    parts = [
        f"{field}={_compact_harbinger_metric(evidence[field])}"
        for field in fields
        if field in evidence
    ]
    return "; ".join(parts)


def _annotate_harbinger_payload(
    harbinger: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    annotated = dict(payload)
    evidence = annotated.get("evidence") if isinstance(annotated.get("evidence"), dict) else {}
    annotated["harbinger"] = harbinger
    annotated["evidence_summary"] = _harbinger_evidence_summary(harbinger, evidence)
    return annotated


def _leading_indicator_mitigation_items(
    triggered_harbingers: Iterable[str],
    scorecard: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for harbinger in triggered_harbingers:
        card = scorecard[harbinger]
        items.append(
            {
                "harbinger": harbinger,
                "evidence_summary": card["evidence_summary"],
                "mitigation": card["mitigation"],
                "next_action": card["next_action"],
                "evidence": card["evidence"],
            }
        )
    return items


def _format_harbinger_reporting_detail(harbinger: str, card: dict[str, Any]) -> str:
    state = "triggered" if card.get("triggered") else "clear"
    evidence_summary = str(card.get("evidence_summary") or "").strip()
    mitigation = str(card.get("mitigation") or "").strip()
    next_action = str(card.get("next_action") or "").strip()
    detail_parts = [f"{harbinger}: {state}"]
    if evidence_summary:
        detail_parts.append(f"evidence: {evidence_summary}")
    if mitigation:
        detail_parts.append(f"mitigation: {mitigation}")
    if next_action:
        detail_parts.append(f"next_action: {next_action}")
    return "; ".join(detail_parts)


def _build_leading_indicator_report(drift_check: dict[str, Any]) -> dict[str, Any]:
    metrics = drift_check.get("metrics") if isinstance(drift_check, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    scorecard = metrics.get("harbinger_scorecard") if isinstance(metrics, dict) else {}
    scorecard = scorecard if isinstance(scorecard, dict) else {}
    triggered_harbingers = [
        str(item)
        for item in metrics.get("triggered_harbingers", [])
        if str(item).strip()
    ]
    harbingers: dict[str, dict[str, Any]] = {}
    for harbinger in _LEADING_INDICATOR_HARBINGERS:
        raw_card = scorecard.get(harbinger)
        card = dict(raw_card) if isinstance(raw_card, dict) else {}
        evidence = card.get("evidence") if isinstance(card.get("evidence"), dict) else {}
        evidence_summary = str(card.get("evidence_summary") or "").strip()
        if not evidence_summary and evidence:
            evidence_summary = _harbinger_evidence_summary(harbinger, evidence)
        report_card = {
            "triggered": bool(card.get("triggered")),
            "severity": str(
                card.get("severity") or ("fail" if card.get("triggered") else "none")
            ),
            "evidence_summary": evidence_summary,
            "mitigation": str(card.get("mitigation") or ""),
            "next_action": str(card.get("next_action") or ""),
            "evidence": evidence,
        }
        report_card["reporting_detail"] = _format_harbinger_reporting_detail(
            harbinger,
            report_card,
        )
        harbingers[harbinger] = report_card

    return {
        "contract_version": LEADING_INDICATOR_REPORT_CONTRACT_VERSION,
        "score": drift_check.get("score"),
        "status": drift_check.get("status"),
        "detail": drift_check.get("detail"),
        "triggered_harbingers": triggered_harbingers,
        "operator_value_score_series": metrics.get("operator_value_score_series") or [],
        "stabilization_hold": metrics.get("stabilization_hold") or {},
        "recommended_mitigations": metrics.get("recommended_mitigations") or [],
        "harbingers": harbingers,
    }


def _detect_critical_slowing_down(
    scores: list[float],
    stabilization_hold: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    stabilization_hold = stabilization_hold or _detect_stabilization_hold(scores)
    recent_scores = [round(score, 4) for score in scores[-5:]]
    prior_peak = max(scores[:-1]) if len(scores) > 1 else (scores[-1] if scores else None)
    current_score = scores[-1] if scores else None
    recovery_gap = (prior_peak - current_score) if prior_peak is not None and current_score is not None else 0.0
    deltas = [scores[idx] - scores[idx - 1] for idx in range(1, len(scores))]
    recent_deltas = deltas[-3:]
    flat_or_negative_count = sum(1 for delta in recent_deltas if delta <= 0.01)
    average_recent_delta = sum(recent_deltas) / len(recent_deltas) if recent_deltas else None
    active_signal = (
        len(scores) >= 5
        and recovery_gap >= 0.05
        and len(recent_deltas) >= 3
        and flat_or_negative_count == len(recent_deltas)
    )
    triggered = active_signal and not stabilization_hold["settled"]

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "sample_count": len(scores),
            "recent_scores": recent_scores,
            "prior_peak": _rounded_float(prior_peak),
            "current_score": _rounded_float(current_score),
            "recovery_gap": round(recovery_gap, 4),
            "recent_deltas": [round(delta, 4) for delta in recent_deltas],
            "average_recent_delta": _rounded_float(average_recent_delta),
            "flat_or_negative_delta_count": flat_or_negative_count,
            "active_signal": active_signal,
            "stabilization_hold_active": stabilization_hold["active"],
            "settled_operator_value_state": stabilization_hold["state"],
            "stabilization_hold": stabilization_hold,
        },
        mitigation="Stop expanding self-improvement scope until the lagging operator-value signal recovers.",
        next_action="Select the next maintenance item only if it restores operator decision support plus verified change evidence.",
    )


def _detect_variance_explosion(
    scores: list[float],
    stabilization_hold: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    stabilization_hold = stabilization_hold or _detect_stabilization_hold(scores)
    recent_scores = scores[-4:]
    baseline_scores = scores[:-4]
    baseline_stddev = _population_stddev(baseline_scores)
    recent_stddev = _population_stddev(recent_scores)
    recent_range = max(recent_scores) - min(recent_scores) if recent_scores else 0.0
    active_signal = (
        len(scores) >= 6
        and len(recent_scores) >= 4
        and recent_range >= 0.2
        and recent_stddev >= max(0.08, baseline_stddev * 3)
    )
    triggered = active_signal and not stabilization_hold["settled"]

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "sample_count": len(scores),
            "baseline_scores": [round(score, 4) for score in baseline_scores[-4:]],
            "recent_scores": [round(score, 4) for score in recent_scores],
            "baseline_stddev": round(baseline_stddev, 4),
            "recent_stddev": round(recent_stddev, 4),
            "recent_range": round(recent_range, 4),
            "active_signal": active_signal,
            "stabilization_hold_active": stabilization_hold["active"],
            "settled_operator_value_state": stabilization_hold["state"],
            "stabilization_hold": stabilization_hold,
        },
        mitigation="Treat the benchmark as unstable and stop optimizing for throughput until score variance narrows.",
        next_action="Run one focused stabilization pass and require the next run to include low-variance evidence before broadening lane selection.",
    )


def _detect_flickering(series: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = _check_status_series(series, "operator_value_alignment")[-6:]
    transition_count = sum(
        1
        for idx in range(1, len(statuses))
        if statuses[idx] != statuses[idx - 1]
    )
    pass_boundary_crossings = sum(
        1
        for idx in range(1, len(statuses))
        if (statuses[idx] == "pass") != (statuses[idx - 1] == "pass")
    )
    triggered = len(statuses) >= 5 and transition_count >= 3 and pass_boundary_crossings >= 2

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "sample_count": len(statuses),
            "recent_statuses": statuses,
            "transition_count": transition_count,
            "pass_boundary_crossings": pass_boundary_crossings,
        },
        mitigation="Do not treat a single passing run as stable while the signal flickers across pass/fail boundaries.",
        next_action="Require two consecutive stable passing runs before raw issue-selection volume is allowed again.",
    )


def _detect_correlation_explosion(series: list[dict[str, Any]]) -> dict[str, Any]:
    current = series[-1] if series else {}
    current_checks = current.get("checks") if current.get("source") == "current" else {}
    check_deltas: dict[str, float] = {}
    for check_id in _LEADING_INDICATOR_CHECK_IDS:
        current_score = _coerce_score((current_checks or {}).get(check_id))
        if current_score is None:
            continue
        previous_score = None
        for entry in reversed(series[:-1]):
            previous_score = _coerce_score((entry.get("checks") or {}).get(check_id))
            if previous_score is not None:
                break
        if previous_score is not None:
            check_deltas[check_id] = round(current_score - previous_score, 4)

    dropped_checks = {
        check_id: delta
        for check_id, delta in check_deltas.items()
        if delta <= -0.05
    }
    triggered = len(dropped_checks) >= 3

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "check_deltas": check_deltas,
            "dropped_check_count": len(dropped_checks),
            "dropped_checks": sorted(dropped_checks),
            "correlated_drop_threshold": -0.05,
        },
        mitigation="Treat simultaneous benchmark-check degradation as coupled risk, not isolated check noise.",
        next_action="Pick a maintenance item that improves the shared evidence path before selecting feature or volume work.",
    )


def _build_leading_indicator_scorecard(
    history: dict[str, Any],
    current_checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    series = _benchmark_indicator_series(history, current_checks)
    operator_scores = _check_score_series(series, "operator_value_alignment")
    stabilization_hold = _detect_stabilization_hold(operator_scores)
    raw_scorecard = {
        "critical_slowing_down": _detect_critical_slowing_down(
            operator_scores,
            stabilization_hold,
        ),
        "variance_explosion": _detect_variance_explosion(
            operator_scores,
            stabilization_hold,
        ),
        "flickering": _detect_flickering(series),
        "correlation_explosion": _detect_correlation_explosion(series),
    }
    scorecard = {
        harbinger: _annotate_harbinger_payload(harbinger, payload)
        for harbinger, payload in raw_scorecard.items()
    }
    triggered_harbingers = [
        harbinger
        for harbinger in _LEADING_INDICATOR_HARBINGERS
        if scorecard[harbinger]["triggered"]
    ]
    return {
        "series_sample_count": len(series),
        "operator_value_score_series": [round(score, 4) for score in operator_scores[-8:]],
        "stabilization_hold": stabilization_hold,
        "scorecard": scorecard,
        "triggered_harbingers": triggered_harbingers,
        "recommended_mitigations": _leading_indicator_mitigation_items(
            triggered_harbingers,
            scorecard,
        ),
    }


def _format_harbinger_mitigation_detail(
    mitigations: Iterable[dict[str, Any]],
    *,
    limit: int = 4,
) -> str:
    parts: list[str] = []
    for item in list(mitigations)[:limit]:
        harbinger = str(item.get("harbinger") or "").strip()
        evidence_summary = str(item.get("evidence_summary") or "").strip()
        mitigation = str(item.get("mitigation") or "").strip()
        if not harbinger:
            continue
        detail = harbinger
        if evidence_summary:
            detail += f" evidence: {evidence_summary}"
        if mitigation:
            detail += f"; mitigation: {mitigation}"
        parts.append(detail)
    return " | ".join(parts)


def _format_execution_throughput_remediation(signal: dict[str, Any]) -> str:
    if not signal.get("remediation_required"):
        return ""
    actions = [
        str(action).strip()
        for action in signal.get("actions", [])
        if str(action).strip()
    ]
    action_detail = "; ".join(actions[:3]) or "repair journal follow-through"
    ctx_detail = (
        "inactive ctx is informational, not the throughput blocker"
        if signal.get("ctx_inactivity_informational")
        else "ctx inactivity is not the throughput blocker"
    )
    capacity = signal.get("capacity_saturation") or {}
    capacity_detail = ""
    if int(capacity.get("fillable_spare_capacity") or 0) > 0:
        capacity_detail = (
            "; "
            f"fillable_spare_capacity={capacity.get('fillable_spare_capacity')} "
            "with independent safe repo-backed candidate(s)"
        )
    return (
        "Execution-loop throughput remediation: "
        f"{signal.get('recent_completed_codex_count')} completed Codex run(s) vs "
        f"{signal.get('recent_journal_work_item_count')} journal work item(s); "
        f"{ctx_detail}{capacity_detail}; next actions: {action_detail}."
    )


def _execution_throughput_remediation_payload(signal: dict[str, Any]) -> dict[str, Any]:
    if not signal.get("remediation_required"):
        return {"required": False}
    return {
        "required": True,
        "blocking_surface": signal.get("blocking_surface"),
        "recent_completed_codex_count": signal.get("recent_completed_codex_count"),
        "recent_journal_work_item_count": signal.get("recent_journal_work_item_count"),
        "journal_to_codex_ratio": signal.get("journal_to_codex_ratio"),
        "actions": signal.get("actions") or [],
        "capacity_saturation": signal.get("capacity_saturation") or {},
        "ctx_status": signal.get("ctx_status"),
        "ctx_active_count": signal.get("ctx_active_count"),
        "ctx_inactivity_blocking": False,
        "detail": _format_execution_throughput_remediation(signal),
    }


def _evaluate_leading_indicator_drift_check(
    operator_value_check: dict[str, Any],
    history: dict[str, Any],
    current_checks: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    current_score = float(operator_value_check.get("score") or 0.0)
    operator_value_metrics = dict(operator_value_check.get("metrics") or {})
    execution_throughput = dict(operator_value_metrics.get("execution_throughput") or {})
    execution_throughput_detail = _format_execution_throughput_remediation(
        execution_throughput
    )
    prior_scores = _history_check_scores(history, "operator_value_alignment")
    previous_score = prior_scores[-1] if prior_scores else None
    delta = round(current_score - previous_score, 4) if previous_score is not None else None
    regressing = delta is not None and delta < _LEADING_INDICATOR_WARN_DELTA
    materially_regressing = (
        delta is not None and delta <= _LEADING_INDICATOR_FAIL_DELTA
    )
    indicator_payload = _build_leading_indicator_scorecard(
        history,
        current_checks or {"operator_value_alignment": operator_value_check},
    )
    triggered_harbingers = indicator_payload["triggered_harbingers"]

    if triggered_harbingers:
        score = max(0.2, 0.6 - (0.15 * len(triggered_harbingers)))
        if regressing:
            score = min(score, 0.5)
        mitigation_detail = _format_harbinger_mitigation_detail(
            indicator_payload["recommended_mitigations"],
        )
        detail = (
            "Leading indicators triggered: "
            + ", ".join(triggered_harbingers)
            + "; run mitigation before expanding self-improvement scope."
        )
        if mitigation_detail:
            detail += " " + mitigation_detail
    elif indicator_payload["stabilization_hold"]["active"]:
        score = 0.85
        detail = (
            "Operator-value leading indicator is in stabilization hold after a degraded plateau; "
            "keep operator-value guardrails active until recovery."
        )
    elif previous_score is None:
        score = 1.0
        detail = "No prior operator-value score; drift not assessed."
    elif materially_regressing:
        score = 0.5
        detail = "Operator-value alignment is regressing; keep quantity guardrail active."
    elif regressing:
        score = 0.65
        detail = (
            "Operator-value alignment moved slightly lower without a triggered "
            "harbinger; keep watching the guardrail without blocking recovery."
        )
    else:
        score = 1.0
        detail = "Operator-value leading indicator is stable or improving."

    if execution_throughput_detail:
        detail += " " + execution_throughput_detail
        score = min(score, 0.6)

    metrics = operator_value_metrics
    metrics.update(
        {
            "previous_operator_value_score": (
                round(previous_score, 4) if previous_score is not None else None
            ),
            "current_operator_value_score": round(current_score, 4),
            "operator_value_delta": delta,
            "prior_operator_value_sample_count": len(prior_scores),
            "leading_indicator_contract_version": "harbinger_scorecard.v2",
            "series_sample_count": indicator_payload["series_sample_count"],
            "operator_value_score_series": indicator_payload["operator_value_score_series"],
            "stabilization_hold": indicator_payload["stabilization_hold"],
            "triggered_harbingers": triggered_harbingers,
            "harbinger_scorecard": indicator_payload["scorecard"],
            "recommended_mitigations": indicator_payload["recommended_mitigations"],
            "execution_throughput_remediation": _execution_throughput_remediation_payload(
                execution_throughput
            ),
        }
    )
    drift_check = _build_benchmark_item(
        "leading_indicator_drift",
        "Leading-indicator drift",
        score=score,
        weight=20,
        detail=detail,
        critical=True,
        metrics=metrics,
    )
    drift_check["report"] = _build_leading_indicator_report(drift_check)
    return drift_check


def _build_issue_selection_summary(
    checks: dict[str, dict[str, Any]],
    gate: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    guardrail_checks = {
        name: check
        for name, check in checks.items()
        if name in {
            "reliability_gate",
            "execution_loop",
            "anti_make_work_check",
            "operator_value_alignment",
            "leading_indicator_drift",
        }
        and check.get("status") != "pass"
    }
    quantity_guardrail_active = bool(guardrail_checks)
    reliability_blocked = "reliability_gate" in guardrail_checks
    execution_blocked = "execution_loop" in guardrail_checks
    gate = gate or {}
    ctx_remediation = gate.get("ctx_remediation") or {}
    ontology_repair = (gate.get("ontology") or {}).get("external_repair") or {}
    operator_metrics = (checks.get("operator_value_alignment") or {}).get("metrics") or {}
    drift_metrics = (checks.get("leading_indicator_drift") or {}).get("metrics") or {}
    execution_throughput = operator_metrics.get("execution_throughput") or {}
    execution_remediation = (
        drift_metrics.get("execution_throughput_remediation")
        or _execution_throughput_remediation_payload(execution_throughput)
    )
    execution_actions = [
        str(action).strip()
        for action in execution_remediation.get("actions", [])
        if str(action).strip()
    ]
    capacity_saturation = (
        execution_remediation.get("capacity_saturation")
        or execution_throughput.get("capacity_saturation")
        or {}
    )
    fillable_capacity = int(capacity_saturation.get("fillable_spare_capacity") or 0)
    remediation_actions = [
        str(item.get("action") or "").strip()
        for item in (ctx_remediation, ontology_repair)
        if item.get("required") and str(item.get("action") or "").strip()
    ]
    remediation_actions.extend(execution_actions)
    if reliability_blocked:
        recommended_focus = "self-improvement evidence freshness repair"
        detail = (
            "Repair self-improvement evidence freshness before selecting throughput or operator-value work: "
            + "; ".join(remediation_actions or ["inspect reliability gate provenance"])
        )
    elif execution_remediation.get("required"):
        if fillable_capacity > 0:
            recommended_focus = "parallel safe repo-backed execution"
            detail = (
                "Fill spare execution capacity with independent safe repo-backed candidates "
                "while selected PR/review-held work remains separate: "
                f"{fillable_capacity} fillable slot(s), "
                f"{capacity_saturation.get('safe_repo_backed_candidate_count')} safe candidate(s). "
                "Continue journal follow-through for completed Codex deliveries. "
                + "; ".join(execution_actions)
            )
        else:
            recommended_focus = "Codex delivery journal follow-through"
            detail = (
                "Prioritize completed Codex delivery follow-through before selecting more issue volume: "
                f"{execution_remediation.get('recent_completed_codex_count')} recent completed Codex run(s), "
                f"{execution_remediation.get('recent_journal_work_item_count')} journal work item(s). "
                "Inactive ctx evidence is informational on this host. "
                + "; ".join(execution_actions)
            )
    elif execution_blocked:
        execution = checks.get("execution_loop") or {}
        action = str(
            ((execution.get("metrics") or {}).get("next_throughput_action"))
            or "convert planned self-improvement work into completed Codex delivery plus journal evidence"
        )
        recommended_focus = "self-improvement execution follow-through"
        detail = "Restore execution-loop throughput before selecting raw volume work: " + action
    elif quantity_guardrail_active:
        recommended_focus = "operator decision support plus verified system change"
        detail = (
            "Do not select issues because they increase task count; "
            "prefer work with operator decision support and verified change evidence."
        )
    else:
        recommended_focus = "normal lane selection"
        detail = "Benchmark guardrails permit normal lane selection."

    return {
        "quantity_guardrail_active": quantity_guardrail_active,
        "suppress_raw_throughput_selection": quantity_guardrail_active,
        "blocked_checks": sorted(guardrail_checks),
        "recommended_focus": recommended_focus,
        "detail": detail,
        "remediation_actions": remediation_actions,
        "execution_throughput": execution_remediation,
        "parallel_repo_backed_selection_allowed": not reliability_blocked,
        "parallel_repo_backed_selection_blocker": (
            "reliability_gate" if reliability_blocked else None
        ),
    }


def _operator_decision_support_summary(metrics: dict[str, Any]) -> str:
    fields = [
        str(item)
        for item in metrics.get("operator_decision_support_fields", [])
        if str(item).strip()
    ]
    if fields:
        examples = metrics.get("operator_decision_support_examples") or []
        first = examples[0] if examples else {}
        source = first.get("source") or "unknown"
        item_id = first.get("id") or "unknown"
        return (
            "Operator decision-support evidence captured "
            f"({', '.join(fields)}) in {source}:{item_id}."
        )

    missing = metrics.get("missing_operator_decision_support_examples") or []
    if missing:
        examples = [
            f"{item.get('source')}:{item.get('id') or 'unknown'}"
            for item in missing[:3]
        ]
        return (
            "Operator decision-support evidence missing for verified system changes: "
            + ", ".join(examples)
            + "."
        )

    if int(metrics.get("assessed_work_item_count") or 0) == 0:
        return "No claimed work items required operator decision-support evidence."
    return "No explicit operator decision-support evidence was captured."


def _build_operator_summary(
    checks: dict[str, dict[str, Any]],
    issue_selection: dict[str, Any],
) -> dict[str, str]:
    execution = checks["execution_loop"]
    operator_value = checks["operator_value_alignment"]
    drift = checks["leading_indicator_drift"]
    operator_value_metrics = operator_value.get("metrics") or {}
    return {
        "execution_loop": str(execution.get("detail") or ""),
        "operator_value_alignment": str(operator_value.get("detail") or ""),
        "operator_decision_support_evidence": _operator_decision_support_summary(
            operator_value_metrics
        ),
        "leading_indicator_drift": str(drift.get("detail") or ""),
        "issue_selection": str(issue_selection.get("detail") or ""),
    }


def evaluate_self_improvement_benchmark(
    *,
    journal_path: Path = DEFAULT_JOURNAL_PATH,
    codex_runs_path: Path = DEFAULT_CODEX_RUNS_PATH,
    ctx_bindings_path: Path = DEFAULT_CTX_BINDINGS_PATH,
    ontology_root: Path = DEFAULT_ONTOLOGY_ROOT,
    history_path: Path = DEFAULT_BENCHMARK_HISTORY_PATH,
    now: Optional[datetime] = None,
    freshness_hours: int = DEFAULT_FRESHNESS_HOURS,
    active_stale_hours: int = DEFAULT_ACTIVE_STALE_HOURS,
    persist: bool = True,
) -> dict[str, Any]:
    current = now or datetime.now(tz=timezone.utc)
    gate = evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        ontology_root=ontology_root,
        now=current,
        freshness_hours=freshness_hours,
        active_stale_hours=active_stale_hours,
    )
    source_statuses = [
        str((entry or {}).get("status") or "")
        for entry in (gate.get("sources") or {}).values()
        if isinstance(entry, dict)
    ]
    stale_source_count = sum(status in {"stale", "missing", "degraded"} for status in source_statuses)

    reliability_score = 1.0
    reliability_score -= 0.2 * stale_source_count
    reliability_score -= 0.15 if gate.get("stale_active_codex") else 0.0
    reliability_score -= 0.15 if gate.get("stale_active_ctx") else 0.0
    reliability_score -= 0.15 if gate.get("planning_contradictions") else 0.0
    reliability_score -= (
        0.15
        if str((gate.get("ontology") or {}).get("status") or "") in {"stale", "missing", "degraded"}
        else 0.0
    )
    if gate.get("status") == "degraded":
        reliability_score = min(reliability_score, 0.45)

    detail = (
        "Reliability floor is healthy."
        if gate.get("status") == "healthy"
        else "; ".join(
            gate.get("reasons")
            or gate.get("warnings")
            or gate.get("contradictions")
            or ["Reliability floor degraded."]
        )
    )
    reliability_gate = _build_benchmark_item(
        "reliability_gate",
        "Reliability gate",
        score=reliability_score,
        weight=25,
        detail=detail,
        critical=True,
        metrics={
            "gate_status": gate.get("status"),
            "stale_source_count": stale_source_count,
            "warning_count": len(gate.get("warnings") or []),
            "contradiction_count": len(gate.get("contradictions") or []),
            "freshness_spread_hours": gate.get("freshness_spread_hours"),
            "ctx_remediation_required": bool(
                (gate.get("ctx_remediation") or {}).get("required")
            ),
        },
    )
    execution_loop = _evaluate_execution_loop_check(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        now=current,
    )
    anti_make_work_check = _evaluate_anti_make_work_check(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        now=current,
        freshness_hours=freshness_hours,
    )
    operator_value_alignment = _evaluate_operator_value_alignment_check(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        now=current,
        freshness_hours=freshness_hours,
    )
    history = _load_benchmark_history(history_path)
    leading_indicator_drift = _evaluate_leading_indicator_drift_check(
        operator_value_alignment,
        history,
        {
            "reliability_gate": reliability_gate,
            "anti_make_work_check": anti_make_work_check,
            "operator_value_alignment": operator_value_alignment,
        },
    )
    checks = {
        "reliability_gate": reliability_gate,
        "execution_loop": execution_loop,
        "anti_make_work_check": anti_make_work_check,
        "operator_value_alignment": operator_value_alignment,
        "leading_indicator_drift": leading_indicator_drift,
    }
    project_score = _weighted_project_score(checks)
    critical_failures = [
        name
        for name, check in checks.items()
        if check.get("critical") and check.get("status") == "fail"
    ]
    issue_selection = _build_issue_selection_summary(checks, gate)
    operator_summary = _build_operator_summary(checks, issue_selection)
    previous_project_score = _latest_history_project_score(history)
    direction = _score_direction(project_score, previous_project_score, threshold=0.1)
    trend = "single_run" if previous_project_score is None else direction
    if leading_indicator_drift.get("status") == "fail":
        direction = "negative"
        trend = "regressing"
    operator_value_metrics = operator_value_alignment.get("metrics", {})
    operator_value_checks = {
        "operator_decision_support_rate": (
            operator_value_metrics.get("operator_decision_support_rate")
        ),
        "verified_system_change_rate": (
            operator_value_metrics.get("verified_system_change_rate")
        ),
        "aligned_work_rate": (
            operator_value_metrics.get("aligned_work_rate")
        ),
        "operator_value_score": operator_value_alignment.get("score"),
        "operator_decision_support_evidence": (
            operator_value_metrics.get("operator_decision_support_examples") or []
        ),
        "missing_operator_decision_support_fields": (
            operator_value_metrics.get("missing_operator_decision_support_fields") or []
        ),
        "missing_operator_decision_support": (
            operator_value_metrics.get("missing_operator_decision_support_examples") or []
        ),
    }
    journal_reporting_contract = _build_journal_reporting_contract(
        _load_json(journal_path),
    )

    benchmark = {
        "contract_version": BENCHMARK_CONTRACT_VERSION,
        "generated_at": current.isoformat(),
        "project_score": project_score,
        "score": project_score,
        "direction": direction,
        "trend": trend,
        "gate": gate,
        "checks": checks,
        "critical_failures": critical_failures,
        "execution_loop": {
            "status": execution_loop.get("status"),
            "score": execution_loop.get("score"),
            "metrics": execution_loop.get("metrics"),
            "next_throughput_action": (
                (execution_loop.get("metrics") or {}).get("next_throughput_action")
            ),
        },
        "operator_value_score": operator_value_alignment.get("score"),
        "operator_value_checks": operator_value_checks,
        "journal_reporting_contract": journal_reporting_contract,
        "leading_indicators": leading_indicator_drift.get("report")
        or _build_leading_indicator_report(leading_indicator_drift),
        "anti_make_work": {
            "status": anti_make_work_check.get("status"),
            "score": anti_make_work_check.get("score"),
            "flags": [
                item.get("issue")
                for item in anti_make_work_check.get("metrics", {}).get("shallow_examples", [])
                if item.get("issue")
            ],
        },
        "issue_selection": issue_selection,
        "summary": operator_summary,
        "history_path": str(history_path),
    }

    if persist:
        history["runs"].append(
            {
                "generated_at": benchmark["generated_at"],
                "project_score": benchmark["project_score"],
                "direction": benchmark["direction"],
                "critical_failures": benchmark["critical_failures"],
                "operator_value_score": benchmark["operator_value_score"],
                "operator_value_checks": benchmark["operator_value_checks"],
                "issue_selection": benchmark["issue_selection"],
                "checks": {
                    name: _benchmark_history_check_snapshot(name, check)
                    for name, check in checks.items()
                },
            }
        )
        history["runs"] = history["runs"][-_BENCHMARK_HISTORY_LIMIT:]
        _save_benchmark_history(history_path, history)

    return benchmark


def _coerce_path(value: Optional[Path | str], default: Path) -> Path:
    return Path(value).expanduser() if value else default


def _pipeline_benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "score": benchmark.get("score"),
        "project_score": benchmark.get("project_score"),
        "direction": benchmark.get("direction"),
        "trend": benchmark.get("trend"),
        "critical_failures": benchmark.get("critical_failures"),
    }
    drift = (benchmark.get("checks") or {}).get("leading_indicator_drift") or {}
    drift_metrics = drift.get("metrics") or {}
    if drift:
        execution_remediation = drift_metrics.get("execution_throughput_remediation") or {}
        summary["leading_indicator_drift"] = {
            "score": drift.get("score"),
            "status": drift.get("status"),
            "triggered_harbingers": drift_metrics.get("triggered_harbingers") or [],
            "harbinger_report": drift.get("report") or _build_leading_indicator_report(drift),
            "recommended_mitigations": drift_metrics.get("recommended_mitigations") or [],
            "execution_throughput_remediation": execution_remediation,
        }
    execution_metrics = ((benchmark.get("checks") or {}).get("execution_loop") or {}).get("metrics") or {}
    if execution_metrics:
        summary["execution_loop_follow_through"] = {
            "pending_journal_follow_through_count": execution_metrics.get(
                "pending_journal_follow_through_count"
            ),
            "pending_journal_follow_through_codex_runs": execution_metrics.get(
                "pending_journal_follow_through_codex_runs"
            ) or [],
            "journal_operator_support_codex_run_count": execution_metrics.get(
                "journal_operator_support_codex_run_count"
            ),
        }
    reporting_contract = benchmark.get("journal_reporting_contract") or {}
    if reporting_contract:
        summary["journal_reporting_contract"] = {
            "contract_version": reporting_contract.get("contract_version"),
            "status": reporting_contract.get("status"),
            "active_focus_count": len(reporting_contract.get("active_focus") or []),
            "recent_outcome_count": len(reporting_contract.get("recent_outcomes") or []),
        }
    return summary


def _pipeline_top_candidate(
    benchmark: dict[str, Any],
    candidate_limit: int,
) -> Optional[dict[str, Any]]:
    checks = benchmark.get("checks") or {}
    blocked = [
        check_id
        for check_id in (benchmark.get("issue_selection") or {}).get("blocked_checks", [])
        if check_id in checks
    ][: max(1, int(candidate_limit or 1))]
    if not blocked:
        return None

    check_id = blocked[0]
    check = checks.get(check_id) or {}
    label = str(check.get("label") or check_id.replace("_", " ")).strip()
    detail = str(check.get("detail") or "").strip()
    candidate = {
        "candidate_source": "benchmark",
        "candidate_id": check_id,
        "benchmark_id": check_id,
        "lane": "Maintenance",
        "status": "ready",
        "priority": 1 if check.get("critical") else 2,
        "title": f"Repair {label.lower()}",
        "score": check.get("score"),
        "detail": detail,
        "target_surface": "hermes-agent self-improvement reliability floor",
        "verification": (
            "Rerun self_improvement_pipeline and confirm the selected benchmark "
            "check passes without weakening reliability gates."
        ),
    }
    if check_id == "leading_indicator_drift":
        metrics = check.get("metrics") or {}
        candidate["triggered_harbingers"] = metrics.get("triggered_harbingers") or []
        candidate["recommended_mitigations"] = metrics.get("recommended_mitigations") or []
        candidate["harbinger_scorecard"] = metrics.get("harbinger_scorecard") or {}
        candidate["leading_indicator_report"] = (
            check.get("report") or _build_leading_indicator_report(check)
        )
        candidate["execution_throughput_remediation"] = (
            metrics.get("execution_throughput_remediation") or {}
        )
    return candidate


def _coerce_candidate_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _candidate_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _candidate_identifier(candidate: dict[str, Any]) -> str:
    for key in _BACKLOG_CANDIDATE_ID_KEYS:
        text = _candidate_string(candidate.get(key))
        if text:
            return text
    return ""


def _candidate_title(candidate: dict[str, Any]) -> str:
    for key in ("title", "summary", "name"):
        text = _candidate_string(candidate.get(key))
        if text:
            return text
    return _candidate_identifier(candidate) or "Untitled backlog candidate"


def _candidate_planning_field_present(candidate: dict[str, Any], aliases: Iterable[str]) -> bool:
    for alias in aliases:
        value = candidate.get(alias)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and any(
            _candidate_planning_field_present({"value": item}, ("value",))
            for item in value
        ):
            return True
        if isinstance(value, dict) and any(
            _candidate_string(value.get(key))
            for key in ("body", "comment", "content", "description", "text", "title", "value")
        ):
            return True
    return False


def _build_linear_planning_surface(backlog_candidates: Any) -> dict[str, Any]:
    candidates = _coerce_candidate_records(backlog_candidates)
    required_fields = sorted(_LINEAR_PLANNING_SURFACE_FIELD_ALIASES)
    missing_field_counts = {field: 0 for field in required_fields}
    issue_samples: list[dict[str, Any]] = []

    for candidate in candidates:
        missing_fields = [
            field
            for field in required_fields
            if not _candidate_planning_field_present(
                candidate,
                _LINEAR_PLANNING_SURFACE_FIELD_ALIASES[field],
            )
        ]
        for field in missing_fields:
            missing_field_counts[field] += 1
        if missing_fields and len(issue_samples) < _LINEAR_PLANNING_SURFACE_SAMPLE_LIMIT:
            issue_samples.append(
                {
                    "candidate_id": _candidate_identifier(candidate),
                    "title": _candidate_title(candidate),
                    "missing_fields": missing_fields,
                }
            )

    expected_field_count = len(candidates) * len(required_fields)
    missing_field_count = sum(missing_field_counts.values())
    score = 1.0 if expected_field_count == 0 else round(
        (expected_field_count - missing_field_count) / expected_field_count,
        4,
    )
    return {
        "surface": "linear_planning_surface",
        "status": "pass" if missing_field_count == 0 else "partial",
        "score": score,
        "candidate_count": len(candidates),
        "required_fields": required_fields,
        "missing_field_counts": {
            field: count for field, count in missing_field_counts.items() if count
        },
        "issue_samples": issue_samples,
        "detail": (
            "No Linear backlog candidates were provided."
            if not candidates
            else "Linear backlog candidates expose lane, verification, and active status-comment planning fields."
            if missing_field_count == 0
            else "Linear backlog candidates are missing planning fields; inspect issue_samples for exact candidates and fields."
        ),
    }


def _iter_candidate_strings(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, dict):
        for key in ("name", "title", "value", "label", "id", "state", "status", "type"):
            child = value.get(key)
            if isinstance(child, (dict, list)):
                yield from _iter_candidate_strings(child)
                continue
            text = _candidate_string(child)
            if text:
                yield text
        for key in ("node", "nodes", "edge", "edges"):
            yield from _iter_candidate_strings(value.get(key))
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_candidate_strings(item)
        return
    text = _candidate_string(value)
    if text:
        yield text


def _candidate_labels(candidate: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for key in ("label", "labels", "label_names", "labelNames", "tags"):
        labels.update(text.lower() for text in _iter_candidate_strings(candidate.get(key)))
    return labels


def _candidate_status_values(candidate: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in _BACKLOG_CANDIDATE_STATUS_KEYS:
        values.update(text.lower() for text in _iter_candidate_strings(candidate.get(key)))
    return values


def _candidate_has_terminal_state(statuses: set[str]) -> bool:
    normalized = {_normalize_evidence_key(status) for status in statuses}
    return bool(normalized & _BACKLOG_CANDIDATE_TERMINAL_STATES)


def _candidate_repo_values(candidate: dict[str, Any]) -> list[str]:
    repos: list[str] = []
    for key in _BACKLOG_CANDIDATE_REPO_KEYS:
        repos.extend(_candidate_string(text) for text in _iter_candidate_strings(candidate.get(key)))
    return [repo for repo in repos if repo]


def _candidate_project_values(candidate: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("project", "project_name", "projectName", "project_status", "projectStatus"):
        values.update(text.lower() for text in _iter_candidate_strings(candidate.get(key)))
    return values


def _candidate_bool(candidate: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = candidate.get(key)
        if isinstance(value, str):
            if value.strip().lower() in {"1", "true", "yes", "y"}:
                return True
            continue
        if bool(value):
            return True
    return False


def _candidate_has_human_owner(candidate: dict[str, Any], labels: set[str]) -> bool:
    if labels & _BACKLOG_CANDIDATE_HUMAN_OWNER_LABELS:
        return True
    normalized_labels = {_normalize_evidence_key(label) for label in labels}
    if "owner_human" in normalized_labels:
        return True
    owner_type = _candidate_string(
        candidate.get("owner_type") or candidate.get("ownerType") or candidate.get("ownership")
    ).lower()
    if owner_type == "human":
        return True
    owner = candidate.get("owner")
    if isinstance(owner, str) and owner.strip().lower() in {"human", "owner:human"}:
        return True
    if isinstance(owner, dict):
        for key in ("type", "kind", "ownership"):
            if _candidate_string(owner.get(key)).lower() == "human":
                return True
    return False


def _candidate_has_hermes_delegate_residue(
    candidate: dict[str, Any],
    labels: set[str],
) -> bool:
    normalized_labels = {_normalize_evidence_key(label) for label in labels}
    if labels & _BACKLOG_CANDIDATE_HERMES_DELEGATE_LABELS:
        return True
    if normalized_labels & {
        "delegate_codex",
        "delegate_hermes",
        "delegated_codex",
        "delegated_hermes",
        "hermes_delegate",
    }:
        return True

    for key in _BACKLOG_CANDIDATE_HERMES_DELEGATE_KEYS:
        for text in _iter_candidate_strings(candidate.get(key)):
            normalized = _normalize_evidence_key(text)
            if normalized in {"codex", "delegate_codex", "hermes", "hermes_delegate"}:
                return True
    return False


def _candidate_cleanup_reason(
    candidate: dict[str, Any],
    reasons: Iterable[str],
) -> Optional[str]:
    reason_set = set(reasons)
    labels = _candidate_labels(candidate)
    if not reason_set or not _candidate_has_hermes_delegate_residue(candidate, labels):
        return None
    if reason_set & {
        "duplicate",
        "ignored_project",
        "not_actionable_state",
        "owner_human",
        "selected_or_active",
    }:
        return "stale_hermes_ownership_residue"
    return None


def _candidate_filter_reasons(
    candidate: dict[str, Any],
    *,
    selected_candidate_ids: set[str],
) -> list[str]:
    candidate_id = _candidate_identifier(candidate)
    labels = _candidate_labels(candidate)
    statuses = _candidate_status_values(candidate)
    projects = _candidate_project_values(candidate)
    repo_values = _candidate_repo_values(candidate)
    reasons: list[str] = []

    if candidate_id and candidate_id in selected_candidate_ids:
        reasons.append("selected_or_active")
    elif _candidate_bool(candidate, *_BACKLOG_CANDIDATE_SELECTED_KEYS):
        reasons.append("selected_or_active")

    if (
        _candidate_bool(candidate, "ignored_project", "project_ignored", "ignoredProject")
        or labels & _BACKLOG_CANDIDATE_IGNORED_PROJECT_LABELS
        or "ignored" in projects
    ):
        reasons.append("ignored_project")

    if _candidate_has_human_owner(candidate, labels):
        reasons.append("owner_human")

    if (
        _candidate_bool(candidate, "duplicate", "is_duplicate", "isDuplicate")
        or "duplicate" in labels
        or "duplicate" in statuses
    ):
        reasons.append("duplicate")

    if _candidate_has_terminal_state(statuses):
        reasons.append("not_actionable_state")

    if (
        _candidate_bool(candidate, "repo_unresolved", "repoUnresolved", "repository_unresolved")
        or labels & _BACKLOG_CANDIDATE_REPO_UNRESOLVED_LABELS
        or not repo_values
    ):
        reasons.append("repo_unresolved")

    return reasons


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, str]:
    priority = candidate.get("priority")
    try:
        priority_value = int(priority)
    except (TypeError, ValueError):
        priority_value = 999
    return priority_value, _candidate_identifier(candidate) or _candidate_title(candidate)


def _parallel_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = _candidate_identifier(candidate)
    repos = _candidate_repo_values(candidate)
    return {
        "candidate_source": _candidate_string(candidate.get("candidate_source"))
        or _candidate_string(candidate.get("source"))
        or "backlog",
        "candidate_id": candidate_id,
        "issue_id": candidate_id,
        "title": _candidate_title(candidate),
        "repo": repos[0],
        "repos": repos,
        "lane": _candidate_string(candidate.get("lane")) or "Implementation",
        "status": _candidate_string(candidate.get("status")) or "ready",
        "priority": candidate.get("priority"),
        "target_surface": "repo-backed self-improvement backlog",
        "safety": {
            "ignored_project": False,
            "not_actionable_state": False,
            "owner_human": False,
            "duplicate": False,
            "repo_unresolved": False,
        },
    }


def _filter_reason_counts(filtered: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in filtered:
        for reason in item.get("reasons") or []:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _resolve_parallel_capacity(
    *,
    available_capacity: Optional[int],
    candidate_limit: int,
    top_candidate: Optional[dict[str, Any]],
) -> int:
    if available_capacity is not None:
        return max(0, int(available_capacity))
    selected_slots = 1 if top_candidate else 0
    return max(0, int(candidate_limit or 1) - selected_slots)


def _build_parallel_backlog_selection(
    *,
    benchmark: dict[str, Any],
    backlog_candidates: Any,
    selected_candidate_ids: Optional[Iterable[Any]],
    available_capacity: Optional[int],
    candidate_limit: int,
    top_candidate: Optional[dict[str, Any]],
) -> dict[str, Any]:
    requested_candidates = _coerce_candidate_records(backlog_candidates)
    selected_ids = {
        _candidate_string(candidate_id)
        for candidate_id in (selected_candidate_ids or [])
        if _candidate_string(candidate_id)
    }
    capacity = _resolve_parallel_capacity(
        available_capacity=available_capacity,
        candidate_limit=candidate_limit,
        top_candidate=top_candidate,
    )
    issue_selection = benchmark.get("issue_selection") or {}
    reliability_blocked = (
        issue_selection.get("parallel_repo_backed_selection_blocker") == "reliability_gate"
    )

    if reliability_blocked:
        return {
            "available_capacity": capacity,
            "selected_candidate_ids": sorted(selected_ids),
            "requested_backlog_candidate_count": len(requested_candidates),
            "eligible_backlog_candidate_count": 0,
            "filtered_backlog_candidate_count": 0,
            "filtered_reasons": {},
            "candidates": [],
            "saturation_state": "blocked_by_reliability_gate",
            "linear_planning_surface": _build_linear_planning_surface(requested_candidates),
            "guardrail_scope": (
                "Reliability repair blocks new repo-backed work; non-reliability "
                "review guardrails remain separate from parallel lane selection."
            ),
        }

    filtered: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate in requested_candidates:
        reasons = _candidate_filter_reasons(
            candidate,
            selected_candidate_ids=selected_ids,
        )
        if reasons:
            reason_list = sorted(set(reasons))
            filtered_candidate = {
                "candidate_id": _candidate_identifier(candidate),
                "title": _candidate_title(candidate),
                "reasons": reason_list,
            }
            cleanup_reason = _candidate_cleanup_reason(candidate, reason_list)
            if cleanup_reason:
                filtered_candidate["cleanup_reason"] = cleanup_reason
            filtered.append(filtered_candidate)
            continue
        eligible.append(candidate)

    selected = [
        _parallel_candidate_payload(candidate)
        for candidate in sorted(eligible, key=_candidate_sort_key)[:capacity]
    ]
    if capacity <= 0:
        saturation_state = "no_spare_capacity"
    elif selected:
        saturation_state = "spare_capacity_filled"
    elif requested_candidates:
        saturation_state = "no_safe_repo_backed_candidates"
    else:
        saturation_state = "no_backlog_candidates"

    return {
        "available_capacity": capacity,
        "selected_candidate_ids": sorted(selected_ids),
        "requested_backlog_candidate_count": len(requested_candidates),
        "eligible_backlog_candidate_count": len(eligible),
        "filtered_backlog_candidate_count": len(filtered),
        "filtered_reasons": _filter_reason_counts(filtered),
        "filtered_candidates": filtered[:10],
        "candidates": selected,
        "saturation_state": saturation_state,
        "linear_planning_surface": _build_linear_planning_surface(requested_candidates),
        "guardrail_scope": (
            "Quality guardrails suppress raw task-count selection but do not "
            "serialize independent repo-backed candidates that pass safety filters."
        ),
    }


def _format_pipeline_summary(
    *,
    benchmark: dict[str, Any],
    top_candidate: Optional[dict[str, Any]],
    parallel_selection: Optional[dict[str, Any]] = None,
) -> str:
    checks = benchmark.get("checks") or {}
    reliability = checks.get("reliability_gate") or {}
    execution = checks.get("execution_loop") or {}
    drift = checks.get("leading_indicator_drift") or {}
    lines = [
        "Self-improvement pipeline:",
        f"- score={benchmark.get('score')}",
        f"- reliability_gate={reliability.get('score')} {reliability.get('status')}",
        f"- execution_loop={execution.get('score')} {execution.get('status')}",
        f"- leading_indicator_drift={drift.get('score')} {drift.get('status')}",
    ]
    reporting_contract = benchmark.get("journal_reporting_contract") or {}
    if reporting_contract:
        lines.append(
            "- journal_reporting_contract="
            f"{reporting_contract.get('status')} "
            f"focus={len(reporting_contract.get('active_focus') or [])} "
            f"outcomes={len(reporting_contract.get('recent_outcomes') or [])}"
        )
    drift_metrics = drift.get("metrics") or {}
    leading_indicator_report = drift.get("report") or (
        _build_leading_indicator_report(drift) if drift else {}
    )
    if leading_indicator_report:
        harbinger_states = []
        for harbinger in _LEADING_INDICATOR_HARBINGERS:
            card = (leading_indicator_report.get("harbingers") or {}).get(harbinger) or {}
            state = "triggered" if card.get("triggered") else "clear"
            harbinger_states.append(f"{harbinger}:{state}")
        lines.append("- leading_indicator_watchlist=" + ", ".join(harbinger_states))
    triggered_harbingers = drift_metrics.get("triggered_harbingers") or []
    if triggered_harbingers:
        lines.append(
            "- leading_indicator_harbingers="
            + ", ".join(str(item) for item in triggered_harbingers)
        )
        for item in drift_metrics.get("recommended_mitigations") or []:
            harbinger = str(item.get("harbinger") or "").strip()
            evidence_summary = str(item.get("evidence_summary") or "").strip()
            mitigation = str(item.get("mitigation") or "").strip()
            next_action = str(item.get("next_action") or "").strip()
            if not harbinger:
                continue
            line = f"- leading_indicator_{harbinger}:"
            if evidence_summary:
                line += f" evidence={evidence_summary};"
            if mitigation:
                line += f" mitigation={mitigation};"
            if next_action:
                line += f" next_action={next_action}"
            lines.append(line.rstrip(";"))
    execution_remediation = drift_metrics.get("execution_throughput_remediation") or {}
    if execution_remediation.get("required"):
        lines.append(
            "- execution_throughput_remediation="
            f"{execution_remediation.get('recent_completed_codex_count')} completed Codex run(s), "
            f"{execution_remediation.get('recent_journal_work_item_count')} journal work item(s), "
            f"blocker={execution_remediation.get('blocking_surface')}"
        )
        execution_metrics = (execution.get("metrics") or {}) if execution else {}
        pending_count = execution_metrics.get("pending_journal_follow_through_count")
        if pending_count is not None:
            lines.append(f"- pending_journal_follow_through_count={pending_count}")
        pending_runs = execution_metrics.get("pending_journal_follow_through_codex_runs") or []
        if pending_runs:
            run_ids = [str(item.get("id") or item.get("run_id")) for item in pending_runs[:5]]
            lines.append("- pending_journal_follow_through_codex_runs=" + ", ".join(run_ids))
        for action in execution_remediation.get("actions") or []:
            lines.append(f"- execution_throughput_action={action}")
    critical = benchmark.get("critical_failures") or []
    if critical:
        lines.append(f"- critical_failures={', '.join(str(item) for item in critical)}")
    if top_candidate:
        lines.append(f"- top_candidate={top_candidate.get('candidate_id')}")
    else:
        lines.append("- top_candidate=None")
    if parallel_selection and parallel_selection.get("requested_backlog_candidate_count"):
        parallel_candidates = parallel_selection.get("candidates") or []
        lines.append(
            "- parallel_candidates="
            f"{len(parallel_candidates)}/{parallel_selection.get('available_capacity')} "
            f"state={parallel_selection.get('saturation_state')}"
        )
        filter_counts = parallel_selection.get("filtered_reasons") or {}
        if filter_counts:
            lines.append(
                "- parallel_candidate_filters="
                + ", ".join(
                    f"{reason}={filter_counts[reason]}"
                    for reason in sorted(filter_counts)
                )
            )
    return "\n".join(lines)


def evaluate_self_improvement_pipeline(
    *,
    journal_path: Optional[Path | str] = None,
    codex_runs_path: Optional[Path | str] = None,
    ctx_bindings_path: Optional[Path | str] = None,
    ontology_root: Optional[Path | str] = None,
    history_path: Optional[Path | str] = None,
    now: Optional[datetime] = None,
    freshness_hours: int = DEFAULT_FRESHNESS_HOURS,
    active_stale_hours: int = DEFAULT_ACTIVE_STALE_HOURS,
    persist: bool = True,
    candidate_limit: int = 3,
    available_capacity: Optional[int] = None,
    selected_candidate_ids: Optional[Iterable[Any]] = None,
    backlog_candidates: Optional[list[dict[str, Any]]] = None,
    auto_repair_linear: Optional[bool] = None,
    auto_close_resolved: Optional[bool] = None,
) -> dict[str, Any]:
    current = now or datetime.now(tz=timezone.utc)
    benchmark = evaluate_self_improvement_benchmark(
        journal_path=_coerce_path(journal_path, DEFAULT_JOURNAL_PATH),
        codex_runs_path=_coerce_path(codex_runs_path, DEFAULT_CODEX_RUNS_PATH),
        ctx_bindings_path=_coerce_path(ctx_bindings_path, DEFAULT_CTX_BINDINGS_PATH),
        ontology_root=_coerce_path(ontology_root, DEFAULT_ONTOLOGY_ROOT),
        history_path=_coerce_path(history_path, DEFAULT_BENCHMARK_HISTORY_PATH),
        now=current,
        freshness_hours=freshness_hours,
        active_stale_hours=active_stale_hours,
        persist=persist,
    )
    top_candidate = _pipeline_top_candidate(benchmark, candidate_limit)
    parallel_selection = _build_parallel_backlog_selection(
        benchmark=benchmark,
        backlog_candidates=backlog_candidates,
        selected_candidate_ids=selected_candidate_ids,
        available_capacity=available_capacity,
        candidate_limit=candidate_limit,
        top_candidate=top_candidate,
    )
    linear_requested = bool(auto_repair_linear) or bool(auto_close_resolved)
    pipeline = {
        "contract_version": BENCHMARK_CONTRACT_VERSION,
        "evaluated_at": current.isoformat(),
        "runtime_surface": "hermes-agent-core",
        "benchmark_before": _pipeline_benchmark_summary(benchmark),
        "benchmark": benchmark,
        "reporting_contract": benchmark.get("journal_reporting_contract"),
        "leading_indicators": benchmark.get("leading_indicators"),
        "linear": {
            "available": False,
            "error": (
                "Linear writeback is not part of the Hermes core self-improvement pipeline."
                if linear_requested
                else None
            ),
            "managed_issues": [],
            "closed_issues": [],
            "repairs": [],
        },
        "top_candidate": top_candidate,
        "capacity": parallel_selection,
        "parallel_candidates": parallel_selection.get("candidates") or [],
    }
    pipeline["summary_markdown"] = _format_pipeline_summary(
        benchmark=benchmark,
        top_candidate=top_candidate,
        parallel_selection=parallel_selection,
    )
    return pipeline


def self_improvement_evidence_gate(
    journal_path: Optional[str] = None,
    codex_runs_path: Optional[str] = None,
    ctx_bindings_path: Optional[str] = None,
    ontology_root: Optional[str] = None,
    now: Optional[str] = None,
    freshness_hours: Optional[int] = None,
    active_stale_hours: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    gate = evaluate_self_improvement_evidence(
        journal_path=Path(journal_path).expanduser() if journal_path else DEFAULT_JOURNAL_PATH,
        codex_runs_path=Path(codex_runs_path).expanduser() if codex_runs_path else DEFAULT_CODEX_RUNS_PATH,
        ctx_bindings_path=Path(ctx_bindings_path).expanduser() if ctx_bindings_path else DEFAULT_CTX_BINDINGS_PATH,
        ontology_root=Path(ontology_root).expanduser() if ontology_root else DEFAULT_ONTOLOGY_ROOT,
        now=_parse_time(now) if now else None,
        freshness_hours=int(freshness_hours) if freshness_hours else DEFAULT_FRESHNESS_HOURS,
        active_stale_hours=int(active_stale_hours) if active_stale_hours else DEFAULT_ACTIVE_STALE_HOURS,
    )
    return json.dumps({"success": True, "gate": gate, "task_id": task_id})


def self_improvement_benchmark(
    journal_path: Optional[str] = None,
    codex_runs_path: Optional[str] = None,
    ctx_bindings_path: Optional[str] = None,
    ontology_root: Optional[str] = None,
    history_path: Optional[str] = None,
    now: Optional[str] = None,
    freshness_hours: Optional[int] = None,
    active_stale_hours: Optional[int] = None,
    persist: Optional[bool] = None,
    task_id: Optional[str] = None,
) -> str:
    benchmark = evaluate_self_improvement_benchmark(
        journal_path=Path(journal_path).expanduser() if journal_path else DEFAULT_JOURNAL_PATH,
        codex_runs_path=Path(codex_runs_path).expanduser() if codex_runs_path else DEFAULT_CODEX_RUNS_PATH,
        ctx_bindings_path=Path(ctx_bindings_path).expanduser() if ctx_bindings_path else DEFAULT_CTX_BINDINGS_PATH,
        ontology_root=Path(ontology_root).expanduser() if ontology_root else DEFAULT_ONTOLOGY_ROOT,
        history_path=Path(history_path).expanduser() if history_path else DEFAULT_BENCHMARK_HISTORY_PATH,
        now=_parse_time(now) if now else None,
        freshness_hours=int(freshness_hours) if freshness_hours else DEFAULT_FRESHNESS_HOURS,
        active_stale_hours=int(active_stale_hours) if active_stale_hours else DEFAULT_ACTIVE_STALE_HOURS,
        persist=True if persist is None else bool(persist),
    )
    return json.dumps({"success": True, "benchmark": benchmark, "task_id": task_id})


def self_improvement_pipeline(
    journal_path: Optional[str] = None,
    codex_runs_path: Optional[str] = None,
    ctx_bindings_path: Optional[str] = None,
    ontology_root: Optional[str] = None,
    history_path: Optional[str] = None,
    now: Optional[str] = None,
    freshness_hours: Optional[int] = None,
    active_stale_hours: Optional[int] = None,
    persist: Optional[bool] = None,
    candidate_limit: Optional[int] = None,
    available_capacity: Optional[int] = None,
    selected_candidate_ids: Optional[list[str]] = None,
    backlog_candidates: Optional[list[dict[str, Any]]] = None,
    auto_repair_linear: Optional[bool] = None,
    auto_close_resolved: Optional[bool] = None,
    task_id: Optional[str] = None,
) -> str:
    pipeline = evaluate_self_improvement_pipeline(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=_parse_time(now) if now else None,
        freshness_hours=int(freshness_hours) if freshness_hours else DEFAULT_FRESHNESS_HOURS,
        active_stale_hours=int(active_stale_hours) if active_stale_hours else DEFAULT_ACTIVE_STALE_HOURS,
        persist=True if persist is None else bool(persist),
        candidate_limit=int(candidate_limit) if candidate_limit else 3,
        available_capacity=(
            int(available_capacity) if available_capacity is not None else None
        ),
        selected_candidate_ids=selected_candidate_ids,
        backlog_candidates=backlog_candidates,
        auto_repair_linear=auto_repair_linear,
        auto_close_resolved=auto_close_resolved,
    )
    return json.dumps({"success": True, "pipeline": pipeline, "task_id": task_id})


registry.register(
    name="self_improvement_evidence_gate",
    toolset="self_improvement",
    schema=SELF_IMPROVEMENT_EVIDENCE_SCHEMA,
    handler=lambda args, **kw: self_improvement_evidence_gate(
        journal_path=args.get("journal_path"),
        codex_runs_path=args.get("codex_runs_path"),
        ctx_bindings_path=args.get("ctx_bindings_path"),
        ontology_root=args.get("ontology_root"),
        now=args.get("now"),
        freshness_hours=args.get("freshness_hours"),
        active_stale_hours=args.get("active_stale_hours"),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="self_improvement_pipeline",
    toolset="self_improvement",
    schema=SELF_IMPROVEMENT_PIPELINE_SCHEMA,
    handler=lambda args, **kw: self_improvement_pipeline(
        journal_path=args.get("journal_path"),
        codex_runs_path=args.get("codex_runs_path"),
        ctx_bindings_path=args.get("ctx_bindings_path"),
        ontology_root=args.get("ontology_root"),
        history_path=args.get("history_path"),
        now=args.get("now"),
        freshness_hours=args.get("freshness_hours"),
        active_stale_hours=args.get("active_stale_hours"),
        persist=args.get("persist"),
        candidate_limit=args.get("candidate_limit"),
        available_capacity=args.get("available_capacity"),
        selected_candidate_ids=args.get("selected_candidate_ids"),
        backlog_candidates=args.get("backlog_candidates"),
        auto_repair_linear=args.get("auto_repair_linear"),
        auto_close_resolved=args.get("auto_close_resolved"),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="self_improvement_benchmark",
    toolset="self_improvement",
    schema=SELF_IMPROVEMENT_BENCHMARK_SCHEMA,
    handler=lambda args, **kw: self_improvement_benchmark(
        journal_path=args.get("journal_path"),
        codex_runs_path=args.get("codex_runs_path"),
        ctx_bindings_path=args.get("ctx_bindings_path"),
        ontology_root=args.get("ontology_root"),
        history_path=args.get("history_path"),
        now=args.get("now"),
        freshness_hours=args.get("freshness_hours"),
        active_stale_hours=args.get("active_stale_hours"),
        persist=args.get("persist"),
        task_id=kw.get("task_id"),
    ),
)
