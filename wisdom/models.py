"""Typed models for Hermes Wisdom Kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args


Category = Literal["business", "investing", "health", "life", "inbox"]
SourceType = Literal[
    "thought",
    "voice",
    "podcast",
    "book",
    "article",
    "meeting",
    "quote",
    "conversation",
    "other",
]
CaptureStatus = Literal["active", "archived"]
ReviewStatus = Literal["unreviewed", "reviewed", "accepted", "dismissed", "applied", "archived"]
ApplicationStatus = Literal["proposed", "accepted", "dismissed", "archived"]
ApplicationType = Literal[
    "task_proposal",
    "reminder_proposal",
    "principle",
    "checklist",
    "client_language",
    "investment_rule",
    "health_experiment",
    "writing_idea",
    "decision_rule",
]

VALID_CATEGORIES = set(get_args(Category))
VALID_SOURCE_TYPES = set(get_args(SourceType))
VALID_REVIEW_STATUSES = set(get_args(ReviewStatus))
VALID_APPLICATION_TYPES = set(get_args(ApplicationType))


@dataclass(frozen=True)
class WisdomConfig:
    enabled: bool = True
    db_path: Path | None = None
    capture_mode: str = "explicit"
    max_results: int = 5
    interpret_timeout_seconds: float = 5.0
    interpretation_mode: str = "deterministic"
    application_mode: str = "deterministic"
    apply_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class TriggerMatch:
    prefix: str
    cleaned_text: str
    category_hint: Category | None = None
    source_hint: SourceType | None = None


@dataclass(frozen=True)
class Classification:
    category: Category
    source_type: SourceType
    title: str
    confidence: float
    importance_score: float | None = None
    novelty_score: float | None = None
    actionability_score: float | None = None


@dataclass(frozen=True)
class CaptureRecord:
    id: int
    raw_event_id: int
    created_at: float
    updated_at: float
    title: str
    original_text: str
    cleaned_text: str | None
    category: Category
    source_type: SourceType
    status: CaptureStatus
    review_status: ReviewStatus
    reviewed_at: float | None
    accepted_at: float | None
    dismissed_at: float | None
    applied_at: float | None
    confidence: float
    importance_score: float | None
    novelty_score: float | None
    actionability_score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterpretationRecord:
    id: int
    capture_id: int
    created_at: float
    summary: str
    insight: str | None
    why_it_matters: str | None
    possible_application: str | None
    counterpoint: str | None
    confidence: float
    method: str
    model_used: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApplicationRecord:
    id: int
    capture_id: int
    created_at: float
    application_type: ApplicationType
    title: str
    body: str
    status: ApplicationStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureOutcome:
    status: Literal["captured", "blocked_secret", "ignored", "disabled", "error"]
    capture: CaptureRecord | None = None
    message: str | None = None


@dataclass(frozen=True)
class StatusSnapshot:
    enabled: bool
    capture_mode: str
    db_path: Path
    fts_available: bool
    counts: dict[str, int]
    last_capture_at: float | None
