"""Internal application proposal generation for Wisdom captures."""

from __future__ import annotations

from wisdom.db import WisdomDB
from wisdom.models import ApplicationRecord, CaptureRecord


def create_application_proposals(
    db: WisdomDB,
    capture_id: int,
    *,
    mode: str = "deterministic",
    timeout: float = 30.0,
) -> list[ApplicationRecord]:
    capture = db.get_capture(capture_id)
    if capture is None:
        return []
    existing = db.list_applications(capture_id)
    if existing:
        return existing
    proposals = None
    if str(mode or "").strip().lower() == "llm":
        from wisdom.llm_apply import llm_application_proposals

        proposals = llm_application_proposals(capture, timeout=timeout)
    if proposals is None:
        proposals = deterministic_application_dicts(capture)
    return db.insert_applications(capture_id=capture_id, applications=proposals)


def deterministic_application_dicts(capture: CaptureRecord) -> list[dict[str, object]]:
    text = (capture.cleaned_text or capture.original_text).strip()
    short = _short(text)
    if capture.category == "business":
        return [
            _app(
                "client_language",
                "Client language",
                (
                    'client-facing line: "This review is not just a record of what happened. '
                    'Its job is to help decide what should be done next." Use this for x10x, '
                    f"client reporting, reviews, or trust-building communication. Seed: {short}"
                ),
            ),
            _app(
                "principle",
                "Business principle",
                (
                    "Client reporting should reduce decision uncertainty, not merely describe past "
                    f"performance. Every client touchpoint should answer what changes now. Seed: {short}"
                ),
            ),
            _app(
                "task_proposal",
                "Business process proposal",
                (
                    'Add a "What this means now" section to client reports, PMS/AIF reviews, '
                    f"or sales follow-ups so the idea becomes a process improvement. Seed: {short}"
                ),
            ),
        ]
    if capture.category == "investing":
        return [
            _app(
                "investment_rule",
                "Investment rule",
                (
                    "Do not size a position based only on thesis confidence. Size it based on "
                    f"survivability, liquidity, downside path, and forced-exit risk. Seed: {short}"
                ),
            ),
            _app(
                "checklist",
                "Investment checklist",
                (
                    "Before sizing:\n"
                    "1. What loss can I survive?\n"
                    "2. What adverse move breaks the trade?\n"
                    "3. What forces exit?\n"
                    "4. Is liquidity adequate?\n"
                    "5. Am I sizing by conviction or survivability?"
                ),
            ),
            _app(
                "decision_rule",
                "Risk decision rule",
                (
                    "If the thesis is attractive but survivability, liquidity, downside path, or "
                    f"forced-exit risk is unclear, reduce size or pass. Seed: {short}"
                ),
            ),
        ]
    if capture.category == "health":
        return [
            _app(
                "health_experiment",
                "Health experiment",
                (
                    f"{_health_experiment_text(text)} Use the first 7 days as calibration, then "
                    f"keep or discard the rule based on the pattern. Seed: {short}"
                ),
            ),
            _app(
                "decision_rule",
                "Decision-quality rule",
                (
                    "Avoid major investing or business decisions after poor sleep, low energy, or "
                    f"the state named here unless urgency is real. Seed: {short}"
                ),
            ),
            _app(
                "principle",
                "Personal health rule",
                (
                    "Treat cognition as state-dependent, not constant. Tracking question: did "
                    f"sleep, food, energy, or stress change decision confidence today? Seed: {short}"
                ),
            ),
        ]
    if capture.category == "life":
        return [
            _app(
                "principle",
                "Life principle",
                (
                    "Before building a new system, identify the uncomfortable action the system "
                    f"may be helping avoid. Seed: {short}"
                ),
            ),
            _app(
                "writing_idea",
                "Reflection prompt",
                (
                    "Reflection prompt / writing seed: What conversation or decision am I avoiding "
                    f"by optimizing the process? Start from: {short}"
                ),
            ),
            _app(
                "decision_rule",
                "Decision rule",
                (
                    "Do the uncomfortable human action first, or name why it cannot be done now, "
                    f"before adding another system. Seed: {short}"
                ),
            ),
        ]
    return [
        _app(
            "principle",
            "Principle candidate",
            f"Do not promote this yet. Keep it only if it can change a decision or behavior on review. Seed: {short}",
        ),
        _app(
            "writing_idea",
            "Writing idea",
            f"Use this as a writing seed only if it still feels specific, surprising, or reusable on review: {short}",
        ),
    ]


def _app(application_type: str, title: str, body: str) -> dict[str, object]:
    return {
        "application_type": application_type,
        "title": title,
        "body": body,
        "status": "proposed",
        "metadata": {"generator_version": 2},
    }


def _health_experiment_text(text: str) -> str:
    lowered = text.lower()
    if "sleep" in lowered:
        return "Track sleep quality and major decision confidence for 14 days."
    return "Track the input named in this note and major decision confidence for 14 days."


def _short(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 180:
        return compact
    return compact[:177].rstrip() + "..."
