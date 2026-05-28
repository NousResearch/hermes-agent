"""Internal application proposal generation for Wisdom captures."""

from __future__ import annotations

from wisdom.db import WisdomDB
from wisdom.models import ApplicationRecord, CaptureRecord


def create_application_proposals(db: WisdomDB, capture_id: int) -> list[ApplicationRecord]:
    capture = db.get_capture(capture_id)
    if capture is None:
        return []
    existing = db.list_applications(capture_id)
    if existing:
        return existing
    proposals = _proposals_for_capture(capture)
    return db.insert_applications(capture_id=capture_id, applications=proposals)


def _proposals_for_capture(capture: CaptureRecord) -> list[dict[str, object]]:
    text = (capture.cleaned_text or capture.original_text).strip()
    short = _short(text)
    if capture.category == "business":
        return [
            _app(
                "client_language",
                "Client language",
                f"Use this as a client-facing line, then explain the practical implication: {short}",
            ),
            _app(
                "principle",
                "Business principle",
                f"Turn this into an operating principle: every client touchpoint should answer what changes now, not only what happened. Seed: {short}",
            ),
            _app(
                "task_proposal",
                "Business process proposal",
                f"Review one report, pitch, or meeting script and add a section that operationalizes this idea: {short}",
            ),
        ]
    if capture.category == "investing":
        return [
            _app(
                "investment_rule",
                "Investment rule",
                f"Do not act on thesis quality alone. Convert the idea into a rule about survivability, downside path, liquidity, and forced-exit risk. Seed: {short}",
            ),
            _app(
                "checklist",
                "Investment checklist",
                "Before sizing: 1. What loss can I survive? 2. What adverse move breaks the trade? "
                "3. What forces exit? 4. Is liquidity adequate? 5. Am I sizing by conviction or survivability?",
            ),
            _app(
                "decision_rule",
                "Decision rule",
                f"Use this as a pre-trade pause: name the decision error this prevents before allocating capital. Seed: {short}",
            ),
        ]
    if capture.category == "health":
        return [
            _app(
                "health_experiment",
                "Health experiment",
                f"Run a small reversible experiment for 7 days, track the input, decision quality, and energy effect, then keep or discard it. Seed: {short}",
            ),
            _app(
                "decision_rule",
                "Decision-quality rule",
                f"Use this as a guardrail before important decisions: check whether the state described here is distorting judgment. Seed: {short}",
            ),
            _app("principle", "Health principle", f"Treat this as a personal operating constraint until disproven: {short}"),
        ]
    if capture.category == "life":
        return [
            _app("principle", "Life principle", f"Preserve the principle in portable form: {short}"),
            _app("writing_idea", "Writing idea", f"Use this as a short essay seed. Start with the concrete situation, then the hidden pattern: {short}"),
            _app("decision_rule", "Decision rule", f"Turn this into a personal rule that changes a future choice, not just a reflection: {short}"),
        ]
    return [
        _app("principle", "Principle candidate", f"Review whether this has a durable rule inside it: {short}"),
        _app("writing_idea", "Writing idea", f"Use this as a writing seed only if it still feels specific on review: {short}"),
    ]


def _app(application_type: str, title: str, body: str) -> dict[str, object]:
    return {
        "application_type": application_type,
        "title": title,
        "body": body,
        "status": "proposed",
        "metadata": {"generator_version": 1},
    }


def _short(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 180:
        return compact
    return compact[:177].rstrip() + "..."
