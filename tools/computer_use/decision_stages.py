"""Pluggable decision-lane stages: reranker, Jev, and state rendering (#113850)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from tools.computer_use.decision_lane import (
    Decision,
    ElementCandidate,
    SemanticState,
)
from tools.computer_use.system_one import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ENDPOINT,
    TransportError,
    ask,
    classify_http_status,
)

_ACTION_CRITERIA = {
    "click": "Click or press a button, link, or control",
    "type": "Type into a text field (caller supplies text separately)",
    "key": "Press a keyboard shortcut or special key",
    "scroll": "Scroll the window or a scrollable region",
    "wait": "Wait for loading or animation to finish",
    "done": "Goal already achieved on the current screen",
    "escalate": "Cannot proceed safely; hand back to the main planner",
}

_TEXT_FIELD_ROLES = frozenset(
    {"textfield", "textarea", "searchfield", "combobox", "text", "entry"}
)


def render_state_text(state: SemanticState, candidates: tuple[ElementCandidate, ...]) -> str:
    """Semantic state for System One. No pixels, no secrets."""
    lines: list[str] = []
    if state.goal_hint:
        lines.append(f"GOAL: {state.goal_hint}")
    if state.busy:
        lines.append("STATUS: busy or loading")
    lines.append(f"SCREEN ({len(candidates)} elements):")
    for cand in candidates:
        if not cand.enabled:
            continue
        label = (cand.label or "").replace("\n", " ")[:60]
        role = cand.role or "element"
        lines.append(f"  [{cand.ref}] {role} '{label}'")
    return "\n".join(lines)


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def reranker_stage(state: SemanticState, candidates: tuple[ElementCandidate, ...]) -> Decision | None:
    """Local semantic reranker: label overlap against goal_hint. Abstains on ambiguity."""
    hint = (state.goal_hint or "").strip()
    if not hint:
        return None
    live = [c for c in candidates if c.enabled]
    if not live:
        return None
    hint_lower = hint.lower()
    exact = [c for c in live if c.label.strip().lower() == hint_lower]
    if len(exact) == 1:
        return Decision(action="click", target_ref=exact[0].ref, confidence=0.88, backend="reranker")
    hint_tokens = _tokenize(hint)
    if not hint_tokens:
        return None
    scored: list[tuple[float, ElementCandidate]] = []
    for cand in live:
        label_tokens = _tokenize(cand.label)
        if not label_tokens:
            continue
        overlap = len(hint_tokens & label_tokens) / max(len(hint_tokens), 1)
        if hint_lower in cand.label.lower() or cand.label.lower() in hint_lower:
            overlap = max(overlap, 0.75)
        if overlap >= 0.5:
            scored.append((overlap, cand))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].ref))
    best_score, best = scored[0]
    if len(scored) > 1 and abs(scored[1][0] - best_score) < 0.05:
        return None  # ambiguous
    action = "type" if (best.role or "").lower() in _TEXT_FIELD_ROLES else "click"
    return Decision(
        action=action,
        target_ref=best.ref,
        confidence=min(0.87, 0.65 + best_score * 0.3),
        backend="reranker",
    )


def _jev_api_key() -> str:
    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        if val := os.environ.get(name, "").strip():
            return val
    return ""


def _jev_base_url() -> str:
    for name in ("TYPESAFE_BASE_URL", "JEV_BASE_URL"):
        if val := os.environ.get(name, "").strip().rstrip("/"):
            return val
    return DEFAULT_BASE_URL


def _jev_model() -> str:
    return os.environ.get("TYPESAFE_MODEL", os.environ.get("JEV_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL


def http_transport(payload: dict[str, Any]) -> dict[str, Any]:
    """POST to TypeSafe System One. Raises TransportError on failure."""
    api_key = _jev_api_key()
    if not api_key:
        raise TransportError("missing TYPESAFE_API_KEY / JEV_API_KEY")
    url = f"{_jev_base_url()}{ENDPOINT}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(os.environ.get("TYPESAFE_TIMEOUT", "15"))) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        kind = classify_http_status(exc.code)
        raise TransportError(f"HTTP {exc.code} ({kind})", kind=kind) from exc
    except urllib.error.URLError as exc:
        raise TransportError(str(exc.reason or exc)) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TransportError("invalid JSON response") from exc
    if not isinstance(parsed, dict):
        raise TransportError("response must be a JSON object")
    return parsed


def build_jev_questions(candidates: tuple[ElementCandidate, ...]) -> dict[str, dict]:
    """Factorized questions per #113850 — no flattened action×target space."""
    live = [c for c in candidates if c.enabled]
    target_criteria: dict[str, str] = {"none": "No specific element target"}
    for cand in live:
        label = (cand.label or "").replace("\n", " ")[:40]
        target_criteria[cand.ref] = f"{cand.role or 'element'} '{label}'"
    return {
        "action": {
            "type": "choice",
            "instructions": "What is the best next action toward the GOAL?",
            "criteria": dict(_ACTION_CRITERIA),
        },
        "target": {
            "type": "choice",
            "instructions": "Which visible element is the action target? Choose none for wait/scroll/key/done/escalate.",
            "criteria": target_criteria,
        },
        "needs_vision": {
            "type": "noul",
            "instructions": "Does this step require seeing pixels/screenshot evidence not in the text state?",
            "criteria": {
                "true": "Pixels or visual layout are necessary",
                "false": "Accessibility text state is enough",
            },
        },
        "needs_generation": {
            "type": "noul",
            "instructions": "Does this step require free-form text generation (not selecting an existing value)?",
            "criteria": {
                "true": "Must compose novel text",
                "false": "Can use existing labels/values or caller-provided inputs",
            },
        },
        "done": {
            "type": "noul",
            "instructions": "Is the GOAL already achieved on the current screen?",
            "criteria": {
                "true": "Goal appears complete",
                "false": "More work remains",
            },
        },
    }


def _choice_confidence(answer) -> float:
    if answer.confidence is not None:
        return float(answer.confidence)
    if answer.probabilities and answer.choice:
        return float(answer.probabilities.get(answer.choice, 0.0))
    return 0.0


def _noul_true_prob(answer) -> float:
    return float(answer.probabilities.get("true", answer.noul or 0.0))


def parse_jev_response(parsed, candidates: tuple[ElementCandidate, ...]) -> Decision | None:
    """Map System One answers to a lane Decision. Abstains on low confidence."""
    answers = parsed.answers
    if "done" in answers and _noul_true_prob(answers["done"]) >= 0.85:
        conf = _noul_true_prob(answers["done"])
        return Decision(action="done", done=True, confidence=conf, backend="jev")
    action_ans = answers.get("action")
    if action_ans is None or not action_ans.choice:
        return None
    action = action_ans.choice
    if action not in _ACTION_CRITERIA:
        return None
    action_conf = _choice_confidence(action_ans)
    target_ref = None
    target_conf = 1.0
    if action in {"click", "type"}:
        target_ans = answers.get("target")
        if target_ans is None or not target_ans.choice or target_ans.choice == "none":
            return None
        target_ref = target_ans.choice
        live_refs = {c.ref for c in candidates if c.enabled}
        if target_ref not in live_refs:
            return None
        target_conf = _choice_confidence(target_ans)
    needs_vision = _noul_true_prob(answers["needs_vision"]) >= 0.5 if "needs_vision" in answers else False
    needs_generation = _noul_true_prob(answers["needs_generation"]) >= 0.5 if "needs_generation" in answers else False
    confidence = min(action_conf, target_conf) if action in {"click", "type"} else action_conf
    return Decision(
        action=action,
        target_ref=target_ref,
        needs_vision=needs_vision,
        needs_generation=needs_generation,
        confidence=confidence,
        backend="jev",
    )


def jev_stage(state: SemanticState, candidates: tuple[ElementCandidate, ...]) -> Decision | None:
    """Jev / TypeSafe System One stage. Abstains on transport or parse errors."""
    if not _jev_api_key():
        return None
    questions = build_jev_questions(candidates)
    state_text = render_state_text(state, candidates)
    try:
        parsed = ask(http_transport, state_text, questions, model=_jev_model())
    except (TransportError, Exception):
        return None
    return parse_jev_response(parsed, candidates)
