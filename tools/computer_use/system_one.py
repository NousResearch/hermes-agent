"""TypeSafe System One protocol adapter for computer-use decision lane (#113850).

Pure functions + explicit transport injection. Parses choice/noul/score answers with
full probability distributions. No network in this module — callers inject transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

ENDPOINT = "/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai"

DOCUMENTED_ERROR_STATUS = {
    401: "unauthorized",
    422: "unprocessable_entity",
    429: "rate_limited",
    529: "overloaded",
}

USAGE_NUMERIC_ALLOWLIST = ("input_tokens", "output_tokens")


class ProtocolError(ValueError):
    """Malformed request or answer relative to the published schema."""


class TransportError(RuntimeError):
    """Injected transport failed. Never treated as a probability."""

    def __init__(self, message: str, *, kind: str = "transport") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass(frozen=True)
class Answer:
    id: str
    kind: str
    raw: dict[str, Any]
    probabilities: dict[str, float]
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    confidence: float | None = None
    legend: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedResponse:
    model: str
    answers: dict[str, Answer]
    usage: Usage
    abstain: bool = False
    abstain_reason: str | None = None


Transport = Callable[[dict[str, Any]], dict[str, Any]]


def build_request(
    state: Any,
    questions: Mapping[str, dict],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if not questions:
        raise ProtocolError("questions map must be non-empty")
    for key, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in {"noul", "choice", "score"}:
            raise ProtocolError(f"question {key!r} missing documented type")
        if "instructions" not in q:
            raise ProtocolError(f"question {key!r} missing instructions")
        if q["type"] == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise ProtocolError(f"choice {key!r} needs criteria map")
        if q["type"] == "score":
            criteria = q.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ProtocolError(f"score {key!r} needs >=2 level descriptions")
        if q["type"] == "noul" and any(k in q for k in ("true", "false")) and "criteria" not in q:
            raise ProtocolError(
                f"noul {key!r} puts true/false at top level; official schema nests them under criteria"
            )
    return {"state": state, "model": model, "questions": dict(questions)}


def _finite_prob(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{where}: probability must be a number")
    x = float(value)
    if x != x or x in (float("inf"), float("-inf")):
        raise ProtocolError(f"{where}: nonfinite probability")
    if x < 0.0 or x > 1.0:
        raise ProtocolError(f"{where}: probability out of [0,1]")
    return x


def _require_distribution(probs: Any, expected_keys: set[str], *, where: str) -> dict[str, float]:
    if not isinstance(probs, dict) or not probs:
        raise ProtocolError(f"{where}: probabilities map required")
    if set(probs) != expected_keys:
        raise ProtocolError(f"{where}: probabilities keys must match criteria exactly")
    out = {k: _finite_prob(v, where=f"{where}.{k}") for k, v in probs.items()}
    total = sum(out.values())
    if abs(total - 1.0) > 1e-6:
        raise ProtocolError(f"{where}: probabilities must sum to 1 (got {total})")
    return out


def parse_answer(qid: str, question: Mapping[str, Any], raw: Mapping[str, Any]) -> Answer:
    if not isinstance(raw, dict):
        raise ProtocolError(f"{qid}: answer must be object")
    kind = question.get("type")
    if raw.get("type") != kind:
        raise ProtocolError(f"{qid}: answer type {raw.get('type')!r} != question type {kind!r}")
    payload = dict(raw)
    if kind == "noul":
        noul = _finite_prob(raw.get("noul"), where=f"{qid}.noul")
        dist = {"true": noul, "false": round(1.0 - noul, 12)}
        if "probabilities" in raw:
            dist = _require_distribution(raw["probabilities"], {"true", "false"}, where=qid)
            if abs(dist["true"] - noul) > 1e-6:
                raise ProtocolError(f"{qid}: noul value disagrees with probabilities.true")
        return Answer(id=qid, kind="noul", raw=payload, probabilities=dist, noul=noul)
    if kind == "choice":
        criteria = question.get("criteria") or {}
        dist = _require_distribution(raw.get("probabilities"), set(criteria), where=qid)
        choice = raw.get("choice")
        if choice not in dist:
            raise ProtocolError(f"{qid}: choice {choice!r} not in probabilities")
        argmax = max(dist.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if choice != argmax and dist[choice] < dist[argmax]:
            raise ProtocolError(f"{qid}: choice is not a highest-probability option")
        conf = raw.get("confidence")
        confidence = None if conf is None else _finite_prob(conf, where=f"{qid}.confidence")
        return Answer(
            id=qid, kind="choice", raw=payload, probabilities=dist,
            choice=str(choice), confidence=confidence,
        )
    if kind == "score":
        levels = question.get("criteria") or []
        keys = {str(i) for i in range(len(levels))}
        dist = _require_distribution(raw.get("probabilities"), keys, where=qid)
        legend = raw.get("legend")
        if not isinstance(legend, dict) or set(legend) != keys:
            raise ProtocolError(f"{qid}: legend keys must match level indices")
        for i, desc in enumerate(levels):
            if legend[str(i)] != desc:
                raise ProtocolError(f"{qid}: legend[{i}] != criteria[{i}]")
        score = raw.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ProtocolError(f"{qid}: score must be a number")
        expected = sum(int(k) * p for k, p in dist.items())
        if abs(float(score) - expected) > 1e-4:
            raise ProtocolError(f"{qid}: score is not probability-weighted level index")
        conf = raw.get("confidence")
        confidence = None if conf is None else _finite_prob(conf, where=f"{qid}.confidence")
        return Answer(
            id=qid, kind="score", raw=payload, probabilities=dist,
            score=float(score), confidence=confidence, legend=dict(legend),
        )
    raise ProtocolError(f"{qid}: unknown type")


def parse_systemone_response(
    raw: Mapping[str, Any],
    questions: Mapping[str, dict],
    *,
    abstain_if_missing: bool = False,
) -> ParsedResponse:
    if not isinstance(raw, dict):
        raise ProtocolError("response must be object")
    answers_raw = raw.get("answers")
    if not isinstance(answers_raw, dict):
        raise ProtocolError("answers map required")
    missing = set(questions) - set(answers_raw)
    extra = set(answers_raw) - set(questions)
    if extra:
        raise ProtocolError(f"unexpected answer ids: {sorted(extra)}")
    if missing and not abstain_if_missing:
        raise ProtocolError(f"missing answers: {sorted(missing)}")
    parsed: dict[str, Answer] = {}
    for qid, q in questions.items():
        if qid not in answers_raw:
            continue
        parsed[qid] = parse_answer(qid, q, answers_raw[qid])
    usage_raw = raw.get("usage") or {}
    if not isinstance(usage_raw, dict):
        raise ProtocolError("usage must be object")
    usage_kwargs = {}
    for key in USAGE_NUMERIC_ALLOWLIST:
        val = usage_raw.get(key, 0)
        if val is None:
            val = 0
        if isinstance(val, bool) or type(val) is not int or val < 0:
            raise ProtocolError(f"usage.{key} must be a non-negative int")
        usage_kwargs[key] = val
    abstain = bool(missing) if abstain_if_missing else False
    return ParsedResponse(
        model=str(raw.get("model") or ""),
        answers=parsed,
        usage=Usage(**usage_kwargs),
        abstain=abstain,
        abstain_reason="missing_answers" if abstain else None,
    )


def classify_http_status(status: int) -> str:
    return DOCUMENTED_ERROR_STATUS.get(status, "undocumented_http")


def ask(
    transport: Transport,
    state: Any,
    questions: Mapping[str, dict],
    *,
    model: str = DEFAULT_MODEL,
) -> ParsedResponse:
    """Injected transport: callable payload -> raw JSON body. No network here."""
    payload = build_request(state, questions, model=model)
    try:
        raw = transport(payload)
    except TransportError:
        raise
    except Exception as exc:
        raise TransportError(f"injected transport failed: {exc}") from exc
    return parse_systemone_response(raw, questions)
