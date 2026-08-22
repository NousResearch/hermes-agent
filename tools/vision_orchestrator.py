#!/usr/bin/env python3
"""Vision Orchestrator — unified entry point for the local vision policy
layer (V0.1 inactive foundation).

Pipeline (frozen V0.1):

    VisionRequest
        → planner (deterministic model-slot selection)
        → unified invocation (exactly ONE model call)
        → deterministic quality evaluation
        → VisionResult with escalation *recommendation* only

Stage B1 guarantees:

- at most one model invocation per ``analyze_image`` request;
- escalation is RECOMMENDED, never executed;
- ``click_authorized`` is always false — no clicking, no navigation;
- the feature is disabled by default (``vision_router.enabled: false``);
- ``auxiliary.vision`` behavior and the active Taobao Skill are untouched.

``analyze_image`` is an internal callable for tests and later integration —
it is NOT registered as a model-visible tool in this stage.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from tools.ollama_generate_vision_client import (
    NATIVE_GENERATE_DEFAULT_PROFILE,
    NATIVE_GENERATE_STRICT_SCHEMA,
    TASK_STRICT_SCHEMAS,
    TRANSPORT_OLLAMA_NATIVE_GENERATE,
    TRANSPORT_OPENAI_COMPATIBLE,
    invoke_native_generate,
)
from tools.ollama_vision_client import invoke_vision_model, prepare_image
from tools.vision_policy import (
    DEFAULT_MODEL_SLOTS,
    DEFAULT_TIMEOUTS,
    PRECISION_EXECUTION_PROFILE,
    PROMPT_TEMPLATE_IDS,
    CoordinateContext,
    ExecutionStatus,
    ModelSlot,
    PolicyBlockedError,
    QualityDecision,
    VisionMode,
    VisionRequest,
    VisionTask,
    build_result,
    evaluate_quality,
    plan_vision,
    resolve_default_transport,
    serialize,
)

# ---------------------------------------------------------------------------
# Prompt templates (versioned IDs — see PROMPT_TEMPLATE_IDS in policy).
# Prompt text lives here so it can evolve independently of the policy layer.
# ---------------------------------------------------------------------------

_BASE_INSTRUCTION = (
    "You are analyzing a screenshot for a research workflow.\n"
    "Separate DIRECTLY OBSERVED content from UNCERTAIN INTERPRETATION and "
    "INFERRED meaning. If you are not sure, say so explicitly with an "
    "uncertainty marker. Do not invent content you cannot see.\n"
    "Respond in JSON only, using this schema where applicable:\n"
    '{"observed_text": [...], "targets": [{"label": "...", "bbox_px": [x1,y1,x2,y2], '
    '"point_px": [x,y]}], "evidence": "...", "uncertainty": "...", '
    '"inference": "..."}\n'
)

_TASK_PROMPTS = {
    "SCENE_DESCRIBE": (
        _BASE_INSTRUCTION
        + "TASK: describe the overall scene. Fields: observation (what is "
        "directly visible), inference (your interpretation), uncertainty "
        "(what is unclear)."
    ),
    "UI_READ": (
        _BASE_INSTRUCTION
        + "TASK: read the requested interface text exactly. Provide exact "
        "quotations in observed_text. Do not paraphrase prices, buttons, "
        "login prompts, or field values."
    ),
    "UI_LOCATE": (
        _BASE_INSTRUCTION
        + "TASK: locate the target element and return its pixel bbox and "
        "center point in the source image coordinate space. Include "
        "target-nearby text as evidence. bbox_px = [left, top, right, "
        "bottom]."
    ),
    "EXACT_OCR": (
        _BASE_INSTRUCTION
        + "TASK: transcribe the exact text in the requested region "
        "character-for-character. Preserve line order. Do not summarize."
    ),
    "EVIDENCE_VERIFY": (
        _BASE_INSTRUCTION
        + "TASK: verify the claim against what is directly visible. "
        "Provide evidence (what you see), observation vs inference, and "
        "flag any unresolved contradiction. Never claim an action was "
        "performed."
    ),
}


def _raw_base64_from_data_url(data_url: str) -> str:
    """Strip the ``data:image/...;base64,`` prefix from a prepared data URL.

    The native generate contract requires RAW base64 in the ``images`` array
    (no data-URL prefix). The data URL produced by ``prepare_image`` encodes
    exactly the transport-compatible bytes (WebP already converted to PNG),
    so stripping the prefix reuses the validated two-layer image contract
    without re-encoding.
    """
    if "," not in data_url:
        raise ValueError("Malformed data URL from image preparation.")
    return data_url.split(",", 1)[1]


def _build_prompt(request: VisionRequest) -> str:
    task_prompt = _TASK_PROMPTS.get(request.task.value, _BASE_INSTRUCTION)
    parts = [
        task_prompt,
        f"QUESTION: {request.question}" if request.question else "",
    ]
    if request.required_outputs:
        parts.append(f"REQUIRED OUTPUTS: {', '.join(request.required_outputs)}")
    if request.region:
        parts.append(f"REGION: {request.region}")
    return "\n".join(p for p in parts if p)


def _parse_structured(raw_text: str, task: Any) -> Dict[str, Any]:
    """Best-effort parse of the model's JSON response.

    Returns an extracted dict with the fields the quality gates inspect.
    When the response cannot be parsed as JSON at all, returns a minimal
    dict carrying ``_parse_failed: True`` so the quality evaluator can
    classify it as NOT_EVALUATED rather than an escalation candidate.
    Never trusted as authoritative — quality gates decide.

    Supports the bounded fenced-response contract: raw JSON object/array,
    or ONE complete Markdown-fenced block containing valid JSON
    (`` ```json`` / `` ``` `` with any case for the language tag).
    Whitespace outside the single fence is allowed; arbitrary prose before
    or after the fence, multiple fenced blocks, and unclosed fences are
    rejected (``_parse_failed``). No regex-only extraction of the first
    ``{...}`` from arbitrary prose is performed.
    """
    text = (raw_text or "").strip()
    fenced = _strip_single_fenced_block(text)
    if fenced is None:
        # Not a fenced response (or an invalid fence) — try raw JSON only.
        fenced = text
    extracted: Dict[str, Any] = {}
    try:
        parsed = json.loads(fenced)
        if isinstance(parsed, dict):
            extracted = parsed
        else:
            extracted = {"_parse_failed": True, "raw_text": raw_text}
    except Exception:  # noqa: BLE001 — non-JSON responses handled by gates
        extracted = {"_parse_failed": True, "raw_text": raw_text}
    return extracted


def _strip_single_fenced_block(text: str):
    """Return the inner text of exactly one complete Markdown code fence
    containing valid JSON, or ``None`` when the input is not a single
    well-formed fenced block.

    Contract:

    - optional whitespace outside the single fence is allowed;
    - language tag matching is case-insensitive (``json``, ``JSON``, ...);
    - a generic fence (`` ``` `` with no tag) containing valid JSON is
      accepted;
    - arbitrary prose before or after the fence is rejected;
    - multiple fenced blocks are rejected;
    - an unclosed fence is rejected;
    - malformed fenced content is rejected (caller then fails JSON parse).

    Returns the fenced inner text (already JSON-parseable) or ``None``.
    """
    lines = text.splitlines()
    if not lines:
        return None
    first = lines[0].strip()
    if not first.startswith("```"):
        return None  # not fenced — caller falls back to raw JSON
    # Parse the opening fence tag (allow ```json, ```JSON, ``` plain).
    opener = first[3:].strip().lower()
    if opener and opener != "json":
        return None  # fenced with a non-JSON language tag → not our contract
    # The final line must close the fence.
    if len(lines) < 2 or lines[-1].strip() != "```":
        return None  # unclosed fence → rejected
    # Any ``` line inside the body means multiple fences → rejected.
    body = lines[1:-1]
    for ln in body:
        if ln.strip().startswith("```"):
            return None  # second fence → rejected
    inner = "\n".join(body).strip()
    if not inner:
        return None  # empty fenced content → rejected
    # The body must itself be valid JSON — if not, reject the fence so the
    # caller reports a parse failure (no partial extraction).
    try:
        json.loads(inner)
    except Exception:  # noqa: BLE001
        return None
    return inner


def _recommended_next_slot(
    task: Any,
    initial_slot: ModelSlot,
    quality: Dict[str, Any],
) -> Optional[ModelSlot]:
    """Deterministic escalation *recommendation* (never executed).

    The recommendation is driven by the actual failed gates, not merely by
    the current model slot:
    - FAST_VLM UI_READ/UI_LOCATE missing or uncertain evidence → PRECISION_VLM;
    - PRECISION_VLM with a text-specific failure (exact text missing) → OCR;
    - PRECISION_VLM coordinate failure (out of bounds / ordering / crop) →
      NOT OCR — a text specialist cannot fix coordinates;
    - semantic contradiction or refusal → HUMAN_REVIEW (None).
    """
    decision = quality.get("quality_decision")
    if decision != QualityDecision.ESCALATE_RECOMMENDED.value:
        return None

    failed = set(quality.get("failed_gates") or [])

    # Coordinate failures are never OCR's job.
    coordinate_gates = {
        "coordinates_in_bounds",
        "bbox_or_point_present",
        "coordinate_space_declared",
        "x_out_of_bounds",
        "y_out_of_bounds",
        "bbox_left_ge_right",
        "bbox_top_ge_bottom",
        "point_outside_bbox",
        "crop_origin_negative",
        "crop_width_nonpositive",
        "crop_height_nonpositive",
        "crop_exceeds_source_width",
        "crop_exceeds_source_height",
        "coordinate_space_missing_dimensions",
    }
    if failed & coordinate_gates:
        return ModelSlot.PRECISION_VLM

    # Text-specific failure: OCR is the right follow-up.
    text_gates = {"required_text_present", "text_non_empty"}
    if initial_slot == ModelSlot.PRECISION_VLM and (failed & text_gates):
        return ModelSlot.OCR

    # Default: a higher-precision VLM can resolve missing/uncertain evidence.
    if initial_slot == ModelSlot.FAST_VLM:
        return ModelSlot.PRECISION_VLM
    if task.value == "EXACT_OCR":
        return ModelSlot.PRECISION_VLM
    return ModelSlot.PRECISION_VLM


async def analyze_image(
    request: VisionRequest,
    *,
    model_slots: Optional[Dict[ModelSlot, str]] = None,
    timeouts: Optional[Dict[ModelSlot, float]] = None,
    enabled: Optional[bool] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    task_id: Optional[str] = None,
    transport: Optional[str] = None,
    native_generate_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical entry point: one request → one (or zero) model call → result.

    ``transport`` selects the invocation path. ``None`` (default) = AUTO:
    resolved per selected slot — PRECISION_VLM binds to
    ``TRANSPORT_OLLAMA_NATIVE_GENERATE`` (``PRECISION_EXECUTION_PROFILE``,
    );
    FAST_VLM / OCR keep the OpenAI-compatible path. An explicit
    ``TRANSPORT_OPENAI_COMPATIBLE`` override is honored for PRECISION_VLM
    diagnostics; an explicit ``TRANSPORT_OLLAMA_NATIVE_GENERATE`` override is
    honored anywhere. Unknown identities raise ``ValueError`` (safe failure).
    No automatic fallback between transports ever occurs.

    ``native_generate_options`` (optional) overrides the bound Precision
    profile (``PRECISION_EXECUTION_PROFILE``: num_ctx 32768, num_predict
    4000, temperature 0.1, task-specific strict JSON Schema, timeout 120s).

    Feature-flag semantics:
    - ``enabled=None`` (default): resolve the effective ``vision_router``
      feature flag from the standard configuration path. The runtime default
      is DISABLED — this faithfully follows the feature flag.
    - ``enabled=False``: returns POLICY_BLOCKED without resolving,
      downloading, normalizing, or encoding the image and without calling
      the model.
    - ``enabled=True``: explicit internal override for mocked tests and
      future controlled validation. Never model-visible; never changes
      ``auxiliary.vision``.
    """
    if enabled is None:
        from hermes_cli.config import load_config

        cfg = load_config()
        enabled = vision_router_enabled(cfg)

    if transport is not None and transport not in (
        TRANSPORT_OPENAI_COMPATIBLE,
        TRANSPORT_OLLAMA_NATIVE_GENERATE,
    ):
        raise ValueError(f"Unknown transport identity: {transport!r}")
    transport_arg = transport

    # Early-return paths (router disabled / policy blocked) carry a
    # conservative transport placeholder — no slot resolution occurred.
    placeholder_transport = transport_arg or TRANSPORT_OPENAI_COMPATIBLE
    transport_info = {
        "requested_transport": placeholder_transport,
        "selected_transport": placeholder_transport,
        "endpoint_family": (
            "ollama_native_generate"
            if transport_arg == TRANSPORT_OLLAMA_NATIVE_GENERATE
            else "openai_compatible"
        ),
        "native_generate_used": (
            transport_arg == TRANSPORT_OLLAMA_NATIVE_GENERATE
        ),
    }

    if not enabled:
        return {
            "execution_status": ExecutionStatus.POLICY_BLOCKED.value,
            "quality_decision": QualityDecision.NOT_EVALUATED.value,
            "answer": "",
            "structured": {},
            "initial_model_slot": None,
            "final_model_slot": None,
            "actual_model": "",
            "recommended_next_slot": None,
            "human_review_required": False,
            "click_authorized": False,
            "quality": {"passed_gates": [], "failed_gates": ["vision_router_disabled"]},
            "coordinate_context": {},
            "trace": [],
            "logical_model_calls": 0,
            "transport": transport_info,
        }

    slots = model_slots or dict(DEFAULT_MODEL_SLOTS)
    timeouts = timeouts or dict(DEFAULT_TIMEOUTS)

    # Planner (deterministic; may raise PolicyBlockedError).
    try:
        plan = plan_vision(request, model_slots=slots, timeouts=timeouts)
    except PolicyBlockedError as pbe:
        return {
            "execution_status": ExecutionStatus.POLICY_BLOCKED.value,
            "quality_decision": QualityDecision.NOT_EVALUATED.value,
            "answer": "",
            "structured": {},
            "initial_model_slot": None,
            "final_model_slot": None,
            "actual_model": "",
            "recommended_next_slot": (
                pbe.recommended_slot.value if pbe.recommended_slot else None
            ),
            "human_review_required": False,
            "click_authorized": False,
            "quality": {"passed_gates": [], "failed_gates": [pbe.reason]},
            "coordinate_context": {},
            "trace": [],
            "transport": transport_info,
        }

    trace: List[Dict[str, Any]] = []
    initial_slot = plan.selected_slot
    actual_model = plan.selected_model_identity

    # Transport resolution: explicit override wins; otherwise AUTO per slot
    # (PRECISION_VLM → bound native-generate profile; FAST/OCR → OpenAI).
    effective_transport = (
        transport_arg
        if transport_arg is not None
        else resolve_default_transport(initial_slot)
    )
    native = effective_transport == TRANSPORT_OLLAMA_NATIVE_GENERATE
    transport_info = {
        "requested_transport": (
            transport_arg or "AUTO"
        ),
        "selected_transport": effective_transport,
        "endpoint_family": (
            "ollama_native_generate" if native else "openai_compatible"
        ),
        "native_generate_used": native,
    }

    # Image preparation (reuses existing helpers).
    try:
        (
            data_url,
            width_px,
            height_px,
            mime,
            normalized_sha256,
            transport_meta,
        ) = await prepare_image(
            request.image_source, task_id=task_id
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "execution_status": ExecutionStatus.INVALID_RESPONSE.value,
            "quality_decision": QualityDecision.NOT_EVALUATED.value,
            "answer": "",
            "structured": {},
            "initial_model_slot": initial_slot.value,
            "final_model_slot": initial_slot.value,
            "actual_model": actual_model,
            "recommended_next_slot": None,
            "human_review_required": False,
            "click_authorized": False,
            "quality": {"passed_gates": [], "failed_gates": ["image_preparation_failed"]},
            "coordinate_context": (
                request.coordinate_context.to_dict() if request.coordinate_context else {}
            ),
            "trace": [],
            "logical_model_calls": 0,
            "transport": transport_info,
        }

    cc = request.coordinate_context or CoordinateContext()
    if width_px is not None and cc.source_width_px is None:
        cc.source_width_px = width_px
    if height_px is not None and cc.source_height_px is None:
        cc.source_height_px = height_px

    prompt = _build_prompt(request)
    t0 = time.monotonic()
    if native:
        if not base_url:
            raise ValueError("OLLAMA_NATIVE_GENERATE transport requires base_url")
        native_opts = dict(NATIVE_GENERATE_DEFAULT_PROFILE)
        # Bound Precision profile values are the defaults; per-call options
        # override them. Task-specific strict schema resolves by task.
        native_opts.update(
            {
                "num_ctx": PRECISION_EXECUTION_PROFILE["num_ctx"],
                "num_predict": PRECISION_EXECUTION_PROFILE["num_predict"],
                "temperature": PRECISION_EXECUTION_PROFILE["temperature"],
                "format": TASK_STRICT_SCHEMAS.get(
                    request.task.value, NATIVE_GENERATE_STRICT_SCHEMA
                ),
            }
        )
        native_opts.update(native_generate_options or {})
        invocation = await invoke_native_generate(
            model=actual_model,
            prompt=prompt,
            image_raw_base64=_raw_base64_from_data_url(data_url),
            num_ctx=int(native_opts.get("num_ctx") or 32768),
            num_predict=int(native_opts.get("num_predict") or 4000),
            temperature=float(native_opts.get("temperature") or 0.1),
            seed=int(native_opts["seed"]) if native_opts.get("seed") is not None else None,
            format_spec=native_opts.get("format"),
            base_url=base_url,
            timeout_seconds=float(native_opts.get("timeout_seconds") or plan.timeout_seconds),
            transport_retries=int(native_opts.get("transport_retries") or 0),
        )
    else:
        invocation = await invoke_vision_model(
            model=actual_model,
            prompt=prompt,
            image_data_url=data_url,
            timeout_seconds=plan.timeout_seconds,
            base_url=base_url,
            api_key=api_key,
        )
    latency_ms = int((time.monotonic() - t0) * 1000)

    execution_status = ExecutionStatus(invocation["execution_status"])
    trace_entry = {
        "attempt": 1,
        "model_slot": initial_slot.value,
        "actual_model": actual_model,
        "latency_ms": latency_ms,
        "selection_reason": plan.selection_reason,
        "execution_status": execution_status.value,
        "normalized_image_sha256": normalized_sha256,
        "transport_image_sha256": transport_meta.get("transport_image_sha256"),
        "transport_mime_type": transport_meta.get("transport_mime_type"),
        "transport_transcoded": transport_meta.get("transport_transcoded"),
        "requested_transport": transport_info["requested_transport"],
        "selected_transport": transport_info["selected_transport"],
        "endpoint_family": transport_info["endpoint_family"],
        "native_generate_used": native,
    }
    if native:
        trace_entry.update(
            {
                "content_source": invocation.get("content_source"),
                "thinking_fallback_used": invocation.get("thinking_fallback_used"),
                "response_character_count": invocation.get("response_character_count"),
                "thinking_character_count": invocation.get("thinking_character_count"),
                "total_ms": invocation.get("total_ms"),
                "prompt_eval_ms": invocation.get("prompt_eval_ms"),
                "eval_ms": invocation.get("eval_ms"),
                "prompt_eval_count": invocation.get("prompt_eval_count"),
                "eval_count": invocation.get("eval_count"),
                "done_reason": invocation.get("done_reason"),
                "transport_attempts": invocation.get("transport_attempts"),
            }
        )
    trace.append(trace_entry)

    # Exactly one logical model call: if the single invocation failed,
    # evaluate as NOT_EVALUATED (no retry). Note: ``max_model_calls=1``
    # means ONE logical model-selection/inference operation by the
    # Orchestrator; the existing auxiliary client may still perform bounded
    # lower-level transport retries (auxiliary.transient_retries) for
    # transient timeout/connection/server failures — those are transport
    # retries, not additional logical model escalations.
    if execution_status != ExecutionStatus.SUCCESS:
        quality = {
            "quality_decision": QualityDecision.NOT_EVALUATED.value,
            "passed_gates": [],
            "failed_gates": [f"execution_{execution_status.value.lower()}"],
        }
        return build_result(
            execution_status=execution_status,
            quality=quality,
            answer="",
            structured={},
            initial_slot=initial_slot,
            final_slot=initial_slot,
            actual_model=actual_model,
            recommended_next_slot=None,
            coordinate_context=cc,
            trace=trace,
            image_sha256=normalized_sha256,
            logical_model_calls=1,
            transport_meta=transport_meta,
            transport=transport_info,
        )

    if native:
        raw_text = invocation.get("extracted_content") or ""
    else:
        raw_text = invocation.get("raw_text") or ""
    extracted = _parse_structured(raw_text, request.task)
    if (
        request.task == VisionTask.EXACT_OCR
        and extracted.get("_parse_failed")
        and raw_text.strip()
    ):
        # OCR models (glm-ocr) naturally return plain transcription, not
        # JSON. The exact text IS the OCR result — accept it canonically
        # (diagnostic finding D, HERMES_VISION_FULL_CORPUS_QUALITY_FINDINGS_
        # DIAGNOSTIC_V0_1) instead of classifying NOT_EVALUATED.
        extracted = {
            "observed_text": [raw_text.strip()],
            "_ocr_plain_text": True,
        }
    quality = evaluate_quality(
        request.task,
        response_text=raw_text,
        required_outputs=request.required_outputs,
        extracted=extracted,
        coordinates={
            "x": _first_point_x(extracted),
            "y": _first_point_y(extracted),
            "bbox": _first_bbox(extracted),
            "coordinate_space": cc.coordinate_space,
        },
        coordinate_context=cc,
    )
    recommended = _recommended_next_slot(request.task, initial_slot, quality)

    return build_result(
        execution_status=ExecutionStatus.SUCCESS,
        quality=quality,
        answer=raw_text,
        structured=extracted,
        initial_slot=initial_slot,
        final_slot=initial_slot,
        actual_model=actual_model,
        recommended_next_slot=recommended,
        coordinate_context=cc,
        trace=trace,
        image_sha256=normalized_sha256,
        logical_model_calls=1,
        transport_meta=transport_meta,
        transport=transport_info,
    )


# -- coordinate extraction helpers ------------------------------------------


def _first_target(extracted: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    targets = extracted.get("targets") or []
    if isinstance(targets, list) and targets and isinstance(targets[0], dict):
        return targets[0]
    return None


def _first_bbox(extracted: Dict[str, Any]) -> Optional[List[float]]:
    t = _first_target(extracted)
    if t is None:
        return None
    bbox = t.get("bbox_px") or t.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(v) for v in bbox]
    return None


def _first_point_x(extracted: Dict[str, Any]) -> Optional[float]:
    t = _first_target(extracted)
    if t is None:
        return None
    pt = t.get("point_px") or t.get("point")
    if isinstance(pt, list) and len(pt) >= 1:
        try:
            return float(pt[0])
        except (TypeError, ValueError):
            return None
    return None


def _first_point_y(extracted: Dict[str, Any]) -> Optional[float]:
    t = _first_target(extracted)
    if t is None:
        return None
    pt = t.get("point_px") or t.get("point")
    if isinstance(pt, list) and len(pt) >= 2:
        try:
            return float(pt[1])
        except (TypeError, ValueError):
            return None
    return None


# -- runtime feature-flag gate ----------------------------------------------

_DEFAULT_ENABLED = False


def vision_router_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the feature flag from a config dict (or default False).

    The default is always False; the live runtime configuration is not
    modified by this module. ``auxiliary.vision`` behavior is never touched.
    The canonical section is ``auxiliary.vision_router`` (legacy top-level
    ``vision_router`` accepted only when the nested mapping is absent).
    """
    from tools.vision_policy import resolve_vision_router_enabled

    return resolve_vision_router_enabled(config)


# -- serialization convenience ----------------------------------------------


def analyze_image_from_dict(
    data: Dict[str, Any],
    **kwargs: Any,
) -> str:
    """JSON convenience wrapper around :func:`analyze_image`."""
    request = VisionRequest.from_dict(data)
    result = asyncio_run(analyze_image(request, **kwargs))
    return serialize(result)


def asyncio_run(coro: Any) -> Any:
    """Run a coroutine, reusing an existing loop when present (tests)."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Already inside an event loop (e.g. tests with pytest-asyncio or
        # an agent context): return the coroutine for the caller to await.
        return coro
    return asyncio.run(coro)
