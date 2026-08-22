#!/usr/bin/env python3
"""Vision Policy — deterministic policy layer for the local vision
Orchestrator (V0.1 inactive foundation).

Stage B1 implements the *inactive* foundation only:

- pure Vision-Need policy (no Browser / model / network);
- canonical enums and request/plan/result schemas;
- deterministic task + criticality + mode → model-slot selection;
- backward-compatible legacy role aliases;
- deterministic post-response quality gates;
- escalation *recommendation* without escalation execution.

The feature is disabled by default (``vision_router.enabled: false``) and
must never modify ``auxiliary.vision`` behavior or any active Skill.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
import os as _os
from urllib.parse import urlparse as _urlparse
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vision Router configuration resolver (config-path alignment repair)
#
# Canonical path: config["auxiliary"]["vision_router"].
# Legacy fallback: top-level config["vision_router"] ONLY when the nested
# mapping is absent. Both paths present -> nested wins. No merging of
# conflicting values. Malformed values fail closed.
# ---------------------------------------------------------------------------

_VR_ROUTER_KEYS = ("enabled", "ocr_excerpt_chars", "ocr_page_chars",
                   "per_workflow_max_calls")


def resolve_vision_router_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve the Vision Router configuration section.

    Precedence (deterministic):
    1. ``config["auxiliary"]["vision_router"]`` when present and a mapping
       (canonical);
    2. top-level ``config["vision_router"]`` mapping ONLY when the nested
       mapping is absent (legacy compatibility);
    3. anything else -> empty mapping (fail closed).
    """
    if not isinstance(config, dict):
        return {}
    aux = config.get("auxiliary")
    if isinstance(aux, dict):
        nested = aux.get("vision_router")
        if isinstance(nested, dict):
            return nested
    legacy = config.get("vision_router")
    if isinstance(legacy, dict):
        return legacy
    return {}


def resolve_vision_router_enabled(config: Optional[Dict[str, Any]]) -> bool:
    """Effective Router flag from the canonical section.

    Malformed or non-boolean values fail closed to False.
    """
    router = resolve_vision_router_config(config)
    val = router.get("enabled", False)
    return val if isinstance(val, bool) else False


def resolve_vision_router_value(
    config: Optional[Dict[str, Any]], key: str, default: Any
) -> Any:
    """Read one Router-section value through the shared resolver."""
    router = resolve_vision_router_config(config)
    if key not in _VR_ROUTER_KEYS:
        return default
    return router.get(key, default)


# ---------------------------------------------------------------------------
# Trusted Ollama endpoint resolution (native base_url wiring repair)
#
# Source precedence (deterministic, no merging):
# 1. explicit trusted config: auxiliary.vision.base_url (the accepted
#    OpenAI-compatible Vision service endpoint — same Ollama service);
# 2. OLLAMA_HOST environment convention (Ollama's standard host:port env);
# 3. otherwise None -> fail closed.
#
# The returned value is the NATIVE root (scheme://host:port) — the /v1
# OpenAI-compatible suffix is stripped because the native transport calls
# {base_url}/api/generate at the root.
# ---------------------------------------------------------------------------

def resolve_ollama_base_url(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Trusted Ollama service root for the native /api/generate transport.

    Server-controlled only: never derived from model-visible arguments.
    Malformed values fail closed to None.
    """
    base: Optional[str] = None
    if isinstance(config, dict):
        aux = config.get("auxiliary")
        if isinstance(aux, dict):
            vis = aux.get("vision")
            if isinstance(vis, dict):
                v = vis.get("base_url")
                if isinstance(v, str) and v.strip():
                    base = v.strip()
    if not base:
        env = _os.environ.get("OLLAMA_HOST", "")
        if isinstance(env, str) and env.strip():
            base = env.strip()
    if not base:
        return None
    return _normalize_ollama_base(base)


def _normalize_ollama_base(base: str) -> Optional[str]:
    """Normalize to scheme://host:port. Rejects file://, paths, embedded
    credentials, unsupported schemes, and empty hosts. Deterministic."""
    if not base or not isinstance(base, str):
        return None
    candidate = base if "://" in base else f"http://{base}"
    try:
        parsed = _urlparse(candidate)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    return f"{parsed.scheme}://{parsed.hostname}:{port}"

# ---------------------------------------------------------------------------

# Canonical enums (V0.1 frozen set)
# ---------------------------------------------------------------------------


class VisionNeedDecision(str, enum.Enum):
    VISION_NOT_NEEDED = "VISION_NOT_NEEDED"
    VISION_REQUIRED = "VISION_REQUIRED"


class VisionTask(str, enum.Enum):
    SCENE_DESCRIBE = "SCENE_DESCRIBE"
    UI_READ = "UI_READ"
    UI_LOCATE = "UI_LOCATE"
    EXACT_OCR = "EXACT_OCR"
    EVIDENCE_VERIFY = "EVIDENCE_VERIFY"


class VisionCriticality(str, enum.Enum):
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class VisionMode(str, enum.Enum):
    AUTO = "AUTO"
    FAST = "FAST"
    PRECISION = "PRECISION"
    OCR = "OCR"


class ModelSlot(str, enum.Enum):
    FAST_VLM = "FAST_VLM"
    PRECISION_VLM = "PRECISION_VLM"
    OCR = "OCR"


class ExecutionStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    ENDPOINT_UNAVAILABLE = "ENDPOINT_UNAVAILABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    POLICY_BLOCKED = "POLICY_BLOCKED"


# Transport identities (single source of truth for the Vision layer).
# The native Ollama ``/api/generate`` transport is bound as the default
# execution profile for the PRECISION_VLM slot ; the
# OpenAI-compatible transport remains the default for FAST_VLM / OCR and
# remains selectable for PRECISION_VLM via an explicit override.
TRANSPORT_OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"
TRANSPORT_OLLAMA_NATIVE_GENERATE = "OLLAMA_NATIVE_GENERATE"


def resolve_default_transport(slot: "ModelSlot") -> str:
    """Default transport for a model slot when no explicit override exists.

    PRECISION_VLM → native ``/api/generate`` (bound profile);
    FAST_VLM / OCR → OpenAI-compatible (unchanged legacy behavior).
    """
    if slot == ModelSlot.PRECISION_VLM:
        return TRANSPORT_OLLAMA_NATIVE_GENERATE
    return TRANSPORT_OPENAI_COMPATIBLE


class QualityDecision(str, enum.Enum):
    PASS = "PASS"
    ESCALATE_RECOMMENDED = "ESCALATE_RECOMMENDED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    NOT_EVALUATED = "NOT_EVALUATED"


class PolicyReasonCode(str, enum.Enum):
    """Deterministic planner/escalation reason codes (non-sensitive)."""

    AUTO_MATRIX = "auto_matrix"
    EXPLICIT_FAST = "explicit_fast"
    EXPLICIT_PRECISION = "explicit_precision"
    EXPLICIT_OCR = "explicit_ocr"
    HIGH_CRITICALITY_OVERRIDE = "high_criticality_override"
    HIGH_CRITICALITY_POLICY_BLOCK = "high_criticality_policy_block"
    OCR_FOR_EXACT_TEXT = "ocr_for_exact_text"
    FAST_INVALID_FOR_HIGH = "fast_invalid_for_high"


# ---------------------------------------------------------------------------
# Legacy role compatibility (deterministic, explicit case handling)
# ---------------------------------------------------------------------------

_LEGACY_ALIASES = {
    "FAST_VISION_MODEL": ModelSlot.FAST_VLM,
    "PRECISION_VISION_MODEL": ModelSlot.PRECISION_VLM,
    "LAYOUT_OCR_MODEL": ModelSlot.OCR,
    # Lower/any-case acceptance — explicit normalization.
    "fast_vision_model": ModelSlot.FAST_VLM,
    "precision_vision_model": ModelSlot.PRECISION_VLM,
    "layout_ocr_model": ModelSlot.OCR,
    # Canonical names accepted directly (idempotent).
    "FAST_VLM": ModelSlot.FAST_VLM,
    "PRECISION_VLM": ModelSlot.PRECISION_VLM,
    "OCR": ModelSlot.OCR,
    "fast_vlm": ModelSlot.FAST_VLM,
    "precision_vlm": ModelSlot.PRECISION_VLM,
    "ocr": ModelSlot.OCR,
}


def resolve_model_slot(value: Any) -> Optional[ModelSlot]:
    """Resolve a canonical slot name or legacy alias to a ModelSlot.

    Deterministic and explicit about case handling. Returns ``None`` for
    unknown values (callers decide how to treat them).
    """
    if isinstance(value, ModelSlot):
        return value
    if value is None:
        return None
    key = str(value).strip()
    return _LEGACY_ALIASES.get(key)


# ---------------------------------------------------------------------------
# Default model-identity catalog (configuration-driven at runtime; these
# defaults are role *references*, not private endpoints).
# ---------------------------------------------------------------------------

DEFAULT_MODEL_SLOTS = {
    ModelSlot.FAST_VLM: "qwen2.5vl",
    ModelSlot.PRECISION_VLM: "qwen3.6:27b",
    ModelSlot.OCR: "glm-ocr",
}

# Timeout safety ceilings (seconds) — NOT expected latency. Dynamic timeout
# selection (image size / crop / task) is explicitly deferred to a later
# stage.
DEFAULT_TIMEOUTS = {
    ModelSlot.FAST_VLM: 45,
    ModelSlot.PRECISION_VLM: 120,
    ModelSlot.OCR: 45,
}

# PRECISION_VLM execution profile — the permanently bound default for the
# Precision slot (task HERMES_VISION_PRECISION_NATIVE_GENERATE_PROFILE_
# BINDING_V0_1). Applies when the selected slot is PRECISION_VLM and the
# caller supplies no explicit transport override; an explicit
# OPENAI_COMPATIBLE override remains honored for diagnostics.
# Profile values are explicit — no reliance on Ollama defaults, model
# maximum context, or global OLLAMA_CONTEXT_LENGTH.
PRECISION_EXECUTION_PROFILE: Dict[str, Any] = {
    "model": DEFAULT_MODEL_SLOTS[ModelSlot.PRECISION_VLM],
    "transport": TRANSPORT_OLLAMA_NATIVE_GENERATE,
    "endpoint_family": "ollama_native_generate",
    "num_ctx": 32768,
    "num_predict": 4000,
    "temperature": 0.1,
    "stream": False,
    "timeout_seconds": DEFAULT_TIMEOUTS[ModelSlot.PRECISION_VLM],
    "structured_output": True,
}

# V0.1: exactly one model invocation per analyze_image request.
MAX_MODEL_CALLS = 1

# Prompt-template versions (versioned IDs; prompt bodies live in the
# orchestrator so they can evolve without touching the policy layer).
PROMPT_TEMPLATE_IDS = {
    VisionTask.SCENE_DESCRIBE: "vision-b1-scene-describe-v1",
    VisionTask.UI_READ: "vision-b1-ui-read-v1",
    VisionTask.UI_LOCATE: "vision-b1-ui-locate-v1",
    VisionTask.EXACT_OCR: "vision-b1-exact-ocr-v1",
    VisionTask.EVIDENCE_VERIFY: "vision-b1-evidence-verify-v1",
}

# ---------------------------------------------------------------------------
# Request / Plan / Result schemas
# ---------------------------------------------------------------------------


@dataclass
class CoordinateContext:
    """Coordinate-space metadata. Source-image pixel space is canonical.

    Browser viewport context is recorded when supplied so a later executor
    can map image pixels → CSS px → screen px. The Orchestrator never clicks.
    """

    source_width_px: Optional[int] = None
    source_height_px: Optional[int] = None
    crop_x_px: int = 0
    crop_y_px: int = 0
    crop_width_px: Optional[int] = None
    crop_height_px: Optional[int] = None
    viewport_width_css: Optional[int] = None
    viewport_height_css: Optional[int] = None
    device_scale_factor: Optional[float] = None
    coordinate_space: str = "SOURCE_IMAGE_PIXELS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_width_px": self.source_width_px,
            "source_height_px": self.source_height_px,
            "crop_x_px": self.crop_x_px,
            "crop_y_px": self.crop_y_px,
            "crop_width_px": self.crop_width_px,
            "crop_height_px": self.crop_height_px,
            "viewport_width_css": self.viewport_width_css,
            "viewport_height_css": self.viewport_height_css,
            "device_scale_factor": self.device_scale_factor,
            "coordinate_space": self.coordinate_space,
        }


@dataclass
class VisionRequest:
    """Canonical request. Never carries credentials, cookies, private page
    HTML, or authentication state."""

    request_id: str
    image_source: str
    task: VisionTask
    mode: VisionMode = VisionMode.AUTO
    criticality: VisionCriticality = VisionCriticality.NORMAL
    question: str = ""
    required_outputs: List[str] = field(default_factory=list)
    region: Optional[str] = None
    hints: Dict[str, Any] = field(default_factory=dict)
    coordinate_context: Optional[CoordinateContext] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionRequest":
        cc = data.get("coordinate_context")
        return cls(
            request_id=str(data.get("request_id") or ""),
            image_source=str(data.get("image_source") or ""),
            task=VisionTask(str(data.get("task") or VisionTask.SCENE_DESCRIBE.value)),
            mode=VisionMode(str(data.get("mode") or VisionMode.AUTO.value)),
            criticality=VisionCriticality(
                str(data.get("criticality") or VisionCriticality.NORMAL.value)
            ),
            question=str(data.get("question") or ""),
            required_outputs=list(data.get("required_outputs") or []),
            region=data.get("region"),
            hints=dict(data.get("hints") or {}),
            coordinate_context=CoordinateContext(**cc) if isinstance(cc, dict) else None,
        )


@dataclass
class VisionPlan:
    """Deterministic plan produced by the planner (no network)."""

    selected_slot: ModelSlot
    selected_model_identity: str
    selection_reason: str
    prompt_template_id: str
    quality_gates: List[str]
    timeout_seconds: float
    max_model_calls: int = MAX_MODEL_CALLS
    escalation_may_be_recommended: bool = True
    click_authorized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_slot": self.selected_slot.value,
            "selected_model_identity": self.selected_model_identity,
            "selection_reason": self.selection_reason,
            "prompt_template_id": self.prompt_template_id,
            "quality_gates": list(self.quality_gates),
            "timeout_seconds": self.timeout_seconds,
            "max_model_calls": self.max_model_calls,
            "escalation_may_be_recommended": self.escalation_may_be_recommended,
            "click_authorized": self.click_authorized,
        }


# ---------------------------------------------------------------------------
# Vision-Need policy (pure; no Browser / model / network)
# ---------------------------------------------------------------------------

# Inputs the pure policy may consider. Values are explicit booleans supplied
# by the caller (typically the Agent strategy layer), NOT derived by calling
# any external system.
DOM_SUFFICIENT_HINTS = (
    "dom_sufficient",
    "accessibility_sufficient",
)


def decide_vision_need(
    *,
    dom_sufficient: bool = False,
    accessibility_sufficient: bool = False,
    image_only_content: bool = False,
    canvas_content: bool = False,
    screenshot_evidence_required: bool = False,
    precise_coordinates_required: bool = False,
    explicit_image_request: bool = False,
    visual_state_required: bool = False,
) -> VisionNeedDecision:
    """Decide whether vision is required.

    Pure policy: considers only explicit caller-supplied booleans. Never
    calls Browser, a model, or the network.
    """
    if dom_sufficient or accessibility_sufficient:
        return VisionNeedDecision.VISION_NOT_NEEDED
    if (
        image_only_content
        or canvas_content
        or screenshot_evidence_required
        or precise_coordinates_required
        or explicit_image_request
        or visual_state_required
    ):
        return VisionNeedDecision.VISION_REQUIRED
    return VisionNeedDecision.VISION_NOT_NEEDED


# ---------------------------------------------------------------------------
# Planner — deterministic task/criticality/mode → model-slot matrix
# ---------------------------------------------------------------------------

# mode=AUTO matrix: (task, criticality) → slot
_AUTO_MATRIX = {
    (VisionTask.SCENE_DESCRIBE, VisionCriticality.NORMAL): ModelSlot.FAST_VLM,
    (VisionTask.SCENE_DESCRIBE, VisionCriticality.HIGH): ModelSlot.PRECISION_VLM,
    (VisionTask.UI_READ, VisionCriticality.NORMAL): ModelSlot.FAST_VLM,
    (VisionTask.UI_READ, VisionCriticality.HIGH): ModelSlot.PRECISION_VLM,
    (VisionTask.UI_LOCATE, VisionCriticality.NORMAL): ModelSlot.FAST_VLM,
    (VisionTask.UI_LOCATE, VisionCriticality.HIGH): ModelSlot.PRECISION_VLM,
    (VisionTask.EXACT_OCR, VisionCriticality.NORMAL): ModelSlot.OCR,
    (VisionTask.EXACT_OCR, VisionCriticality.HIGH): ModelSlot.OCR,
    (VisionTask.EVIDENCE_VERIFY, VisionCriticality.NORMAL): ModelSlot.PRECISION_VLM,
    (VisionTask.EVIDENCE_VERIFY, VisionCriticality.HIGH): ModelSlot.PRECISION_VLM,
}

# Tasks for which an explicit FAST mode is never safe at HIGH criticality.
_HIGH_TASKS_REQUIRING_PRECISION = {
    VisionTask.UI_READ,
    VisionTask.UI_LOCATE,
    VisionTask.EVIDENCE_VERIFY,
}

# Explicit OCR is valid for EXACT_OCR and for text-specific UI_READ requests
# whose required outputs are exact text.
_OCR_OK_TASKS = {VisionTask.EXACT_OCR, VisionTask.UI_READ}


def plan_vision(
    request: VisionRequest,
    model_slots: Optional[Dict[ModelSlot, str]] = None,
    timeouts: Optional[Dict[ModelSlot, float]] = None,
) -> VisionPlan:
    """Deterministic planner. No network, no model call."""
    slots = model_slots or dict(DEFAULT_MODEL_SLOTS)
    timeouts = timeouts or dict(DEFAULT_TIMEOUTS)
    task = request.task
    criticality = request.criticality
    mode = request.mode

    # Explicit mode behavior -------------------------------------------------
    if mode == VisionMode.PRECISION:
        slot = ModelSlot.PRECISION_VLM
        reason = PolicyReasonCode.EXPLICIT_PRECISION.value
    elif mode == VisionMode.OCR:
        if task not in _OCR_OK_TASKS:
            # OCR is a text specialist; refuse for non-text tasks rather than
            # silently degrading.
            raise PolicyBlockedError(
                f"EXPLICIT_OCR_INVALID_TASK:{task.value}",
                recommended_slot=ModelSlot.PRECISION_VLM,
            )
        slot = ModelSlot.OCR
        reason = PolicyReasonCode.EXPLICIT_OCR.value
    elif mode == VisionMode.FAST:
        if (
            criticality == VisionCriticality.HIGH
            and task in _HIGH_TASKS_REQUIRING_PRECISION
        ):
            # Safety policy (frozen): FAST requested for a HIGH-criticality
            # UI_READ / UI_LOCATE / EVIDENCE_VERIFY must not silently force
            # the low-precision model. Deterministic behavior: POLICY_BLOCKED
            # with the recommended compatible slot. (This is the safest
            # interpretation of the accepted design and matches Hermes'
            # safety-over-override semantics.)
            raise PolicyBlockedError(
                PolicyReasonCode.FAST_INVALID_FOR_HIGH.value,
                recommended_slot=ModelSlot.PRECISION_VLM,
            )
        slot = ModelSlot.FAST_VLM
        reason = PolicyReasonCode.EXPLICIT_FAST.value
    else:  # AUTO
        slot = _AUTO_MATRIX.get((task, criticality), ModelSlot.PRECISION_VLM)
        reason = PolicyReasonCode.AUTO_MATRIX.value

    model_identity = slots.get(slot, DEFAULT_MODEL_SLOTS[slot])
    timeout = timeouts.get(slot, DEFAULT_TIMEOUTS[slot])

    return VisionPlan(
        selected_slot=slot,
        selected_model_identity=model_identity,
        selection_reason=reason,
        prompt_template_id=PROMPT_TEMPLATE_IDS[task],
        quality_gates=_quality_gates_for(task),
        timeout_seconds=float(timeout),
        max_model_calls=MAX_MODEL_CALLS,
        escalation_may_be_recommended=True,
        click_authorized=False,
    )


def _quality_gates_for(task: VisionTask) -> List[str]:
    return {
        VisionTask.SCENE_DESCRIBE: [
            "response_present",
            "observation_non_empty",
            "observation_inference_distinguished",
            "no_excessive_hedging",
        ],
        VisionTask.UI_READ: [
            "response_present",
            "required_text_present",
            "no_contradictory_text",
            "no_explicit_unreadable",
        ],
        VisionTask.UI_LOCATE: [
            "response_present",
            "target_found",
            "target_unique",
            "bbox_or_point_present",
            "coordinates_in_bounds",
            "bbox_ordering_valid",
            "coordinate_space_declared",
        ],
        VisionTask.EXACT_OCR: [
            "response_present",
            "text_non_empty",
            "region_match",
            "no_placeholder",
        ],
        VisionTask.EVIDENCE_VERIFY: [
            "response_present",
            "observation_inference_distinguished",
            "evidence_provided",
            "no_unresolved_contradiction",
            "no_action_claimed",
        ],
    }.get(task, ["response_present"])


# ---------------------------------------------------------------------------
# Policy-blocked error
# ---------------------------------------------------------------------------


class PolicyBlockedError(Exception):
    """Raised when the deterministic policy refuses a request.

    Carries a non-sensitive reason code and the recommended compatible slot.
    """

    def __init__(
        self,
        reason: str,
        *,
        recommended_slot: Optional[ModelSlot] = None,
    ):
        super().__init__(reason)
        self.reason = reason
        self.recommended_slot = recommended_slot


# ---------------------------------------------------------------------------
# Coordinate validation helpers (pure)
# ---------------------------------------------------------------------------

SOURCE_IMAGE_PIXELS = "SOURCE_IMAGE_PIXELS"


def validate_coordinates(
    *,
    x: Optional[float],
    y: Optional[float],
    width: Optional[int],
    height: Optional[int],
    bbox: Optional[List[float]] = None,
    crop_x: int = 0,
    crop_y: int = 0,
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
) -> List[str]:
    """Validate pixel coordinates against source-image bounds.

    Returns a list of failed-gate reason codes (empty = valid).

    HALF-OPEN pixel-space contract (documented in DESIGN.md):
    - ``SOURCE_IMAGE_PIXELS``: integer pixel grid, origin (0,0) top-left;
    - a POINT (x, y) satisfies 0 <= x < W and 0 <= y < H — x == W is
      INVALID (points are pixel positions);
    - a BBOX is half-open: 0 <= left < right <= W and
      0 <= top < bottom <= H — right == W IS a valid bbox edge;
    - a CROP is half-open: crop_x >= 0, crop_width > 0 and
      crop_x + crop_width <= W — a crop ending exactly at W is valid;
    - a point inside a bbox satisfies left <= x < right and
      top <= y < bottom;
    - crop-local coordinates are mapped back to source-image space before
      validation when a crop is used.
    """
    failures: List[str] = []

    # Source dimensions must be present and positive when coordinates are
    # returned.
    if width is None or height is None:
        failures.append("coordinate_space_missing_dimensions")
        return failures
    if width <= 0 or height <= 0:
        failures.append("coordinate_space_nonpositive_dimensions")
        return failures

    # Crop validation: origin >= 0, dimensions > 0, and the crop must lie
    # inside the source image (half-open: crop end may equal W/H).
    if crop_x < 0 or crop_y < 0:
        failures.append("crop_origin_negative")
    if crop_width is not None and crop_width <= 0:
        failures.append("crop_width_nonpositive")
    if crop_height is not None and crop_height <= 0:
        failures.append("crop_height_nonpositive")
    if crop_width is not None and crop_x + crop_width > width:
        failures.append("crop_exceeds_source_width")
    if crop_height is not None and crop_y + crop_height > height:
        failures.append("crop_exceeds_source_height")

    if bbox is not None and len(bbox) == 4:
        left, top, right, bottom = bbox
        # Half-open strict ordering: left < right and top < bottom
        # (zero-area bbox is rejected).
        if left >= right:
            failures.append("bbox_left_ge_right")
        if top >= bottom:
            failures.append("bbox_top_ge_bottom")
        # Bbox edges: 0 <= left < right <= W (right == W is valid).
        if not (0 <= left < right <= width):
            failures.append("bbox_x_out_of_bounds")
        if not (0 <= top < bottom <= height):
            failures.append("bbox_y_out_of_bounds")
        if x is not None:
            # Point inside bbox: left <= x < right, top <= y < bottom.
            if not (left <= x < right):
                failures.append("point_outside_bbox")
            if y is not None and not (top <= y < bottom):
                failures.append("point_outside_bbox")

    if x is not None:
        # Map crop-local → source-image coordinates. Half-open: x < width.
        src_x = x + (crop_x or 0)
        if not (0 <= src_x < width):
            failures.append("x_out_of_bounds")
    if y is not None:
        src_y = y + (crop_y or 0)
        if not (0 <= src_y < height):
            failures.append("y_out_of_bounds")
    return failures


def validate_normalized_point(x_norm: float, y_norm: float) -> List[str]:
    """Validate a normalized point.

    Half-open: 0.0 <= x_norm < 1.0 and 0.0 <= y_norm < 1.0. A value of
    1.0 is INVALID — it does not denote a pixel center.
    """
    failures: List[str] = []
    if not (0.0 <= x_norm < 1.0):
        failures.append("x_norm_out_of_bounds")
    if not (0.0 <= y_norm < 1.0):
        failures.append("y_norm_out_of_bounds")
    return failures


# ---------------------------------------------------------------------------
# Quality evaluation (deterministic; no second model call)
# ---------------------------------------------------------------------------


def evaluate_quality(
    task: VisionTask,
    *,
    response_text: Optional[str],
    required_outputs: List[str],
    extracted: Optional[Dict[str, Any]] = None,
    coordinates: Optional[Dict[str, Any]] = None,
    coordinate_context: Optional[CoordinateContext] = None,
) -> Dict[str, Any]:
    """Evaluate one model response against the task's quality gates.

    Returns ``{"quality_decision": ..., "passed_gates": [...],
    "failed_gates": [...]}``. Never calls another model.
    """
    passed: List[str] = []
    failed: List[str] = []
    extracted = extracted or {}
    coordinates = coordinates or {}

    if not response_text or not str(response_text).strip():
        failed.append("response_present")
        return {
            "quality_decision": QualityDecision.NOT_EVALUATED.value,
            "passed_gates": passed,
            "failed_gates": failed,
        }
    passed.append("response_present")

    # Classification B: explicit refusal / inability / non-analysis response.
    # A syntactically present but non-analytic response is NOT a usable
    # escalation candidate — it is HUMAN_REVIEW_REQUIRED (no safe automatic
    # follow-up is warranted for a model that declined to analyze).
    if _is_refusal_or_inability(response_text):
        failed.append("refusal_or_inability")
        return {
            "quality_decision": QualityDecision.HUMAN_REVIEW_REQUIRED.value,
            "passed_gates": passed,
            "failed_gates": failed,
        }

    # Classification A: required structured response cannot be parsed.
    # ``extracted`` carries a marker from the orchestrator when JSON parsing
    # failed entirely — treat as NOT_EVALUATED (no evaluable content).
    if extracted.get("_parse_failed"):
        failed.append("structured_parse_failed")
        return {
            "quality_decision": QualityDecision.NOT_EVALUATED.value,
            "passed_gates": passed,
            "failed_gates": failed,
        }

    # Shared: required outputs present.
    missing_required = [o for o in required_outputs if o not in extracted]
    if missing_required:
        failed.append("required_outputs_missing")
    else:
        passed.append("required_outputs_present")

    # Shared: schema valid when structured output required.
    if task in (VisionTask.UI_LOCATE, VisionTask.UI_READ, VisionTask.EVIDENCE_VERIFY):
        if _has_contradiction(extracted):
            failed.append("unresolved_contradiction")
        else:
            passed.append("no_unresolved_contradiction")

    # Shared: no placeholder / empty result.
    if _is_placeholder(response_text):
        failed.append("placeholder_result")

    # Task-specific gates ----------------------------------------------------
    if task == VisionTask.SCENE_DESCRIBE:
        if not extracted.get("observation"):
            failed.append("observation_non_empty")
        else:
            passed.append("observation_non_empty")
        if extracted.get("inference") is not None or extracted.get("uncertainty"):
            passed.append("observation_inference_distinguished")
        else:
            failed.append("observation_inference_distinguished")
        if _excessive_hedging(response_text):
            failed.append("no_excessive_hedging")
        else:
            passed.append("no_excessive_hedging")

    elif task == VisionTask.UI_READ:
        text_value, text_status = _resolve_ui_read_text(extracted)
        if text_value:
            passed.append("required_text_present")
        else:
            failed.append("required_text_present")
        if text_status == "conflict":
            failed.append("text_field_conflict")
        if _has_explicit_unreadable(response_text):
            failed.append("no_explicit_unreadable")
        else:
            passed.append("no_explicit_unreadable")

    elif task == VisionTask.UI_LOCATE:
        targets = extracted.get("targets") or []
        if targets:
            passed.append("target_found")
        else:
            failed.append("target_found")
        if len(targets) == 1:
            passed.append("target_unique")
        elif len(targets) > 1:
            failed.append("target_unique")
        if coordinates.get("bbox") or coordinates.get("point"):
            passed.append("bbox_or_point_present")
        else:
            failed.append("bbox_or_point_present")
        if coordinates.get("coordinate_space"):
            passed.append("coordinate_space_declared")
        else:
            failed.append("coordinate_space_declared")
        # Validate coordinates.
        cc = coordinate_context or CoordinateContext()
        coord_failures = validate_coordinates(
            x=coordinates.get("x"),
            y=coordinates.get("y"),
            width=cc.source_width_px,
            height=cc.source_height_px,
            bbox=coordinates.get("bbox"),
            crop_x=cc.crop_x_px,
            crop_y=cc.crop_y_px,
            crop_width=cc.crop_width_px,
            crop_height=cc.crop_height_px,
        )
        if coord_failures:
            failed.extend(coord_failures)
        else:
            passed.append("coordinates_in_bounds")

    elif task == VisionTask.EXACT_OCR:
        text = extracted.get("observed_text")
        if text and str(text).strip():
            passed.append("text_non_empty")
        else:
            failed.append("text_non_empty")
        if extracted.get("region_match"):
            passed.append("region_match")
        elif region_mismatch_detected(extracted):
            failed.append("region_match")
        else:
            passed.append("region_match")

    elif task == VisionTask.EVIDENCE_VERIFY:
        if extracted.get("observation") is not None:
            passed.append("observation_inference_distinguished")
        else:
            failed.append("observation_inference_distinguished")
        if extracted.get("evidence"):
            passed.append("evidence_provided")
        else:
            failed.append("evidence_provided")
        if extracted.get("action_claimed"):
            failed.append("no_action_claimed")
        else:
            passed.append("no_action_claimed")

    # Quality decision -------------------------------------------------------
    # Deterministic classification of failures into safe escalation candidates
    # vs. human-review cases. A failure is only ESCALATE_RECOMMENDED when a
    # task-relevant partial result exists and a specific higher-precision or
    # OCR follow-up would plausibly resolve the missing evidence. Generic
    # prose, placeholder output, and unresolved contradictions are NOT safe
    # escalation candidates.
    if failed:
        if _is_escalation_candidate(failed, extracted):
            decision = QualityDecision.ESCALATE_RECOMMENDED.value
        else:
            decision = QualityDecision.HUMAN_REVIEW_REQUIRED.value
    else:
        decision = QualityDecision.PASS.value

    return {
        "quality_decision": decision,
        "passed_gates": sorted(set(passed)),
        "failed_gates": sorted(set(failed)),
    }


# -- small deterministic helpers --------------------------------------------


def _normalize_text_value(value: Any) -> Any:
    """Normalize a text-field value for deterministic comparison.

    Lists → tuple of stripped non-empty strings; strings → stripped string;
    anything else (dict, int, None, ...) → ``None``. Never raises; non-string
    values fail safely.
    """
    if isinstance(value, list):
        items = tuple(str(v).strip() for v in value if str(v).strip())
        return items or None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _resolve_ui_read_text(
    extracted: Dict[str, Any],
) -> Tuple[Any, str]:
    """Resolve the canonical UI_READ text field at one explicit boundary.

    Canonical field: ``observed_text`` — the established internal Vision
    field (Prompt schema example, UI_READ/EXACT_OCR gates, calibration and
    existing tests all use it). Legacy alias: ``visible_text`` — accepted
    ONLY through this resolver (introduced by the native-generate strict
    schema; backward-compatible normalization).

    Returns ``(value, status)`` where ``status`` is one of:

    - ``"canonical"`` — ``observed_text`` present and non-empty;
    - ``"alias"`` — ``observed_text`` missing/empty, ``visible_text``
      non-empty (legacy compatibility);
    - ``"conflict"`` — both fields present with materially different
      normalized values (never silently choose; caller marks contradiction);
    - ``"missing"`` — neither field is usable.
    """
    observed = extracted.get("observed_text")
    visible = extracted.get("visible_text")
    observed_norm = _normalize_text_value(observed)
    visible_norm = _normalize_text_value(visible)
    if observed_norm and visible_norm:
        if observed_norm == visible_norm:
            return observed, "canonical"
        return None, "conflict"
    if observed_norm:
        return observed, "canonical"
    if visible_norm:
        return visible, "alias"
    return None, "missing"


def _has_contradiction(extracted: Dict[str, Any]) -> bool:
    """True only for an EXPLICIT contradiction signal.

    Diagnostic finding B (HERMES_VISION_FULL_CORPUS_QUALITY_FINDINGS_
    DIAGNOSTIC_V0_1): repeated identical entries inside ``observed_text``
    are NOT contradictions — UI screenshots legitimately repeat navigation
    labels and page text, and models may enumerate the same visible label
    more than once. Treating duplication as contradiction produced four
    false HUMAN_REVIEW_REQUIRED results in the corpus.

    The only reliable contradiction signal is an explicit model-flagged
    ``contradiction`` field (EVIDENCE_VERIFY task schema) — never inferred
    from mere repetition.
    """
    flagged = extracted.get("contradiction")
    if isinstance(flagged, str) and flagged.strip():
        return True
    return False


def _is_refusal_or_inability(text: str) -> bool:
    """Detect explicit refusal / inability / non-analysis responses.

    Bounded deterministic signal set (English + Chinese). A model that
    declined to analyze is not a safe escalation candidate — the response
    contains no task-relevant observation a follow-up model could build on
    deterministically.
    """
    t = str(text).strip().lower()
    signals = (
        # English
        "i cannot analyze",
        "i can't analyze",
        "cannot analyze this image",
        "unable to analyze",
        "i am unable to read",
        "i cannot read the image",
        "unable to read the image",
        "the image is unavailable",
        "no image was provided",
        "i cannot see the requested text",
        "cannot see the requested text",
        "i cannot view images",
        "i can't view images",
        "i'm sorry",
        "i am sorry",
        # Chinese
        "无法分析这张图片",
        "无法分析图片",
        "看不清",
        "无法读取",
        "无法识别",
        "无法查看图片",
        "不能分析",
        "图片不可用",
        "没有提供图片",
    )
    return any(s in t for s in signals)


def _is_escalation_candidate(failed_gates: List[str], extracted: Dict[str, Any]) -> bool:
    """A failure is a safe escalation candidate only when:
    - no unresolved contradiction exists (contradiction is human-review);
    - no placeholder/refusal/parse failure exists;
    - no required action was falsely claimed;
    - a task-relevant partial result actually exists.
    """
    never_escalate = {
        "unresolved_contradiction",
        "text_field_conflict",
        "placeholder_result",
        "refusal_or_inability",
        "structured_parse_failed",
        "no_action_claimed",
        "action_claimed",
    }
    if any(g in never_escalate for g in failed_gates):
        return False
    if not extracted:
        return False
    # A purely cosmetic failure (e.g. hedging only) with no other gate
    # failures would already be PASS; any remaining failure here implies
    # missing/uncertain evidence, which a precision/OCR follow-up can
    # address.
    return True


def _is_placeholder(text: str) -> bool:
    t = str(text).strip().lower()
    placeholders = (
        "unable to see",
        "cannot see",
        "i cannot view",
        "i can't view",
        "not visible in",
        "no image provided",
        "i cannot analyze",
        "i can't analyze",
        "cannot analyze",
        "unable to analyze",
        "无法分析",
        "看不清",
        "无法读取",
        "无法识别",
    )
    return any(p in t for p in placeholders)


def _excessive_hedging(text: str) -> bool:
    t = str(text).strip().lower()
    hedge_words = ("maybe", "perhaps", "possibly", "i think", "probably", "似乎", "可能", "大概")
    hits = sum(1 for w in hedge_words if w in t)
    return hits >= 3


def _has_explicit_unreadable(text: str) -> bool:
    t = str(text).strip().lower()
    markers = (
        "unreadable",
        "cannot read",
        "can't read",
        "illegible",
        "blurry",
        "看不清",
        "无法辨认",
        "无法读取",
    )
    return any(m in t for m in markers)


def region_mismatch_detected(extracted: Dict[str, Any]) -> bool:
    """True when the OCR response explicitly contradicts the requested
    region (e.g. quoted text clearly outside the crop)."""
    return bool(extracted.get("region_mismatch"))


# ---------------------------------------------------------------------------
# Serializable result / trace builders
# ---------------------------------------------------------------------------


def build_result(
    *,
    execution_status: ExecutionStatus,
    quality: Dict[str, Any],
    answer: str = "",
    structured: Optional[Dict[str, Any]] = None,
    initial_slot: ModelSlot,
    final_slot: ModelSlot,
    actual_model: str,
    recommended_next_slot: Optional[ModelSlot] = None,
    coordinate_context: Optional[CoordinateContext] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
    image_sha256: Optional[str] = None,
    logical_model_calls: int = 0,
    transport_meta: Optional[Dict[str, Any]] = None,
    transport: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical VisionResult. Never contains image bytes, base64, cookies,
    credentials, full prompts, page HTML, or auth tokens.

    ``image_sha256`` is the SHA-256 of the EXACT normalized image bytes used
    for the model request (non-sensitive metadata only) — this is the
    canonical calibration identity. ``logical_model_calls`` counts
    Orchestrator-level model invocations; it does NOT count lower-level
    transport retries performed inside the existing auxiliary client.

    ``transport_meta`` (optional) carries the two-layer transport hash
    contract: ``transport_image_sha256`` (SHA-256 of the exact bytes encoded
    in the model-request data URL), ``transport_mime_type`` (MIME declared in
    that data URL) and ``transport_transcoded`` (true only when canonical
    bytes were converted for endpoint compatibility, e.g. WebP → PNG).

    ``transport`` (optional) carries the transport-selection metadata for the
    invocation path used :
    ``requested_transport``, ``selected_transport``, ``endpoint_family`` and
    ``native_generate_used``. Non-sensitive; never contains endpoint
    credentials or raw output.
    """
    quality_decision = quality.get("quality_decision") or QualityDecision.NOT_EVALUATED.value
    human_review = quality_decision == QualityDecision.HUMAN_REVIEW_REQUIRED.value
    return {
        "execution_status": execution_status.value,
        "quality_decision": quality_decision,
        "answer": answer,
        "structured": structured or {},
        "initial_model_slot": initial_slot.value,
        "final_model_slot": final_slot.value,
        "actual_model": actual_model,
        "recommended_next_slot": (
            recommended_next_slot.value if recommended_next_slot else None
        ),
        "human_review_required": human_review,
        "click_authorized": False,
        "quality": {
            "passed_gates": quality.get("passed_gates", []),
            "failed_gates": quality.get("failed_gates", []),
        },
        "coordinate_context": (
            coordinate_context.to_dict() if coordinate_context else {}
        ),
        "normalized_image_sha256": image_sha256,
        "transport_image_sha256": (transport_meta or {}).get("transport_image_sha256"),
        "transport_mime_type": (transport_meta or {}).get("transport_mime_type"),
        "transport_transcoded": (transport_meta or {}).get("transport_transcoded"),
        "transport": transport or {},
        "logical_model_calls": logical_model_calls,
        "trace": trace or [],
    }


def serialize(obj: Any) -> str:
    """JSON-serialize a result dict (enums already converted to values)."""
    return json.dumps(obj, ensure_ascii=False, default=str)
