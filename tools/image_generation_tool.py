#!/usr/bin/env python3
"""Image generation via FAL.ai (model picked in ``hermes tools``, persisted to ``image_gen.model``).

``_build_fal_payload()`` / ``_build_fal_edit_payload()`` translate unified inputs into the
``FAL_MODELS`` payload filtered to its ``supports`` whitelist so models never receive rejected
keys. Clarity upscaling is strictly per-call opt-in: default-on degraded text/CJK/faces.
"""

import json
import logging
import os
import datetime
import threading
import uuid
from typing import Any, Dict, Optional

# Imported lazily by _load_fal_client() (~64 ms on every CLI cold start); a test-monkeypatched
# value short-circuits the loader.
fal_client: Any = None


def _load_fal_client() -> Any:
    """Lazily import fal_client into the module global (idempotent; keeps a test-installed mock)."""
    global fal_client
    if fal_client is None:
        from tools.fal_common import import_fal_client
        fal_client = import_fal_client()
    return fal_client


from tools.debug_helpers import DebugSession
from tools.fal_common import _ManagedFalSyncClient, _extract_http_status, _normalize_fal_queue_url_format
from tools.image_generation_catalog import (
    DEFAULT_ASPECT_RATIO, DEFAULT_MODEL, FAL_MODELS, UPSCALER_CREATIVITY, UPSCALER_DEFAULT_PROMPT,
    UPSCALER_FACTOR, UPSCALER_GUIDANCE_SCALE, UPSCALER_MODEL, UPSCALER_NEGATIVE_PROMPT,
    UPSCALER_NUM_INFERENCE_STEPS, UPSCALER_RESEMBLANCE, UPSCALER_SAFETY_CHECKER, VALID_ASPECT_RATIOS,
)
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    NOUS_MANAGED_PROVIDER, fal_key_is_configured, managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message, read_selection, selection_error)

logger = logging.getLogger(__name__)

_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")
_managed_fal_client = None
_managed_fal_client_config = None
_managed_fal_client_lock = threading.Lock()


# --- Managed FAL gateway (Nous Subscription) ---
def _resolve_managed_fal_gateway():
    """Managed gateway config for the stored `hermes tools` selection, or ``None`` for direct FAL.

    ``"nous"`` (or legacy ``use_gateway: true``) → managed ONLY (unreachable = selection-naming
    error, never a silent FAL_KEY fallback). Other stored provider → direct ONLY (missing FAL_KEY
    = error naming the selection). Never configured → autodetect: direct if FAL_KEY, else managed.
    """
    selected = read_selection("image_gen")
    if selected == NOUS_MANAGED_PROVIDER:
        gateway = resolve_managed_tool_gateway("fal-queue")
        if gateway is None:
            raise ValueError(selection_error(
                "image_gen", NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or unreachable)"))
        return gateway
    if selected is not None:
        if fal_key_is_configured():
            return None
        raise ValueError(selection_error("image_gen", selected, "FAL_KEY is not set"))
    # Never-configured category: legacy credential autodetect (do NOT persist).
    return None if fal_key_is_configured() else resolve_managed_tool_gateway("fal-queue")


def _get_managed_fal_client(managed_gateway):
    """Reuse the managed FAL client so its internal httpx.Client is not leaked per call."""
    global _managed_fal_client, _managed_fal_client_config
    client_config = (managed_gateway.gateway_origin.rstrip("/"), managed_gateway.nous_user_token)
    with _managed_fal_client_lock:
        if _managed_fal_client is None or _managed_fal_client_config != client_config:
            # Resolved on this module so monkeypatching ``image_generation_tool.fal_client`` still applies.
            _managed_fal_client = _ManagedFalSyncClient(
                _load_fal_client(), key=managed_gateway.nous_user_token,
                queue_run_origin=managed_gateway.gateway_origin)
            _managed_fal_client_config = client_config
        return _managed_fal_client


class ImageGenerationInterrupted(Exception):
    """Raised when the user interrupts while a FAL job is in flight."""


def _wait_fal_result(handler, *, poll_seconds: float = 0.5):
    """Interrupt-aware ``handler.get()``: the SDK blocks 30-60s, hiding user interrupts.

    The get runs on a daemon worker; the interrupt bit is polled between join slices and on
    interrupt the worker is abandoned (remote job keeps running).
    """
    from tools.interrupt import is_interrupted
    result_box: list = []
    error_box: list = []
    def _get():
        try:
            result_box.append(handler.get())
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
            error_box.append(exc)
    worker = threading.Thread(target=_get, daemon=True, name="fal-result-wait")
    worker.start()
    while worker.is_alive():
        if is_interrupted():
            raise ImageGenerationInterrupted(
                "Image generation interrupted by user — abandoned the in-flight FAL job.")
        worker.join(timeout=poll_seconds)
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


def _submit_fal_request(model: str, arguments: Dict[str, Any]):
    """Submit a FAL request using direct credentials or the managed queue gateway."""
    _load_fal_client()
    request_headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = _resolve_managed_fal_gateway()
    if managed_gateway is None:
        return fal_client.submit(model, arguments=arguments, headers=request_headers)
    try:
        return _get_managed_fal_client(managed_gateway).submit(
            model, arguments=arguments, headers=request_headers)
    except Exception as exc:
        # A managed-gateway 4xx usually means the portal doesn't proxy this model
        # (allowlist miss, billing gate): give remediation instead of a raw httpx error.
        status = _extract_http_status(exc)
        if status is not None and 400 <= status < 500:
            gateway_message = ""
            if status in {401, 402, 403}:
                gateway_message = "\n\n" + nous_tool_gateway_unavailable_message(
                    "managed FAL image generation", force_fresh=True)
            raise ValueError(
                f"Nous Subscription gateway rejected model '{model}' (HTTP {status}). This model "
                f"may not yet be enabled on the Nous Portal's FAL proxy. Either:\n"
                f"  • Set FAL_KEY in your environment to use FAL.ai directly, or\n"
                f"  • Pick a different model via `hermes tools` → Image Generation."
                f"{gateway_message}") from exc
        raise


# --- Config readers, model resolution + payload construction ---
def _read_image_gen_key(key: str) -> Optional[str]:
    """Return the stripped ``image_gen.<key>`` string from config.yaml, or None."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        value = section.get(key) if isinstance(section, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception as exc:
        logger.debug("Could not read image_gen.%s: %s", key, exc)
    return None


def _read_configured_image_model():
    """``image_gen.model`` from config.yaml, or None."""
    return _read_image_gen_key("model")


def _read_configured_image_provider():
    """``image_gen.provider`` from config.yaml, or None (unset keeps the in-tree FAL fallback even
    when other providers are registered; ``"fal"`` routes via ``plugins/image_gen/fal/``).

    We only consult the plugin registry when this is explicitly set — an unset value keeps users on the
    in-tree FAL fallback even when other providers happen to be registered (e.g. a user has OPENAI_API_KEY
    set for other features but never asked for OpenAI image gen). ``"fal"`` explicitly routes through
    ``plugins/image_gen/fal/`` (which delegates back into this module's pipeline via call-time indirection —
    see issue #26241).
    """
    return _read_image_gen_key("provider")


def _plugin_provider_name() -> Optional[str]:
    """Configured provider that must go through the plugin registry; None for unset/fal/nous."""
    configured = _read_configured_image_provider()
    if not configured or configured in ("fal", NOUS_MANAGED_PROVIDER):
        return None
    return configured


def _resolve_fal_model() -> tuple:
    """Return ``(model_id, meta)`` for the configured FAL model, falling back to DEFAULT_MODEL (warned) when unknown."""
    # FAL_IMAGE_MODEL is an undocumented escape hatch (backward-compat for tests/scripts).
    model_id = _read_image_gen_key("model") or os.getenv("FAL_IMAGE_MODEL", "").strip()
    if model_id and model_id not in FAL_MODELS:
        logger.warning("Unknown FAL model '%s' in config; falling back to %s", model_id, DEFAULT_MODEL)
        model_id = None
    model_id = model_id or DEFAULT_MODEL
    return model_id, FAL_MODELS[model_id]


_SIZE_KEY_BY_STYLE = {"image_size_preset": "image_size", "gpt_literal": "image_size",
                      "aspect_ratio": "aspect_ratio"}


def _build_payload(model_id, prompt, aspect_ratio, seed, overrides, image_urls=None) -> Dict[str, Any]:
    """Text-to-image / edit payload (``image_urls`` selects edit mode): defaults + native size
    spec + overrides, filtered to the model whitelist.

    Edit endpoints mostly auto-infer size, so the size key is sent only when ``edit_supports``
    lists it. ``prompt`` (and ``image_urls`` on edits) survive a whitelist gap: every FAL
    endpoint requires them, so a catalog mistake can't send a broken request.
    """
    meta = FAL_MODELS[model_id]
    edit = image_urls is not None
    supports = (meta.get("edit_supports") or set()) if edit else meta["supports"]
    sizes = meta["sizes"]
    aspect = (aspect_ratio or DEFAULT_ASPECT_RATIO).lower().strip()
    if aspect not in sizes:
        aspect = DEFAULT_ASPECT_RATIO
    payload: Dict[str, Any] = dict(meta.get("defaults", {}))
    payload["prompt"] = (prompt or "").strip()
    required = {"prompt"}
    if edit:
        payload["image_urls"] = list(image_urls)
        required.add("image_urls")
    size_key = _SIZE_KEY_BY_STYLE.get(meta["size_style"])
    if size_key is None and not edit:
        raise ValueError(f"Unknown size_style: {meta['size_style']!r}")
    if size_key is not None and (not edit or size_key in supports):
        payload[size_key] = sizes[aspect]
    if isinstance(seed, int):
        payload["seed"] = seed
    payload.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return {k: v for k, v in payload.items() if k in supports or k in required}


def _build_fal_payload(model_id, prompt, aspect_ratio=DEFAULT_ASPECT_RATIO, seed=None, overrides=None):
    """FAL text-to-image payload for ``model_id`` from unified inputs."""
    return _build_payload(model_id, prompt, aspect_ratio, seed, overrides)


def _build_fal_edit_payload(model_id, prompt, image_urls, aspect_ratio=DEFAULT_ASPECT_RATIO,
                            seed=None, overrides=None):
    """FAL *edit* (image-to-image) payload: ``image_urls`` + prompt, filtered to ``edit_supports``."""
    return _build_payload(model_id, prompt, aspect_ratio, seed, overrides, image_urls=image_urls)


# --- Upscaler ---
def _upscale_image(image_url: str, original_prompt: str) -> Optional[Dict[str, Any]]:
    """Upscale via FAL's Clarity Upscaler; None on failure (caller keeps the original)."""
    try:
        logger.info("Upscaling image with Clarity Upscaler...")
        handler = _submit_fal_request(UPSCALER_MODEL, arguments={
            "image_url": image_url, "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR, "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY, "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER})
        result = _wait_fal_result(handler)
        if result and "image" in result:
            up = result["image"]
            logger.info("Image upscaled successfully to %sx%s",
                        up.get("width", "unknown"), up.get("height", "unknown"))
            return {
                "url": up["url"], "width": up.get("width", 0), "height": up.get("height", 0),
                "upscaled": True, "upscale_factor": UPSCALER_FACTOR}
        logger.error("Upscaler returned invalid response")
        return None
    except ImageGenerationInterrupted:
        # A user interrupt must not degrade into a silent "use original" fallback.
        raise
    except Exception as e:
        logger.error("Error upscaling image: %s", e, exc_info=True)
        return None


# --- Artifact path hinting for non-local terminal backends ---
_CONTAINER_HOME_ENVS = {"DockerEnvironment", "SingularityEnvironment", "ModalEnvironment"}
# No env yet: only deterministic cache roots translate side-effect free (SSH: tilde path; its
# first sync uploads the cache file).
_CACHE_BASE_BY_BACKEND = {"docker": "/root/.hermes", "singularity": "/root/.hermes",
                          "modal": "/root/.hermes", "ssh": "~/.hermes"}


def _looks_like_absolute_file_path(value: str) -> bool:
    if not value or not isinstance(value, str) or value.lower().startswith(("http://", "https://", "data:")):
        return False
    return os.path.isabs(value) or (len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"})


def _active_terminal_env(task_id: str | None):
    try:
        from tools.terminal_tool_lifecycle import get_active_env
        return get_active_env(task_id or "default")
    except Exception as exc:  # noqa: BLE001 - artifact hinting must not break generation
        logger.debug("Could not inspect active terminal environment: %s", exc)
        return None


def _agent_cache_base_for_env(env: Any) -> str | None:
    if env is not None:
        # Optional extension hook: an environment may expose its own agent-visible cache root.
        explicit = getattr(env, "agent_visible_cache_base", None)
        if callable(explicit):
            try:
                value = explicit()
                if value:
                    return str(value).rstrip("/")
            except Exception as exc:  # noqa: BLE001
                logger.debug("active env agent_visible_cache_base failed: %s", exc)
        remote_home = getattr(env, "_remote_home", None)
        if remote_home:
            return f"{str(remote_home).rstrip('/')}/.hermes"
        if env.__class__.__name__ in _CONTAINER_HOME_ENVS:
            return "/root/.hermes"
    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    return _CACHE_BASE_BY_BACKEND.get(backend)


def _postprocess_image_generate_result(raw: str, task_id: str | None = None) -> str:
    """Annotate successful local results: ``image`` stays the host/gateway-deliverable path;
    ``agent_visible_image`` is the same file as seen by a non-local terminal backend."""
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return raw
    if not isinstance(payload, dict) or not payload.get("success"):
        return raw
    image = payload.get("image")
    if not isinstance(image, str) or not _looks_like_absolute_file_path(image):
        return raw
    env = _active_terminal_env(task_id)
    cache_base = _agent_cache_base_for_env(env)
    if not cache_base:
        return raw
    try:
        from tools.credential_files import map_cache_path_to_container
        agent_path = map_cache_path_to_container(image, container_base=cache_base)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not translate image cache path for backend: %s", exc)
        return raw
    if not agent_path or agent_path == image:
        return raw
    sync_manager = getattr(env, "_sync_manager", None)
    if sync_manager is not None:
        try:
            sync_manager.sync(force=True)
        except Exception as exc:  # noqa: BLE001 - keep generation success; log for operators
            logger.warning("Could not force-sync generated image artifact: %s", exc)
    payload.setdefault("host_image", image)
    payload.setdefault("agent_visible_image", agent_path)
    return json.dumps(payload, ensure_ascii=False)


# --- Tool entry point ---
def _format_images(images: list, should_upscale: bool, prompt: str) -> list:
    """Normalize FAL result images, optionally chaining the upscaler (falls back to the original on failure)."""
    formatted = []
    for img in images:
        if not (isinstance(img, dict) and "url" in img):
            continue
        if should_upscale:
            upscaled = _upscale_image(img["url"], prompt.strip())
            if upscaled:
                formatted.append(upscaled)
                continue
            logger.warning("Using original image as fallback (upscale failed)")
        formatted.append({
            "url": img["url"], "width": img.get("width", 0), "height": img.get("height", 0),
            "upscaled": False})
    return formatted


def _prepare_fal_request(model_id, meta, prompt, aspect_ratio, seed, overrides, source_images):
    """Validate inputs and return ``(endpoint, arguments)``; raises ValueError with the user-facing message."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt is required and must be a non-empty string")
    # A stored-but-broken selection raises the selection-naming error from
    # _resolve_managed_fal_gateway(); only never-configured reports "no backend at all".
    if not (fal_key_is_configured() or _resolve_managed_fal_gateway()):
        raise ValueError(_build_no_backend_setup_message())
    edit_endpoint, display = meta.get("edit_endpoint"), meta.get("display", model_id)
    # Fail clearly rather than silently dropping sources and producing an unrelated picture.
    if source_images and not edit_endpoint:
        raise ValueError(
            f"Model '{display}' ({model_id}) is not capable of image-to-image / editing. "
            f"Provide a text-only prompt (omit image_url), or switch to an edit-capable model "
            f"via `hermes tools` → Image Generation.")
    aspect_lc = (aspect_ratio or DEFAULT_ASPECT_RATIO).lower().strip()
    if aspect_lc not in VALID_ASPECT_RATIOS:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_lc = DEFAULT_ASPECT_RATIO
    if source_images:
        # Clamp reference count to the model's declared cap.
        max_refs = int(meta.get("max_reference_images") or 1)
        clamped_sources = source_images[:max_refs] if max_refs > 0 else source_images
        arguments = _build_fal_edit_payload(
            model_id, prompt, clamped_sources, aspect_lc, seed=seed, overrides=overrides)
        logger.info("Editing image with %s (%s) — %d source image(s), prompt: %s",
                    display, edit_endpoint, len(clamped_sources), prompt[:80])
        return edit_endpoint, arguments
    arguments = _build_fal_payload(model_id, prompt, aspect_lc, seed=seed, overrides=overrides)
    logger.info("Generating image with %s (%s) — prompt: %s", display, model_id, prompt[:80])
    return model_id, arguments


def image_generate_tool(
    prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: Optional[int] = None, guidance_scale: Optional[float] = None,
    num_images: Optional[int] = None, output_format: Optional[str] = None,
    seed: Optional[int] = None, image_url: Optional[str] = None,
    reference_image_urls: Optional[list] = None, upscale: Optional[bool] = None) -> str:
    """Generate (or, with source images + an ``edit_endpoint`` model, edit) an image via FAL.

    Extra kwargs are overrides filtered per-model via ``supports`` / ``edit_supports`` (dropped
    silently so callers survive model switches). Returns JSON ``{"success", "image", "modality",
    "error", "error_type"}``.
    """
    model_id, meta = _resolve_fal_model()
    refs = reference_image_urls if isinstance(reference_image_urls, (list, tuple)) else []
    source_images = [c.strip() for c in (image_url, *refs) if isinstance(c, str) and c.strip()]
    use_edit = bool(source_images) and bool(meta.get("edit_endpoint"))
    modality = "image" if use_edit else "text"
    overrides: Dict[str, Any] = {
        "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale,
        "num_images": num_images, "output_format": output_format}
    debug_call_data = {
        "model": model_id,
        "parameters": {"prompt": prompt, "aspect_ratio": aspect_ratio, **overrides, "seed": seed,
                       "modality": modality, "source_images": len(source_images)},
        "error": None, "success": False, "images_generated": 0, "generation_time": 0}
    start_time = datetime.datetime.now()

    def finish(generation_time: float, response: Dict[str, Any]) -> str:
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        return json.dumps(response, indent=2, ensure_ascii=False)
    try:
        endpoint, arguments = _prepare_fal_request(
            model_id, meta, prompt, aspect_ratio, seed,
            {k: v for k, v in overrides.items() if v is not None}, source_images)
        result = _wait_fal_result(_submit_fal_request(endpoint, arguments=arguments))
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        if not result or "images" not in result:
            raise ValueError("Invalid response from FAL.ai API — no images returned")
        images = result.get("images", [])
        if not images:
            raise ValueError("No images were generated")
        # Explicit ``upscale`` wins, including for edits; the catalog default never upscales
        # edits (Clarity is a text-to-image pass and must not silently alter compositions).
        if upscale is not None:
            should_upscale = bool(upscale)
        else:
            should_upscale = bool(meta.get("upscale", False)) and not use_edit
        formatted_images = _format_images(images, should_upscale, prompt)
        if not formatted_images:
            raise ValueError("No valid image URLs returned from API")
        upscaled_count = sum(1 for img in formatted_images if img.get("upscaled"))
        logger.info("Generated %s image(s) in %.1fs (%s upscaled) via %s [%s]",
                    len(formatted_images), generation_time, upscaled_count, endpoint, modality)
        debug_call_data["success"] = True
        debug_call_data["images_generated"] = len(formatted_images)
        return finish(generation_time, {
            "success": True,
            "image": formatted_images[0]["url"],
            "modality": modality,
            "upscaled": bool(formatted_images[0].get("upscaled"))})
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        debug_call_data["error"] = error_msg
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        return finish(generation_time,
                      {"success": False, "image": None, "error": str(e), "error_type": type(e).__name__})


def check_fal_api_key() -> bool:
    """True if the selected FAL backend (never configured: any FAL backend) is available.

    A stored-but-broken selection reports False here (registry gating); the naming error
    surfaces at call time from ``_resolve_managed_fal_gateway``.
    """
    try:
        gateway = _resolve_managed_fal_gateway()
    except ValueError:
        return False
    return bool(gateway) or fal_key_is_configured()


def _build_no_backend_setup_message() -> str:
    """Actionable no-backend error: FAL_KEY signup, managed-gateway status, plugin alternative."""
    managed = managed_nous_tools_enabled()
    lines = ["Image generation is unavailable in this environment.", "", "Missing requirements:"]
    if managed:
        lines.append("  - FAL_KEY is not set and the managed FAL gateway is unreachable")
    else:
        lines.append("  - FAL_KEY environment variable is not set")
        if gateway_message := nous_tool_gateway_unavailable_message("managed FAL image generation"):
            lines.append(f"  - {gateway_message}")
    lines += ["", "To enable image generation, do one of:",
              "  1. Get a free API key at https://fal.ai and set FAL_KEY=<your-key> "
              "(then restart the session)"]
    if managed:
        lines.append("  2. Sign in to a Nous account that has the managed FAL gateway enabled "
                     "(`hermes setup`)")
    lines.append("  3. Configure a different image_gen provider via `hermes tools` → Image Generation "
                 "(run `hermes plugins list` to see installed backends)")
    return "\n".join(lines)


def _get_plugin_provider(name: str, *, force: bool = False):
    """Discover plugins (local import: importing this module must not trigger discovery) and return the named provider."""
    from agent.image_gen_registry import get_provider
    from hermes_cli.plugins import _ensure_plugins_discovered
    if force:
        _ensure_plugins_discovered(force=True)
    else:
        _ensure_plugins_discovered()
    return get_provider(name)


def check_image_generation_requirements() -> bool:
    """True if FAL or the explicitly configured image backend is available."""
    try:
        if check_fal_api_key():
            # Lazy import doubles as the SDK presence check: ImportError falls through to plugins.
            _load_fal_client()
            return True
    except ImportError:
        pass
    configured = _plugin_provider_name()
    if configured is None:
        return False
    # Probe only the selected plugin: a cloud key alone must not opt a user into a paid backend.
    try:
        provider = _get_plugin_provider(configured)
        return bool(provider and provider.is_available())
    except Exception:
        return False


# --- Registry ---
from tools.registry import registry, tool_error

IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    # Placeholder: description AND params are rebuilt at get_tool_definitions() time by
    # _build_dynamic_image_schema() from the active backend's capabilities. Edit-only args
    # and upscale are advertised ONLY when supported; the handler accepts them regardless
    # (replay compat + teaching errors).
    "description": (
        "Generate images from text prompts. The active model's edit/reference "
        "capabilities are rendered at serving time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The text prompt describing the desired image (text-to-"
                    "image) or the edit to apply (image-to-image). Be detailed "
                    "and descriptive."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": "The aspect ratio of the generated image. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            # image_url / reference_image_urls / upscale are added per-capability; never statically.
        },
        # See #95681.
        "required": ["prompt"],
    },
}


# --- Plugin provider dispatch + managed-mode Krea routing ---
def _provider_error(error: str, error_type: str) -> str:
    """JSON error envelope shared by every provider-dispatch failure path."""
    return json.dumps({"success": False, "image": None, "error": error, "error_type": error_type})


def _provider_result(result, contract_error: str) -> str:
    """JSON-encode a provider's dict result; anything else is a contract violation."""
    if not isinstance(result, dict):
        return _provider_error(contract_error, "provider_contract")
    return json.dumps(result)


def _add_provider_kwargs(kwargs, image_url, reference_image_urls, upscale, model=None) -> Dict[str, Any]:
    """Add the optional ``provider.generate(**kwargs)`` args in place (edit args only when supplied)."""
    if model:
        kwargs["model"] = model
    if isinstance(image_url, str) and image_url.strip():
        kwargs["image_url"] = image_url.strip()
    if reference_image_urls is not None:
        from agent.image_gen_provider import normalize_reference_images
        norm_refs = normalize_reference_images(reference_image_urls)
        if norm_refs:
            kwargs["reference_image_urls"] = norm_refs
    if upscale is not None:
        kwargs["upscale"] = bool(upscale)
    return kwargs


def _dispatch_to_plugin_provider(
    prompt: str, aspect_ratio: str, image_url: Optional[str] = None,
    reference_image_urls: Optional[list] = None, upscale: Optional[bool] = None):
    """JSON result from the selected plugin provider, or ``None`` to fall through to in-tree FAL
    (provider unset / ``"fal"`` / ``"nous"``). Providers without ``upscale`` ignore it via ``**kwargs``."""
    configured = _plugin_provider_name()
    if configured is None:
        return None
    try:
        provider = _get_plugin_provider(configured)
    except Exception as exc:
        logger.debug("image_gen plugin dispatch skipped: %s", exc)
        return None
    if provider is None:
        # Long-lived sessions may have discovered plugins before a bundled backend
        # was patched in or config changed: retry once with a forced refresh.
        try:
            provider = _get_plugin_provider(configured, force=True)
        except Exception as exc:
            logger.debug("image_gen plugin force-refresh skipped: %s", exc)
    if provider is None:
        return _provider_error(
            f"image_gen.provider='{configured}' is set but no plugin registered that name. "
            f"Run `hermes plugins list` to see available image gen backends.", "provider_not_registered")
    pname = getattr(provider, "name", "?")
    kwargs: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio}
    try:
        _add_provider_kwargs(kwargs, image_url, reference_image_urls, upscale,
                             model=_read_configured_image_model())
        result = provider.generate(**kwargs)
    except Exception as exc:
        # A TypeError from generate() predating image_url support (third-party plugin not yet
        # updated): text-to-image keeps working; surface a clear note when an edit was requested.
        is_type_error = isinstance(exc, TypeError)
        if is_type_error and ("image_url" in kwargs or "reference_image_urls" in kwargs):
            logger.warning("image_gen provider '%s' rejected image-to-image kwargs "
                           "(signature too narrow): %s", pname, exc)
            return _provider_error(
                f"Provider '{pname}' does not support image-to-image / editing (its generate() "
                f"signature is out of date with the image_generate schema). Omit image_url for "
                f"text-to-image, or pick a backend that supports editing via `hermes tools` → "
                f"Image Generation.", "modality_unsupported")
        logger.warning("Image gen provider '%s' raised%s: %s", pname,
                       " TypeError" if is_type_error else "", exc)
        return _provider_error(f"Provider '{pname}' error: {exc}", "provider_exception")
    return _provider_result(result, "Provider returned a non-dict result")


# Native ``krea-2-*`` ids are served by the Krea managed gateway (managed mode only —
# direct/BYO users keep their pipeline); ``fal-ai/krea/v2/*`` catalog ids stay on FAL.
_KREA_NATIVE_MODELS = {"krea-2-medium", "krea-2-large", "krea-2-medium-turbo"}


def _normalize_krea_model(model_id: Optional[str]) -> Optional[str]:
    """Return the native Krea plugin model id when ``model_id`` is ``krea-2-*``."""
    candidate = model_id.strip() if isinstance(model_id, str) else None
    return candidate if candidate in _KREA_NATIVE_MODELS else None


def _maybe_route_managed_krea(
    prompt: str, aspect_ratio: str, image_url: Optional[str] = None,
    reference_image_urls: Optional[list] = None, upscale: Optional[bool] = None) -> Optional[str]:
    """JSON result from the managed Krea gateway, or ``None`` to fall through.

    Fires only for a native ``krea-2-*`` model with no ``image_gen.provider`` other than
    ``"nous"`` stored (a picker choice dispatches normally) and a resolvable Krea gateway.
    """
    configured_provider = _read_configured_image_provider()
    if configured_provider is not None and configured_provider != NOUS_MANAGED_PROVIDER:
        return None
    normalized = _normalize_krea_model(_read_configured_image_model())
    if normalized is None:
        return None
    try:
        from plugins.image_gen.krea import _resolve_managed_krea_gateway
        if _resolve_managed_krea_gateway() is None:
            return None
        provider = _get_plugin_provider("krea")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Managed Krea routing unavailable: %s", exc)
        return None
    if provider is None:
        return None
    kwargs: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio, "model": normalized}
    try:
        _add_provider_kwargs(kwargs, image_url, reference_image_urls, upscale)
        result = provider.generate(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Managed Krea routing failed: %s", exc)
        return _provider_error(f"Managed Krea generation error: {exc}", "provider_exception")
    return _provider_result(result, "Krea provider returned a non-dict result")


def _confine_source_images(image_url, reference_image_urls, task_id, *, permitted: tuple = ("image",)):
    """Resolve path-like sources to ``data:`` URLs under a non-local terminal backend.

    Routes through ``tools.image_source`` (in-sandbox exec-read, media-cache host reads,
    credential guard) so generation obeys the same confinement as vision. URLs/data: pass
    through; local backend is a no-op. Returns ``(image_url, reference_image_urls, error_json_or_None)``.
    """
    if (os.getenv("TERMINAL_ENV") or "local").strip().lower() in ("", "local"):
        return image_url, reference_image_urls, None
    from model_tools import _run_async
    from tools.image_source import ImageResolutionError, resolve_local_source_to_data_url

    def resolve(ref):
        return _run_async(resolve_local_source_to_data_url(ref, task_id, permitted=permitted))
    try:
        if isinstance(image_url, str) and image_url.strip():
            image_url = resolve(image_url)
        if isinstance(reference_image_urls, (list, tuple)):
            reference_image_urls = [resolve(r) if isinstance(r, str) else r for r in reference_image_urls]
    except ImageResolutionError as exc:
        return image_url, reference_image_urls, _provider_error(
            f"Could not read source image: {exc}", type(exc).__name__)
    return image_url, reference_image_urls, None


def _handle_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for image generation")
    aspect_ratio = args.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    upscale = args.get("upscale")
    task_id = kw.get("task_id")
    # Confinement chokepoint BEFORE any dispatch: every route receives sandbox-confined bytes.
    image_url, reference_image_urls, confine_error = _confine_source_images(
        args.get("image_url"), args.get("reference_image_urls"), task_id)
    if confine_error is not None:
        return confine_error
    # Order matters: explicit plugin provider (incl. "krea"), then model-driven managed Krea
    # interception (only when no provider is set, so BYO/direct FAL stays untouched), then FAL.
    sources = dict(image_url=image_url, reference_image_urls=reference_image_urls,
                   upscale=upscale if isinstance(upscale, bool) else None)
    raw = None
    for route in (_dispatch_to_plugin_provider, _maybe_route_managed_krea, image_generate_tool):
        raw = route(prompt, aspect_ratio, **sources)
        if raw is not None:
            break
    return _postprocess_image_generate_result(raw, task_id=task_id)


# --- Dynamic schema — reflect the active backend's image-to-image capability ---
# Advertising edit capability up front saves a wasted turn. Memoized by config.yaml mtime in
# model_tools.get_tool_definitions(), so it rebuilds on switch.
_NO_CAPABILITIES = {"modalities": ["text"], "max_reference_images": 0, "supports_upscale": False}


def _active_image_capabilities() -> Dict[str, Any]:
    """Best-effort capabilities of the active backend/model; never raises.

    Mirrors runtime dispatch: a set ``image_gen.provider`` asks that plugin, else the FAL
    catalog. Fail-closed: an undeclared capability is advertised as absent.
    """
    info: Dict[str, Any] = dict(_NO_CAPABILITIES)
    configured_provider = _read_configured_image_provider()
    if configured_provider and configured_provider != "fal":
        try:
            provider = _get_plugin_provider(configured_provider)
            if provider is not None:
                try:
                    caps = provider.capabilities() or {}
                except Exception:  # noqa: BLE001
                    caps = {}
                info["provider"] = provider.display_name
                info["model"] = _read_configured_image_model() or (provider.default_model() or "")
                if caps.get("modalities"):
                    info["modalities"] = list(caps["modalities"])
                if caps.get("max_reference_images"):
                    info["max_reference_images"] = int(caps["max_reference_images"])
                # Plugins opt in explicitly; absent = no upscale param.
                info["supports_upscale"] = bool(caps.get("supports_upscale"))
                return info
        except Exception:  # noqa: BLE001
            pass
    # In-tree FAL path (provider unset or == "fal"); _resolve_fal_model() never raises.
    model_id, meta = _resolve_fal_model()
    can_edit = bool(meta.get("edit_endpoint"))
    info["provider"] = "FAL.ai"
    info["model"] = meta.get("display", model_id)
    info["modalities"] = ["text", "image"] if can_edit else ["text"]
    info["max_reference_images"] = int(meta.get("max_reference_images") or 1) if can_edit else 0
    # Clarity is available on request for ANY catalog model (``upscale`` is only the default).
    info["supports_upscale"] = True
    return info


# Param snippets assembled per-capability by _build_dynamic_image_schema.
_IMAGE_URL_PARAM = {
    "type": "string",
    "description": (
        "Source image to edit/transform (image-to-image). A public URL or "
        "an absolute local file path from the conversation. Omit for "
        "text-to-image."
    ),
}

_UPSCALE_PARAM = {
    "type": "boolean",
    "description": (
        "Post-generation high-resolution pass (~2x, extra cost/latency), "
        "off by default. A creative enhancer that can alter fine detail "
        "(rendered text, faces) — use only when resolution matters more "
        "than fidelity."
    ),
}


def _build_dynamic_image_schema() -> Dict[str, Any]:
    """Render description AND params from the active model's capabilities; args it cannot
    honor are NOT advertised (the handler still accepts them for replay compat)."""
    base_desc = (
        "Generate high-quality images from text prompts{edit_clause}. "
        "Returns the result in the `image` field — a URL or an absolute "
        "file path; reference it in your response using the current "
        "platform's file-delivery convention."
    )
    info = _active_image_capabilities()
    max_refs = int(info.get("max_reference_images") or 0)
    can_edit = "image" in set(info.get("modalities") or ["text"])
    static_props = IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
    properties: Dict[str, Any] = {
        "prompt": static_props["prompt"], "aspect_ratio": static_props["aspect_ratio"]}
    if can_edit:
        edit_clause = ", or edit / transform an existing image by passing image_url"
        properties["image_url"] = _IMAGE_URL_PARAM
        if max_refs > 1:
            properties["reference_image_urls"] = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max_refs,
                "description": (
                    f"Up to {max_refs} additional reference images (style, "
                    "character, or composition) guiding an edit. URLs or "
                    "absolute local paths."
                ),
            }
    else:
        edit_clause = " (text-to-image only — the active model cannot edit existing images)"
    if info.get("supports_upscale"):
        properties["upscale"] = _UPSCALE_PARAM
    return {"description": base_desc.format(edit_clause=edit_clause),
            "parameters": {"type": "object", "properties": properties, "required": ["prompt"]}}


registry.register(
    name="image_generate", toolset="image_gen", schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate, check_fn=check_image_generation_requirements, requires_env=[],
    is_async=False,   # sync fal_client API to avoid "Event loop is closed" in gateway
    emoji="🎨", dynamic_schema_overrides=_build_dynamic_image_schema,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def is_krea_model(model_id: Optional[str]) -> bool:
    """True when ``model_id`` is a native Krea plugin id (``krea-2-*``)."""
    return _normalize_krea_model(model_id) is not None
# ---- END PLUGIN-COMPAT ----
