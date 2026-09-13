"""browser_vision helpers: Lightpanda pre-route, native provider vision, auxiliary-LLM screenshot analysis.

Split out of ``tools/browser_tool.py``. Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_cli.config import cfg_get
from tools.browser_tool_origin import origin as _bt
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_lightpanda_fallback as _lp


def _vision_mode_label() -> str:
    _cp = _cloud._get_cloud_provider()
    return "local" if _cp is None else f"cloud ({_cp.display_name})"


def _lightpanda_vision_preroute(
    effective_task_id: str, annotate: bool, screenshot_path: Path,
) -> Tuple[bool, Optional[str], Path]:
    """Capture the vision screenshot via the Chrome fallback when Lightpanda is the engine
    (it has no graphical renderer). Returns ``(prerouted, fallback_warning, path)``;
    on fallback failure ``prerouted`` is False and the caller takes the normal
    screenshot path (forcing Chrome) so the standard fallback metadata still applies."""
    engine = _cloud._get_browser_engine()
    if engine != "lightpanda" or not _cloud._should_inject_engine(engine):
        return False, None, screenshot_path
    _bt.logger.debug("browser_vision: pre-routing screenshot to Chrome (engine=lightpanda)")
    screenshot_args = ["--annotate"] if annotate else []
    fb_result = _lp._chrome_fallback_screenshot(effective_task_id, screenshot_args, _bt._get_command_timeout())
    fb_result = _lp._annotate_lightpanda_fallback(fb_result, _bt._LP_VISION_FALLBACK_REASON)
    if not fb_result.get("success"):
        _bt.logger.warning("Lightpanda Chrome fallback vision screenshot failed: %s", fb_result.get("error"))
        return False, None, screenshot_path
    fb_path = fb_result.get("data", {}).get("path", "")
    if fb_path and os.path.exists(fb_path):
        import uuid as uuid_mod
        from hermes_constants import get_hermes_dir

        screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        persistent_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
        shutil.copy2(fb_path, persistent_path)
        screenshot_path = persistent_path
    return True, fb_result.get("fallback_warning"), screenshot_path


def _native_vision_result(
    screenshot_path: Path, question: str, annotate: bool,
    result: Dict[str, Any], lp_fallback_warning: Optional[str],
) -> Dict[str, Any]:
    """Multimodal tool-result envelope: the main model inspects the pixels itself.

    The embed is baked into history and re-sent every later turn, so apply the same
    proactive resize as vision_analyze's native path (skipped when already under
    both caps; without Pillow it fails open to the raw bytes).
    """
    from tools.vision_tools import (
        _EMBED_MAX_DIMENSION,
        _EMBED_TARGET_BYTES,
        _build_native_vision_tool_result,
        _resize_image_for_vision,
    )

    data_url = _resize_image_for_vision(screenshot_path, mime_type="image/png", max_base64_bytes=_EMBED_TARGET_BYTES,
                                        max_dimension=_EMBED_MAX_DIMENSION, force_jpeg=True)
    native_result = _build_native_vision_tool_result(image_url=str(screenshot_path), question=question,
                                                     image_data_url=data_url,
                                                     image_size_bytes=screenshot_path.stat().st_size)
    meta = native_result.setdefault("meta", {})
    meta["screenshot_path"] = str(screenshot_path)
    if lp_fallback_warning:
        meta["fallback_warning"] = lp_fallback_warning
    if annotate and result.get("data", {}).get("annotations"):
        meta["annotations"] = result["data"]["annotations"]
    native_result["text_summary"] = f"{native_result.get('text_summary', '')} Screenshot path: {screenshot_path}".strip()
    return native_result


def _analyze_screenshot_with_aux_llm(screenshot_path: Path, question: str) -> str:
    """One-shot aux vision-LLM analysis (not baked into history), secret-redacted.

    Full resolution first; on a size-related provider rejection the image is
    downscaled once and retried. ``auxiliary.vision.timeout/temperature`` — local
    vision models can take well over 30s, so the default timeout is generous.
    """
    import base64

    vision_prompt = (
        f"You are analyzing a screenshot of a web browser.\n\n"
        f"User's question: {question}\n\n"
        f"Provide a detailed and helpful answer based on what you see in the screenshot. "
        f"If there are interactive elements, describe them. If there are verification challenges "
        f"or CAPTCHAs, describe what type they are and what action might be needed. "
        f"Focus on answering the user's specific question."
    )
    _screenshot_bytes = screenshot_path.read_bytes()
    _screenshot_b64 = base64.b64encode(_screenshot_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{_screenshot_b64}"
    vision_model = _bt._get_vision_model()
    _bt.logger.debug("browser_vision: analysing screenshot (%d bytes)", len(_screenshot_bytes))

    vision_timeout = 120.0
    vision_temperature = 0.1
    try:
        from hermes_cli.config import load_config
        _vision_cfg = cfg_get(load_config(), "auxiliary", "vision", default={})
        if _vision_cfg.get("timeout") is not None:
            vision_timeout = float(_vision_cfg["timeout"])
        if _vision_cfg.get("temperature") is not None:
            vision_temperature = float(_vision_cfg["temperature"])
    except Exception:
        pass

    from agent.auxiliary_client import call_llm  # lazy: heavy client, only needed on the vision path

    call_kwargs = {
        "task": "vision", "temperature": vision_temperature, "timeout": vision_timeout,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    }
    if vision_model:
        call_kwargs["model"] = vision_model
    try:
        response = call_llm(**call_kwargs)
    except Exception as _api_err:
        from tools.vision_tools import _is_image_size_error, _resize_image_for_vision, _RESIZE_TARGET_BYTES
        if not (_is_image_size_error(_api_err) and len(data_url) > _RESIZE_TARGET_BYTES):
            raise
        _bt.logger.info("Vision API rejected screenshot (%.1f MB); auto-resizing to ~%.0f MB and retrying...",
                        len(data_url) / (1024 * 1024), _RESIZE_TARGET_BYTES / (1024 * 1024))
        data_url = _resize_image_for_vision(screenshot_path, mime_type="image/png")
        call_kwargs["messages"][0]["content"][1]["image_url"]["url"] = data_url
        response = call_llm(**call_kwargs)

    from agent.redact import redact_sensitive_text  # the LLM may have read secrets off the screenshot
    return redact_sensitive_text((response.choices[0].message.content or "").strip())
