"""Semantic Readiness evaluation for browser pages.

Replaces arbitrary ``sleep(5)`` loops with observable conditions:
selector readiness, text presence, minimum content length, and DOM settlement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import urlsplit
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def build_readiness_expression(
    *,
    selector: Optional[str] = None,
    all_text: Optional[List[str]] = None,
    any_text: Optional[List[str]] = None,
    min_text_length: Optional[int] = None,
) -> str:
    """Generate a lightweight, self-contained JavaScript readiness probe."""
    config = {
        "selector": selector or "",
        "all_text": all_text or [],
        "any_text": any_text or [],
        "min_text_length": min_text_length or 0,
    }
    cfg_json = json.dumps(config)
    return f"""
    (() => {{
        const cfg = {cfg_json};
        const res = {{
            ready: false,
            dom_ready: document.readyState === 'complete' || document.readyState === 'interactive',
            selector_found: true,
            all_text_found: true,
            any_text_found: true,
            text_length: 0,
            missing: []
        }};
        const bodyText = (document.body && document.body.innerText) ? document.body.innerText : '';
        res.text_length = bodyText.trim().length;

        if (cfg.selector) {{
            res.selector_found = document.querySelector(cfg.selector) !== null;
            if (!res.selector_found) res.missing.push('selector:' + cfg.selector);
        }}

        if (cfg.min_text_length > 0 && res.text_length < cfg.min_text_length) {{
            res.missing.push('min_text_length:' + cfg.min_text_length + ' (current:' + res.text_length + ')');
        }}

        if (cfg.all_text.length > 0) {{
            for (const t of cfg.all_text) {{
                if (!bodyText.includes(t)) {{
                    res.all_text_found = false;
                    res.missing.push('text:' + t);
                }}
            }}
        }}

        if (cfg.any_text.length > 0) {{
            const foundAny = cfg.any_text.some(t => bodyText.includes(t));
            res.any_text_found = foundAny;
            if (!foundAny) res.missing.push('any_text:' + cfg.any_text.join('|'));
        }}

        res.ready = res.dom_ready &&
                    res.selector_found &&
                    res.all_text_found &&
                    res.any_text_found &&
                    (cfg.min_text_length <= 0 || res.text_length >= cfg.min_text_length);

        return res;
    }})()
    """


def wait_for_condition(
    evaluate_fn: Callable[[str], Any],
    *,
    selector: Optional[str] = None,
    all_text: Optional[List[str]] = None,
    any_text: Optional[List[str]] = None,
    min_text_length: Optional[int] = None,
    timeout_seconds: float = 15.0,
    poll_interval: float = 0.3,
) -> Dict[str, Any]:
    """Poll the browser page using evaluate_fn until conditions are satisfied or timeout occurs."""
    expression = build_readiness_expression(
        selector=selector,
        all_text=all_text,
        any_text=any_text,
        min_text_length=min_text_length,
    )
    start_time = time.monotonic()
    last_res: Dict[str, Any] = {}

    while time.monotonic() - start_time < max(0.5, timeout_seconds):
        try:
            raw = evaluate_fn(expression)
            if isinstance(raw, str):
                try:
                    last_res = json.loads(raw)
                except Exception:
                    last_res = {"ready": False, "raw": raw}
            elif isinstance(raw, dict):
                last_res = raw
            else:
                last_res = {"ready": False}

            if last_res.get("ready"):
                last_res["elapsed_seconds"] = round(time.monotonic() - start_time, 3)
                return last_res
        except Exception as exc:
            last_res = {"ready": False, "error": str(exc)}

        time.sleep(poll_interval)

    last_res["timeout"] = True
    last_res["elapsed_seconds"] = round(time.monotonic() - start_time, 3)
    return last_res



@dataclass(slots=True)
class BrowserReadinessContract:
    path: str
    required_semantic_targets: list[str] = field(default_factory=list)
    require_auth: bool = False
    require_hydration: bool = False
    entity_identity: str | None = None
    timeout_seconds: float = 30

    def evaluate(self, observation: dict) -> dict:
        if not isinstance(observation, dict):
            return {"ready": False, "code": "not_ready", "diagnostic": browser_diagnostic({}, "not_ready")}
        if observation.get("captcha_required"):
            code = "captcha_required"
        elif observation.get("auth_required") or (self.require_auth and observation.get("authenticated") is not True):
            code = "auth_required"
        elif observation.get("surface") in {"canvas", "non_dom"}:
            code = "unsupported_surface"
        elif urlsplit(str(observation.get("url", ""))).path != self.path:
            code = "not_ready"
        elif self.require_hydration and observation.get("hydrated") is not True:
            code = "hydration_timeout"
        elif self.entity_identity is not None and observation.get("entity_identity") != self.entity_identity:
            code = "dom_drift"
        elif set(self.required_semantic_targets) - set(observation.get("semantic_targets", [])):
            code = "dom_drift"
        else:
            code = "ready"
        return {"ready": code == "ready", "code": code,
                "diagnostic": browser_diagnostic(observation, code) if code != "ready" else None}


def browser_diagnostic(observation: dict, failure_code: str) -> dict:
    classification = {"auth_required": "AUTH_REQUIRED", "dom_drift": "DOM_DRIFT",
                      "captcha_required": "AUTH_REQUIRED",
                      "hydration_timeout": "HYDRATION_PENDING", "unsupported_surface": "CANVAS_OR_NON_DOM_SURFACE"}
    return {"classification": classification.get(failure_code, "UNKNOWN_BROWSER_STATE"),
            "url": str(observation.get("url", ""))[:2048],
            "readyState": observation.get("readyState"),
            "title": str(observation.get("title", ""))[:200],
            "semantic_targets": list(observation.get("semantic_targets", []))[:50],
            "overlay": bool(observation.get("overlay")), "hydrated": observation.get("hydrated"),
            "recipe_fingerprint": observation.get("recipe_fingerprint")}
