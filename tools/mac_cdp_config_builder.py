#!/usr/bin/env python3
"""Build guarded Mac CDP sidecar configs for Kagura browser automation.

This generator keeps production/browser form work on the non-intrusive path:
read-only DOM inventory first, then guarded fill config with allowSubmit=false.
It writes configs under the Mac shared-worker root by default so they can be
executed by mac_run_shared_python without foreground GUI focus takeover.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

DEFAULT_SHARED_ROOT = Path("/Users/tora/.hermes/profiles/kagura/local-worker/shared")
DEFAULT_PROFILE = DEFAULT_SHARED_ROOT / "kagura-chrome-cdp-profile"
DEFAULT_PORT = 9223
SECRET_PATTERNS = [
    r"password",
    r"passwd",
    r"token\s*=",
    r"secret\s*=",
    r"apikey",
    r"api_key",
    r"authorization\s*:",
    r"bearer\s+",
    r"sk-[A-Za-z0-9_-]{8,}",
    r"ghp_[A-Za-z0-9_]{8,}",
    r"xoxb-[A-Za-z0-9-]{8,}",
    r"\botp\b",
    r"\btotp\b",
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{80,}(?![A-Za-z0-9_-])",
]
FIELD_KINDS = {"value", "textContent", "innerText"}
MODES = {"inventory", "fill"}
FIXED_VALIDATION_EXPRESSION = "({ok:true})"


def _approval_token(session_id: str) -> str:
    return f"APPROVED:{session_id}"


def _side_effect_policy(mode: str) -> dict[str, Any]:
    """Return the approval policy agents should surface before execution."""
    if mode == "inventory":
        return {
            "approvalRequiredBeforeRun": False,
            "operationClass": "read-only inventory",
            "knownSideEffects": [
                "loads the target URL in a dedicated CDP browser profile",
                "may create normal network access logs, analytics events, or cookies",
                "does not type, click, submit, save, or intentionally mutate fields",
            ],
        }
    return {
        "approvalRequiredBeforeRun": True,
        "operationClass": "guarded fill without submit",
        "knownSideEffects": [
            "loads the target URL in a dedicated CDP browser profile",
            "types supplied values into selected fields",
            "does not click submit/save buttons because allowSubmit is forced false",
            "the target site may still autosave or trigger oninput/onchange side effects",
            "runner marks the fill unsafe if post-fill network/storage/submit activity is detected",
        ],
        "approvalInstruction": (
            "Before running this fill config, explicitly tell the user the likely "
            "side effects and obtain approval for this specific URL and field set."
        ),
    }


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be absolute http/https")
    return url


def _host_allowed(host: str, allowed: list[str]) -> bool:
    normalized = host.lower().strip(".")
    for domain in allowed:
        d = str(domain).lower().strip().strip(".")
        if not d:
            continue
        if normalized == d:
            return True
        # Deliberately do NOT allow arbitrary subdomains for production form work.
        # If a subdomain is intended, include it explicitly in allowedDomains.
    return False


def _append_session_id(url: str, session_id: str | None) -> str:
    if not session_id:
        return url
    parsed = urlparse(url)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    pairs = [(k, v) for k, v in pairs if k != "sessionId"]
    pairs.append(("sessionId", session_id))
    return urlunparse(parsed._replace(query=urlencode(pairs)))


def _looks_like_secret(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return any(re.search(pat, text, re.IGNORECASE) for pat in SECRET_PATTERNS)


def _sanitize_prefix(raw: str | None, mode: str) -> str:
    base = raw or ("cdp-readonly-inventory" if mode == "inventory" else "cdp-form-fill")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-.")
    return cleaned or ("cdp-readonly-inventory" if mode == "inventory" else "cdp-form-fill")


def _validate_shared_output(path: Path, shared_root: Path) -> str:
    resolved = path.expanduser().resolve()
    root = shared_root.expanduser().resolve()
    if not (resolved == root or root in resolved.parents):
        raise ValueError(f"output path outside shared root: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _validate_fields(fields: Any) -> list[dict[str, Any]]:
    if not isinstance(fields, list) or not fields:
        raise ValueError("fill mode requires at least one field")
    checked: list[dict[str, Any]] = []
    for idx, field in enumerate(fields):
        if not isinstance(field, dict):
            raise ValueError(f"field[{idx}] must be an object")
        selector = field.get("selector")
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError(f"field[{idx}] requires non-empty selector")
        kind = field.get("kind", "value")
        if kind not in FIELD_KINDS:
            raise ValueError(f"field[{idx}] unsupported kind: {kind}")
        if "value" not in field:
            raise ValueError(f"field[{idx}] requires value")
        if _looks_like_secret(field.get("value")):
            raise ValueError(f"field[{idx}] rejected as possible secret payload")
        checked.append({"selector": selector, "kind": kind, "value": str(field.get("value", ""))})
    return checked


def build_config(spec: dict[str, Any], shared_root: str | Path = DEFAULT_SHARED_ROOT) -> dict[str, Any]:
    """Return a guarded sidecar config from a compact browser-operation spec."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    mode = spec.get("mode", "inventory")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    url = _require_http_url(str(spec.get("url") or ""))
    parsed = urlparse(url)
    allowed = spec.get("allowedDomains") or []
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("allowedDomains is required and must list exact hostnames")
    if not _host_allowed(parsed.hostname or "", [str(x) for x in allowed]):
        raise ValueError(f"host {parsed.hostname!r} not in allowedDomains")
    if spec.get("allowSubmit") is True:
        raise ValueError("allowSubmit=true is not permitted by this guarded generator")
    session_id = str(spec.get("sessionId") or "").strip()
    if mode == "fill" and not session_id:
        raise ValueError("fill mode requires a non-empty sessionId from a prior inventory pass")
    if mode == "fill" and spec.get("validationExpression"):
        raise ValueError(
            "custom validationExpression is not permitted for guarded fill configs; "
            "arbitrary JavaScript could submit, save, or otherwise mutate the page"
        )

    root = _as_path(shared_root)
    prefix = _sanitize_prefix(spec.get("outputPrefix"), mode)
    url = _append_session_id(url, session_id or None)

    cfg: dict[str, Any] = {
        "url": url,
        "port": int(spec.get("port", DEFAULT_PORT)),
        "profile": str(spec.get("profile") or DEFAULT_PROFILE),
        "windowSize": str(spec.get("windowSize", "1280,900")),
        "pageReadyTimeoutSec": int(spec.get("pageReadyTimeoutSec", 12)),
        "allowSubmit": False,
        "sideEffectPolicy": _side_effect_policy(mode),
        "allowedDomains": [str(x).lower().strip().strip(".") for x in allowed if str(x).strip()],
    }

    if mode == "inventory":
        cfg.update(
            {
                "readOnly": True,
                "minFields": int(spec.get("minFields", 1)),
                "outputPath": _validate_shared_output(root / f"{prefix}-readonly-inventory-result.json", root),
                "screenshotPath": _validate_shared_output(root / f"{prefix}-readonly-inventory-screenshot.png", root),
            }
        )
    else:
        cfg.update(
            {
                "sessionId": session_id,
                "fields": _validate_fields(spec.get("fields")),
                "postFillWaitSec": float(spec.get("postFillWaitSec", 1.0)),
                "postFillNetworkDrainSec": float(spec.get("postFillNetworkDrainSec", 0.5)),
                "validationExpression": FIXED_VALIDATION_EXPRESSION,
                "outputPath": _validate_shared_output(root / f"{prefix}-form-fill-result.json", root),
                "screenshotPath": _validate_shared_output(root / f"{prefix}-form-fill-screenshot.png", root),
            }
        )
        approval_token = str(spec.get("approvalToken") or "").strip()
        if approval_token:
            if approval_token != _approval_token(session_id):
                raise ValueError("approvalToken must match APPROVED:<sessionId>")
            cfg["sideEffectApproval"] = {
                "approved": True,
                "token": approval_token,
                "scope": "url+fields+sessionId",
            }
    if _looks_like_secret({k: v for k, v in cfg.items() if k not in {"profile", "outputPath", "screenshotPath"}}):
        raise ValueError("config rejected as possible secret payload")
    return cfg


def write_config(spec: dict[str, Any], shared_root: str | Path = DEFAULT_SHARED_ROOT, out_path: str | Path | None = None) -> Path:
    cfg = build_config(spec, shared_root=shared_root)
    root = _as_path(shared_root)
    mode = spec.get("mode", "inventory")
    if out_path is None:
        name = "cdp-readonly-inventory-config.json" if mode == "inventory" else "cdp-form-fill-config.json"
        path = root / name
    else:
        path = _as_path(out_path)
    _validate_shared_output(path, root)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build guarded Mac CDP sidecar config JSON")
    parser.add_argument("spec", help="Path to compact operation spec JSON")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--out", help="Output config path under shared root")
    parser.add_argument("--print", action="store_true", help="Print generated config to stdout")
    args = parser.parse_args(argv)
    spec_path = Path(args.spec).expanduser()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out = write_config(spec, shared_root=args.shared_root, out_path=args.out)
    if args.print:
        print(out.read_text(encoding="utf-8"), end="")
    else:
        print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
