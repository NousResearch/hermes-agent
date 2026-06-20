#!/usr/bin/env python3
"""Shared KIE (kie.ai) client for EasyHermes native media generation.

This is the single source of truth for talking to kie.ai's job API. It backs
the KIE image-gen and video-gen provider plugins (``plugins/image_gen/kie``,
``plugins/video_gen/kie``) and the website/slides/web-video tools.

It is a thin, synchronous port of the Kari/Langflow ``lfx.kari_media`` module,
with the same two modes:

- **relay** (EasyHermes desktop, logged in): ``KARI_HUB_URL`` + ``KARI_WORKSPACE_TOKEN``
  are set → upload / createTask / recordInfo go to the cloud hub
  ``{KARI_HUB_URL}/api/v1/kari/kie/*`` with ``X-Kari-Workspace-Token``. The client
  never holds ``KIE_API_KEY``; the hub injects the real key and bills authoritatively.
- **direct** (self-hosted single instance, no hub): call kie.ai directly with a
  locally-held ``KIE_API_KEY``.

Job flow (relay passes through the same shapes)
-----------------------------------------------
1. ``POST .../createTask`` with ``{"model", "input"}`` → ``{"code": 200, "data": {"taskId"}}``.
2. Poll ``GET .../recordInfo?taskId=...`` until ``state`` is ``success``/``succeeded``
   (returns ``resultJson.resultUrls``) or ``fail`` (raises with ``failMsg``).

Auth: relay → ``X-Kari-Workspace-Token``; direct → ``Authorization: Bearer ${KIE_API_KEY}``.

Env:
    KARI_HUB_URL + KARI_WORKSPACE_TOKEN  → relay through the cloud hub (no local key).
    KIE_API_KEY            direct mode only — the kie.ai key (held by the backend).
    KIE_JOBS_BASE_URL      default https://api.kie.ai (direct mode).
    KIE_UPLOAD_BASE  /  KIE_STREAM_UPLOAD_BASE
                           default https://kieai.redpandaai.co (direct mode).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KieError(RuntimeError):
    """Raised for any KIE configuration / request / task failure."""


class KieNotConfiguredError(KieError):
    """Raised when ``KIE_API_KEY`` is absent — distinguishes config from runtime errors."""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def kie_key() -> str:
    """Return ``KIE_API_KEY`` or raise :class:`KieNotConfiguredError`."""
    key = (os.getenv("KIE_API_KEY") or "").strip()
    if not key:
        raise KieNotConfiguredError(
            "KIE_API_KEY is not set. Set it in the backend environment to enable "
            "image / video generation."
        )
    return key


def kie_key_is_configured() -> bool:
    """Return True when media generation can run — via the cloud-hub relay
    (``KARI_HUB_URL`` + ``KARI_WORKSPACE_TOKEN``) or a locally-held ``KIE_API_KEY``."""
    return is_relay() or bool((os.getenv("KIE_API_KEY") or "").strip())


def _jobs_base() -> str:
    return (os.getenv("KIE_JOBS_BASE_URL") or "https://api.kie.ai").rstrip("/")


def _upload_base() -> str:
    return (
        os.getenv("KIE_UPLOAD_BASE")
        or os.getenv("KIE_STREAM_UPLOAD_BASE")
        or "https://kieai.redpandaai.co"
    ).rstrip("/")


def _hub_base() -> str:
    """Cloud-hub base URL for KIE relay; empty when running as a local key-holder."""
    return (os.getenv("KARI_HUB_URL") or "").rstrip("/")


def _workspace_token() -> str:
    return (os.getenv("KARI_WORKSPACE_TOKEN") or "").strip()


def is_relay() -> bool:
    """Relay through the cloud hub when both the hub URL and a workspace token are set.

    Mirrors ``lfx.kari_media`` workspace mode: the client never holds ``KIE_API_KEY`` —
    upload / createTask / recordInfo go to ``{hub}/api/v1/kari/kie/*`` with the workspace
    token, and the hub injects the real key and performs authoritative billing.
    """
    return bool(_hub_base() and _workspace_token())


def _hub_headers() -> Dict[str, str]:
    return {"X-Kari-Workspace-Token": _workspace_token(), "Accept": "application/json"}


def _load_httpx() -> Any:
    """Import httpx lazily so importing this module never adds CLI cold-start cost."""
    import httpx  # noqa: PLC0415 — intentional lazy import

    return httpx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def as_url_list(value: Any) -> List[str]:
    """Normalize a str / list-of-str into a clean list of non-empty URLs."""
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if item and str(item).strip()]


def upload(abs_path: str, *, timeout: float = 120.0) -> str:
    """Upload a local reference file and return its public URL.

    Relay mode → POST to the hub ``/api/v1/kari/kie/upload`` with the workspace token
    (hub returns ``{"url": ...}``). Direct mode → POST straight to kie.ai's upload API.
    """
    httpx = _load_httpx()
    fname = os.path.basename(abs_path)
    with open(abs_path, "rb") as fh:
        content = fh.read()
    if is_relay():
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{_hub_base()}/api/v1/kari/kie/upload",
                headers=_hub_headers(),
                files={"file": (fname, content)},
                data={"fileName": fname},
            )
        resp.raise_for_status()
        url = (resp.json() or {}).get("url")
        if not url:
            raise KieError(f"KIE upload (hub relay) returned no url: {resp.text[:200]}")
        return url
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{_upload_base()}/api/file-stream-upload",
            headers={"Authorization": f"Bearer {kie_key()}"},
            files={"file": (fname, content)},
            data={"uploadPath": "kari", "fileName": fname},
        )
    resp.raise_for_status()
    payload = resp.json()
    url = (payload.get("data") or {}).get("fileUrl") or payload.get("fileUrl")
    if not url:
        raise KieError(f"KIE upload returned no fileUrl: {payload}")
    return url


def run(
    model: str,
    input_obj: Dict[str, Any],
    *,
    max_wait: float = 300.0,
    interval: float = 3.0,
) -> List[str]:
    """Create a kie.ai job and poll to completion. Returns ``resultUrls``.

    Synchronous (blocks the calling tool for up to ``max_wait`` seconds) — this
    is the expected shape for a generation tool call.
    """
    httpx = _load_httpx()
    if is_relay():
        # 经云端中枢:不持 KIE_API_KEY,带容器 token;中枢注入密钥 + 权威扣费。
        base = f"{_hub_base()}/api/v1/kari/kie"
        create_url = f"{base}/createTask"
        record_url = f"{base}/recordInfo"
        post_headers = {**_hub_headers(), "Content-Type": "application/json"}
        get_headers = _hub_headers()
    else:
        base = _jobs_base()
        create_url = f"{base}/api/v1/jobs/createTask"
        record_url = f"{base}/api/v1/jobs/recordInfo"
        key = kie_key()
        post_headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        get_headers = {"Authorization": f"Bearer {key}"}

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(create_url, headers=post_headers, json={"model": model, "input": input_obj})
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 200:
            raise KieError(f"KIE createTask failed: {data.get('msg')}")
        task_id = (data.get("data") or {}).get("taskId")
        if not task_id:
            raise KieError("KIE createTask returned no taskId")

        logger.info("kie job created model=%s task=%s", model, task_id)
        waited = 0.0
        while waited < max_wait:
            time.sleep(interval)
            waited += interval
            poll = client.get(record_url, headers=get_headers, params={"taskId": task_id})
            poll.raise_for_status()
            record = (poll.json() or {}).get("data") or {}
            state = str(record.get("state", "")).strip().lower()
            if state in ("success", "succeeded"):
                result_json = record.get("resultJson") or "{}"
                try:
                    parsed = json.loads(result_json)
                except (ValueError, TypeError):
                    # 中枢/上游返回的 resultJson 不该崩成裸 JSONDecodeError —— 视为空结果。
                    parsed = {}
                urls = (parsed.get("resultUrls", []) if isinstance(parsed, dict) else []) or []
                logger.info("kie job %s done: %d url(s)", task_id, len(urls))
                return urls
            if state == "fail":
                raise KieError(record.get("failMsg") or "KIE task failed")
    raise KieError(f"KIE task timed out after {max_wait:.0f}s")


def resolve_ref(url: Optional[str], local_path: Optional[str]) -> str:
    """Return a public URL for a reference image: pass through a URL, or upload a local path."""
    clean = (url or "").strip()
    if clean:
        return clean
    if local_path:
        return upload(local_path)
    return ""
