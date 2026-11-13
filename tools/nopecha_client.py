"""NopeCHA Recognition API client -- provider-specific HTTP layer only.

Talks to NopeCHA's Recognition API to turn a CAPTCHA image/audio into click
coordinates or text. Reference docs (verified live against the real API,
2026-11-12 -- not guessed):
    https://developers.nopecha.com/guides/recognition/
    https://nopecha.com/api-reference/

Deliberately does NOT call NopeCHA's Token API (sitekey + URL -> a solved
token ready to submit to a site's widget) -- that is a different product
tier, requires a paid key with no free option, and is exactly the
"automatic token injection" this integration must not do. This module only
ever returns a recognition RESULT (what to click / what text is shown); it
never submits anything back to a third-party site.

IMPLEMENTATION STATUS (Phase 2, 2026-11-13): only ``recaptcha`` has a
confirmed live end-to-end round trip (submit -> job id -> poll -> real
result, against NopeCHA's own demo image, Phase 1 pilot). The other six
Recognition-API types are fully spec'd in RECOGNITION_SPECS from the
official reference (field names verified by reading the live HTML
reference, not guessed) but still raise NotImplementedError pending a real
round-trip test each -- blocked as of this writing by the free-tier IP
ban described below hitting even the local dev machine (not just the VPS).
Flip ``implemented`` to True for a type only after it has been exercised
against a real NopeCHA job, matching how 'recaptcha' was validated.

Auth modes
----------
Free (default -- no key / NOPECHA_API_KEY unset or empty):
    No Authorization header sent. NopeCHA buckets usage by the caller's
    public IP. Confirmed live: 100 credits per ~23h rolling window (the
    ``duration`` field from GET /v1/status is 82800 seconds, NOT a fixed
    daily/midnight reset -- do not assume otherwise anywhere in this
    module). Free credits are rejected outright for IPs NopeCHA considers
    high-risk -- confirmed live from ndr-production (Hetzner
    91.99.219.247): HTTP 403 {"error":12,"message":"Banned IP"}.
    IMPORTANT (found 2026-11-13): this is not just a datacenter-IP problem.
    The SAME residential IP that completed a real recaptcha job cleanly on
    2026-11-12 came back "Banned IP" less than 24h later, after only a
    handful of direct API calls -- with no browser/extension involved.
    This strongly suggests NopeCHA's free-tier risk model flags the
    *calling pattern* (raw scripted API usage without a key), not just IP
    reputation. Treat free mode as unreliable for this integration's
    actual usage pattern in practice, not merely "avoid datacenter IPs" --
    a paid NOPECHA_API_KEY should be treated as the realistic default for
    production, with free mode as a best-effort fallback only.

Paid (NOPECHA_API_KEY set):
    Sent as ``Authorization: Bearer <key>``.

Error codes (app-level ``error`` field in the JSON body, confirmed against
the live API reference and against real 429/403 responses hit during
testing):
    10  invalid_request          HTTP 400
    17  client_outdated          HTTP 400 (reserved, unused by the provider)
    15  invalid_api_key          HTTP 401
    18  plan_upgrade_required    HTTP 402
    16  insufficient_credit      HTTP 403 (key has run out of credit)
    12  ip_ineligible            HTTP 403 (free-plan IP banned/high-risk)
    14  job_incomplete           HTTP 409 (poll again; provider recommends ~500ms)
    19  endpoint_discontinued    HTTP 410
    11  rate_limited             HTTP 429
    9   unknown_error            HTTP 500

Quota awareness (Phase 2)
--------------------------
``get_status_cached`` keeps a short-TTL, on-disk cache of the last known
``/v1/status`` result (keyed by a hash of whichever key/mode was used --
never the raw key), shared across every Hermes process on this host so
concurrent sessions don't each independently re-check quota or submit a
request that's already known to fail. ``record_call_result`` keeps a
best-effort on-disk call-stats counter (success/failure counts, last
request id, last error) for observability -- see
``tools/nopecha_recognition_tool.py`` for where it's called. Both are
best-effort: concurrent processes can race on the stats file (lost update,
not corruption -- writes are atomic), which is an accepted tradeoff for
advisory counters, not billing-critical state.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

NOPECHA_API_BASE = "https://api.nopecha.com"

# Conservative client-side cap. NopeCHA's reference doesn't publish a hard
# per-image limit; this just keeps a misbehaving caller (e.g. a full-page
# screenshot instead of a cropped challenge) from sending something absurd.
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB, pre-base64

# How long a cached /v1/status result is trusted before re-checking live.
# Short enough to stay accurate (credit resets over ~hours, not seconds),
# long enough to absorb a burst of calls across concurrent sessions.
QUOTA_CACHE_TTL_S = 30.0

# App-level error code -> (symbolic reason, human description).
_ERROR_REASONS: Dict[int, tuple] = {
    10: ("invalid_request", "Request is invalid -- check body/query parameters."),
    17: ("client_outdated", "Client must update to continue using the service."),
    15: ("invalid_api_key", "The provided API key is invalid."),
    18: ("plan_upgrade_required", "This feature is unavailable on the current plan."),
    16: ("insufficient_credit", "The API key has run out of credit."),
    12: ("ip_ineligible", "This IP address is ineligible for the Free plan (high-risk/VPN/proxy)."),
    14: ("job_incomplete", "Job is not finished yet -- retry shortly."),
    19: ("endpoint_discontinued", "This endpoint has been discontinued."),
    11: ("rate_limited", "Rate limit reached for this endpoint."),
    9: ("unknown_error", "An unknown error occurred on the provider's side."),
}

# Error reasons that are PERSISTENT -- retrying the real request won't help,
# so a fresh cached quota check showing one of these can fail fast instead
# of making a doomed submit call. Anything else (network_error, rate_limited,
# unknown_error, job_incomplete) is treated as possibly-transient and must
# not block the real attempt.
_PERSISTENT_ERROR_REASONS = {
    "ip_ineligible", "invalid_api_key", "insufficient_credit",
    "plan_upgrade_required", "endpoint_discontinued",
}

# Recognition-API type specs. `implemented=False` entries are intentionally
# unused -- see module docstring ("IMPLEMENTATION STATUS").
RECOGNITION_SPECS: Dict[str, Dict[str, Any]] = {
    "recaptcha": {"requires": ["task", "image_data"], "optional": ["grid"], "implemented": True},
    "funcaptcha": {"requires": ["task", "image_data"], "optional": [], "implemented": False},
    "funcaptcha_match": {"requires": ["task", "image_data"], "optional": [], "implemented": False},
    "awscaptcha": {"requires": ["audio_data"], "optional": [], "implemented": False},
    "geetest": {"requires": ["task", "image_data"], "optional": [], "implemented": False},
    "lemincaptcha": {"requires": ["task", "image_data"], "optional": [], "implemented": False},
    "textcaptcha": {"requires": ["image_data"], "optional": [], "implemented": False},
}

_DEFAULT_TIMEOUT = 15.0


class NopeCHAError(Exception):
    """Normalized NopeCHA API error."""

    def __init__(self, reason: str, message: str, code: Optional[int] = None, http_status: Optional[int] = None):
        self.reason = reason
        self.message = message
        self.code = code
        self.http_status = http_status
        super().__init__(f"[{reason}] {message} (app_code={code}, http={http_status})")


def _resolve_api_key(explicit_key: Optional[str] = None) -> str:
    """Resolve the API key to use. Empty string == free mode (not an error).

    Precedence: explicit argument > NOPECHA_API_KEY env/config.
    Mirrors ``tools/tool_backend_helpers.py::fal_key_is_configured``'s
    env-first-then-.env-file pattern so CLI setup and tool-side checks agree.
    """
    if explicit_key:
        return explicit_key.strip()
    value = os.getenv("NOPECHA_API_KEY")
    if value is None:
        try:
            from hermes_cli.config import get_env_value
            value = get_env_value("NOPECHA_API_KEY")
        except Exception:
            value = None
    return (value or "").strip()


def _key_fingerprint(key: str) -> str:
    """Never persist a raw key to disk -- cache files are keyed by this instead."""
    if not key:
        return "free"
    return "key:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _auth_headers(key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _raise_for_error(resp: requests.Response) -> None:
    """Parse a non-2xx NopeCHA response and raise a normalized NopeCHAError."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    code = body.get("error")
    message = body.get("message", resp.text[:200] if resp.text else "no body")
    reason, _desc = _ERROR_REASONS.get(code, ("unknown_error", f"Unrecognized app error code {code!r}"))
    raise NopeCHAError(reason=reason, message=message, code=code, http_status=resp.status_code)


def _request_with_retry(method: str, url: str, *, max_attempts: int = 2, **kwargs) -> requests.Response:
    """Bounded retry for TRANSIENT NETWORK failures only.

    Does not retry on any HTTP response the server actually returned
    (including 429/403/etc) -- those are rejections, not transport failures,
    and spec explicitly says not to retry rejected challenges.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.debug("nopecha_client: transient network error (attempt %d/%d): %s", attempt, max_attempts, exc)
                time.sleep(0.5 * attempt)
    raise NopeCHAError(reason="network_error", message=str(last_exc), code=None, http_status=None)


def get_status(key: Optional[str] = None, timeout: float = 10.0) -> Dict[str, Any]:
    """GET /v1/status -- quota/credit/reset-interval check. Does not consume
    a recognition credit; safe to call proactively before submitting a job.
    """
    resolved_key = _resolve_api_key(key)
    resp = _request_with_retry(
        "GET", f"{NOPECHA_API_BASE}/v1/status",
        headers=_auth_headers(resolved_key), timeout=timeout,
    )
    if not resp.ok:
        _raise_for_error(resp)
    return resp.json()


# ─── Shared on-disk cache (quota status + call stats) ──────────────────────
# Shared across every Hermes process on this host via a plain file under
# HERMES_HOME/cache/nopecha/ -- not a database, not a lock service. Good
# enough for "avoid an obviously-doomed duplicate request" and "rough
# success/failure counters", not meant for billing-accurate concurrency.

def _cache_dir() -> Path:
    from hermes_constants import get_hermes_home
    d = get_hermes_home() / "cache" / "nopecha"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _quota_cache_path() -> Path:
    return _cache_dir() / "quota_status.json"


def _stats_cache_path() -> Path:
    return _cache_dir() / "call_stats.json"


def get_status_cached(key: Optional[str] = None, *, force: bool = False) -> Dict[str, Any]:
    """``get_status`` with a short-TTL on-disk cache shared across processes.

    Returns the same shape as ``get_status`` (plus internal ``_cached_at``/
    ``_fingerprint`` bookkeeping keys, harmless for callers that just read
    ``credit``/``quota``/``duration``). Raises NopeCHAError exactly like
    ``get_status`` when there is no usable cache and the live call fails.
    """
    resolved_key = _resolve_api_key(key)
    fingerprint = _key_fingerprint(resolved_key)
    path = _quota_cache_path()

    if not force:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - cached.get("_cached_at", 0)
            if cached.get("_fingerprint") == fingerprint and 0 <= age < QUOTA_CACHE_TTL_S:
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    status = get_status(key=resolved_key)
    status["_cached_at"] = time.time()
    status["_fingerprint"] = fingerprint
    try:
        from utils import atomic_json_write
        atomic_json_write(path, status)
    except Exception as exc:
        logger.debug("nopecha_client: quota cache write failed (non-fatal): %s", exc)
    return status


def record_call_result(
    challenge_type: str, *, success: bool, request_id: Optional[str] = None,
    code: Optional[int] = None, reason: Optional[str] = None,
) -> None:
    """Best-effort on-disk success/failure counters, shared across processes.

    Never pass challenge image/audio content or the API key here -- this
    file is meant to be safe to read/share for observability.
    """
    path = _stats_cache_path()
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        stats = {"success_count": 0, "failure_count": 0, "by_type": {}}

    stats.setdefault("by_type", {})
    type_stats = stats["by_type"].setdefault(challenge_type, {"success_count": 0, "failure_count": 0})

    if success:
        stats["success_count"] = stats.get("success_count", 0) + 1
        type_stats["success_count"] += 1
    else:
        stats["failure_count"] = stats.get("failure_count", 0) + 1
        type_stats["failure_count"] += 1
        stats["last_error_code"] = code
        stats["last_error_reason"] = reason

    stats["last_request_id"] = request_id
    stats["updated_at"] = time.time()

    try:
        from utils import atomic_json_write
        atomic_json_write(path, stats)
    except Exception as exc:
        logger.debug("nopecha_client: stats write failed (non-fatal): %s", exc)


def get_call_stats() -> Dict[str, Any]:
    """Read back the shared call-stats counter. Empty defaults if never written."""
    try:
        return json.loads(_stats_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"success_count": 0, "failure_count": 0, "by_type": {}}


def _preflight_quota_check(key: str) -> None:
    """Fail fast on a known-persistent quota problem using a fresh cache.

    Only ever short-circuits on PERSISTENT error reasons (banned IP,
    invalid key, out of credit, ...) from a cache still inside its TTL, or
    on a cached credit count of exactly 0. Anything else -- stale cache,
    no cache, a transient error, or credit > 0 -- falls through silently
    and lets the real submit attempt proceed; this check must never mask
    a request that might actually succeed.
    """
    try:
        status = get_status_cached(key=key)
    except NopeCHAError as exc:
        if exc.reason in _PERSISTENT_ERROR_REASONS:
            raise
        return  # transient quota-check failure -- don't block the real attempt
    except Exception:
        return
    if status.get("credit") == 0:
        raise NopeCHAError(
            reason="insufficient_credit",
            message=(
                f"Cached quota check shows 0 credit remaining "
                f"(checked {time.time() - status.get('_cached_at', 0):.0f}s ago) -- "
                f"skipping request to avoid a guaranteed-failed call."
            ),
            code=16,
        )


def submit_recognition(
    challenge_type: str,
    *,
    task: Optional[str] = None,
    grid: Optional[str] = None,
    image_data: Optional[List[str]] = None,
    audio_data: Optional[List[str]] = None,
    key: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """Submit a recognition job. Returns the job ID (NopeCHA's ``data`` field).

    See module docstring "IMPLEMENTATION STATUS" for which challenge_type
    values are actually wired end-to-end.
    """
    spec = RECOGNITION_SPECS.get(challenge_type)
    if spec is None:
        raise NopeCHAError(
            reason="unsupported_challenge_type",
            message=f"'{challenge_type}' is not a NopeCHA Recognition API type. "
                    f"Supported types: {sorted(RECOGNITION_SPECS)}.",
        )
    if not spec["implemented"]:
        raise NotImplementedError(
            f"nopecha_client: challenge_type '{challenge_type}' is defined but not yet "
            f"validated end-to-end against a real job. See module docstring."
        )

    payload: Dict[str, Any] = {}
    if task is not None:
        payload["task"] = task
    if grid is not None:
        payload["grid"] = grid
    if image_data is not None:
        payload["image_data"] = image_data
    if audio_data is not None:
        payload["audio_data"] = audio_data

    missing = [f for f in spec["requires"] if f not in payload]
    if missing:
        raise NopeCHAError(
            reason="invalid_request",
            message=f"Missing required field(s) for '{challenge_type}': {missing}.",
        )

    resolved_key = _resolve_api_key(key)
    _preflight_quota_check(resolved_key)

    resp = _request_with_retry(
        "POST", f"{NOPECHA_API_BASE}/v1/recognition/{challenge_type}",
        json=payload, headers=_auth_headers(resolved_key), timeout=timeout,
    )
    if not resp.ok:
        _raise_for_error(resp)
    body = resp.json()
    job_id = body.get("data")
    if not job_id:
        raise NopeCHAError(reason="unknown_error", message=f"Submit succeeded but no job id in response: {body!r}")
    return job_id


def poll_recognition(
    challenge_type: str,
    job_id: str,
    *,
    key: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_wait: float = 20.0,
    poll_interval: float = 0.5,
) -> Any:
    """Poll GET /v1/recognition/{type}?id=... until the job completes.

    Retries on app error 14 (job_incomplete) -- the provider's documented
    "not ready yet" signal -- up to ``max_wait`` seconds, honoring their
    recommended ~500ms spacing. Any other error is raised immediately
    (not retried) -- a rejected/invalid job will never become valid by
    waiting.
    """
    resolved_key = _resolve_api_key(key)
    deadline = time.time() + max_wait
    while True:
        resp = _request_with_retry(
            "GET", f"{NOPECHA_API_BASE}/v1/recognition/{challenge_type}",
            params={"id": job_id}, headers=_auth_headers(resolved_key), timeout=timeout,
        )
        if resp.ok:
            return resp.json().get("data")
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if body.get("error") == 14 and time.time() < deadline:
            time.sleep(poll_interval)
            continue
        if body.get("error") == 14:
            raise NopeCHAError(
                reason="recognition_timeout",
                message=f"Job {job_id} still incomplete after {max_wait}s.",
                code=14, http_status=resp.status_code,
            )
        _raise_for_error(resp)
