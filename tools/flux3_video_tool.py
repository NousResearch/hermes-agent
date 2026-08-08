"""Native BFL FLUX 3 video generation tools, backed by the Nous tool gateway.

These are native tools in the ``image_generate`` mold: schemas and
descriptions are pinned here as build-time facts, the handlers speak the
gateway's own REST contract, and ``check_fn`` hides the whole toolset only when
there is no Nous sign-in to call it with — never on entitlement, which is the
gateway's to rule on. No runtime discovery, and no server-supplied schema is
ever consulted — that is the point of the design.

The wire is two calls against the gateway's managed mount, and it names the
vendor but not the vendor's API:

- ``POST {base}/generations`` with ``{mode, prompt, ...}`` -> ``{id, status,
  guidance}``
- ``GET {base}/generations/<id>`` -> the job state plus ``guidance``

``guidance`` is the gateway's live policy channel: exact waits, what to do
next, and how to deliver a finished clip ship from the server so they cannot
drift from what it actually enforces. Handlers surface it verbatim as the
tool's result text, and surface ``error.message`` the same way on a refusal.

Media inputs: handlers know their own media fields explicitly. A local file
path is resolved through :func:`tools.image_source.resolve_image_source`
(sandbox confinement, credential guard) and delivered via the Nous upload
protocol (presign, direct PUT to storage, ``nous-upload:<token>`` reference).
URLs pass through untouched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Optional

from tools.registry import registry
from tools.managed_tool_gateway import (
    build_managed_media_uploader,
    managed_gateway_auth_headers,
    managed_vendor_endpoints,
    peek_nous_access_token,
    read_nous_access_token,
)

logger = logging.getLogger(__name__)

_TOOLSET = "bfl"
_VENDOR = "bfl"

# Submit sits behind the gateway's upstream call plus upload-reference
# resolution, so it is given a generous read timeout. A poll passes its own,
# much shorter one (see _POLL_READ_TIMEOUT_SECONDS): the job endpoint answers
# at once, and a poll allowed to hang this long would spend the whole call.
_TRANSPORT_READ_TIMEOUT_SECONDS = 180.0
_TRANSPORT_CONNECT_TIMEOUT_SECONDS = 10.0

_SIGN_IN_MESSAGE = (
    "BFL video generation needs a Nous Portal sign-in. "
    "Ask the user to run `hermes model` and sign in to Nous, then retry."
)

# ---------------------------------------------------------------------------
# Local-path detection (moved from the retired MCP media-argument walker)
# ---------------------------------------------------------------------------

# Drive-absolute Windows paths: ``C:\Users\...`` / ``C:/Users/...``.
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")

# A whole string of base64 alphabet (optionally line-wrapped, optionally
# padded). Base64's alphabet includes "/", so an inline JPEG payload always
# starts with "/9j/" — which reads as an absolute path unless caught first.
_BASE64_PAYLOAD = re.compile(r"^[A-Za-z0-9+/\r\n]+={0,2}[\r\n]*$")
# Real filesystem paths are short; base64 of even a thumbnail runs to
# thousands of characters. Anything this long and alphabet-pure is a payload.
_MIN_BASE64_PAYLOAD_LENGTH = 256


def _looks_like_local_path(value: str) -> bool:
    """True for things we should read off disk rather than forward as-is.

    Deliberately narrow: only explicitly rooted paths and ``file://`` qualify.
    Bare names are ambiguous with opaque ids, URLs pass through, and base64
    payloads (checked first — a JPEG's base64 starts ``/9j/``) fall through
    untouched.
    """
    if len(value) >= _MIN_BASE64_PAYLOAD_LENGTH and _BASE64_PAYLOAD.match(value):
        return False
    if value.startswith("file://"):
        return True
    if value == "~" or value.startswith(("~/", "~\\")):
        return True
    if value.startswith(("/", "./", "../", ".\\", "..\\")):
        return True
    if _WINDOWS_DRIVE_PATH.match(value) or value.startswith("\\\\"):
        return True
    return False


def _display_path(path: str) -> str:
    """A value safe to embed in an error message shown to the model."""
    return path if len(path) <= 200 else f"{path[:200]}… ({len(path)} characters)"


# ---------------------------------------------------------------------------
# Gateway transport
# ---------------------------------------------------------------------------

def _endpoints() -> Optional[dict]:
    """Absolute URLs for the managed BFL routes, or None when unreachable."""
    return managed_vendor_endpoints(_VENDOR)


async def _call_gateway(
    method: str,
    url: str,
    json_body: Optional[dict] = None,
    read_timeout: Optional[float] = None,
) -> str:
    """One REST round trip, rendered as this tool's result.

    The gateway's ``guidance`` (on success) and ``error.message`` (on a
    refusal) are both written for a model to act on, so they are surfaced
    verbatim. A refusal is a normal outcome the model can respond to — being
    throttled is not a broken tool — so only genuinely unreadable responses
    become ``error``.

    Those unreadable ones carry ``transport_error`` as well. A refusal is the
    gateway's ruling on the request; a transport failure is the absence of one,
    and says nothing about the job. The poll loop tells them apart on that key
    rather than on the wording of a message.
    """
    import httpx

    headers = managed_gateway_auth_headers(url)
    if not headers:
        return json.dumps({"error": _SIGN_IN_MESSAGE})
    headers["Content-Type"] = "application/json"

    timeout = httpx.Timeout(
        _TRANSPORT_CONNECT_TIMEOUT_SECONDS,
        read=_TRANSPORT_READ_TIMEOUT_SECONDS if read_timeout is None else read_timeout,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, headers=headers, json=json_body)
    except Exception as exc:
        return json.dumps({
            "error": f"Could not reach the video-generation gateway: {type(exc).__name__}: {exc}",
            "transport_error": True,
        })

    if response.status_code == 401:
        return json.dumps({"error": _SIGN_IN_MESSAGE, "needs_reauth": True})

    try:
        payload = response.json()
    except Exception:
        payload = None

    if not isinstance(payload, dict):
        # An edge or a proxy answering in HTML rather than the gateway itself,
        # which is what a 502 or 504 in front of it looks like from here.
        return json.dumps({
            "error": f"The video-generation gateway answered HTTP {response.status_code} with an unreadable body.",
            "transport_error": True,
        })

    if response.status_code >= 400:
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        message = error.get("message") or f"the gateway refused the request (HTTP {response.status_code})"
        out: dict = {"error": str(message)}
        if isinstance(error.get("details"), dict):
            out["details"] = error["details"]
        return json.dumps(out, ensure_ascii=False)

    guidance = payload.pop("guidance", None)
    return json.dumps(
        {"result": guidance or "Request accepted.", "details": payload},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Media delivery
# ---------------------------------------------------------------------------

# Every media field the gateway understands, and the media kind each accepts.
# `input_image` and `input_images` are interchangeable server-side, so both
# must be sanitized whatever the mode.
_MEDIA_FIELDS = {
    "input_image": ("image",),
    "input_images": ("image",),
    "input_video": ("video",),
}
# The vendor takes at most ten keyframes. Checked here as well as server-side
# so an over-long list is refused before it spends the caller's upload quota.
_MAX_IMAGES = 10


# One get_result call looks repeatedly rather than a fixed twice. The job
# endpoint answers immediately — there is no long poll — so every second
# between looks is a second a finished clip goes unnoticed, and many
# short-spaced looks per call cut both that notice delay and the number of
# times the model has to decide to keep waiting.
#
# Two bounds keep the loop inside the agent's per-tool ceiling. model_tools'
# async bridge abandons a tool at 300s and reports it to the model as a bare
# "TimeoutError:" — no job id, no sign the generation is still alive — which is
# the worst answer this tool can give, so neither bound may approach it.
# _CALL_BACKSTOP_SECONDS is the wall-clock guarantee over the whole handler;
# _POLL_BUDGET_SECONDS stops new looks earlier still, and the difference
# between them is what a clip finishing on the last look has to download in.
_CALL_BACKSTOP_SECONDS = 240.0
_POLL_BUDGET_SECONDS = 180.0
# The gap between looks, and so the notice delay on a finished job. The
# gateway's poll limiter allows 120 a minute per principal, and it only ever
# has one generation of ours to answer for, so this cadence spends about a
# tenth of what it permits.
_POLL_GAP_SECONDS = 5.0
# The budget is counted as it is spent — the waits and the time each look
# actually takes — rather than read off a wall clock. A slow gateway therefore
# costs looks instead of overrunning the call, and the loop stays testable
# without a fake clock.
#
# Waits are taken in slices so they stay answerable. Nothing outside a tool can
# end a call that has already started — the executor only checks for an
# interrupt between tools — so a tool that blocks this long watches the flag
# itself.
_POLL_WAIT_SLICE_SECONDS = 1.0
# A poll's own read timeout. It has to clear the gateway's server-side poll
# budget, which bounds one status read at 45s across its retries and its
# regional redirect hops: cutting a poll off before the server would give up
# turns a slow-but-healthy read into a transport error, and an error ends the
# loop. Still far below the submit path's patience, so a wedged poll cannot
# quietly spend the whole call either.
_POLL_READ_TIMEOUT_SECONDS = 60.0
# How many looks in a row may fail to reach the gateway before the loop gives
# up on the call. A blip costs a look rather than the whole remaining budget:
# ending on the first one hands the model an error it can only answer by
# polling again immediately, which is the burst this loop exists to prevent.
# Bounded so a gateway that is genuinely down is reported promptly instead of
# being retried for minutes.
_MAX_CONSECUTIVE_TRANSPORT_ERRORS = 3

# Mirrors the gateway's BFL statuses
_TERMINAL_POLL_STATUSES = frozenset(
    {"Ready", "Error", "Request Moderated", "Content Moderated", "Task not found"}
)


def _poll_is_finished(raw: str) -> bool:
    """True when there is nothing to wait for: done, refused, or unreadable."""
    try:
        payload = json.loads(raw)
    except Exception:
        return True
    if not isinstance(payload, dict) or "error" in payload:
        # A refusal carries its own guidance (a limit, a dead job, a bad id),
        # and sleeping on it would only delay showing the model what to do.
        # The one exception is a throttle, which states a wait the loop can
        # absorb — the caller checks _retry_after_seconds before asking here.
        return True
    details = payload.get("details")
    status = details.get("status") if isinstance(details, dict) else None
    return not isinstance(status, str) or status in _TERMINAL_POLL_STATUSES


def _retry_after_seconds(raw: str) -> Optional[float]:
    """How long the gateway asked us to wait, when a refusal is a throttle.

    A throttle is the one refusal worth absorbing here rather than handing
    back. Returning it ends the call, and the model it lands on has no clock —
    told to wait, it asks again immediately — so a rate limit answered that way
    produces a tighter loop than the one that tripped it. The gateway sends the
    wait as a number alongside the message, so there is nothing to parse out of
    prose.
    """
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict) or "error" not in payload:
        return None
    details = payload.get("details")
    value = details.get("retryAfterSeconds") if isinstance(details, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _is_transport_error(raw: str) -> bool:
    """True when the gateway did not answer, as opposed to answering "no".

    Set by ``_call_gateway`` on the paths where nothing readable came back, so
    this reads a flag rather than matching on the text of a message.
    """
    try:
        payload = json.loads(raw)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("transport_error") is True


async def _wait_between_looks(seconds: float) -> bool:
    """Hold the call open until the next look; False if the user interrupted."""
    from tools.interrupt import is_interrupted

    remaining = seconds
    while remaining > 0:
        if is_interrupted():
            return False
        this_slice = min(_POLL_WAIT_SLICE_SECONDS, remaining)
        await asyncio.sleep(this_slice)
        remaining -= this_slice
    return True


def _warm_nous_token() -> None:
    """Refresh the Nous token once, before any parallel upload needs it.

    ``read_nous_access_token`` takes no lock and, when a refresh fails, falls
    back to returning the stale cached token. Uploading in parallel therefore
    had every request discover the token was expiring at the same instant and
    fire its own refresh; the rotating refresh token means the first wins and
    the rest quietly send the stale bearer, which the gateway answers with 401.
    One warm-up call first puts a fresh token in the cache, so the fan-out all
    reads it instead of racing for it.
    """
    try:
        read_nous_access_token()
    except Exception as exc:  # pragma: no cover — the real read retries below
        logger.debug("Nous token warm-up failed before parallel uploads: %s", exc)


async def _prepare_media(args: dict, task_id: Optional[str]) -> dict:
    """Replace local paths with upload references in every media field.

    Deliberately covers all media fields rather than the one this mode
    expects. The gateway accepts `input_image` and `input_images`
    interchangeably, so a value left unsanitized in the "wrong" field still
    reaches the vendor — as a raw local path, which both fails the generation
    and discloses the user's directory layout to a third party.
    """
    prepared = dict(args or {})
    _warm_nous_token()
    for field, permitted in _MEDIA_FIELDS.items():
        value = prepared.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            if len(value) > _MAX_IMAGES:
                raise ValueError(f"{field} takes at most {_MAX_IMAGES} images; you passed {len(value)}.")
            # Uploaded together rather than one after another: each entry is a
            # presign round trip plus a full-file PUT, and ten keyframes in
            # sequence took long enough that a submit looked hung.
            prepared[field] = list(
                await asyncio.gather(*(_deliver_media(entry, permitted, task_id) for entry in value))
            )
        else:
            prepared[field] = await _deliver_media(value, permitted, task_id)
    return prepared


def _without_media(args: dict) -> dict:
    """Drop media fields entirely — for the mode that takes none.

    The gateway ignores them for text-to-video, so stripping here matches its
    behaviour while avoiding an upload the caller never needed.
    """
    return {k: v for k, v in dict(args or {}).items() if k not in _MEDIA_FIELDS}


async def _deliver_media(value, permitted: tuple, task_id: Optional[str]):
    """Replace a local path with a ``nous-upload:`` reference; pass URLs through.

    Raises ``ValueError`` with a model-readable sentence when the file cannot
    be read or uploaded — the caller turns that into the tool's error payload.
    """
    if not isinstance(value, str) or not _looks_like_local_path(value):
        return value

    from tools.image_source import ImageResolutionError, ResolveContext, resolve_image_source

    try:
        resolved = await resolve_image_source(value, ResolveContext(task_id=task_id), permitted=permitted)
    except ImageResolutionError as exc:
        raise ValueError(f"Could not read {_display_path(value)}: {exc}") from exc

    endpoints = _endpoints()
    uploader = None
    if endpoints is not None:
        uploader = build_managed_media_uploader(endpoints["base_url"], endpoints["upload_path"])
    if uploader is None:
        raise ValueError("Local file uploads are not available in this build; pass a URL instead.")

    try:
        return await uploader(resolved.data, resolved.mime)
    except Exception as exc:
        raise ValueError(f"Could not upload {_display_path(value)}: {exc}") from exc


# ---------------------------------------------------------------------------
# Saving the finished clip
# ---------------------------------------------------------------------------

# Generous: a 50MB clip over a slow link. Bounded per call by what is left of
# the backstop, so this is a ceiling rather than the figure actually used.
_DOWNLOAD_READ_TIMEOUT_SECONDS = 300.0
_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 15.0
# Kept clear of the backstop so the download's own timeout fires first: that
# way a stalled save is reported as one, instead of being cancelled mid-write
# with the call's answer lost.
_DOWNLOAD_GRACE_SECONDS = 5.0
# A rejection page is a few hundred bytes of XML; a clip is megabytes. Anything
# smaller than this is not the video, whatever the HTTP status said.
_MIN_PLAUSIBLE_VIDEO_BYTES = 64 * 1024
# Enough collision suffixes to be useful, few enough to fail fast if something
# is generating files in a loop.
_MAX_FILENAME_ATTEMPTS = 50


def _download_read_timeout(started: float) -> float:
    """What is left of the call for a download, never more than the ceiling.

    Without this the download's own generous timeout outlives the agent's
    per-tool ceiling, and the "saving failed, poll again to retry" answer below
    is never reached: the bridge kills the call first and the model is told
    only "TimeoutError", with no indication the clip exists and is one poll
    away.

    Must not invent time past what remains: clamping upward used to schedule a
    download the outer ``asyncio.wait_for`` then cancelled, answering with a
    false ``_still_generating`` while the job was already Ready.
    """
    left = _CALL_BACKSTOP_SECONDS - (time.monotonic() - started) - _DOWNLOAD_GRACE_SECONDS
    return max(0.0, min(_DOWNLOAD_READ_TIMEOUT_SECONDS, left))


async def _save_if_ready(raw: str, save_to, started: float) -> str:
    """Download a finished clip and swap the signed URL for a local path.

    The URL is handled here rather than by the model on purpose. It is long and
    percent-encoded, and re-keying it into a shell command dropped characters
    often enough to be the main failure mode of this tool — a corrupted
    signature is rejected, but `curl` still writes the rejection body to the
    output file and exits 0, so it read as success. Passing the exact string
    from this response removes the transcription step entirely.

    It also keeps a bearer credential out of the transcript: the signed URL
    grants the clip to anyone holding it for the next 15-60 minutes, and
    returning it put it in the model's context and the saved conversation. The
    gateway already scrubs presigned URLs for *input* media for exactly this
    reason; this makes the output side match.
    """
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(payload, dict):
        return raw

    details = payload.get("details")
    if not isinstance(details, dict) or details.get("status") != "Ready":
        return raw

    result = details.get("result")
    url = result.get("sample") if isinstance(result, dict) else None
    if not isinstance(url, str) or not url.strip():
        return raw

    # Dropped whether or not the save succeeds. A retry re-polls, which mints a
    # fresh URL, so nothing is lost by not handing this one to the model.
    result.pop("sample", None)

    try:
        target, size = await _download_video(url.strip(), save_to, started)
    except Exception as exc:
        payload["result"] = (
            f"The clip finished but saving it failed: {type(exc).__name__}: {exc}. "
            "Poll this job again to retry the download; the job itself is unaffected."
        )
        return json.dumps(payload, ensure_ascii=False)

    details["saved_path"] = str(target)
    details["saved_bytes"] = size
    payload["result"] = _delivery_lead_in(target) + str(payload.get("result") or "")
    return json.dumps(payload, ensure_ascii=False)


def _delivery_lead_in(target) -> str:
    """Opens the result text, ahead of the gateway's own delivery guidance.

    On a messaging platform the tag is spelled out rather than described. The
    model has to reproduce this path exactly or the file is not sent, and the
    reply is published either way — the tag is stripped from the text whether
    or not it named a real file, so a wrong path reads to the user as a
    message that simply forgot the attachment. Handing over the finished line
    removes the step where that goes wrong; only this side knows the path.
    """
    if _delivers_as_an_attachment():
        return (
            f"Saved to {target}. To deliver it, copy the next line into your reply exactly as "
            f"written, alone on its own line, with nothing added around it:\n"
            f"MEDIA:{target}\n"
        )
    return f"Saved to {target}. "


async def _download_video(url: str, save_to, started: float) -> tuple:
    """Stream the clip to disk, returning (path, bytes).

    SSRF-guarded, for the same reason the upload PUT is: this URL comes from
    the vendor by way of the gateway, and it is fetched from the user's own
    machine. Real result URLs are public CDN objects, which the guard allows.
    """
    import httpx

    from tools.url_safety import create_ssrf_safe_async_client

    target = _resolve_destination(save_to, _filename_from_url(url))
    # Written under a .part name and renamed only once it is complete and
    # plausible, so a failed download can never leave something that looks like
    # a playable file behind.
    partial = target.with_name(target.name + ".part")
    timeout = httpx.Timeout(_DOWNLOAD_CONNECT_TIMEOUT_SECONDS, read=_download_read_timeout(started))

    try:
        async with create_ssrf_safe_async_client(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)

        size = partial.stat().st_size
        if size < _MIN_PLAUSIBLE_VIDEO_BYTES:
            raise ValueError(f"the download returned only {size} bytes, which is not a video")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return target, size


def _filename_from_url(url: str) -> str:
    from pathlib import PurePosixPath
    from urllib.parse import unquote, urlsplit

    name = PurePosixPath(unquote(urlsplit(url).path)).name
    # The path segment is vendor-controlled, so keep only a plain filename.
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")[:120]
    return name or "flux3-video.mp4"


def _delivers_as_an_attachment() -> bool:
    """True on a surface where the clip is received rather than opened off disk.

    Deferred to the shared classifier so this tool cannot drift from the rest
    of the codebase about what counts as a chat channel. It matters here that
    the API server and webhooks are *not* one: they carry a platform value but
    no attachment channel, and neither strips an unfulfilled MEDIA: tag out of
    the reply, so treating them as messaging puts the literal tag in front of
    the caller.
    """
    try:
        from gateway.session_context import session_is_messaging_surface

        return session_is_messaging_surface()
    except Exception:
        return False


def _default_directory():
    """Where a clip lands when the caller named no location.

    On a messaging platform the user has no filesystem — the only way they
    ever see the clip is as an attachment — so it goes to the gateway's own
    video cache, which is an unconditionally allowed delivery root. Downloads
    is not: an operator running HERMES_MEDIA_DELIVERY_STRICT=1 delivers only
    from the cache roots, so a clip saved to Downloads there is dropped on the
    way out and the user is shown a reply with nothing attached.
    """
    from pathlib import Path

    if _delivers_as_an_attachment():
        try:
            from hermes_constants import get_hermes_dir

            return get_hermes_dir("cache/videos", "video_cache")
        except Exception:
            logger.debug("Could not resolve the video cache dir; using Downloads", exc_info=True)
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.cwd()


def _resolve_destination(save_to, filename: str):
    """Where to write, honouring an explicit request and never overwriting."""
    from pathlib import Path

    if isinstance(save_to, str) and save_to.strip():
        requested = Path(save_to.strip()).expanduser()
        if requested.is_dir() or save_to.rstrip().endswith(("/", "\\")):
            directory, name = requested, filename
        else:
            directory, name = requested.parent, requested.name
    else:
        directory, name = _default_directory(), filename

    directory.mkdir(parents=True, exist_ok=True)
    return _free_path(directory / name)


def _free_path(candidate):
    """`name.mp4` -> `name-2.mp4` -> `name-3.mp4` … so nothing is clobbered."""
    if not candidate.exists():
        return candidate
    for suffix in range(2, _MAX_FILENAME_ATTEMPTS + 2):
        sibling = candidate.with_name(f"{candidate.stem}-{suffix}{candidate.suffix}")
        if not sibling.exists():
            return sibling
    raise ValueError(f"could not find a free filename next to {candidate}")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _submit_args(mode: str, args: dict) -> dict:
    """The wire body: the model's arguments, minus Nones, plus our mode.

    Arguments pass through as the model gave them; the gateway owns validation
    and the translation onto the vendor's own fields.
    """
    body = {k: v for k, v in dict(args or {}).items() if v is not None}
    body["mode"] = mode
    return body


async def _submit(mode: str, args: dict) -> str:
    endpoints = _endpoints()
    if endpoints is None:
        return _error("BFL video generation is not available in this build.")
    return await _call_gateway("POST", f"{endpoints['base_url']}/generations", _submit_args(mode, args))


async def _handle_text_to_video(args: dict, **kwargs) -> str:
    return await _submit("text_to_video", _without_media(args))


async def _handle_image_to_video(args: dict, **kwargs) -> str:
    try:
        prepared = await _prepare_media(args, kwargs.get("task_id"))
    except ValueError as exc:
        return _error(str(exc))
    return await _submit("image_to_video", prepared)


async def _handle_keyframes_to_video(args: dict, **kwargs) -> str:
    images = (args or {}).get("input_images")
    if not isinstance(images, list) or not images:
        return _error("input_images must be a non-empty list of 1-10 images (local paths or URLs).")
    try:
        prepared = await _prepare_media(args, kwargs.get("task_id"))
    except ValueError as exc:
        return _error(str(exc))
    return await _submit("keyframes_to_video", prepared)


async def _handle_video_continuation(args: dict, **kwargs) -> str:
    try:
        prepared = await _prepare_media(args, kwargs.get("task_id"))
    except ValueError as exc:
        return _error(str(exc))
    return await _submit("video_continuation", prepared)


def _still_generating(job_id: str) -> str:
    """The backstop's answer: an ordinary "call again", never a raised timeout."""
    return json.dumps(
        {
            "result": (
                "Still generating. This call reached its own time limit, which the job is "
                f"unaffected by — call bfl_flux3_get_result again with id={job_id} to keep "
                "waiting."
            ),
            "details": {"id": job_id, "status": "Generating"},
        },
        ensure_ascii=False,
    )


async def _poll_until_done(url: str, save_to, started: float) -> str:
    """Look until the job settles, the budget runs out, or the user stops.

    The waiting is absorbed here rather than asked of the model. A model has no
    clock: told to wait it emits "I'll check back in a minute" and its next
    action lands immediately, so guidance produced a burst of polls rather than
    a paced one. Waiting inside the call cannot be skipped, needs no shell, and
    works the same on every platform.
    """
    spent = 0.0
    unanswered = 0
    while True:
        look_started = time.monotonic()
        raw = await _call_gateway("GET", url, read_timeout=_POLL_READ_TIMEOUT_SECONDS)
        spent += time.monotonic() - look_started

        if _is_transport_error(raw):
            # The job is upstream and unaffected by our failure to ask about
            # it, so a blip costs this look and the loop tries again.
            unanswered += 1
            if unanswered >= _MAX_CONSECUTIVE_TRANSPORT_ERRORS:
                return raw
            gap = _POLL_GAP_SECONDS
        else:
            unanswered = 0
            throttled_for = _retry_after_seconds(raw)
            if throttled_for is None and _poll_is_finished(raw):
                return await _save_if_ready(raw, save_to, started)
            # Never faster than our own cadence, however short a wait the
            # gateway names: its number is a floor on politeness, not a licence
            # to hammer.
            gap = _POLL_GAP_SECONDS if throttled_for is None else max(throttled_for, _POLL_GAP_SECONDS)

        if gap <= 0 or spent + gap >= _POLL_BUDGET_SECONDS:
            # Out of budget. A still-generating status carries the gateway's own
            # "call again"; a throttle we could not outwait carries its wait.
            return raw
        if not await _wait_between_looks(gap):
            # Interrupted mid-wait: hand back the status we already have rather
            # than spending a round trip the user has just asked us to stop for.
            return raw
        spent += gap


async def _handle_get_result(args: dict, **kwargs) -> str:
    job_id = (args or {}).get("id")
    if not isinstance(job_id, str) or not job_id.strip():
        return _error("id is required: the job id returned by the generate tool.")
    endpoints = _endpoints()
    if endpoints is None:
        return _error("BFL video generation is not available in this build.")
    from urllib.parse import quote

    job_id = job_id.strip()
    url = f"{endpoints['base_url']}/generations/{quote(job_id, safe='')}"
    save_to = (args or {}).get("save_to")
    started = time.monotonic()

    # The loop stops itself once its budget is spent, but a look already in
    # flight still runs to completion, and a download follows it. This is the
    # wall-clock guarantee over all of that: whatever stalls inside, the model
    # is answered from here rather than by the async bridge, whose own timeout
    # arrives as a bare "TimeoutError:".
    try:
        return await asyncio.wait_for(
            _poll_until_done(url, save_to, started), timeout=_CALL_BACKSTOP_SECONDS
        )
    except asyncio.TimeoutError:
        return _still_generating(job_id)


async def _handle_prompting_guide(args: dict, **kwargs) -> str:
    return FLUX3_PROMPTING_GUIDE


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def _has_nous_credential() -> bool:
    """True when a Nous bearer is on hand, without spending a refresh to learn it.

    Two lookups, because the transport itself has two.
    ``peek_nous_access_token`` covers the env override and the active store's
    cached token. A profile that was never logged into separately has neither,
    and reads the credential from the global-root ``auth.json`` — the same
    fallback ``resolve_nous_access_token`` takes when the transport refreshes.
    Probing only the first would hide the tools from a profile whose calls
    would have gone through perfectly well.

    Neither lookup validates or refreshes the token: an expired credential is
    the gateway's 401 to report, and that answer already asks for a sign-in.
    """
    if peek_nous_access_token():
        return True
    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
    except Exception:
        return False
    token = state.get("access_token")
    return isinstance(token, str) and bool(token.strip())


def check_bfl_requirements() -> bool:
    """Visible to anyone signed in to Nous; the gateway rules on the rest.

    No entitlement check. What an account may generate — plan, credits, per
    account limits — is the gateway's decision, and it refuses with a reason
    written for the model to act on; deciding it a second time here can only
    disagree with the server and hide the tools from someone entitled to them.

    A sign-in is still required, because the gateway takes a Nous bearer and
    nothing else: with no credential every call could only ever answer "sign
    in", so the six schemas would be pure cost on every API call.

    Stays a pair of file reads — no portal probe, no OAuth refresh. Behind the
    registry's 30s cache this still runs on every CLI start, gateway session
    and cron tick.
    """
    try:
        if _endpoints() is None:
            return False
        return _has_nous_credential()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pinned schemas
# ---------------------------------------------------------------------------

_ASPECT_RATIOS = ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]
_RESOLUTIONS = ["720p"]

_GUIDE_POINTER = "Read bfl_flux3_prompting_guide before your first generation. "
_MEDIA_SENTENCE = (
    "Media fields accept a local file path (uploaded automatically to Nous-managed temporary "
    "storage and deleted when the generation finishes) or a URL. "
)
_OVERRIDE_SENTENCE = "All guidance is defaults: explicit user instructions override it."


def _shared_submit_properties() -> dict:
    return {
        "prompt": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Generation brief in plain prose (it is interpreted and expanded by a reasoning "
                "harness). Order it subject, distinguishing visual specifics, action, camera, "
                "lighting, environment, audio, style. Audio is generated by default — say "
                '"no music" if unwanted.'
            ),
        },
        "aspect_ratio": {
            "type": "string",
            "enum": list(_ASPECT_RATIOS),
            "default": "auto",
            "description": 'Output aspect ratio. "auto" lets the model choose.',
        },
        "duration": {
            "oneOf": [
                {"type": "integer", "minimum": 5, "maximum": 20},
                {"type": "string", "const": "auto"},
            ],
            "default": "auto",
            "description": 'Clip duration in whole seconds (5-20), or "auto".',
        },
        "resolution": {
            "type": "string",
            "enum": list(_RESOLUTIONS),
            "default": "720p",
            "description": "Output resolution bin.",
        },
        "generate_audio": {
            "type": "boolean",
            "default": True,
            "description": "Generate synchronized audio.",
        },
        "grounding": {
            "type": "boolean",
            "default": True,
            "description": "Allow a short research pass before generation.",
        },
        "seed": {
            "type": "integer",
            "minimum": 0,
            "maximum": 4294967295,
            "description": "Optional reproducibility seed.",
        },
        "version": {
            "type": "string",
            "description": 'Model version pin. Defaults to "latest".',
        },
    }


TEXT_TO_VIDEO_SCHEMA = {
    "name": "bfl_flux3_text_to_video",
    "description": (
        _GUIDE_POINTER
        + "FLUX 3 text-to-video: generates a clip (with audio) from the prompt alone. Nothing but "
        "the prompt anchors the subject here, so research anything with a real, checkable "
        "appearance before writing it — whatever you leave unspecified is filled in for you. "
        "Generation takes several minutes: this returns a job id immediately; poll "
        "bfl_flux3_get_result. "
        + _OVERRIDE_SENTENCE
    ),
    "parameters": {
        "type": "object",
        "properties": _shared_submit_properties(),
        "required": ["prompt"],
        "additionalProperties": False,
    },
}

IMAGE_TO_VIDEO_SCHEMA = {
    "name": "bfl_flux3_image_to_video",
    "description": (
        _GUIDE_POINTER
        + "FLUX 3 image-to-video: animates one image as the literal opening frame — those pixels "
        "are frame 0 and the clip moves from there. "
        + _MEDIA_SENTENCE
        + "Returns a job id; poll bfl_flux3_get_result. "
        + _OVERRIDE_SENTENCE
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_shared_submit_properties(),
            "input_image": {
                "type": "string",
                "minLength": 1,
                "description": "Exactly one opening-frame image: a local file path or a URL (PNG/JPEG/WebP, up to 10MB).",
            },
        },
        "required": ["prompt", "input_image"],
        "additionalProperties": False,
    },
}

KEYFRAMES_TO_VIDEO_SCHEMA = {
    "name": "bfl_flux3_keyframes_to_video",
    "description": (
        _GUIDE_POINTER
        + "FLUX 3 keyframe video: a storyboard of 1-10 images pinned at chosen frame positions "
        "(24fps). Name the subject and describe the motion that carries it between the pins, and "
        "keep the subject consistent across every pinned image. "
        + _MEDIA_SENTENCE
        + "Returns a job id; poll bfl_flux3_get_result. "
        + _OVERRIDE_SENTENCE
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_shared_submit_properties(),
            "input_images": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 10,
                "description": "1-10 keyframe images: local file paths or URLs (PNG/JPEG/WebP, up to 10MB each).",
            },
            "keyframe_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 480},
                "minItems": 1,
                "maxItems": 10,
                "description": (
                    "One unique non-negative frame index per image (24fps). Each must be at most "
                    'duration×24, so set an explicit duration rather than "auto" whenever you pin '
                    'indices — "auto" resolves to 5, 10, 15 or 20 seconds and an index past the '
                    "length it picks is rejected."
                ),
            },
        },
        "required": ["prompt", "input_images", "keyframe_indices"],
        "additionalProperties": False,
    },
}

VIDEO_CONTINUATION_SCHEMA = {
    "name": "bfl_flux3_video_continuation",
    "description": (
        _GUIDE_POINTER
        + "FLUX 3 video continuation: the new generation picks up from the input clip's final "
        'frames. Open the prompt with "Continue this video from its final frames:", re-establish '
        "the subject and the moment it ended on, then describe what happens next. input_video "
        "must be an mp4 of at most 50MB and 15 seconds, and the generated segment tops out at 15s "
        "too — chain a second continuation for a longer sequence. duration is the "
        "new segment only. "
        + _MEDIA_SENTENCE
        + "Returns a job id; poll bfl_flux3_get_result. "
        + _OVERRIDE_SENTENCE
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_shared_submit_properties(),
            "input_video": {
                "type": "string",
                "minLength": 1,
                "description": "The clip to continue: a local file path or a URL. mp4 only, at most 50MB and 15 seconds.",
            },
        },
        "required": ["prompt", "input_video"],
        "additionalProperties": False,
    },
}

GET_RESULT_SCHEMA = {
    "name": "bfl_flux3_get_result",
    "description": (
        "Poll a FLUX 3 video job by the job id a generate tool returned. Generation takes minutes "
        "and a long Generating phase is normal. This call waits for you while the job runs, so it "
        "may run for several minutes; if it returns still generating, just call it again. Do not "
        "sleep between calls. "
        "On Ready the clip is downloaded for you and the response gives its local path; your only "
        "remaining step is to deliver that file as the response describes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "minLength": 1,
                "description": "Job id from a previous bfl_flux3_* generate call.",
            },
            "save_to": {
                "type": "string",
                "description": (
                    "Where to save the finished clip: a directory or a full file path. Set this "
                    "only when the user asked for a particular location; the default is "
                    "~/Downloads. An existing file is never overwritten."
                ),
            },
        },
        "required": ["id"],
        "additionalProperties": False,
    },
}

PROMPTING_GUIDE_SCHEMA = {
    "name": "bfl_flux3_prompting_guide",
    "description": (
        "Read this before your first FLUX 3 generation. The prompting and grounding guide: how "
        "to research a subject so it renders as itself, how to assemble a prompt, which generate "
        "tool fits, and how to save and deliver the finished clip. Takes no arguments and spends "
        "no generation budget."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


# ---------------------------------------------------------------------------
# Pinned prompting guide (methodology only — policy numbers such as waits and
# limits arrive live in the gateway's tool responses, so they cannot drift)
# ---------------------------------------------------------------------------

FLUX3_PROMPTING_GUIDE = """# FLUX 3 video generation — how to get the best results

Everything here is guidance, not policy: the user's explicit instructions
always win. If they say skip the research, save somewhere specific, or deliver
differently — do it their way without arguing. Only the server's own validation
and limits are non-negotiable.

Start from the user's own wording. The prompt is rewritten by a reasoning
harness before anything is generated, so restyling it yourself just stacks a
second rewrite on top and gives their intent another chance to drift. There are
two reasons to add anything: grounding facts for a subject with a real,
checkable appearance, and the behaviours below, which they had no way to know
about. Absent those, send what they wrote.

## Grounding

Grounding is your job, and it is the highest-leverage step in the workflow.
What you specify is preserved; what you leave out is filled in for you. So
when a subject has a real, checkable appearance, research it first and put
what you found into the prompt — that is what makes the result yours rather
than an approximation of it.

Ground whenever someone who knows the subject could watch the clip and say
"that's not what that is": a named person or place, a landmark, a particular
vehicle or machine, anything technical, anything culturally specific, or a
period setting. Skip it for generics ("a dog on a beach") — there is no fact
to get wrong.

To ground: search for VISUAL references, where three good photographs beat any
amount of prose. If you can analyze images, analyze what you find. Then put
only what a camera could see into the prompt: silhouette and proportion,
materials and finish, specific colours, distinctive details, era-correct
context.

## How this model behaves

The prompt is read by a reasoning harness rather than a tag encoder, so keyword
tricks and word order do nothing. Write plain prose at whatever length the
brief deserves; a single line is a valid prompt.

Audio is generated by default, whether or not you mention it, so leaving it out
gets you invented sound rather than silence. Name the ambient sound, the music
and any speech separately — each lands as its own layer — and say "no music"
when you do not want it.

A quoted line becomes speech only if a speaker is visible on camera. Without
one it tends to render as burned-in text instead, so quote the line, describe
the speaker, and add "no on-screen text, no subtitles".

Multi-shot sequences work inside a single generation: "SHOT ONE ... HARD CUT.
SHOT TWO ..." produces real cuts, and "one continuous unbroken shot" gets an
uncut take. Consecutive shots have to contrast in scale, location or colour or
the cut will not read as one — near-identical coverage blends back into a
continuous take.

## Choosing a tool

- No input media -> bfl_flux3_text_to_video.
- Animate one image as the literal opening frame ->
  bfl_flux3_image_to_video (those pixels are frame 0, and the clip moves from
  there). Name the subject and its distinguishing specifics here as you would
  anywhere else: only frame 0 is pinned, every frame after it is generated,
  and what you leave out is the model's call rather than yours.
- Several images pinned at chosen moments -> bfl_flux3_keyframes_to_video
  (name the subject and describe the motion that carries it between the pins;
  keep it consistent across every pin, since a mismatch becomes a visible
  morph mid-clip). Pin exact indices and you want an explicit duration too —
  every index must fall within duration×24.
- Keep going from where a clip ends, or chain segments ->
  bfl_flux3_video_continuation. Open with "Continue this video from its final
  frames:", re-establish the subject and the moment the clip ended on, then
  describe what happens next; duration is the new segment only, so plan
  segments of 15 seconds or less to feed outputs back in.

## Media inputs

Pass local file paths directly — never hand-encode file contents into an argument.
URLs also work. Limits per file:
images 10MB (PNG/JPEG/WebP), video one mp4 of at most 50MB and 15 seconds.
Do not pre-shrink files to fit imagined caps; oversized pixel dimensions are
auto-downscaled and output tops out at 720p.

## Workflow

Submit returns a job id immediately — the video does not exist yet. Poll
bfl_flux3_get_result with that id; generation takes several minutes and a long
Generating phase is normal, not a stall. Nothing reaches disk before the job is
Ready, so checking folders mid-run tells you nothing.

The waiting is not yours to do. bfl_flux3_get_result takes the pause itself
while a job is still running, so one call can occupy several minutes and comes
back within seconds of the job finishing. If it returns still generating, just
call it again — no sleeping, no interval to judge, nothing to time.

A job survives client restarts: re-poll the same id rather than resubmitting,
which would only spend your budgets on duplicate work.

## Save and deliver

bfl_flux3_get_result saves the clip itself and returns saved_path. The download
is not yours to do and no URL is handed to you to fetch. Pass save_to only when
the user named a location; otherwise it lands in ~/Downloads, and an existing
file is never overwritten. If saving fails the response says so — poll the same
job again to retry, which is safe and spends no generation budget.

Then deliver that file so the clip plays inline. Which markup plays inline is the
host's decision, so check your system prompt or platform instructions and use
exactly the form they give; the common ones are a MEDIA: tag alone on its own
line and a markdown embed. Two things break it. Write the real absolute path,
with ~ expanded. And keep the markup plain: wrapping it in bold, backticks or a
code fence, or rewriting it as a [link](path), turns an inline player into
literal text or a click-target. That is the most common way this step fails,
and it reads as success because the filename is on screen. Where the
instructions call for no delivery markup, or the host has no such mechanism,
follow them and state the absolute path in plain text.

Report what you did rather than what the job says it did: the echoed prompt
field describes intent and often overstates what the render preserved.
"""


# ---------------------------------------------------------------------------
# Registration (auto-discovered: top-level registry.register calls)
# ---------------------------------------------------------------------------

registry.register(
    name="bfl_flux3_text_to_video",
    toolset=_TOOLSET,
    schema=TEXT_TO_VIDEO_SCHEMA,
    handler=_handle_text_to_video,
    check_fn=check_bfl_requirements,
    requires_env=[],
    is_async=True,
    emoji="🎬",
)

registry.register(
    name="bfl_flux3_image_to_video",
    toolset=_TOOLSET,
    schema=IMAGE_TO_VIDEO_SCHEMA,
    handler=_handle_image_to_video,
    check_fn=check_bfl_requirements,
    requires_env=[],
    is_async=True,
    emoji="🎬",
)

registry.register(
    name="bfl_flux3_keyframes_to_video",
    toolset=_TOOLSET,
    schema=KEYFRAMES_TO_VIDEO_SCHEMA,
    handler=_handle_keyframes_to_video,
    check_fn=check_bfl_requirements,
    requires_env=[],
    is_async=True,
    emoji="🎬",
)

registry.register(
    name="bfl_flux3_video_continuation",
    toolset=_TOOLSET,
    schema=VIDEO_CONTINUATION_SCHEMA,
    handler=_handle_video_continuation,
    check_fn=check_bfl_requirements,
    requires_env=[],
    is_async=True,
    emoji="🎬",
)

registry.register(
    name="bfl_flux3_get_result",
    toolset=_TOOLSET,
    schema=GET_RESULT_SCHEMA,
    handler=_handle_get_result,
    check_fn=check_bfl_requirements,
    requires_env=[],
    is_async=True,
    emoji="🎬",
)

registry.register(
    name="bfl_flux3_prompting_guide",
    toolset=_TOOLSET,
    schema=PROMPTING_GUIDE_SCHEMA,
    handler=_handle_prompting_guide,
    check_fn=check_bfl_requirements,
    requires_env=[],
    is_async=True,
    emoji="📖",
)
