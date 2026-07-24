"""QQ空间 (QZone) 说说 publishing tool.

Registers one LLM-callable tool, ``qzone_publish``, which posts a 说说
(status update) — text, attached local images, and/or an AI-generated
image — to the configured QQ account's QQ空间. Image generation reuses
Hermes's existing ``image_generate`` backend (FAL / OpenAI / xAI); no new
model integration is introduced.

QQ has no official open API for publishing 说说 — the old QZone OpenAPI
``emotion`` interface was deprecated years ago. This tool therefore drives
the QZone *web* endpoints (``cgi_upload_image`` + ``emotion_cgi_publish_v6``)
directly, which need a logged-in QQ session (cookies) plus a ``g_tk`` CSRF
token.

Rather than running its own QR login, the tool *borrows* the login state
from a running OneBot implementation (NapCat / Lagrange.Core). OneBot v11
exposes ``get_login_info`` and ``get_cookies`` actions; NapCat/Lagrange
return real QZone cookies (including ``p_skey`` / ``skey``) for the QQ
account the bot is signed in as. The ``g_tk`` token is then computed
locally from ``p_skey`` with the standard QZone hash.

Configuration (environment variables):
- ``ONEBOT_HTTP_URL``    -- base URL of the OneBot HTTP API, e.g.
                            ``http://127.0.0.1:3000`` (required).
- ``ONEBOT_ACCESS_TOKEN``-- optional bearer token if the OneBot HTTP
                            server has ``access-token`` configured.

Caveats: this relies on reverse-engineered web endpoints. Cookies expire,
Tencent risk-control may require a captcha, and automated posting violates
Tencent's Terms of Service and carries an account-ban risk. The image
upload + ``richval`` wire format in particular is community-reverse-
engineered and may need adjustment if Tencent changes it. Failures are
surfaced verbatim to the model; the tool never silently retries.

Only included in the ``qzone`` toolset, so it has zero cost for users on
other platforms.
"""

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from tools.onebot_client import (
    onebot_base_url as _onebot_base_url,
    onebot_call as _onebot_call,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# QZone web endpoints. Fixed constants — never built from user input, so
# there is no SSRF surface here.
QZONE_PUBLISH_URL = (
    "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com"
    "/cgi-bin/emotion_cgi_publish_v6"
)
QZONE_UPLOAD_URL = "https://up.qzone.qq.com/cgi-bin/upload/cgi_upload_image"

# The QZone cookie domain to request from OneBot. NapCat/Lagrange return the
# *.qq.com cookie jar (uin / skey / p_skey / ...) for this domain.
_QZONE_COOKIE_DOMAIN = "user.qzone.qq.com"

# A desktop UA — QZone serves a different (mobile) flow to mobile UAs.
_DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Image attachment limits. QZone说说 accepts at most 9 images.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_MAX_IMAGES = 9
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB

# Default aspect ratio for AI-generated 说说 images. QZone shows feed
# images best near-square, so this differs from image_generate's own default.
_DEFAULT_GEN_ASPECT = "square"

_QZONE_TIMEOUT = 20
_QZONE_UPLOAD_TIMEOUT = 60


# ---------------------------------------------------------------------------
# OneBot credentials
#
# The OneBot v11 HTTP client (``_onebot_call`` / ``_onebot_base_url``) is
# imported from ``tools.onebot_client`` so it is shared with the qq_voice tool
# rather than duplicated. The helpers below borrow the QZone-specific login
# state (uin + cookies) from that connection.
# ---------------------------------------------------------------------------


def _get_login_uin() -> str:
    """Return the QQ number (uin) the OneBot instance is logged in as."""
    data = _onebot_call("get_login_info")
    uin = data.get("user_id")
    if not uin:
        raise RuntimeError("OneBot get_login_info returned no user_id.")
    return str(uin)


def _get_qzone_cookie_string() -> str:
    """Return the raw QZone cookie string from OneBot's get_cookies action."""
    data = _onebot_call("get_cookies", {"domain": _QZONE_COOKIE_DOMAIN})
    cookies = (data.get("cookies") or "").strip()
    if not cookies:
        raise RuntimeError(
            "OneBot get_cookies returned an empty cookie string — the QQ "
            "login state may be stale; re-login the NapCat/Lagrange client."
        )
    return cookies


# ---------------------------------------------------------------------------
# QZone primitives (pure, unit-tested)
# ---------------------------------------------------------------------------

def _compute_gtk(p_skey: str) -> int:
    """Compute the QZone ``g_tk`` CSRF token from the ``p_skey`` cookie.

    This is the long-standing QZone hash (DJB-style) over ``p_skey``.
    """
    h = 5381
    for ch in p_skey:
        h += (h << 5) + ord(ch)
    return h & 0x7FFFFFFF


def _extract_cookie_value(cookie_str: str, key: str) -> str | None:
    """Return the value of ``key`` from a ``k=v; k2=v2`` cookie string."""
    for part in cookie_str.split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name == key:
            return value
    return None


def _extract_pic_info(data: dict) -> dict:
    """Pull the fields needed for ``richval`` out of an upload response.

    QZone has shipped several response shapes over the years; values are
    fetched leniently with fallbacks so a missing optional field degrades
    gracefully rather than raising.
    """
    return {
        "albumid": data.get("albumid", ""),
        "lloc": data.get("lloc") or data.get("photoid", ""),
        "sloc": data.get("sloc") or data.get("photoid", ""),
        "type": data.get("type", 0),
        "width": data.get("width", 0),
        "height": data.get("height", 0),
        "url": data.get("url") or data.get("pre", ""),
    }


def _build_richval(pic_infos: list) -> str:
    """Build the ``richval`` string for an image 说说.

    Reverse-engineered wire format: one comma-delimited segment per image,
    segments joined by a TAB. If Tencent changes the format, this is the
    single place to fix.
    """
    segments = []
    for pic in pic_infos:
        segments.append(
            ",{albumid},{lloc},{sloc},{type},{height},{width},,{height},{width}".format(
                albumid=pic.get("albumid", ""),
                lloc=pic.get("lloc", ""),
                sloc=pic.get("sloc", ""),
                type=pic.get("type", 0),
                height=pic.get("height", 0),
                width=pic.get("width", 0),
            )
        )
    return "\t".join(segments)


def _build_publish_form(text: str, uin: str, pic_infos: list | None = None) -> dict:
    """Build the form body for the QZone emotion_publish endpoint."""
    form = {
        "syn_tweet_verson": "1",
        "paramstr": "1",
        "pic_template": "",
        "richtype": "",
        "richval": "",
        "special_url": "",
        "subrichtype": "",
        "who": "1",
        "con": text,
        "feedversion": "1",
        "ver": "1",
        "ugc_right": "1",
        "to_sign": "0",
        "hostuin": str(uin),
        "code_version": "1",
        "format": "json",
        "qzreferrer": f"https://user.qzone.qq.com/{uin}",
    }
    if pic_infos:
        form["richtype"] = "1"
        form["richval"] = _build_richval(pic_infos)
    return form


def _build_upload_form(
    image_b64: str, filename: str, uin: str, skey: str, p_skey: str, gtk: int
) -> dict:
    """Build the form body for the QZone cgi_upload_image endpoint."""
    return {
        "filename": filename,
        "uploadtype": "1",
        "albumtype": "7",
        "exttype": "0",
        "refer": "shuoshuo",
        "output_type": "json",
        "charset": "utf-8",
        "output_charset": "utf-8",
        "upload_hd": "1",
        "hd_width": "2048",
        "hd_height": "10000",
        "hd_quality": "96",
        "backUrls": (
            "http://upbak.photo.qzone.qq.com/cgi-bin/upload/cgi_upload_image,"
            "http://119.147.64.75/cgi-bin/upload/cgi_upload_image"
        ),
        "url": f"{QZONE_UPLOAD_URL}?g_tk={gtk}",
        "base64": "1",
        "zzpaneluin": str(uin),
        "p_uin": str(uin),
        "uin": str(uin),
        "skey": skey,
        "p_skey": p_skey,
        "qzonetoken": "",
        "picfile": image_b64,
    }


def _parse_publish_response(raw: bytes | str) -> dict:
    """Parse the QZone emotion_publish response.

    Returns ``{"ok": True, "tid": ...}`` on success or
    ``{"ok": False, "error": ..., "code": ...}`` otherwise. QZone wraps the
    JSON body in a ``_Callback(...)`` shim in some flows, so the JSON object
    is located by a permissive search rather than parsed from offset 0.

    ``emotion_cgi_publish_v6`` reports success two ways depending on the
    account / endpoint variant: the classic ``{"ret":0,"tid":...}`` shape and
    a newer ``{"code":0,"tid":...,"feedinfo":...}`` shape (verified live —
    NapCat/QZone returns ``code`` with no ``ret``). Either zero status, with a
    non-error ``subcode``, counts as success.
    """
    obj = _extract_json_object(raw)
    if obj is None:
        text = _as_text(raw)
        return {"ok": False, "error": f"unparseable QZone response: {text[:200]}"}

    ret = obj.get("ret")
    code = obj.get("code")
    subcode = obj.get("subcode", 0)
    # Newer QZone responses carry `code` instead of `ret`; fall back to it so
    # a successful post is never mis-reported as a failure.
    status = ret if ret is not None else code
    if status == 0 and subcode in (0, None):
        return {"ok": True, "tid": obj.get("tid") or obj.get("t1_tid"), "raw": obj}

    err = (obj.get("msg") or obj.get("message")
           or f"ret={ret}, code={code}, subcode={subcode}")
    return {"ok": False, "code": status, "error": err, "raw": obj}


def _parse_upload_response(raw: bytes | str) -> dict:
    """Parse the QZone cgi_upload_image response.

    Returns ``{"ok": True, "pic": {...}}`` on success or
    ``{"ok": False, "error": ...}`` otherwise. The response is wrapped in a
    ``frameElement.callback(...)`` shim.
    """
    obj = _extract_json_object(raw)
    if obj is None:
        text = _as_text(raw)
        return {"ok": False, "error": f"unparseable upload response: {text[:200]}"}

    ret = obj.get("ret")
    if ret != 0:
        err = obj.get("msg") or obj.get("message") or f"ret={ret}"
        return {"ok": False, "code": ret, "error": err}

    data = obj.get("data") or {}
    return {"ok": True, "pic": _extract_pic_info(data)}


def _as_text(raw: bytes | str) -> str:
    """Decode a response body to text."""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace").strip()
    return (raw or "").strip()


def _extract_json_object(raw: bytes | str) -> dict | None:
    """Locate and parse the first ``{...}`` JSON object in a response body.

    QZone wraps payloads in JSONP-style shims (``_Callback({...})``,
    ``frameElement.callback({...})``); this finds the JSON regardless.
    """
    text = _as_text(raw)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Local image files
# ---------------------------------------------------------------------------

def _read_image_file(path: str) -> tuple[bytes, str]:
    """Read a local image file, returning ``(bytes, basename)``.

    Raises ``ValueError`` with a human-readable reason for any problem so
    the handler can fail fast before touching the network.
    """
    resolved = os.path.expanduser(str(path))
    if not os.path.isfile(resolved):
        raise ValueError("file not found")
    ext = os.path.splitext(resolved)[1].lower()
    if ext not in _IMAGE_EXTS:
        raise ValueError(
            f"unsupported image type '{ext}' (allowed: {sorted(_IMAGE_EXTS)})"
        )
    size = os.path.getsize(resolved)
    if size == 0:
        raise ValueError("file is empty")
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"image too large ({size} bytes; max {_MAX_IMAGE_BYTES})"
        )
    with open(resolved, "rb") as fh:
        return fh.read(), os.path.basename(resolved)


# ---------------------------------------------------------------------------
# AI image generation
# ---------------------------------------------------------------------------

def _download_image(url: str) -> tuple[bytes, str]:
    """Download a generated image from a URL, returning ``(bytes, filename)``."""
    req = urllib.request.Request(url, headers={"User-Agent": _DESKTOP_UA})
    try:
        with urllib.request.urlopen(req, timeout=_QZONE_UPLOAD_TIMEOUT) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise RuntimeError(f"could not download generated image: {e}") from e

    if not data:
        raise RuntimeError("downloaded generated image is empty")
    if len(data) > _MAX_IMAGE_BYTES:
        raise RuntimeError(
            f"generated image too large ({len(data)} bytes; max {_MAX_IMAGE_BYTES})"
        )

    name = os.path.basename(urllib.parse.urlparse(url).path)
    if not name or os.path.splitext(name)[1].lower() not in _IMAGE_EXTS:
        name = "generated.png"
    return data, name


def _load_image_reference(ref: str) -> tuple[bytes, str]:
    """Resolve an image_generate result (URL or local path) to bytes."""
    ref = str(ref).strip()
    if ref.startswith(("http://", "https://")):
        return _download_image(ref)
    if os.path.isfile(os.path.expanduser(ref)):
        return _read_image_file(ref)
    raise RuntimeError(f"image_generate returned an unusable reference: {ref[:200]}")


def _generate_image(prompt: str, aspect_ratio: str) -> tuple[bytes, str]:
    """Generate an image via Hermes's configured image_generate backend.

    Reuses the in-tree image generation tool (FAL / OpenAI / xAI — whichever
    the user has configured), so no new model integration is introduced.
    Returns ``(bytes, filename)``; raises ``RuntimeError`` on any failure.
    """
    # Imported lazily: image_generation_tool pulls in heavier deps, and a
    # module-level import would couple qzone_tool's import to it.
    from tools.image_generation_tool import (  # noqa: PLC0415 — intentional lazy import
        _handle_image_generate,
        check_image_generation_requirements,
    )

    if not check_image_generation_requirements():
        raise RuntimeError(
            "no image-generation backend is configured — set one up via "
            "`hermes tools` → Image Generation (FAL / OpenAI / xAI)."
        )

    raw = _handle_image_generate({"prompt": prompt, "aspect_ratio": aspect_ratio})
    if isinstance(raw, str):
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"image_generate returned non-JSON: {raw[:200]}") from e
    elif isinstance(raw, dict):
        result = raw
    else:
        raise RuntimeError(
            f"image_generate returned an unexpected type: {type(raw).__name__}"
        )

    if result.get("error"):
        raise RuntimeError(result["error"])
    image_ref = result.get("image")
    if not image_ref:
        raise RuntimeError("image_generate produced no image.")
    return _load_image_reference(image_ref)


# ---------------------------------------------------------------------------
# QZone HTTP requests
# ---------------------------------------------------------------------------

def _qzone_post(url: str, form: dict, cookie: str, uin: str, timeout: int) -> bytes:
    """POST a form-urlencoded body to a QZone endpoint and return the body."""
    data = urllib.parse.urlencode(form).encode("utf-8")
    headers = {
        "Cookie": cookie,
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": f"https://user.qzone.qq.com/{uin}",
        "User-Agent": _DESKTOP_UA,
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001 — best-effort detail only
            pass
        raise RuntimeError(f"QZone HTTP {e.code}. {body}".strip()) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach QZone: {e.reason}") from e


def _qzone_publish_post(form: dict, gtk: int, cookie: str, uin: str) -> bytes:
    """POST a 说说 to QZone and return the raw response body."""
    url = f"{QZONE_PUBLISH_URL}?g_tk={gtk}"
    return _qzone_post(url, form, cookie, uin, _QZONE_TIMEOUT)


def _upload_image(
    image_bytes: bytes, filename: str, uin: str, skey: str, p_skey: str,
    gtk: int, cookie: str,
) -> dict:
    """Upload one image to QZone and return its parsed pic info.

    Raises ``RuntimeError`` if QZone rejects the upload.
    """
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    form = _build_upload_form(image_b64, filename, uin, skey, p_skey, gtk)
    url = f"{QZONE_UPLOAD_URL}?g_tk={gtk}"
    raw = _qzone_post(url, form, cookie, uin, _QZONE_UPLOAD_TIMEOUT)
    result = _parse_upload_response(raw)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "unknown upload error"))
    return result["pic"]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _handle_qzone_publish(args: dict, **kw) -> str:
    """Handler for the qzone_publish tool."""
    text = (args.get("text") or "").strip()
    images = args.get("images") or []
    if isinstance(images, str):  # tolerate a single path passed as a string
        images = [images]
    if not isinstance(images, list):
        return tool_error("qzone_publish 'images' must be a list of file paths.")

    generate = args.get("generate")
    if generate is not None and not isinstance(generate, str):
        return tool_error("qzone_publish 'generate' must be a text prompt string.")
    generate = (generate or "").strip()
    aspect_ratio = (args.get("aspect_ratio") or _DEFAULT_GEN_ASPECT).strip()

    if not text and not images and not generate:
        return tool_error("qzone_publish requires 'text', 'images', or 'generate'.")

    total_images = len(images) + (1 if generate else 0)
    if total_images > _MAX_IMAGES:
        return tool_error(
            f"QZone说说 supports at most {_MAX_IMAGES} images "
            f"(requested {total_images})."
        )

    # Read all local image files up front so a bad path fails before any
    # network call or image generation.
    image_payloads: list[tuple[bytes, str]] = []
    for path in images:
        try:
            image_payloads.append(_read_image_file(path))
        except ValueError as e:
            return tool_error(f"Image '{path}': {e}")

    # Generate an image from the prompt, if requested. Done before touching
    # QZone so a generation failure aborts cleanly.
    if generate:
        try:
            image_payloads.append(_generate_image(generate, aspect_ratio))
        except Exception as e:  # noqa: BLE001 — surface one clear message
            logger.error("qzone_publish: image generation failed: %s", e)
            return tool_error(f"Image generation failed: {e}")

    try:
        uin = _get_login_uin()
        cookie = _get_qzone_cookie_string()
    except Exception as e:  # noqa: BLE001 — surface one clear message to the model
        logger.error("qzone_publish: OneBot credential fetch failed: %s", e)
        return tool_error(f"Could not borrow QQ login state from OneBot: {e}")

    p_skey = _extract_cookie_value(cookie, "p_skey")
    if not p_skey:
        return tool_error(
            "p_skey not found in OneBot cookies — the QQ login state may be "
            "stale or the OneBot client is not fully logged in to QZone."
        )
    skey = _extract_cookie_value(cookie, "skey") or ""
    gtk = _compute_gtk(p_skey)

    # Upload images (if any), collecting pic info for the publish form.
    pic_infos: list[dict] = []
    for image_bytes, filename in image_payloads:
        try:
            pic_infos.append(
                _upload_image(image_bytes, filename, uin, skey, p_skey, gtk, cookie)
            )
        except Exception as e:  # noqa: BLE001 — surface one clear message
            logger.error("qzone_publish: image upload failed (%s): %s", filename, e)
            return tool_error(f"Image upload failed for '{filename}': {e}")

    form = _build_publish_form(text, uin, pic_infos)
    try:
        raw = _qzone_publish_post(form, gtk, cookie, uin)
    except Exception as e:  # noqa: BLE001 — surface one clear message to the model
        logger.error("qzone_publish: QZone request failed: %s", e)
        return tool_error(f"QZone publish request failed: {e}")

    result = _parse_publish_response(raw)
    if result.get("ok"):
        return json.dumps({
            "success": True,
            "tid": result.get("tid"),
            "uin": uin,
            "images": len(pic_infos),
            "generated": bool(generate),
            "message": "说说 published to QQ空间.",
        })
    return tool_error(
        f"QZone rejected the post: {result.get('error')}",
        code=result.get("code"),
    )


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_qzone_available() -> bool:
    """Tool is only exposed when an OneBot HTTP URL is configured."""
    return bool(_onebot_base_url())


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

QZONE_PUBLISH_SCHEMA = {
    "name": "qzone_publish",
    "description": (
        "Publish a 说说 (status update) to the configured QQ account's QQ空间 "
        "(QZone). Supports text, attached local images, and/or an AI-generated "
        "image from a prompt. The QQ login state is borrowed from a running "
        "OneBot (NapCat / Lagrange) instance — no QQ password is needed. Note: "
        "this drives unofficial QZone web endpoints, so it can fail if the "
        "login state is stale or Tencent risk-control intervenes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "The 说说 body text. May be empty when 'images' or "
                    "'generate' is provided; otherwise required."
                ),
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of local image file paths to attach "
                    f"(max {_MAX_IMAGES}). Allowed types: JPG, PNG, GIF, WebP, BMP."
                ),
            },
            "generate": {
                "type": "string",
                "description": (
                    "Optional text prompt — generates an image with the "
                    "configured image-generation backend (FAL / OpenAI / xAI) "
                    "and attaches it to the 说说. Counts toward the "
                    f"{_MAX_IMAGES}-image limit alongside 'images'."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["square", "landscape", "portrait"],
                "description": (
                    "Aspect ratio for the generated image. Only used with "
                    "'generate'. Default: square."
                ),
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="qzone_publish",
    toolset="qzone",
    schema=QZONE_PUBLISH_SCHEMA,
    handler=_handle_qzone_publish,
    check_fn=_check_qzone_available,
    requires_env=["ONEBOT_HTTP_URL"],
    emoji="🐧",
)
