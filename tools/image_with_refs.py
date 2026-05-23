"""AI image generation with character reference立绘.

Sends a text scene prompt + N character reference portraits to an
OpenAI-compatible Responses API endpoint (configured via the standard
``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` env vars), which delegates to
the ``gpt-image-2`` image-generation tool. The model uses the references
for visual anchoring so generated characters look like their canonical
立绘 rather than the model's invented furry/anthro defaults.

Reference立绘 live as PNG files under ``~/.hermes/characters/``, keyed by
short name (e.g. ``grantley.png``, ``algo.png``). Callers select 0-4
references by short name; the tool resolves to file paths, base64-encodes
them, and includes them as ``input_image`` content blocks alongside the
prompt.

Built for the 格兰每日说说 cron job: composes the daily QZone 说说's
companion image. Slow (~2-5 min per call). The streaming Responses API
sometimes drops mid-generation, so the tool wraps each call in a 3-attempt
retry.

The endpoint is whatever ``OPENAI_BASE_URL`` points at (defaults to
``api.openai.com``). Works against the real OpenAI API or any OpenAI-
compatible proxy that supports the Responses API with the ``image_generation``
tool — e.g. ``https://api.cornna.xyz/v1``.
"""

import base64
import json
import logging
import os
import time

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Short-name → ``~/.hermes/characters/<filename>`` mapping. Static so the
# tool's schema can advertise the valid character keys as an enum.
CHARACTER_KEYS = {
    "grantley": "02_grantley_bell.png",   # the tiger persona himself
    "algo":     "01_algo_northrop.png",   # 艾尔戈, owner
    "oscar":    "03_oscar_lawrence.png",  # 铁三角
    "diedrich": "04_diedrich_olsen.png",
    "paul":     "05_paul_pfizner.png",
    "theo":     "06_theo_prince.png",
    "julius":   "07_julius_kinial.png",
    "hermann":  "08_hermann_furst.png",
    "helio":    "09_helio_delatre.png",
    "shayat":   "10_shayat.png",
    "bating":   "11_bating.png",
}

_CHARACTER_DIR = os.path.expanduser("~/.hermes/characters")
_OUTPUT_DIR = os.path.expanduser("~/.hermes/cache/images")

# The Responses-API host model that calls the image_generation tool. The
# actual image work is done by ``IMAGE_MODEL``; this chat model is the
# wrapper that issues the tool call.
CHAT_MODEL = "gpt-5.5"
IMAGE_MODEL = "gpt-image-2"

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by "
    "using the image_generation tool when provided."
)

# Streams sometimes drop mid-generation with an "incomplete chunked read";
# retry transient failures a few times before giving up.
_MAX_ATTEMPTS = 3
_RETRY_SLEEP_SECONDS = 8

_SIZES = {
    "landscape": "1536x1024",
    "square":    "1024x1024",
    "portrait":  "1024x1536",
}

# Cap how many refs go in one request. More refs = larger payload, longer
# stream, more drops, worse quality (model gets confused). Empirically 1-3
# refs reads as best.
_MAX_REFS = 4


# ---------------------------------------------------------------------------
# OpenAI client (lazy)
# ---------------------------------------------------------------------------

def _build_client():
    """Build an OpenAI client for image-gen requests, or None if unconfigured."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = (os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL).strip().rstrip("/")
    try:
        import openai  # noqa: PLC0415 — optional dep, fail clean
    except ImportError:
        return None
    return openai.OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def _load_ref_data_url(name: str) -> str:
    """Encode a character立绘 as a ``data:image/png;base64,...`` URL."""
    filename = CHARACTER_KEYS[name]
    path = os.path.join(_CHARACTER_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"character立绘 missing on disk: {path}. Upload PNGs to "
            "~/.hermes/characters/ before requesting refs."
        )
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _resolve_characters(names):
    """Validate character keys and return their data URLs in order."""
    if not isinstance(names, list):
        raise ValueError(
            f"'characters' must be a list of short names, got {type(names).__name__}"
        )
    if len(names) > _MAX_REFS:
        raise ValueError(
            f"too many refs ({len(names)}); cap is {_MAX_REFS}. More refs slow "
            "generation and degrade prompt adherence."
        )
    bad = [n for n in names if n not in CHARACTER_KEYS]
    if bad:
        valid = ", ".join(sorted(CHARACTER_KEYS.keys()))
        raise ValueError(f"unknown character key(s): {bad}. Valid keys: {valid}")
    return [_load_ref_data_url(n) for n in names]


# ---------------------------------------------------------------------------
# Responses streaming call (one attempt)
# ---------------------------------------------------------------------------

def _generate_once(client, content, size: str, quality: str) -> str | None:
    """One streaming attempt — returns the result image as base64, or None."""
    image_b64 = None
    with client.responses.stream(
        model=CHAT_MODEL,
        store=False,
        instructions=_INSTRUCTIONS,
        input=[{"type": "message", "role": "user", "content": content}],
        tools=[{
            "type": "image_generation",
            "model": IMAGE_MODEL,
            "size": size,
            "quality": quality,
            "output_format": "png",
            "background": "opaque",
            "partial_images": 1,
        }],
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "image_generation"}],
        },
    ) as stream:
        for event in stream:
            et = getattr(event, "type", "")
            if et == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "image_generation_call":
                    r = getattr(item, "result", None)
                    if isinstance(r, str) and r:
                        image_b64 = r
            elif et == "response.image_generation_call.partial_image":
                p = getattr(event, "partial_image_b64", None)
                if isinstance(p, str) and p:
                    image_b64 = p
        final = stream.get_final_response()

    # Sweep the final response for the image-generation result in case the
    # output_item.done event arrived after the for-loop ended.
    for item in getattr(final, "output", None) or []:
        if getattr(item, "type", None) == "image_generation_call":
            r = getattr(item, "result", None)
            if isinstance(r, str) and r:
                image_b64 = r

    return image_b64


def _save_image(b64: str) -> str:
    """Decode and save a base64 PNG; return the absolute path."""
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    name = f"image_with_refs_{int(time.time())}_{os.urandom(2).hex()}.png"
    path = os.path.join(_OUTPUT_DIR, name)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return path


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _handle_image_with_refs(args, **kw) -> str:
    prompt = (args.get("prompt") or "").strip()
    characters = args.get("characters") or []
    aspect_ratio = (args.get("aspect_ratio") or "square").strip().lower()

    if not prompt:
        return tool_error("image_with_refs: 'prompt' is required.")

    try:
        ref_data_urls = _resolve_characters(characters)
    except (ValueError, FileNotFoundError) as e:
        return tool_error(f"image_with_refs: {e}")

    size = _SIZES.get(aspect_ratio, _SIZES["square"])
    quality = "medium"

    client = _build_client()
    if client is None:
        return tool_error(
            "image_with_refs: no OpenAI client — set OPENAI_API_KEY (and "
            "optionally OPENAI_BASE_URL) and ensure the `openai` package is "
            "installed."
        )

    # Build the user content: instruction + scene prompt, then ref images.
    intro = (
        "You are given character reference portraits. For each character that "
        "appears in the scene below, strictly match their species, fur color "
        "and markings, eye color, ear shape, and academy uniform from their "
        "reference image. Render them performing the scene's action in the "
        "described setting, with consistent kemono anthropomorphic anime art "
        "style.\n\n"
        "COMPOSITION DIRECTION (apply unless the scene prompt explicitly "
        "overrides):\n"
        "- Treat this as a candid slice-of-life snapshot, NOT a posed group "
        "portrait. Do NOT line characters up facing the camera.\n"
        "- Each character is in their own DIFFERENT beat — different body "
        "orientation, different gaze direction. They interact with the moment "
        "and with each other, not with the viewer.\n"
        "- Bodies in mid-action poses (reaching, leaning, turning, walking, "
        "sitting sideways), not standing-and-posing. Small natural micro-"
        "expressions, not photo-shoot smiles.\n"
        "- Off-axis camera framing — 3/4 view, over-the-shoulder, slight "
        "high or low angle. Avoid dead-on frontal symmetry.\n"
        "- Bake in incidental ambient detail (a tipped cup, an open book, a "
        "jacket draped on a chair, scattered items on a desk) for a "
        "lived-in, real-room feel.\n"
        "- If the scene involves a FURTIVE / SNEAKY / 偷偷 / 顺手 / "
        "不动声色 action, the actor's posture MUST read as hidden — turned "
        "away, low, hand kept close to body, eyes glancing sideways — and "
        "the other characters MUST be clearly looking elsewhere (out a "
        "window, at a book, at each other) and NEVER at the actor or the "
        "object being placed. Never have everyone staring at the secret "
        "action.\n\n"
    )
    if characters:
        intro += f"Reference characters provided in order: {', '.join(characters)}.\n\n"
    intro += "Scene:\n\n"

    content = [{"type": "input_text", "text": intro + prompt}]
    for url in ref_data_urls:
        content.append({"type": "input_image", "image_url": url, "detail": "high"})

    last_err = None
    img_b64 = None
    for attempt in range(_MAX_ATTEMPTS):
        t0 = time.time()
        try:
            img_b64 = _generate_once(client, content, size, quality)
            dt = time.time() - t0
            if img_b64:
                logger.info(
                    "image_with_refs OK in %.1fs (attempt %d/%d, %d refs)",
                    dt, attempt + 1, _MAX_ATTEMPTS, len(characters),
                )
                break
            last_err = "empty response (no image_generation_call result)"
            logger.warning(
                "image_with_refs empty result on attempt %d/%d after %.1fs",
                attempt + 1, _MAX_ATTEMPTS, dt,
            )
        except Exception as e:  # noqa: BLE001 — surface one clear message
            dt = time.time() - t0
            last_err = f"{type(e).__name__}: {e}"
            logger.warning(
                "image_with_refs attempt %d/%d failed after %.1fs: %s",
                attempt + 1, _MAX_ATTEMPTS, dt, last_err,
            )
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_RETRY_SLEEP_SECONDS)

    if not img_b64:
        return tool_error(
            f"image_with_refs: all {_MAX_ATTEMPTS} attempts failed (last: {last_err})"
        )

    try:
        out_path = _save_image(img_b64)
    except Exception as e:  # noqa: BLE001 — surface one clear message
        return tool_error(f"image_with_refs: could not save image: {e}")

    return json.dumps({
        "success": True,
        "image": out_path,
        "size": size,
        "quality": quality,
        "characters": characters,
        "model": f"{IMAGE_MODEL}-{quality}",
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------

def _check_image_with_refs_available() -> bool:
    """Available when OPENAI_API_KEY is set AND character立绘 dir exists."""
    if not os.path.isdir(_CHARACTER_DIR):
        return False
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

IMAGE_WITH_REFS_SCHEMA = {
    "name": "image_with_refs",
    "description": (
        "Generate an image using gpt-image-2 with optional character "
        "reference立绘 (PNG portraits under ~/.hermes/characters/). Each "
        "named character's立绘 is sent as a visual reference so the "
        "generated image matches their species, fur, eyes, and clothing — "
        "much better likeness than text-only image generation. Returns a "
        "local PNG path. Slow (~2-5 min per call, includes auto-retry); "
        "use for daily 说说 companion images, not high-frequency generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "English image-generation prompt describing the scene, "
                    "lighting, composition, every character's action and "
                    "expression, and style (kemono anthropomorphic anime "
                    "preferred). Detailed prompts (300-1500 chars) work best."
                ),
            },
            "characters": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(CHARACTER_KEYS.keys()),
                },
                "description": (
                    "Optional list of 0-" + str(_MAX_REFS) + " character "
                    "short names to attach as reference立绘. Available keys: "
                    + ", ".join(sorted(CHARACTER_KEYS.keys())) + ". Empty "
                    "list = text-only generation. 1-3 refs is the sweet spot."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["square", "landscape", "portrait"],
                "description": "Output aspect ratio. Default: square.",
            },
        },
        "required": ["prompt"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="image_with_refs",
    toolset="image_refs",
    schema=IMAGE_WITH_REFS_SCHEMA,
    handler=_handle_image_with_refs,
    check_fn=_check_image_with_refs_available,
    requires_env=["OPENAI_API_KEY"],
    emoji="🎨",
)
