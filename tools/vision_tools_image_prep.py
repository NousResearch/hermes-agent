"""Image format detection, normalization and region cropping for vision tools.

Everything here runs BEFORE an image is base64-embedded: a vision tool result is
baked into immutable history and re-sent every turn, so an unsupported media type
or corrupt bytes would wedge the session with a non-retryable 400 on every resume.
"""

from __future__ import annotations

import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_dir

logger = logging.getLogger("tools.vision_tools")

_EXTENSION_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

# Media types the major vision providers (Anthropic in particular) accept
# inline. SVG/BMP/TIFF are rejected with a non-retryable 400.
_ANTHROPIC_SUPPORTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


_MAGIC_MIME_TYPES = (
    (b"\xff\xd8\xff", "image/jpeg"), ((b"GIF87a", b"GIF89a"), "image/gif"), (b"BM", "image/bmp"),
)


def _determine_mime_type(image_path: Path) -> str:
    """MIME type from file extension (defaults to image/jpeg)."""
    return _EXTENSION_MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")


def _detect_image_mime_type_from_bytes(data: bytes) -> Optional[str]:
    """Magic-byte MIME sniff (authoritative; no extension trust). ``None`` for anything without a
    recognized header — including SVG, which has none (the resolver sniffs ``<svg`` itself)."""
    header = data[:64]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        # Reject corrupt PNGs before they can be embedded. Pillow is optional —
        # without it fall back to header-only sniffing; only a failed verify() rejects.
        try:
            from PIL import Image
        except ImportError:
            return "image/png"
        try:
            with Image.open(BytesIO(data)) as image:
                image.verify()
            return "image/png"
        except Exception:
            return None
    for magic, mime in _MAGIC_MIME_TYPES:
        if header.startswith(magic):
            return mime
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _supported_media_types() -> frozenset:
    """Formats the ACTIVE main model's server can decode. The managed llama-server decodes with
    stb_image — no WebP — and an undecodable image part fails SILENTLY (the model confabulates),
    so the set is narrowed there and normalization converts those formats to PNG."""
    try:
        from agent.auxiliary_client import _runtime_main_value as _v
        from hermes_cli.local_runtime.capabilities import ACCEPTED_IMAGE_MIMES, is_managed_provider
        if is_managed_provider(str(_v("provider") or ""), str(_v("base_url") or "")):
            return ACCEPTED_IMAGE_MIMES
    except Exception:  # best-effort narrowing only
        pass
    return _ANTHROPIC_SUPPORTED_MEDIA_TYPES


def _nonempty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _rasterize_svg_to_png(svg_path: Path, out_path: Path) -> bool:
    """Best-effort SVG → PNG via cairosvg, svglib+reportlab, rsvg-convert, inkscape (all soft deps)."""
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(url=str(svg_path), write_to=str(out_path))
        return _nonempty_file(out_path)
    except Exception:
        pass
    try:
        from svglib.svglib import svg2rlg  # type: ignore
        from reportlab.graphics import renderPM  # type: ignore
        drawing = svg2rlg(str(svg_path))
        if drawing is not None:
            renderPM.drawToFile(drawing, str(out_path), fmt="PNG")
            return _nonempty_file(out_path)
    except Exception:
        pass
    import shutil
    import subprocess
    for cmd in (
        ["rsvg-convert", "-o", str(out_path), str(svg_path)],
        ["inkscape", str(svg_path), "--export-type=png", f"--export-filename={out_path}"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
                if _nonempty_file(out_path):
                    return True
            except Exception:
                continue
    return False


def _normalize_to_supported_image(
    image_path: Path, detected_mime: str) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """Ensure an image is in a provider-supported format. Returns ``(path, mime, error)``: the input
    unchanged when supported; ``(new_png_path, "image/png", None)`` after conversion — a temp file
    the CALLER must clean up; ``(None, None, message)`` when impossible. SVG is rasterized; other
    Pillow-readable rasters (BMP, TIFF) re-encode to PNG."""
    if detected_mime in _supported_media_types():
        return image_path, detected_mime, None
    out_dir = get_hermes_dir("cache/vision", "temp_vision_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"converted_{uuid.uuid4()}.png"
    if detected_mime == "image/svg+xml":
        if _rasterize_svg_to_png(image_path, out_path):
            return out_path, "image/png", None
        return None, None, (
            "This is an SVG, which vision models cannot read directly, and no "
            "SVG rasterizer is installed (tried cairosvg, svglib, rsvg-convert, "
            "inkscape). Convert the SVG to PNG first — e.g. open it in a browser "
            "and screenshot it, or install a rasterizer "
            "(`pip install cairosvg`) — then re-run vision_analyze on the PNG.")
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as _img:
            if _img.mode not in ("RGB", "RGBA", "L"):
                _img = _img.convert("RGBA")
            _img.save(out_path, format="PNG")
        if _nonempty_file(out_path):
            return out_path, "image/png", None
    except Exception as _exc:
        logger.warning("Failed to normalize %s image to PNG: %s", detected_mime, _exc)
    return None, None, (
        f"Image format {detected_mime!r} is not supported by the vision API "
        f"and could not be converted to PNG (install Pillow for raster "
        f"conversion). Convert it to PNG or JPEG and try again.")


# Full raster validation runs on untrusted images in a shared CPU executor: bound animated
# work by frame count AND total decoded area so a compact file cannot monopolize a worker.
_VISION_MAX_VALIDATED_FRAME_COUNT = 100
_VISION_MAX_VALIDATED_AGGREGATE_PIXELS = 100_000_000


def _validate_raster_image_decodable(
    image_path: Path,
    max_frames: int = _VISION_MAX_VALIDATED_FRAME_COUNT,
    max_pixels: int = _VISION_MAX_VALIDATED_AGGREGATE_PIXELS) -> Optional[str]:
    """Return an error unless Pillow can fully decode every frame. Header sniffing and ``Image.open``
    only inspect containers: a timed-out download can look like a valid PNG with a truncated pixel
    stream. Without Pillow the image passes unvalidated rather than rejecting everything."""
    try:
        from PIL import Image as _PILImage, ImageSequence as _PILImageSequence
    except ImportError:
        return None
    try:
        with _PILImage.open(image_path) as image:
            image.verify()
        with _PILImage.open(image_path) as image:
            validated_pixels = 0
            for frame_number, frame in enumerate(_PILImageSequence.Iterator(image), start=1):
                if frame_number > max_frames:
                    return (
                        "Image validation rejected animation: "
                        f"frame {frame_number} exceeds the maximum "
                        f"{max_frames} validated frames.")
                next_validated_pixels = validated_pixels + frame.width * frame.height
                if next_validated_pixels > max_pixels:
                    return (
                        "Image validation rejected animation: aggregate decoded "
                        f"pixel count would reach {next_validated_pixels} at frame "
                        f"{frame_number}, exceeding the maximum "
                        f"{max_pixels}.")
                frame.load()
                validated_pixels = next_validated_pixels
    except Exception as exc:
        return f"Image could not be fully decoded: {exc}"
    return None


def _image_exceeds_dimension(image_path: Path, max_dimension: int) -> bool:
    """True if the longest side exceeds ``max_dimension`` px (Anthropic's 8000px per-side cap is
    independent of bytes). False without Pillow or on unreadable files — a missing soft
    dependency must never break the embed path."""
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as _img:
            return max(_img.size) > max_dimension
    except Exception:
        return False


def _crop_image_region(
    image_path: Path, region: Any, offset_out: Optional[dict] = None
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """Crop to ``region`` = [x1, y1, x2, y2] (original-image pixels), BEFORE downscaling so the crop
    gets the full resolution budget. Coordinates clamp to the image bounds; a zero-area/inverted
    region is rejected with an error naming the real dimensions. Returns ``(cropped_temp_path,
    mime, None)`` — caller owns cleanup — or ``(None, None, error)``.
    Ported from QwenLM/qwen-code zoom-image.ts (Apache-2.0)."""
    try:
        from PIL import Image
    except ImportError:
        return None, None, (
            "region cropping requires Pillow (`pip install Pillow`); "
            "retry without the region parameter.")
    if not (isinstance(region, (list, tuple)) and len(region) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in region)):
        return None, None, (
            "Invalid region: expected [x1, y1, x2, y2] as four numbers "
            "(pixel coordinates in the original image).")
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            x1, y1, x2, y2 = (int(v) for v in region)
            cx1, cy1, cx2, cy2 = (max(0, min(v, b)) for v, b in zip((x1, y1, x2, y2), (width, height) * 2))
            if cx2 <= cx1 or cy2 <= cy1:
                return None, None, (
                    f"Invalid region [{x1}, {y1}, {x2}, {y2}]: crops to zero "
                    f"area after clamping to the image bounds. The image is "
                    f"{width}x{height} px — pick x1<x2 and y1<y2 inside "
                    f"[0, 0, {width}, {height}].")
            cropped = img.crop((cx1, cy1, cx2, cy2))
            if offset_out is not None:
                offset_out.update(x=cx1, y=cy1, width=cx2 - cx1, height=cy2 - cy1)
            out_path = image_path.with_name(f"{image_path.stem}_region_{uuid.uuid4().hex[:8]}.png")
            if cropped.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                cropped = cropped.convert("RGB")
            cropped.save(out_path, format="PNG")
            return out_path, "image/png", None
    except Exception as exc:
        return None, None, f"Failed to crop region: {exc}"
