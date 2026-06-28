"""Input normalization for FAL video-to-video tools."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from tools.image_generation_tool import _load_fal_client
from tools.tool_backend_helpers import fal_key_is_configured


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mpeg", ".mpg", ".mkv")
WEBPAGE_EXTENSIONS = (".html", ".htm", ".php", ".asp", ".aspx")


def _path_without_query(value: str) -> str:
    return value.strip().lower().split("?", 1)[0].split("#", 1)[0]


def looks_like_image_url(value: str) -> bool:
    lower = _path_without_query(value)
    return lower.startswith("data:image/") or lower.endswith(IMAGE_EXTENSIONS)


def looks_like_webpage_url(value: str) -> bool:
    raw = value.strip()
    lower = raw.lower()
    if lower.startswith("data:text/html"):
        return True

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if host in {"docs.fal.ai", "docs.fal.co"}:
        return True
    if host.endswith(".fal.ai") and path.startswith("/docs"):
        return True
    return path.endswith(WEBPAGE_EXTENSIONS)


def _local_path_from_video_url(value: str) -> Path | None:
    raw = value.strip()
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https", "data"}:
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme:
        return None
    return Path(raw).expanduser()


def resolve_fal_video_input(video_url: str, *, image_error: str) -> str:
    """Return a FAL-readable video URL, uploading local files when needed."""
    if not video_url or not isinstance(video_url, str) or not video_url.strip():
        raise ValueError("video_url is required")
    if looks_like_image_url(video_url):
        raise ValueError(image_error)
    if looks_like_webpage_url(video_url):
        raise ValueError(
            "video_url must be a direct, publicly downloadable video file URL, "
            "not a documentation page or webpage. Use a .mp4/.webm URL or the "
            "video/output_url returned by video_generate."
        )

    local_path = _local_path_from_video_url(video_url)
    if local_path is None:
        return video_url.strip()

    if not local_path.exists():
        raise ValueError(
            "video_url must be a direct public video URL or an existing local "
            f"video file path; file not found: {local_path}"
        )
    if not local_path.is_file():
        raise ValueError(f"video_url local path is not a file: {local_path}")
    if local_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            "video_url local path must reference a video file "
            f"({', '.join(VIDEO_EXTENSIONS)}): {local_path}"
        )
    if not fal_key_is_configured():
        raise ValueError(
            "Local video file inputs require a direct FAL_KEY so Hermes can "
            "upload the file to FAL storage before processing. Set FAL_KEY or "
            "pass a direct public video URL."
        )

    fal_client = _load_fal_client()
    upload_file = getattr(fal_client, "upload_file", None)
    if upload_file is None:
        raise RuntimeError("fal_client.upload_file is required for local video inputs")

    uploaded_url = upload_file(str(local_path))
    if not isinstance(uploaded_url, str) or not uploaded_url.strip():
        raise RuntimeError("fal_client.upload_file did not return a video URL")
    return uploaded_url.strip()
