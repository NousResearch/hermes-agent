"""Canonical video-analysis runtime and legacy vision-tools compatibility."""

from __future__ import annotations

from tools.video_analysis import core as core

VIDEO_ANALYZE_SCHEMA = core.VIDEO_ANALYZE_SCHEMA
VideoFrame = core.VideoFrame
VideoFrameExtractionError = core.VideoFrameExtractionError
VideoInputNegotiationError = core.VideoInputNegotiationError
_MAX_VIDEO_BASE64_BYTES = core._MAX_VIDEO_BASE64_BYTES
_VIDEO_MIME_TYPES = core._VIDEO_MIME_TYPES
_VIDEO_SIZE_WARN_BYTES = core._VIDEO_SIZE_WARN_BYTES
_detect_video_mime_type = core._detect_video_mime_type
_download_video = core._download_video
_extract_video_frames = core._extract_video_frames
_extract_video_frames_async = core._extract_video_frames_async
_handle_video_analyze = core._handle_video_analyze
_is_path_like_video_source = core._is_path_like_video_source
_materialize_video_from_terminal_backend = (
    core._materialize_video_from_terminal_backend
)
_terminal_backend_is_local = core._terminal_backend_is_local
_video_to_base64_data_url = core._video_to_base64_data_url
video_analyze_tool = core.video_analyze_tool


def _install_legacy_aliases() -> None:
    """Keep tools.vision_tools imports on the canonical runtime owner."""
    from tools import vision_tools

    aliases = {
        "VIDEO_ANALYZE_SCHEMA": VIDEO_ANALYZE_SCHEMA,
        "VideoFrame": VideoFrame,
        "VideoFrameExtractionError": VideoFrameExtractionError,
        "VideoInputNegotiationError": VideoInputNegotiationError,
        "_MAX_VIDEO_BASE64_BYTES": _MAX_VIDEO_BASE64_BYTES,
        "_VIDEO_MIME_TYPES": _VIDEO_MIME_TYPES,
        "_VIDEO_SIZE_WARN_BYTES": _VIDEO_SIZE_WARN_BYTES,
        "_detect_video_mime_type": _detect_video_mime_type,
        "_download_video": _download_video,
        "_extract_video_frames": _extract_video_frames,
        "_extract_video_frames_async": _extract_video_frames_async,
        "_handle_video_analyze": _handle_video_analyze,
        "_is_path_like_video_source": _is_path_like_video_source,
        "_materialize_video_from_terminal_backend": (
            _materialize_video_from_terminal_backend
        ),
        "_terminal_backend_is_local": _terminal_backend_is_local,
        "_video_to_base64_data_url": _video_to_base64_data_url,
        "video_analyze_tool": video_analyze_tool,
    }
    for name, value in aliases.items():
        setattr(vision_tools, name, value)


_install_legacy_aliases()

# Registry ownership intentionally lives in tools/vision_video_analysis.py.
# Built-in discovery imports that bridge after tools.vision_tools, so one
# override settles the canonical handler instead of three import-order writers.

__all__ = [
    "VIDEO_ANALYZE_SCHEMA",
    "VideoFrame",
    "VideoFrameExtractionError",
    "VideoInputNegotiationError",
    "video_analyze_tool",
]
