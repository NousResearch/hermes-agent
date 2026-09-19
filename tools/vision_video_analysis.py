"""Discovery bridge for the sharded video-analysis runtime.

Built-in tool discovery imports self-registering top-level modules in sorted
order. This file sorts after ``vision_tools.py`` and explicitly settles
``video_analyze`` on the package owner after the legacy module is loaded.
"""

from tools import video_analysis
from tools.registry import registry

registry.register(
    name="video_analyze",
    toolset="video",
    schema=video_analysis.VIDEO_ANALYZE_SCHEMA,
    handler=video_analysis._handle_video_analyze,
    check_fn=lambda: video_analysis.core._vision_module().check_vision_requirements(),
    is_async=True,
    emoji="🎬",
    override=True,
)
