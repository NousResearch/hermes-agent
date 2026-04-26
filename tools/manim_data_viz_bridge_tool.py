#!/usr/bin/env python3
"""
Manim data-visualization bridge tool.

Renders a matplotlib/seaborn chart and emits a Manim Community Edition scene
that displays the chart plus optional MathTex explanations. Requires optional
extra: ``pip install 'hermes-agent[data-viz]'``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_DATA_VIZ_OK: bool | None = None


def check_data_viz_bridge() -> bool:
    """True when matplotlib, seaborn, and pandas are importable."""
    global _DATA_VIZ_OK
    if _DATA_VIZ_OK is not None:
        return _DATA_VIZ_OK
    try:
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
        import seaborn  # noqa: F401
        _DATA_VIZ_OK = True
    except ImportError:
        _DATA_VIZ_OK = False
    return _DATA_VIZ_OK


def _handle_manim_data_viz_bridge(args: Dict[str, Any], **kw: Any) -> str:
    if not check_data_viz_bridge():
        return tool_error(
            "Data-viz dependencies missing. Install: pip install 'hermes-agent[data-viz]'"
        )
    from src.skills.creative.manim_data_viz_bridge import DataVizBridgeRequest, ManimDataVizBridge

    raw: Dict[str, Any] = {
        "user_prompt": args.get("user_prompt", ""),
        "viz": args.get("viz"),
    }
    if args.get("manim") is not None:
        raw["manim"] = args["manim"]
    if not raw["user_prompt"]:
        return tool_error("user_prompt is required")
    if raw["viz"] is None:
        return tool_error("viz object is required")
    try:
        req = DataVizBridgeRequest.model_validate(raw)
    except Exception as e:
        return tool_error(f"Invalid manim_data_viz_bridge payload: {e}")
    try:
        out = ManimDataVizBridge().build(req)
    except Exception as e:
        logger.exception("manim_data_viz_bridge build failed")
        return tool_error(f"Bridge build failed: {type(e).__name__}: {e}")
    return tool_result(
        {
            "work_dir": str(out.work_dir),
            "chart_path": out.chart_path.as_posix(),
            "manim_script_path": out.manim_script_path.as_posix(),
            "manim_source": out.manim_source,
            "render_suggestion": out.render_suggestion,
        }
    )


MANIM_DATA_VIZ_BRIDGE_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "name": "manim_data_viz_bridge",
    "description": (
        "Build a Manim CE explainer scene from structured data: renders a seaborn/matplotlib "
        "chart to PNG, writes a runnable Manim script in a temp directory, and returns paths "
        "plus a suggested `manim` CLI line. Map the user's single natural-language request "
        "into the nested `viz` object before calling. Requires optional install "
        "`hermes-agent[data-viz]` and Manim CE on the host to render video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "Verbatim user request (stored as comments in the emitted script).",
            },
            "viz": {
                "type": "object",
                "description": "VizSpecification: chart_type, title, labels, data (exactly one of records, x+y, or matrix).",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["line", "scatter", "bar", "heatmap", "distribution", "regression"],
                    },
                    "title": {"type": "string"},
                    "x_label": {"type": "string", "default": ""},
                    "y_label": {"type": "string", "default": ""},
                    "hue_column": {"type": ["string", "null"]},
                    "seaborn_style": {"type": "string", "default": "darkgrid"},
                    "palette": {"type": "string", "default": "deep"},
                    "x_field": {"type": "string", "default": "x"},
                    "y_field": {"type": "string", "default": "y"},
                    "category_field": {"type": "string", "default": "category"},
                    "value_field": {"type": "string", "default": "value"},
                    "figure_size": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "default": [10.0, 6.0],
                    },
                    "dpi": {"type": "integer", "default": 200},
                    "data": {
                        "type": "object",
                        "description": "Exactly one of: records, x+y arrays, or matrix.",
                        "properties": {
                            "records": {"type": "array", "items": {"type": "object"}},
                            "x": {"type": "array", "items": {"type": "number"}},
                            "y": {"type": "array", "items": {"type": "number"}},
                            "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        },
                    },
                },
                "required": ["chart_type", "title", "data"],
            },
            "manim": {
                "type": "object",
                "description": "Optional ManimSceneOptions.",
                "properties": {
                    "scene_class_name": {"type": "string", "default": "DataVizBridgeScene"},
                    "resolution": {"type": "string", "enum": ["ql", "qm", "qh"], "default": "ql"},
                    "tex_explanations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "LaTeX fragments without $ delimiters.",
                    },
                },
            },
        },
        "required": ["user_prompt", "viz"],
    },
}


registry.register(
    name="manim_data_viz_bridge",
    toolset="data_viz",
    schema=MANIM_DATA_VIZ_BRIDGE_SCHEMA,
    handler=_handle_manim_data_viz_bridge,
    check_fn=check_data_viz_bridge,
    requires_env=[],
    is_async=False,
    description=MANIM_DATA_VIZ_BRIDGE_SCHEMA["description"],
    emoji="📊",
    max_result_size_chars=200_000,
)
