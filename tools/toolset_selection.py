"""Shared toolset-name expansion for model selection and platform suppression."""

LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": ["browser_navigate", "browser_snapshot", "browser_click", "browser_type", "browser_scroll",
                      "browser_back", "browser_press", "browser_get_images", "browser_vision", "browser_console"],
    "cronjob_tools": ["cronjob_manage"],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


def resolve_toolset_selection(name: str, *, disable: bool = False):
    """Resolved tool names, or None for unknown selections; bundles disable only their non-core delta."""
    from toolsets import bundle_non_core_tools, get_toolset, resolve_toolset, validate_toolset
    if validate_toolset(name):
        if disable and (name.startswith("hermes-") or (get_toolset(name) or {}).get("posture")):
            return sorted(bundle_non_core_tools(name))
        return resolve_toolset(name)
    return LEGACY_TOOLSET_MAP.get(name)
