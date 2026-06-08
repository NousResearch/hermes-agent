#!/usr/bin/env python3
"""i18n - Internationalization support for Hermes Agent.

Provides language detection from config and translation functions for
localizing user-facing strings.
"""

import os
from typing import Optional


def get_config_language() -> str:
    """Get the current language setting from config.
    
    Returns:
        Language code: 'zh' for Chinese, 'en' for English (default)
    """
    # Try to read from config without importing the full config module
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if not os.path.exists(config_path):
        return "en"
    
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        # Check approvals.language first, then display.language
        lang = config.get("approvals", {}).get("language")
        if not lang:
            lang = config.get("display", {}).get("language")
        if not lang:
            lang = config.get("language")
        
        return lang if lang in ("zh", "en") else "en"
    except Exception:
        return "en"


def is_chinese() -> bool:
    """Check if current language is Chinese."""
    return get_config_language() == "zh"


def format_zh(text: str, **kwargs) -> str:
    """Format text with Chinese translation if language is set to Chinese.

    Uses locales/cli.yaml for translations as the primary source (extracted
    from the hardcoded dict that was previously inline).  Edit the YAML file
    to add or fix translations instead of modifying this function.
    """
    from pathlib import Path
    import yaml as _yaml_module

    # -- Load YAML translations (lazy, cached on the function object) --
    if not hasattr(format_zh, "_cache"):
        _path = Path(__file__).resolve().parent.parent / "locales" / "cli.yaml"
        _exact: dict[str, str] = {}
        _replace: dict[str, str] = {}
        if _path.is_file():
            try:
                with _path.open("r", encoding="utf-8") as _f:
                    _data = _yaml_module.safe_load(_f) or {}
                _fmt = _data.get("format_zh", {})
                _exact = _fmt.get("exact", {}) or {}
                _replace = _fmt.get("replace", {}) or {}
            except Exception:
                pass
        format_zh._cache = (_exact, _replace)
    else:
        _exact, _replace = format_zh._cache

    if not is_chinese():
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, ValueError):
                pass
        return text

    # 1. Full-message exact match (highest priority)
    if text in _exact:
        result = _exact[text]
        if kwargs:
            try:
                return result.format(**kwargs)
            except (KeyError, ValueError):
                pass
        return result

    # 2. Substring replacements (same logic as before, now YAML-backed)
    result = text
    for en, zh in sorted(_replace.items(), key=lambda x: len(x[0]), reverse=True):
        if en in result:
            result = result.replace(en, zh)

    # 3. Apply format arguments
    if kwargs:
        try:
            result = result.format(**kwargs)
        except (KeyError, ValueError):
            pass

    return result