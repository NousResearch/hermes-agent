#!/usr/bin/env python3
"""
Translation Tool Module - Multi-language Text Translation

Provides text translation between 100+ languages using the deep-translator
library (Google Translate backend, no API key required).

The agent can translate user messages, documents, or any text on the fly,
making Hermes useful for international and multilingual users.

Usage:
    translate_tool(text="Hello world", target="tr")          # -> "Merhaba dünya"
    translate_tool(text="Bonjour", source="fr", target="en") # -> "Hello"
    translate_tool(action="languages")                        # -> list of supported languages
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded translator to avoid import errors if deep-translator isn't installed
_translator_available = None


def _check_translator():
    """Check if deep-translator is installed."""
    global _translator_available
    if _translator_available is None:
        try:
            from deep_translator import GoogleTranslator
            _translator_available = True
        except ImportError:
            _translator_available = False
    return _translator_available


def check_translation_requirements() -> bool:
    """Check if translation tool dependencies are available."""
    return _check_translator()


def translate_tool(
    text: Optional[str] = None,
    target: str = "en",
    source: str = "auto",
    action: str = "translate",
) -> str:
    """
    Translate text between languages or list supported languages.

    Args:
        text: The text to translate.
        target: Target language code (e.g., 'en', 'tr', 'fr', 'de', 'ja').
        source: Source language code or 'auto' for auto-detection.
        action: 'translate' to translate text, 'languages' to list supported languages.

    Returns:
        JSON string with translation result or language list.
    """
    if not _check_translator():
        return json.dumps({
            "error": "deep-translator is not installed. Run: pip install deep-translator"
        }, ensure_ascii=False)

    from deep_translator import GoogleTranslator

    try:
        if action == "languages":
            langs = GoogleTranslator().get_supported_languages(as_dict=True)
            return json.dumps({
                "action": "languages",
                "languages": langs,
                "count": len(langs),
            }, ensure_ascii=False)

        if not text or not text.strip():
            return json.dumps({"error": "No text provided for translation."}, ensure_ascii=False)

        translator = GoogleTranslator(source=source, target=target)
        translated = translator.translate(text)

        return json.dumps({
            "action": "translate",
            "source_language": source,
            "target_language": target,
            "original_text": text,
            "translated_text": translated,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("Translation error: %s", e)
        return json.dumps({
            "error": f"Translation failed: {type(e).__name__}: {e}"
        }, ensure_ascii=False)


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

TRANSLATE_SCHEMA = {
    "name": "translate",
    "description": (
        "Translate text between 100+ languages using Google Translate. "
        "No API key required. Supports auto-detection of source language.\n\n"
        "Actions:\n"
        "- translate: Translate text from source to target language\n"
        "- languages: List all supported language codes\n\n"
        "Common language codes: en (English), tr (Turkish), fr (French), "
        "de (German), es (Spanish), it (Italian), pt (Portuguese), "
        "ru (Russian), ja (Japanese), ko (Korean), zh-CN (Chinese Simplified), "
        "ar (Arabic), hi (Hindi), nl (Dutch), sv (Swedish)\n\n"
        "Example: translate(text='Hello world', target='tr') -> 'Merhaba dünya'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to translate. Required for 'translate' action."
            },
            "target": {
                "type": "string",
                "description": (
                    "Target language code (e.g., 'en', 'tr', 'fr', 'de', 'ja'). "
                    "Defaults to 'en'."
                ),
                "default": "en"
            },
            "source": {
                "type": "string",
                "description": (
                    "Source language code or 'auto' for automatic detection. "
                    "Defaults to 'auto'."
                ),
                "default": "auto"
            },
            "action": {
                "type": "string",
                "enum": ["translate", "languages"],
                "description": (
                    "'translate' to translate text (default), "
                    "'languages' to list all supported language codes."
                ),
                "default": "translate"
            }
        },
        "required": []
    }
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="translate",
    toolset="translation",
    schema=TRANSLATE_SCHEMA,
    handler=lambda args, **kw: translate_tool(
        text=args.get("text"),
        target=args.get("target", "en"),
        source=args.get("source", "auto"),
        action=args.get("action", "translate"),
    ),
    check_fn=check_translation_requirements,
)
