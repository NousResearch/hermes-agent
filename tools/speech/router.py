"""Speech Router: single point of provider selection for TTS requests.

PR1 introduced resolve_provider() as a passthrough. PR2 adds
get_capabilities() so callers (and later PRs -- fallback in PR3, streaming
negotiation in PR8) can ask what a provider can do before using it. Neither
method changes text_to_speech_tool()'s existing behavior; get_capabilities()
has no caller yet outside tests.
"""

from typing import Any, Dict, Optional, Tuple

from tools.speech.capabilities import SpeechCapabilities, get_builtin_capabilities


class SpeechRouter:
    """Resolves which TTS provider a synthesis request should use.

    Today this is a thin wrapper around the existing single-provider config
    read in tools/tts_tool.py. Future PRs add a fallback chain, health
    checks, and streaming negotiation without changing resolve_provider()'s
    return shape.
    """

    def resolve_provider(self) -> Tuple[str, Dict[str, Any]]:
        """Return (provider_name, tts_config) for the current request.

        Deferred import avoids a circular dependency: tools.tts_tool is the
        caller, so it is already fully imported by the time this runs.
        """
        from tools.tts_tool import _get_provider, _load_tts_config

        tts_config = _load_tts_config()
        provider = _get_provider(tts_config)
        return provider, tts_config

    def get_capabilities(self, provider: str) -> Optional[SpeechCapabilities]:
        """Return known capabilities for *provider*, or None if unknown.

        Only covers the ten built-in providers today (see
        tools/speech/capabilities.py). Command/HTTP-generic and plugin
        providers return None here until a later PR adds config-declared
        and live-probed capability sources.
        """
        return get_builtin_capabilities(provider)
