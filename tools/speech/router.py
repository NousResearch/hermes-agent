"""Speech Router: single point of provider selection for TTS requests.

PR1 scope (see "RFC: Speech Router"): passthrough only. resolve_provider()
reproduces exactly what text_to_speech_tool() computed inline before this
change (_load_tts_config() + _get_provider()). No fallback chain, no
capability negotiation, no streaming yet -- those land in later PRs. This
PR only introduces the seam so later PRs have a stable place to grow into.
"""

from typing import Any, Dict, Tuple


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
