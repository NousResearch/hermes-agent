"""Speech Router: single point of provider selection for TTS requests.

PR1 introduced resolve_provider() as a passthrough. PR2 added
get_capabilities(). PR3 adds resolve_provider_chain(): an ordered list of
providers to try, driven by the new ``tts.router.providers`` config key.
When that key is absent -- the default, unopted-in case -- the chain is a
single item identical to what resolve_provider() always returned, so
nothing changes for users who haven't touched the new config.
"""

from typing import Any, Dict, List, Optional, Tuple

from tools.speech.capabilities import SpeechCapabilities, get_builtin_capabilities


class SpeechRouter:
    """Resolves which TTS provider(s) a synthesis request should try.

    Fallback order comes from ``tts.router.providers`` (a list) in
    config.yaml when present; otherwise the chain has exactly one entry,
    matching every pre-Router release. Health checking is deliberately not
    a separate step here -- text_to_speech_tool() moves to the next
    provider in the chain on any synthesis failure (timeout, exception, or
    unsuccessful result), so a provider's own configured timeout doubles as
    its liveness probe. See the Speech Router RFC for why.
    """

    def resolve_provider_chain(self) -> Tuple[List[str], Dict[str, Any]]:
        """Return (ordered provider names to try, tts_config).

        Deferred import avoids a circular dependency: tools.tts_tool is the
        caller, so it is already fully imported by the time this runs.
        """
        from tools.tts_tool import _get_provider, _load_tts_config

        tts_config = _load_tts_config()
        router_config = tts_config.get("router") or {}
        configured = router_config.get("providers")

        if configured:
            chain = [str(name).strip() for name in configured if str(name).strip()]
            if chain:
                return chain, tts_config

        return [_get_provider(tts_config)], tts_config

    def resolve_provider(self) -> Tuple[str, Dict[str, Any]]:
        """Return (provider_name, tts_config) -- the first entry of the chain.

        Kept for callers that only need a single provider; text_to_speech_tool()
        itself now calls resolve_provider_chain() directly to support fallback.
        """
        chain, tts_config = self.resolve_provider_chain()
        return chain[0], tts_config

    def get_capabilities(self, provider: str) -> Optional[SpeechCapabilities]:
        """Return known capabilities for *provider*, or None if unknown.

        Only covers the ten built-in providers today (see
        tools/speech/capabilities.py). Command/HTTP-generic and plugin
        providers return None here until a later PR adds config-declared
        and live-probed capability sources.
        """
        return get_builtin_capabilities(provider)
