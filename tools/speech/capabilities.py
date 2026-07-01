"""Static capability metadata for built-in TTS providers.

PR2 of the Speech Router RFC. Nothing consumes this data yet -- SpeechRouter
does not change its provider-selection behavior because of it. This PR only
makes capabilities queryable so PR3 (fallback chain) and later PRs
(streaming) have grounded facts to select on instead of re-deriving them ad
hoc at call sites.

Values are drawn from what this codebase itself documents or implements,
not from general knowledge about each provider's public API. Fields marked
"not verified" in a comment are conservative defaults (False/None) pending
confirmation -- flag them for correction rather than trusting them blindly.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SpeechCapabilities:
    """What a TTS provider can do, independent of how it's configured.

    ``streaming`` means the provider's API can return audio incrementally
    (confirmed today only for ElevenLabs, see
    ``tools.tts_tool.stream_tts_to_speaker`` -- CLI/TUI only, not yet wired
    to SpeechRouter). ``priority`` is an advisory default; an explicit
    ``tts.router.providers`` order in config.yaml always wins over it.
    """

    local: bool
    streaming: bool
    voice_clone: bool
    multiple_voices: bool
    ssml: bool
    realtime: bool
    languages: str  # "*" for effectively-any, else a short human note
    latency_estimate_ms: Optional[int]
    priority: int  # lower = tried earlier when no explicit order is configured


# SSML: none of the ten builtins implement it. Grounded in the two explicit
# "Do not use SSML" instructions already in this file's own prompt text
# (search for "Do not use SSML") -- no provider path here ever emits or
# expects SSML markup today.
BUILTIN_CAPABILITIES: Dict[str, SpeechCapabilities] = {
    "edge": SpeechCapabilities(
        local=False,  # cloud-backed (Microsoft), but free and keyless
        streaming=False,
        voice_clone=False,
        multiple_voices=True,  # 322 voices, 74 languages (website/docs/.../voice-mode.md)
        ssml=False,
        realtime=False,
        languages="*",
        latency_estimate_ms=1000,  # docs: TTS Provider Comparison table, "~1s"
        priority=100,  # existing code already treats Edge as the universal last-resort default
    ),
    "elevenlabs": SpeechCapabilities(
        local=False,
        streaming=True,  # confirmed: stream_tts_to_speaker() + DEFAULT_ELEVENLABS_STREAMING_MODEL_ID
        voice_clone=True,  # not verified in this codebase's own text -- ElevenLabs' well-documented headline feature, kept True but flagged
        multiple_voices=True,
        ssml=False,
        realtime=False,
        languages="*",
        latency_estimate_ms=2000,  # docs: "~2s"
        priority=10,  # docs list it "Excellent" quality; cheap default to try early when configured
    ),
    "openai": SpeechCapabilities(
        local=False,
        streaming=False,  # not found in this codebase; direct REST call writes a full file
        voice_clone=False,
        multiple_voices=True,  # alloy/echo/fable/onyx/nova/shimmer (DEFAULT_OPENAI_VOICE + config comment)
        ssml=False,
        realtime=False,
        languages="*",
        latency_estimate_ms=1500,  # docs: "~1.5s"
        priority=20,
    ),
    "minimax": SpeechCapabilities(
        local=False,
        streaming=False,  # not verified
        voice_clone=True,  # module docstring: "MiniMax TTS: High-quality with voice cloning"
        multiple_voices=True,  # not verified, conservative
        ssml=False,
        realtime=False,
        languages="*",
        latency_estimate_ms=None,  # not documented
        priority=30,
    ),
    "xai": SpeechCapabilities(
        local=False,
        streaming=False,  # not verified (has `optimize_streaming_latency` tuning knob, but that's a
                            # synchronous-response latency setting, not a chunked/incremental API)
        voice_clone=False,
        multiple_voices=True,  # "Grok voices"
        ssml=False,  # has its own non-SSML expressive-tag syntax instead (_XAI_SPEECH_TAG_RE)
        realtime=False,
        languages="*",
        latency_estimate_ms=None,
        priority=40,
    ),
    "mistral": SpeechCapabilities(
        local=False,
        streaming=False,  # not verified
        voice_clone=False,
        multiple_voices=True,  # not verified, conservative
        ssml=False,
        realtime=False,
        languages="*",
        latency_estimate_ms=None,
        priority=50,
    ),
    "gemini": SpeechCapabilities(
        local=False,
        streaming=False,  # not verified
        voice_clone=False,
        multiple_voices=True,  # module docstring: "30 prebuilt voices"
        ssml=False,
        realtime=False,
        languages="*",
        latency_estimate_ms=None,
        priority=50,
    ),
    "neutts": SpeechCapabilities(
        local=True,
        streaming=False,
        voice_clone=True,  # config.yaml tts.neutts.ref_audio/ref_text = zero-shot voice cloning
        multiple_voices=False,  # one reference voice configured at a time
        ssml=False,
        realtime=False,
        languages="en",  # not verified beyond English; conservative
        latency_estimate_ms=None,  # docs: "Depends on CPU/GPU"
        priority=60,
    ),
    "kittentts": SpeechCapabilities(
        local=True,
        streaming=False,
        voice_clone=False,
        multiple_voices=False,  # not documented; 25MB model, conservative default
        ssml=False,
        realtime=False,
        languages="en",  # not verified, conservative
        latency_estimate_ms=None,
        priority=70,
    ),
    "piper": SpeechCapabilities(
        local=True,
        streaming=False,
        voice_clone=False,
        multiple_voices=True,  # module docstring: "44 languages" implies many voice models
        ssml=False,
        realtime=False,
        languages="*",  # 44 languages per module docstring
        latency_estimate_ms=None,
        priority=70,
    ),
}


def get_builtin_capabilities(provider: str) -> Optional[SpeechCapabilities]:
    """Return capabilities for a built-in provider name, or None if unknown.

    Command/HTTP-generic and plugin providers are out of scope for this
    static registry -- their capabilities come from config
    (``tts.providers.<name>.*``) or, later, a live ``GET /capabilities``
    probe, not from this hardcoded table.
    """
    return BUILTIN_CAPABILITIES.get(provider)
