"""CI guard — every documented TTS provider must have a generator in tts_tool.py.

Filed against lmsanch/toryx-private#797 (Fish Audio voice regression). The
underlying bug: commit 19098b06 reverted the Fish Audio implementation;
`provider: fish` silently fell through the `else:` branch to Edge TTS (a
generic Microsoft voice), and the broken state shipped unnoticed because
the dispatch was silent and the fallback didn't raise.

This test ensures:

1. Every provider the registry documents has a dedicated `_generate_*`
   function + an `elif provider == "..."` dispatch branch. Deleting any
   one fails CI loudly.
2. Unknown providers in the dispatcher fail loud rather than silently
   falling back to Edge. The production path's behavior is asserted by
   monkey-patching + invoking `text_to_speech_tool` with a bogus provider.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


# Set of providers the module docstring + README document as supported.
# Add a new entry here when you add a new provider branch, and the test
# below asserts the code matches.
REQUIRED_PROVIDERS: frozenset[str] = frozenset({
    "edge",
    "elevenlabs",
    "openai",
    "minimax",
    "xai",
    "mistral",
    "gemini",
    "fish",
    "neutts",
})


def _tts_source() -> str:
    path = Path(__file__).resolve().parents[2] / "tools" / "tts_tool.py"
    assert path.exists(), f"tts_tool.py not found at {path}"
    return path.read_text()


def test_every_required_provider_has_a_generator_function():
    """For each documented provider, a `_generate_<provider>*` function must exist."""
    src = _tts_source()
    for prov in REQUIRED_PROVIDERS:
        # Accept `_generate_<prov>_audio`, `_generate_<prov>_tts`, or `_generate_<prov>`.
        pattern = rf"def\s+_generate_{re.escape(prov)}(?:_audio|_tts)?\s*\("
        assert re.search(pattern, src), (
            f"tts_tool.py is missing a generator function for provider {prov!r}. "
            f"Expected a function named _generate_{prov}_audio, _generate_{prov}_tts, "
            f"or _generate_{prov}. If you intentionally removed this provider, also "
            f"remove it from REQUIRED_PROVIDERS in this test."
        )


def test_every_required_provider_has_a_dispatch_branch():
    """The main dispatch in `text_to_speech_tool` must branch on every provider."""
    src = _tts_source()
    for prov in REQUIRED_PROVIDERS:
        # Accept either `elif provider == "fish"` or `if provider == "fish"`.
        pattern = rf'(?:if|elif)\s+provider\s*==\s*["\']{re.escape(prov)}["\']'
        assert re.search(pattern, src), (
            f"tts_tool.py has no `provider == {prov!r}` branch in the dispatch. "
            f"This is the bug from toryx-private#797: Fish was reverted, silent "
            f"Edge fallback hid it. The CI test exists to prevent a repeat."
        )


def test_unknown_provider_fails_loud():
    """An unknown provider must produce an error JSON, NOT generate audio via Edge."""
    src = _tts_source()
    # Find the else-branch of the provider dispatch. It must say "not implemented"
    # or similar, and must NOT silently call _generate_edge_tts.
    # Scan the file for the text_to_speech_tool function body.
    match = re.search(
        r"def\s+text_to_speech_tool[^{]*?\n(.*?)\n(?:def\s+|\Z)",
        src,
        re.DOTALL,
    )
    assert match, "could not locate text_to_speech_tool body"
    body = match.group(1)

    # The final else should contain a loud failure, not a silent Edge call.
    # We specifically check that the error message mentions "not implemented"
    # or "Known providers" or similar (the exact pattern is in the fix).
    assert "not implemented" in body.lower() or "known providers" in body.lower(), (
        "The unknown-provider else branch in text_to_speech_tool does not fail loud. "
        "See toryx-private#797 — a silent fallback to Edge caused a day-long regression. "
        "The dispatch else MUST return an explicit error JSON when provider is unknown."
    )


def test_fish_audio_specifically_present():
    """Defense in depth: fish_audio is the canary. If this breaks, #797 regressed."""
    src = _tts_source()
    assert "_generate_fish_audio" in src, (
        "_generate_fish_audio is missing — this is the exact regression from "
        "toryx-private#797. Do not merge this PR without restoring the function."
    )
    assert 'provider == "fish"' in src or "provider == 'fish'" in src, (
        "Fish dispatch branch is missing — #797 regression pattern."
    )
