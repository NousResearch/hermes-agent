"""Optional LLM pre-rewrite of TTS text (spoken-prose conversion before chunking).

Enabled via ``tts.rewrite.enabled``. The rewrite call routes through Hermes'
built-in auxiliary-LLM stack (``agent.auxiliary_client.call_llm``), so any
provider configured in Hermes works - not just OpenAI-compatible endpoints.
Fail-open by design: any error returns the original text unchanged so synthesis
never blocks. Routing, most specific wins: a ``tts.rewrite.model`` pin (+
optional ``base_url``/``api_key_env``), else ``auxiliary.tts_rewrite.*`` task
config, else auto-route to the main provider/model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Generic default prompts: neutral spoken-prose conversion, no persona, no
# paralinguistic tags. Voice personality is deployment-specific and lives outside
# this file: override via ``tts.rewrite.system_prompt``/``prompt_file`` (or the
# interim twins), else a ``tts_rewrite_prompt.txt`` in the Hermes home, else these.
_GENERIC_REWRITE_SYSTEM_PROMPT = (
    "Rewrite this assistant reply as conversational spoken prose for a TTS voice. "
    "This is re-speaking, not summarizing: keep every point, fact, name and number - "
    "do not drop information. Convert bullets into flowing prose; trim only pure repetition. "
    "LENGTH IS A HARD CONSTRAINT: your output must be at most as long as the input in "
    "characters, ideally slightly under; the more information-dense the input, the leaner "
    "the phrasing must be. Every sentence must add a point, not cushion one. "
    "Match your tone to the content's own mood without exaggerating it. "
    "Clean it for speech: no markdown, no backticks, no brackets or parentheses, spell out "
    "symbols and abbreviations (forty percent, degrees, version dots), and end every "
    "sentence with . ? or !. Output ONLY the rewritten text."
)

_GENERIC_INTERIM_REWRITE_SYSTEM_PROMPT = (
    "Rewrite this mid-turn status update as conversational spoken prose for a TTS voice. "
    "Keep every point and fact, but the update is narrated between tool calls, so brevity "
    "wins - SPEED IS A HARD CONSTRAINT: your output must be at most HALF the input's length "
    "in characters; compress phrasing aggressively, merge sentences, drop pure repetition. "
    "Every sentence must add a point, not cushion one. "
    "Match your tone to the content's own mood without exaggerating it. "
    "Clean it for speech: no markdown, no backticks, no brackets or parentheses, spell out "
    "symbols and abbreviations (forty percent, degrees, version dots), and end every "
    "sentence with . ? or !. Output ONLY the rewritten text."
)

_PROMPT_FILE_MAIN = "tts_rewrite_prompt.txt"
_PROMPT_FILE_INTERIM = "tts_rewrite_interim_prompt.txt"

# Auxiliary-LLM task key: pins provider/model via ``auxiliary.tts_rewrite.*`` config.
_REWRITE_TASK = "tts_rewrite"

_DEFAULT_MIN_CHARS = 200
_DEFAULT_TIMEOUT = 45.0
_DEFAULT_TIMEOUT_INTERIM = 15.0
# Final replies retry once on an implausible answer; interims take a single attempt.
_REWRITE_ATTEMPTS = 2


def _read_prompt_file(path: str) -> str:
    """Read a prompt override file; empty string when missing/unreadable."""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _resolve_system_prompt(cfg: dict, *, interim: bool) -> str:
    """Prompt ladder: inline config -> config file path -> hermes-home file -> generic."""
    inline = str(cfg.get("interim_system_prompt" if interim else "system_prompt") or "").strip()
    if inline:
        return inline
    configured = str(cfg.get("interim_prompt_file" if interim else "prompt_file") or "").strip()
    default_name = _PROMPT_FILE_INTERIM if interim else _PROMPT_FILE_MAIN
    for candidate in ([configured] if configured else []) + [str(get_hermes_home() / default_name)]:
        if not candidate:
            continue
        text = _read_prompt_file(candidate)
        if text:
            return text
    return _GENERIC_INTERIM_REWRITE_SYSTEM_PROMPT if interim else _GENERIC_REWRITE_SYSTEM_PROMPT

# Output-size gate: a rewrite outside these bounds is rejected and the original text
# is spoken instead (0.2x = degenerate/empty output; 2.0x admits the legitimate
# bullets-to-prose expansion of dense inputs, ~1.2x typical).
_MIN_RATIO, _MAX_RATIO = 0.2, 2.0


def _rewrite_config() -> dict:
    """Return the ``tts.rewrite`` config section ({} when unavailable)."""
    try:
        from hermes_cli.config import load_config
        section = (load_config().get("tts") or {}).get("rewrite") or {}
        return section if isinstance(section, dict) else {}
    except Exception as e:
        logger.debug("tts.rewrite config unavailable: %s", e)
        return {}


def rewrite_text_for_speech(text: str, *, interim: bool = False) -> str:
    """Rewrite *text* into TTS-friendly spoken prose; never breaks a synthesis.

    Args:
        text: Reply text to rewrite.
        interim: Use the speed-first variant (half-length target, one attempt, no
            retries, shorter timeout) - mid-turn commentary is spoken live, so a slow
            rewrite is skipped, not waited on.

    Returns:
        The rewritten text, or the original when disabled, too short, failed, or
        implausible (size-gated).
    """
    cfg = _rewrite_config()
    if not cfg.get("enabled"):
        return text
    # Falsy values fall back to the default (min_chars: 0 is a no-op by design).
    min_chars = int(cfg.get("interim_min_chars" if interim else "min_chars",
                            _DEFAULT_MIN_CHARS) or _DEFAULT_MIN_CHARS)
    if not text or len(text) < min_chars:
        return text
    default_timeout = _DEFAULT_TIMEOUT_INTERIM if interim else _DEFAULT_TIMEOUT
    timeout = float(
        cfg.get("interim_timeout" if interim else "timeout",
                default_timeout) or default_timeout)
    system_prompt = _resolve_system_prompt(cfg, interim=interim)
    # Resolve the route once up front: the label also names the route in failure
    # logs, and the pin passes through as explicit args to the aux resolver.
    try:
        model, base_url, api_key, route = _resolve_rewrite_route(cfg)
    except Exception:
        model, base_url, api_key, route = None, None, None, "auto"
    rewritten = None
    try:
        for _attempt in range(1 if interim else _REWRITE_ATTEMPTS):
            rewritten, route = _request_rewrite(
                text, timeout, system_prompt, model=model, base_url=base_url,
                api_key=api_key, interim=interim)
            # gpt-oss misses the size band in both directions and can return
            # empty content; retries empirically land inside the band. Interim
            # takes no retry: one fast attempt, fail-open to raw commentary.
            if _plausible_rewrite(text, rewritten):
                break
            rewritten = None
    except Exception as e:
        logger.warning("TTS rewrite via %s failed (%s); keeping original text", route, e)
        return text
    if not rewritten or not _plausible_rewrite(text, rewritten):
        logger.warning(
            "TTS rewrite via %s rejected (implausible output: %d chars for %d input, ratio %.2f); keeping original text",
            route, len(rewritten or ""), len(text),
            len((rewritten or "").strip()) / max(len(text), 1))
        return text
    logger.info("TTS text rewritten via %s (%d -> %d chars)", route, len(text), len(rewritten))
    return rewritten


def _resolve_rewrite_key(env_var: str) -> str:
    """API key via the shared scope-aware resolver; empty when unset. Never raises."""
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
        return str(resolve_provider_secret(env_var, "") or "").strip()
    except Exception:
        import os
        return (os.getenv(env_var, "") or "").strip()


def _plausible_rewrite(original: str, rewritten: Optional[str]) -> bool:
    """Cheap output gate: non-empty, no code fences, sane length vs the input."""
    stripped = (rewritten or "").strip()
    if not stripped or stripped.count("```"):
        return False
    input_len = len(original) or 1
    return _MIN_RATIO * input_len <= len(stripped) <= _MAX_RATIO * input_len


def _resolve_rewrite_route(cfg: dict) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Resolve the rewrite route; most specific wins.

    Args:
        cfg: The ``tts.rewrite`` config section.

    Returns:
        ``(model, base_url, api_key, label)``. A ``tts.rewrite.model`` pin returns its
        model as the label; otherwise ``(None, None, None, "auto")`` and ``call_llm``
        applies ``auxiliary.tts_rewrite.*`` task config, else auto-routes. An explicit
        ``base_url`` without ``model`` is ignored (legacy configs set both together).
    """
    model = str(cfg.get("model") or "").strip() or None
    base_url = str(cfg.get("base_url") or "").strip() or None
    env_var = str(cfg.get("api_key_env") or "").strip() or None
    if model:
        api_key = _resolve_rewrite_key(env_var) if env_var else None
        return model, base_url, api_key, model
    return None, None, None, "auto"


def _request_rewrite(text: str, timeout: float, system_prompt: str, *,
                     interim: bool = False, model: Optional[str] = None,
                     base_url: Optional[str] = None,
                     api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """One auxiliary-LLM rewrite call.

    Args:
        text: Text to rewrite.
        timeout: Request timeout in seconds.
        system_prompt: Spoken-prose system prompt.
        interim: Passes ``task=None`` to the aux client, which also skips the fallback
            chain - mid-turn commentary never waits on a secondary provider.
        model: Explicit ``tts.rewrite.model`` pin (outranks ``auxiliary.tts_rewrite.*``
            task config inside the resolver); None auto-routes to the main provider.
        base_url: Optional base URL accompanying the model pin.
        api_key: Optional API key accompanying the model pin.

    Returns:
        ``(content | None, route label)``.
    """
    from agent.auxiliary_client import call_llm
    kwargs: Dict[str, Any] = dict(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.6, max_tokens=3000, timeout=timeout,
        task=None if interim else _REWRITE_TASK)
    if model:
        kwargs.update(model=model, base_url=base_url, api_key=api_key)
    response = call_llm(**kwargs)
    label = model or "auto"
    try:
        content = response.choices[0].message.content
        # Auto-route answers name the served model; prefer it for the log label.
        label = str(getattr(response, "model", None) or label)
    except (AttributeError, IndexError, TypeError):
        return None, label
    return (content.strip() if content else None), label
