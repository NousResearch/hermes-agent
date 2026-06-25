"""Lazy-install / import guards for Sakana AI-Scientist runtime deps (aider, etc.)."""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)

LAZY_FEATURE = "tool.ai_scientist"
AIDER_SPEC = "aider-chat==0.86.2"


def aider_importable() -> bool:
    """Return True when the ``aider`` module from ``aider-chat`` is importable."""
    return importlib.util.find_spec("aider.coders") is not None


def _ensure_aider(*, prompt: bool) -> None:
    """Install aider-chat into the active venv (Hermes-pinned overrides apply via pyproject)."""
    if aider_importable():
        return

    from tools.lazy_deps import FeatureUnavailable, _is_satisfied, _venv_pip_install

    if _is_satisfied(AIDER_SPEC) and not aider_importable():
        # Broken partial install — reinstall with dependencies.
        logger.warning("aider-chat metadata present but import failed; reinstalling")

    from tools.lazy_deps import _allow_lazy_installs

    if not _allow_lazy_installs():
        raise FeatureUnavailable(
            LAZY_FEATURE,
            (AIDER_SPEC,),
            "lazy installs disabled (security.allow_lazy_installs=false)",
        )

    logger.info("Lazy-installing %s for AI-Scientist aider loop", AIDER_SPEC)
    result = _venv_pip_install((AIDER_SPEC,))
    if not result.success:
        snippet = (result.stderr or result.stdout or "").strip()[-2000:]
        raise FeatureUnavailable(
            LAZY_FEATURE,
            (AIDER_SPEC,),
            f"aider install failed: {snippet or 'no error output'}",
        )
    if not aider_importable():
        raise FeatureUnavailable(
            LAZY_FEATURE,
            (AIDER_SPEC,),
            "aider-chat installed but `aider.coders` is not importable",
        )


def ensure_ai_scientist_deps(*, prompt: bool = False) -> None:
    """Install AI-Scientist runtime packages (backoff, aider-chat, …) if missing."""
    from tools.lazy_deps import ensure

    ensure(LAZY_FEATURE, prompt=prompt)
    _ensure_aider(prompt=prompt)


def ai_scientist_runtime_ready() -> bool:
    """Non-installing check: vendor tree + aider import surface."""
    try:
        from tools.ai_scientist_tool import AI_SCIENTIST_ENTRYPOINT, AI_SCIENTIST_LAUNCHER

        if not AI_SCIENTIST_ENTRYPOINT.is_file() or not AI_SCIENTIST_LAUNCHER.is_file():
            return False
    except Exception:
        return False
    return aider_importable()
