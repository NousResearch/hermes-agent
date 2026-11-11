"""Real-profile local browsing: snapshot the user's default Chromium profile into a Hermes-owned copy and
launch the user's REAL browser binary on it through the launcher.

The real binary (not the packaged Chromium, and never ``--password-store=basic``/``--use-mock-keychain``)
keeps the OS keychain path intact, so macOS cookies decrypt and the copy stays signed in. One shared
browser per profile home serves every session of that profile (concurrent tasks must not launch rival
Chromiums on one copy dir); it is terminated at exit or when THAT profile turns consent off.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_RP = "browser.use_real_profile is on, but "
# Keep lock contention well inside the agent's 420-second outer tool deadline (an abandoned holder
# may be a daemon worker Python cannot terminate).
_LOCK_TIMEOUT_S = 30.0
# One launch lock per profile home (the real-profile browser and its copy dir are per home): a slow or
# abandoned launch in one profile must never make another wait, then fall back to a signed-out browser.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
# Extra flags on top of the launcher's base set: no hang monitor / popup blocking / repost prompts on an
# unattended browser driving the user's own sessions.
_REAL_PROFILE_EXTRA_FLAGS = ("--disable-hang-monitor", "--disable-popup-blocking", "--disable-prompt-on-repost")


def real_profile_fallback_note(err: str) -> str:
    """Tool-result note when the real profile can't load and a fresh browser serves instead: the browser keeps
    working (a consented user is never left with no browser), and the model is told it is signed out."""
    return (f"{err} Using a fresh browser without the user's logins for this session instead; "
            "tell the user if the task needs their accounts.")


def _unsupported_reason(browser) -> Optional[str]:
    """Why the default browser's profile can't be used, else None. A pre-release channel lives in a profile
    dir we don't resolve; normalizing it to the stable family would drive a DIFFERENT account."""
    from hermes_cli.browser_connect import UNSUPPORTED_CHANNEL
    if browser is None:
        return (_RP + "your default browser is not a supported Chromium browser (Chrome, Edge, Brave, "
                "Brave Origin, Chromium). Real-profile browsing requires a Chromium default; set one or turn the toggle off.")
    if browser == UNSUPPORTED_CHANNEL:
        return (_RP + "your default browser is a pre-release Chromium channel (Beta / Dev / Canary), which "
                "real-profile browsing does not support. Set your default to a "
                "stable Chrome / Edge / Brave / Brave Origin / Chromium, or turn the toggle off.")
    return None


def _snapshot_error(err: str) -> str:
    """A locked profile adds the approved-close command, which the agent must ASK the user about first."""
    from hermes_cli.browser_connect import _PROFILE_LOCKED_PREFIX
    if err and err.startswith(_PROFILE_LOCKED_PREFIX):
        return (err[len(_PROFILE_LOCKED_PREFIX):] + " To close it (only after the user approves — it "
                "quits their browser and loses unsaved tabs), run: `hermes browser close-profile`; the next "
                "browser session then uses the profile.")
    return f"{_RP}{err}"


def _home_lock() -> threading.Lock:
    from hermes_constants import hermes_home_key
    with _locks_guard:
        return _locks.setdefault(hermes_home_key(), threading.Lock())


def real_profile_key() -> str:
    """Launcher key of the CURRENT profile's real-profile browser. Per home: one process serves many profiles,
    so a consent-off profile must never stop another's browser, nor a consenting one reuse its cookie jar."""
    from hermes_constants import hermes_home_key
    return f"hermes-real-profile@{hermes_home_key()}"


def terminate_real_profile_browser() -> None:
    """Stop this profile's real-profile browser if this process launched it (idempotent)."""
    from tools.browser_launcher import stop_browser
    stop_browser(real_profile_key())


def _consent_revoked() -> tuple[None, None]:
    """Consent off: stop the copy-browser and delete the snapshot store (copies of cookies/logins), so
    revoking consent actually removes the credential copies."""
    terminate_real_profile_browser()
    try:
        from hermes_cli.browser_connect import cleanup_real_profile_snapshots
        cleanup_real_profile_snapshots()
    except OSError as exc:
        logger.debug("real-profile cleanup-on-consent-off failed: %s", exc)
    return None, None


def _launch(browser_name: str) -> tuple[Optional[str], Optional[str]]:
    from hermes_cli.browser_connect import chromium_executable, real_profile_copy_dir, snapshot_real_profile
    from tools.browser_launcher import launch_chromium
    from tools.browser_session import ensure_screen_for_headed_chromium
    real_binary = chromium_executable(browser_name)
    if real_binary is None:
        return None, f"{_RP}the real browser binary for '{browser_name}' could not be found. Reinstall it or turn the toggle off."
    snapshot_failure: list[str] = []

    def overlay() -> Optional[str]:
        # The snapshot overlay rewrites Cookies / Login Data, so it runs inside the launcher's cross-process
        # step on the copy dir: never beside another process's copy, never under a live copy-browser (maybe
        # from an earlier Hermes process; the launcher ATTACHES to that one instead).
        copied, err = snapshot_real_profile(browser_name)
        if err or not copied:
            snapshot_failure.append(_snapshot_error(err))
            return snapshot_failure[0]
        return None

    ensure_screen_for_headed_chromium()
    launched, err = launch_chromium(real_profile_key(), executable=real_binary,
                                    user_data_dir=real_profile_copy_dir(browser_name), ephemeral=False,
                                    extra_args=_REAL_PROFILE_EXTRA_FLAGS, timeout=30.0, sandbox_bypass=False,
                                    prepare=overlay)
    if launched is None:
        return None, snapshot_failure[0] if snapshot_failure else f"{_RP}the real-profile browser failed to start: {err}"
    logger.info("real-profile browser ready for %s at %s (%s)", browser_name, launched.http_url, launched.user_data_dir)
    return launched.cdp_url, None


def real_profile_cdp() -> tuple[Optional[str], Optional[str]]:
    """``(cdp_url, error)`` for consented real-profile browsing.

    Snapshot → launch the real binary on the copy (or attach to the one already running) → its CDP URL.
    ``(None, message)`` when it can't (callers fall back to a fresh browser and report the message);
    ``(None, None)`` when consent is off."""
    from tools.browser_session_cloud import use_real_profile
    if not use_real_profile():
        return _consent_revoked()
    # Lightpanda cannot load a Chromium profile; checked before default-browser detection so a host with no
    # Chromium default still reports the actionable engine conflict.
    from tools.browser_lightpanda import using_lightpanda_engine
    if using_lightpanda_engine():
        return None, (_RP + "browser.engine is set to 'lightpanda', which cannot load a real Chromium profile. "
                      "Set browser.engine to 'auto' to use real-profile browsing, or turn the toggle off.")
    lock = _home_lock()
    if not lock.acquire(timeout=_LOCK_TIMEOUT_S):
        return None, (_RP + "the real-profile browser is already being prepared by another call that has not "
                      "finished. Retry after that call completes, or restart Hermes if it was abandoned.")
    try:
        from tools.browser_launcher import get_browser
        running = get_browser(real_profile_key())
        if running is not None:
            return running.cdp_url, None
        from hermes_cli.browser_connect import detect_default_chromium
        browser_name = detect_default_chromium()
        unsupported = _unsupported_reason(browser_name)
        if unsupported:
            return None, unsupported
        return _launch(browser_name)
    finally:
        lock.release()
