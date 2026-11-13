"""Shared CAPTCHA / access-restriction classifier for the browser automation
tools (``tools/browser_tool.py``, and eventually ``tools/browser_camofox.py``).

Single source of truth for "what kind of blocked page is this" so every
browser call site (navigate, snapshot, vision) can reuse the same signals
instead of re-inventing a keyword list per call site. See
``hermes-data/skills/hermes-agent/`` for the agent-facing decision logic that
consumes this.

Classifies a page that DID load into one of:
    "normal"        - nothing unusual.
    "captcha"       - an active verification challenge is blocking the
                      content: a CAPTCHA widget (reCAPTCHA/hCaptcha/
                      Turnstile/FunCAPTCHA/GeeTest/AWS WAF), or a generic
                      "verify you're human" / anti-bot interstitial.
    "access_denied" - the site refused the request for a reason that is NOT
                      a CAPTCHA: a 403/forbidden page, a WAF block page with
                      no challenge widget, geofencing, etc.

Deliberately does NOT handle network/transport failures (DNS, connection
refused, timeout) -- those never reach this module. ``browser_navigate``
already returns ``{"success": False, "error": ...}`` for those before any
page content exists to classify. This module only runs on pages that
actually rendered.

PILOT SCOPE (Phase 1, 2026-11-12): wired into ``browser_navigate`` only.
``browser_snapshot`` / ``browser_vision`` call sites are Phase 2.

Detection signals, strongest to weakest:
  1. Iframe host/title from the CDP supervisor's ``frame_tree`` (see
     ``tools/browser_supervisor.py`` FrameInfo) -- only present when a CDP
     endpoint is attached (Browserbase, ``/browser connect``, or
     ``browser.cdp_url`` config). Not available on the default local
     Playwright backend or Camofox.
  2. Page title keyword match (always available).
  3. Visible accessibility-tree text keyword match (ariaSnapshot -- always
     available, catches challenges that don't change the <title>).
  4. Ambiguous generic "blocked"/"bot detected" wording alone, with no
     captcha-specific corroboration -> classified "access_denied" at low
     confidence, NOT "captcha". This is the explicit fix for "do not
     classify every access-denied page as a CAPTCHA."
"""

from typing import Any, Dict, List, Optional

# Known CAPTCHA/challenge iframe hosts, keyed by provider label used as
# `provider_hint` in the result (maps loosely to NopeCHA recognition `type`
# values, but kept provider-neutral -- this module knows nothing about
# NopeCHA).
CAPTCHA_IFRAME_HOST_PATTERNS: Dict[str, List[str]] = {
    "recaptcha": ["google.com/recaptcha", "recaptcha.net", "gstatic.com/recaptcha"],
    "hcaptcha": ["hcaptcha.com"],
    "turnstile": ["challenges.cloudflare.com"],
    "funcaptcha": ["funcaptcha.com", "arkoselabs.com"],
    "geetest": ["geetest.com"],
    "awswaf": ["captcha-sdk.awswaf.com", "aws-waf"],
    "lemincaptcha": ["leminnow.com", "lemin.ai"],
}

# Explicit CAPTCHA / verification-challenge wording. If this matches, there
# is an active challenge to solve (vs. a flat refusal).
CAPTCHA_KEYWORDS: List[str] = [
    "captcha", "are you a robot", "i'm not a robot", "i am not a robot",
    "verify you are human", "verify that you are human", "prove you are human",
    "verification required", "please verify", "please complete the security check",
    "checking your browser", "just a moment", "attention required",
    "ddos protection", "select all images", "click the checkbox",
]

# Flat refusal wording with NO challenge implied -- access denied, not captcha.
ACCESS_DENIED_KEYWORDS: List[str] = [
    "access denied", "access to this page has been denied", "403 forbidden",
    "forbidden", "you don't have permission", "not available in your country",
    "this content is not available", "request blocked",
]

# Ambiguous on their own -- could be either. Only used as a last resort, and
# resolved to "access_denied" at low confidence rather than guessed as
# "captcha", per the explicit requirement not to over-classify.
AMBIGUOUS_KEYWORDS: List[str] = ["blocked", "bot detected", "unusual traffic"]


def _iter_frames(frame_tree: Optional[Dict[str, Any]]):
    if not frame_tree:
        return
    top = frame_tree.get("top")
    if top:
        yield top
    for child in frame_tree.get("children", []) or []:
        yield child


def _match_any(haystack: str, needles: List[str]) -> List[str]:
    return [n for n in needles if n in haystack]


def classify_page(
    title: str = "",
    url: str = "",
    snapshot_text: Optional[str] = None,
    frame_tree: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify a loaded page as normal / captcha / access_denied.

    Args:
        title: Page <title> (from browser_navigate's nav result).
        url: Final page URL (post-redirect), for logging/context only.
        snapshot_text: Accessibility-tree text (ariaSnapshot), if available.
        frame_tree: The ``frame_tree`` dict from browser_snapshot/supervisor,
            if a CDP endpoint is attached. ``{"top": {...}|None,
            "children": [...], "truncated": bool}``, entries shaped like
            ``FrameInfo.to_dict()`` (url, name, origin, frame_id, is_oopif).

    Returns:
        {"classification": "normal"|"captcha"|"access_denied",
         "confidence": "high"|"medium"|"low",
         "provider_hint": str|None,
         "signals": [str, ...]}
    """
    title_l = (title or "").lower()
    text_l = (snapshot_text or "").lower()
    signals: List[str] = []

    # 1. Iframe-based detection (strongest -- an actual challenge widget is
    #    embedded on the page).
    for frame in _iter_frames(frame_tree):
        furl = str(frame.get("url") or "").lower()
        fname = str(frame.get("name") or "").lower()
        if not furl and not fname:
            continue
        for provider, hosts in CAPTCHA_IFRAME_HOST_PATTERNS.items():
            if any(h in furl for h in hosts):
                signals.append(f"iframe url matched {provider}: {furl!r}")
                return {
                    "classification": "captcha",
                    "confidence": "high",
                    "provider_hint": provider,
                    "signals": signals,
                }
        if any(p in fname for p in ("captcha", "verify", "challenge")):
            signals.append(f"iframe title suggests a challenge: {fname!r}")
            return {
                "classification": "captcha",
                "confidence": "medium",
                "provider_hint": None,
                "signals": signals,
            }

    # 2. Explicit CAPTCHA wording in title or visible text.
    title_captcha_hits = _match_any(title_l, CAPTCHA_KEYWORDS)
    text_captcha_hits = _match_any(text_l, CAPTCHA_KEYWORDS)
    if title_captcha_hits or text_captcha_hits:
        signals.append(f"captcha keywords in title: {title_captcha_hits}" if title_captcha_hits else "")
        signals.append(f"captcha keywords in page text: {text_captcha_hits}" if text_captcha_hits else "")
        signals = [s for s in signals if s]
        return {
            "classification": "captcha",
            "confidence": "high" if title_captcha_hits else "medium",
            "provider_hint": None,
            "signals": signals,
        }

    # 3. Flat access-denied wording (no challenge implied).
    denied_hits = _match_any(title_l, ACCESS_DENIED_KEYWORDS) + _match_any(text_l, ACCESS_DENIED_KEYWORDS)
    if denied_hits:
        signals.append(f"access-denied keywords matched: {denied_hits}")
        return {
            "classification": "access_denied",
            "confidence": "high",
            "provider_hint": None,
            "signals": signals,
        }

    # 4. Ambiguous wording alone -- treat as access_denied at low confidence,
    #    NOT captcha. This is the deliberate fix for over-classification.
    ambiguous_hits = _match_any(title_l, AMBIGUOUS_KEYWORDS)
    if ambiguous_hits:
        signals.append(
            f"ambiguous block wording with no captcha iframe/keyword corroboration: {ambiguous_hits}"
        )
        return {
            "classification": "access_denied",
            "confidence": "low",
            "provider_hint": None,
            "signals": signals,
        }

    return {"classification": "normal", "confidence": "high", "provider_hint": None, "signals": []}
