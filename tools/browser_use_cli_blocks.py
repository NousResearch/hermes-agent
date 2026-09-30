"""Bot-wall detection for browser_exec output.

A blocked page usually reaches the model as an ordinary successful call ("Access Denied",
a challenge interstitial, or chrome's error page for a connection the edge dropped), and
agents then report the task as "blocked" and stop. When the output carries a known wall
signature, browser_exec adds a ``blocked`` field naming the vendor plus the configured
``browser.blocked_page_hint`` so the next step (a real-profile or stealth browser) is in
front of the model even when it never loaded the browser skill.
"""

from __future__ import annotations

import re
from typing import Optional

# (label, pattern). Ambiguous error/gesture text needs nearby wall context;
# a network error or a touch-gesture guide alone is not a bot-wall signature.
_BLOCK_SIGNATURES = (
    ("akamai", re.compile(r"Access Denied[\s\S]{0,400}Reference\s*#\s*[\d.a-f]+", re.I)),
    ("akamai", re.compile(
        r"(?:Access Denied|Reference\s*#)[\s\S]{0,400}ERR_HTTP2_PROTOCOL_ERROR"
        r"|ERR_HTTP2_PROTOCOL_ERROR[\s\S]{0,400}(?:Access Denied|Reference\s*#)", re.I,
    )),
    ("perimeterx", re.compile(r"Access to this page has been denied", re.I)),
    ("perimeterx", re.compile(
        r"(?:PerimeterX|human verification)[\s\S]{0,400}Press\s*&\s*Hold"
        r"|Press\s*&\s*Hold[\s\S]{0,400}(?:PerimeterX|human verification)", re.I,
    )),
    ("cloudflare", re.compile(r"Attention Required!\s*\|\s*Cloudflare|cf-chl-|Verify you are human", re.I)),
    ("cloudflare", re.compile(r"'title':\s*'[^']*Just a moment\.\.\.", re.I)),
    ("datadome", re.compile(r"captcha-delivery\.com|geo\.captcha-delivery", re.I)),
    ("imperva", re.compile(r"Pardon Our Interruption|Request unsuccessful\. Incapsula", re.I)),
    ("generic", re.compile(r"'title':\s*'[^']*(403 Forbidden|Request blocked)", re.I)),
)

_SCAN_MAX_CHARS = 20_000

DEFAULT_HINT = (
    "This page is a bot-protection wall, not the site's content. Do not report the task as "
    "blocked yet: retry the same URL in a browser the site trusts — the user's real Chrome "
    "profile (/browser connect) or a stealth cloud browser provider — then continue there."
)


def detect_block(output: str) -> Optional[str]:
    """Vendor label when ``output`` carries a known bot-wall signature, else None."""
    if not output:
        return None
    window = output[:_SCAN_MAX_CHARS]
    for label, pattern in _BLOCK_SIGNATURES:
        if pattern.search(window):
            return label
    return None


def blocked_page_hint(browser_cfg: dict) -> str:
    """``browser.blocked_page_hint`` from config; empty string disables the hint."""
    hint = browser_cfg.get("blocked_page_hint")
    return DEFAULT_HINT if hint is None else str(hint)
