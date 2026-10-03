"""Deterministic per-app blocklist for the computer_use tool (inspired by Claude Cowork's app blocklist,
support.claude.com article 14128542: "Prevent Claude from accessing certain apps by adding them to a
blocklist. Any requests ... will be automatically denied", with investment/trading/crypto apps blocked
by default).

Matching is token-sequence based, not substring: an app name and a pattern are both split on
non-alphanumeric runs and a pattern matches only as a consecutive token subsequence. So the default
pattern ``kraken`` blocks the Kraken trading app ("Kraken", "Kraken Pro") but never GitKraken (one
token, ``gitkraken``), and ``ledger live`` blocks "Ledger Live" without touching a "Ledger" column in
some spreadsheet app's window title. Deterministic name matching cannot be argued with by the model —
the denial happens at the tool dispatch choke point before any backend call, like the hard-blocked key
combos in tool.py. Like those, this governs the computer_use surface only (the terminal is a separate,
separately-approved surface).

Config (config.yaml, ``computer_use``):
- ``blocked_apps``: list of extra patterns to deny (same token matching).
- ``unblocked_apps``: list of patterns exempted from the built-in sensitive list (false-positive valve).
- ``block_sensitive_apps``: gate for the built-in list below (default True).
"""

from __future__ import annotations

import contextlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Built-in sensitive-app patterns (gated by computer_use.block_sensitive_apps, default ON).
# Mirrors Cowork's default posture — trading/brokerage + cryptocurrency apps — plus credential vaults,
# where a single misdirected click or paste is unrecoverable. Banking is mostly web, covered by the
# real-profile sensitive-cookie excludes in browser real_profile (#97561).
_DEFAULT_SENSITIVE_PATTERNS: Tuple[str, ...] = (
    # investment / trading / brokerage
    "robinhood", "coinbase", "binance", "kraken", "webull", "etrade", "e trade",
    "interactive brokers", "fidelity investments", "charles schwab", "thinkorswim",
    # cryptocurrency wallets
    "ledger live", "trezor", "exodus", "electrum", "metamask", "sparrow wallet",
    # credential vaults / password managers
    "1password", "bitwarden", "keepass", "keepassxc", "lastpass", "dashlane",
    "keychain access", "seahorse",
)


def _cfg() -> Dict[str, Any]:
    """The ``computer_use`` config block ({} when unreadable) — unreadable config keeps the DEFAULT
    sensitive list active (fail closed), since only explicit config can widen access."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        return (load_config() or {}).get("computer_use") or {}
    return {}


def _tokens(name: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def _matches(app_tokens: Sequence[str], pattern: str) -> bool:
    """True when the pattern's tokens appear as a consecutive subsequence of the app's tokens."""
    pat = _tokens(pattern)
    if not pat or len(pat) > len(app_tokens):
        return False
    return any(list(app_tokens[i:i + len(pat)]) == pat for i in range(len(app_tokens) - len(pat) + 1))


def _str_list(value: Any) -> List[str]:
    return [str(v) for v in value if str(v).strip()] if isinstance(value, (list, tuple)) else []


def blocked_app_match(app_name: Optional[str]) -> Optional[str]:
    """The pattern blocking *app_name*, or None. Unknown/empty names fail open (the sticky-target
    guard in tool.py re-checks once a capture names the target)."""
    app_tokens = _tokens(app_name or "")
    if not app_tokens:
        return None
    cfg = _cfg()
    for pattern in _str_list(cfg.get("blocked_apps")):
        if _matches(app_tokens, pattern):
            return pattern
    if cfg.get("block_sensitive_apps", True):
        exempt = _str_list(cfg.get("unblocked_apps"))
        for pattern in _DEFAULT_SENSITIVE_PATTERNS:
            if _matches(app_tokens, pattern) and not any(_matches(app_tokens, e) for e in exempt):
                return pattern
    return None
