"""The env_exfil_* loopback exemption, checked for RE2 without a lookahead, against its lookahead original.

On RE2 the exemption is a matcher (``tools.skills_guard._LoopbackExemptMatcher``) that drops a command match
when a loopback URL starts after it on the same line. The original stdlib patterns below are the reference: on
every generated line the facade must flag what the original flags (a security rule never matches
less, so the exemption never covers more) and nothing else (no new false positives). The corpus is every
sequence of up to three tokens that open, break, restart, complete or nearly complete a loopback URL around
a secret word, behind each rule's command, plus a seeded sample of longer sequences.
"""
import itertools
import random
import re

import pytest

from tools.skills_guard import _COMPILED_THREAT_PATTERNS

_LOOPBACK = r'(?![^\n]*https?://(?:localhost|127\.0\.0\.1|\[::1\]))'
# The patterns as they were before the RE2 rewrite (stdlib lookahead spelling), compiled as the table is.
_ORIGINAL = {
    "env_exfil_curl": r'curl\s+' + _LOOPBACK + r'[^\n]*\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b',
    "env_exfil_wget": r'wget\s+' + _LOOPBACK + r'[^\n]*\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b',
    "env_exfil_fetch": r'fetch\s*\(' + _LOOPBACK + r'[^\n]*\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD)S?\b',
    "env_exfil_httpx": r'httpx?\.(get|post|put|patch)\s*\(' + _LOOPBACK + r'[^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
    "env_exfil_requests": r'requests\.(get|post|put|patch)\s*\(' + _LOOPBACK + r'[^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
}
_COMMANDS = {
    "env_exfil_curl": "curl ", "env_exfil_wget": "wget ", "env_exfil_fetch": "fetch(",
    "env_exfil_httpx": "httpx.post(", "env_exfil_requests": "requests.get(",
}
# Every prefix of each loopback host's URL, the full URLs, near misses, and the secret shapes.
_URL_PREFIXES = sorted({url[:n] for url in ("https://localhost", "http://127.0.0.1", "https://[::1]")
                        for n in range(1, len(url) + 1)})
_TOKENS = [*_URL_PREFIXES, "localhost", "LOCALHOST", "HTTP://LOCALHOST", "localhos", "evil.com/?u=", "x", " ",
           "\n", "/", "$API_KEY", "${GITHUB_TOKEN}", "$SECRETS", "TOKEN", "KEY", "PASSWORD", "$CREDENTIAL", "s",
           "tps://localht", "\u017f"]


def _corpus():
    lines = ["".join(seq) for n in range(1, 4) for seq in itertools.product(_TOKENS, repeat=n)]
    rng = random.Random(20261123)
    lines += ["".join(rng.choices(_TOKENS, k=rng.randint(4, 9))) for _ in range(3000)]
    return lines


_CORPUS = _corpus()


@pytest.mark.parametrize("pid", sorted(_ORIGINAL))
def test_loopback_exemption_matches_its_lookahead_original(pid):
    original = re.compile(_ORIGINAL[pid], re.IGNORECASE)
    rewritten = next(compiled for compiled, rule, *_ in _COMPILED_THREAT_PATTERNS if rule == pid)
    less, more = [], []
    for tail in _CORPUS:
        for head in ("", "x "):
            text = head + _COMMANDS[pid] + tail
            was, now = bool(original.search(text)), bool(rewritten.search(text))
            if was and not now:
                less.append(text)
            elif now and not was:
                more.append(text)
    assert less == [], f"{pid} misses lines the original flags: {less[:5]!r}"
    assert more == [], f"{pid} flags lines the original did not: {more[:5]!r}"
