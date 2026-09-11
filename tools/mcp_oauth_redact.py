"""OAuth query credentials must be scrubbed before any log handler sees them."""
import re

_KEYS = "|".join("".join(f"(?:{char}|%{ord(char):02x})" for char in key)
                 for key in ("code", "state", "error", "error_description"))
_PARAM = re.compile(r"(?i)((?:^|&amp;|[?&#\s\"']|%3f|%26|%23)(?:" + _KEYS + r")(?:=|%3d))([^\s&#\"'<>]*)")
_CANDIDATE = re.compile(r"(?i)(?:^|&amp;|[?&#]|%3f|%26|%23)(?:" + _KEYS + r")(?:=|%3d)")


def contains_oauth_parameters(text):
    return bool(_CANDIDATE.search(text))


def redact_oauth_log(text):
    return _PARAM.sub(r"\1[REDACTED]", text)
