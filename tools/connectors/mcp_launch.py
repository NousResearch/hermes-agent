"""What an MCP consent row will run or contact, worded for the card before the user approves.

Install reads the catalog manifest's transport; enable and authorize read the configured
``mcp_servers`` entry. The line is display-only. It never carries env values or headers, and its
text is scrubbed before it leaves the process: the value of a secret-named flag (``--api-key=…``,
``--token …``, ``-p``/``-k``/``-t`` spaced or attached, an ``Authorization`` header arg), args
shaped like encoded key material (masked whole), URL userinfo, query values and fragment, opaque
URL path segments (hosted MCP URLs often embed the key in the path), then the standard redaction.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Optional

from agent.redact import (
    SECRET_HEADER_NAME_LIST,
    redact_for_egress,
    redact_sensitive_text,
    sanitize_command_for_display,
)

_SECRET_FLAG_RE = re.compile(
    r"^--?[A-Za-z0-9_.-]*(?:key|token|secret|passw(?:or)?d|pass|pwd|auth|credential|bearer|cookie)"
    r"[A-Za-z0-9_.-]*$",
    re.IGNORECASE,
)
# One-letter flags that conventionally take a password, key or token (``mysql -p``, ``-k <key>``),
# in both the spaced (``-p value``) and the attached (``-pvalue``) form.
_SECRET_SHORT_FLAGS = frozenset("pPkt")
# ``Authorization: <scheme> <value>`` and the API-key header family, as one ``-H`` arg.
_SECRET_HEADER_ARG_RE = re.compile(
    rf"^((?:--header=)?(?:Proxy-)?Authorization:\s*(?:[A-Za-z][\w.+-]*\s+)?|(?:--header=)?"
    rf"(?:{'|'.join(SECRET_HEADER_NAME_LIST)})\s*:\s*).+$",
    re.IGNORECASE,
)
# An arg that reads as encoded key material, not a path, package or word: a long base64/url-safe run
# with mixed case and digits, a long letters+digits run, or a long hex run. Masked whole: a
# prefix and suffix of a base64 config blob can be most of a short key.
_BLOB_RE = re.compile(r"[A-Za-z0-9+/_-]{20,}={0,2}")
_HEX_RE = re.compile(r"[0-9a-fA-F]{32,}")


def _is_blob(value: str) -> bool:
    if value.startswith(("/", "-")) or not _BLOB_RE.fullmatch(value):
        return False
    if _HEX_RE.fullmatch(value):
        return True
    has_digit = any(c.isdigit() for c in value)
    mixed = any(c.isupper() for c in value) and any(c.islower() for c in value)
    # A slash-separated run of lowercase words and digits is a path, not base64.
    return has_digit and (mixed or (len(value) >= 32 and "/" not in value)) and any(c.isalpha() for c in value)


def _mask_arg(arg: str) -> tuple[str, bool]:
    """``(shown, mask_next)`` for one argv entry."""
    header = _SECRET_HEADER_ARG_RE.match(arg)
    if header:
        return f"{header.group(1)}***", False
    flag, sep, value = arg.partition("=")
    if _SECRET_FLAG_RE.match(flag):
        return (f"{flag}=***", False) if sep else (arg, True)
    if len(arg) >= 2 and arg[0] == "-" and arg[1] in _SECRET_SHORT_FLAGS:
        return (arg, True) if len(arg) == 2 else (f"{arg[:2]}***", False)
    if sep and _is_blob(value):
        return f"{flag}=***", False
    return ("***" if _is_blob(arg) else arg), False


def _mask_flag_values(args: list[str]) -> list[str]:
    out: list[str] = []
    mask_next = False
    for arg in args:
        # A long flag right after a value-taking flag is the next flag, not its value (``-p --verbose``).
        if mask_next and not arg.startswith("--"):
            out.append("***")
            mask_next = False
            continue
        shown, mask_next = _mask_arg(arg)
        out.append(shown)
    return out


# Tab and newline are the two controls the approval sanitizer keeps; the card draws one line.
_LINE_BREAKS = str.maketrans({"\t": "\\t", "\n": "\\n"})
# Scrubbed first, then cut, so a secret straddling the cut is already masked.
MAX_TARGET_CHARS = 400
_URL_RE = re.compile(r"\b(?:https?|wss?)://[^\s]+", re.IGNORECASE)
# A path segment that reads as a credential, not a route: long and url-safe with letters and digits,
# or shorter but mixed-case with digits (``/v1/Ab3dEf9hIj12/sse``).
_OPAQUE_SEGMENT_RE = re.compile(
    r"(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{20,}"
    r"|(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]{12,}")


def _mask_url(match: re.Match[str]) -> str:
    """Userinfo, every query value and the fragment become ``***``; opaque path segments too. The host,
    path words and query keys stay, so the user still sees where the line connects."""
    url = match.group(0)
    head, sep, rest = url.partition("://")
    rest, hash_, _fragment = rest.partition("#")
    rest, qmark, query = rest.partition("?")
    authority, slash, path = rest.partition("/")
    if "@" in authority:
        authority = "***@" + authority.rpartition("@")[2]
    out = f"{head}{sep}{authority}"
    if slash:
        out += "/" + "/".join("***" if _OPAQUE_SEGMENT_RE.fullmatch(seg) else seg for seg in path.split("/"))
    if qmark:
        out += "?" + "&".join(
            f"{key}=***" if eq else key
            for key, eq, _value in (pair.partition("=") for pair in query.split("&")))
    return out + ("#***" if hash_ else "")


# Tab and newline are the two controls the approval sanitizer keeps; the card draws one line.
_LINE_BREAKS = str.maketrans({"\t": "\\t", "\n": "\\n"})
# Scrubbed first, then cut, so a secret straddling the cut is already masked.
MAX_TARGET_CHARS = 400
_URL_RE = re.compile(r"\b(?:https?|wss?)://[^\s]+", re.IGNORECASE)
# A path segment that reads as a credential, not a route: long and url-safe with letters and digits,
# or shorter but mixed-case with digits (``/v1/Ab3dEf9hIj12/sse``).
_OPAQUE_SEGMENT_RE = re.compile(
    r"(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{20,}"
    r"|(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]{12,}")


def _mask_url_path(match: re.Match[str]) -> str:
    url = match.group(0)
    head, sep, rest = url.partition("://")
    host, slash, path = rest.partition("/")
    if not slash:
        return url
    return f"{head}{sep}{host}/" + "/".join(
        "***" if _OPAQUE_SEGMENT_RE.fullmatch(seg) else seg for seg in path.split("/"))


def _visible(text: str) -> str:
    """Every control, format (bidi overrides/isolates, zero-width, BOM, soft hyphen) and separator
    code point as a visible escape such as ``\\u202e``: the line must read as the argv runs, and the
    user should see that an arg is unusual rather than a cleaned-up version of it."""
    return sanitize_command_for_display(text.translate(_LINE_BREAKS))


def scrub(text: str) -> str:
    text = _URL_RE.sub(_mask_url, " ".join(_visible(text).split()))
    text = redact_for_egress(redact_sensitive_text(text, force=True, redact_url_credentials=True))
    return text if len(text) <= MAX_TARGET_CHARS else text[: MAX_TARGET_CHARS - 1] + "…"


def launch_target(url: Any, command: Any, args: Any) -> Optional[dict[str, str]]:
    """``{"kind": "remote"|"stdio", "target": text}``; the URL wins, as the transport choice does."""
    if isinstance(url, str) and url.strip():
        return {"kind": "remote", "target": scrub(url.strip())}
    if not (isinstance(command, str) and command.strip()):
        return None
    # An argument with whitespace is quoted, so `node "my file.js"` does not read as two arguments.
    words = [command.strip(), *(shlex.quote(a) if any(c.isspace() for c in a) else a
                                for a in _mask_flag_values([str(a) for a in (args or []) if a is not None]))]
    return {"kind": "stdio", "target": scrub(" ".join(words))}


def launch_for(action: str, name: str) -> Optional[dict[str, str]]:
    """The line for one MCP row; None when the name resolves to nothing readable."""
    if action == "install":
        from hermes_cli.mcp_catalog import get_entry

        entry = get_entry(name)
        if entry is None:
            return None
        transport = entry.transport
        return launch_target(transport.url, transport.command, transport.args)
    from hermes_cli.mcp_catalog import installed_servers

    cfg = installed_servers().get(name)
    if not isinstance(cfg, dict):
        return None
    return launch_target(cfg.get("url"), cfg.get("command"), cfg.get("args"))
