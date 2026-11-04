"""What an MCP consent row will run or contact, worded for the card before the user approves.

Install reads the catalog manifest's transport; enable and authorize read the configured
``mcp_servers`` entry. The line is display-only. It never carries the env map or headers, and its
text is scrubbed before it leaves the process: the value of a secret-named flag (``--api-key=…``,
``--token …``), credential-named query params and URL userinfo, long opaque URL path segments
(hosted MCP URLs often embed the key in the path), then the standard redaction.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Optional

from agent.redact import redact_for_egress, redact_sensitive_text, sanitize_command_for_display

_SECRET_FLAG_RE = re.compile(
    r"^--?[A-Za-z0-9_.-]*(?:key|token|secret|passw(?:or)?d|pass|pwd|auth|credential|bearer|cookie)"
    r"[A-Za-z0-9_.-]*$",
    re.IGNORECASE,
)
# Tab and newline are the two controls the approval sanitizer keeps; the card draws one line.
_LINE_BREAKS = str.maketrans({"\t": "\\t", "\n": "\\n"})
# Scrubbed first, then cut, so a secret straddling the cut is already masked.
MAX_TARGET_CHARS = 400
_URL_RE = re.compile(r"\b(?:https?|wss?)://[^\s]+", re.IGNORECASE)
# A path segment that reads as a credential, not a route: long, url-safe, with letters and digits.
_OPAQUE_SEGMENT_RE = re.compile(r"(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{20,}")


def _mask_flag_values(args: list[str]) -> list[str]:
    out: list[str] = []
    mask_next = False
    for arg in args:
        if mask_next:
            out.append("***")
            mask_next = False
            continue
        flag, sep, _value = arg.partition("=")
        if _SECRET_FLAG_RE.match(flag):
            if sep:
                out.append(f"{flag}=***")
                continue
            mask_next = True
        out.append(arg)
    return out


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
    text = _URL_RE.sub(_mask_url_path, " ".join(_visible(text).split()))
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
