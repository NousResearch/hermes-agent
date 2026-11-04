"""What an MCP consent row will run or contact, worded for the card before the user approves.

Install reads the catalog manifest's transport; enable and authorize read the configured
``mcp_servers`` entry. The line is display-only and never carries env values or headers.

The card exists so the user consents to what will run, so hiding execution is worse than showing a
secret. The line is never cut, and a value is masked only when its own name says it is secret: a long
flag such as ``--api-key=…`` / ``--token …`` / ``--token:…``, a ``GITHUB_TOKEN=…`` assignment, an
``Authorization:`` / ``Cookie:`` / API-key header arg, a URL userinfo password, a secret-named URL
query value (``?token=``, ``?key=``). Positional args, short flags (``-p``, ``-k``, ``-t``), packages,
images, scripts and URL paths are always shown, even when one of them is a key: the user approves
that exact command. Invisible and composing code points render as escapes.
"""

from __future__ import annotations

import re
import shlex
import unicodedata
from typing import Any, Optional

from agent.redact import SECRET_HEADER_NAME_LIST, sanitize_command_for_display

# A flag or query name that says its value is secret, after camelCase/snake_case fold to kebab. The
# word ends its segment (``--access-token``, ``--apikey``; not ``--tokenizer``, ``--max-tokens``),
# ``auth`` is a whole segment (not ``--author``), and ``pass``/``pwd`` are the whole name.
_SECRET_NAME_RE = re.compile(
    r"(?:token|secret|passw(?:or)?d|api-?key|credentials?|private-?key|bearer)(?=-|$)"
    r"|(?:^|-)auth(?:orization)?(?=-|$)|^(?:pass|pwd)$")
# Names for where a secret lives or how it is supplied, not the secret: ``--token-file``,
# ``--password-stdin``, ``--auth-type``; ``--no-auth`` is a switch, so the next arg is not its value.
_NOT_A_VALUE_RE = re.compile(
    r"-(?:file|path|dir|env|var|name|url|uri|stdin|prompt|type|mode|method|scheme|helper|cmd|command)$"
    r"|^(?:no|disable|enable|skip|use|with|without)-")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# ``KEY=VALUE`` (an ``env`` prefix, a ``-e`` value): the key's last identifier carries the name.
_ASSIGN_RE = re.compile(r"^([^\s=]*?)([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_SECRET_ENV_RE = re.compile(r"TOKEN|SECRET|PASSW(?:OR)?D|API_?KEY|AUTH(?!OR(?!IZ))|CREDENTIAL|PRIVATE_?KEY",
                            re.IGNORECASE)
# One header arg, alone or as ``--header=…`` / attached ``-H…``: the auth scheme stays, the value goes.
_HEADER_NAMES = "|".join(sorted({"cookie", *SECRET_HEADER_NAME_LIST}, key=len, reverse=True))
_SECRET_HEADER_ARG_RE = re.compile(
    rf"^((?:--header[=:]?|-H)?\s*(?:(?:proxy-)?authorization\s*:\s*(?:(?:bearer|basic|token|digest)\s+)?"
    rf"|(?:{_HEADER_NAMES})\s*:\s*))\S.*$",
    re.IGNORECASE | re.DOTALL,
)
_LONG_FLAG_RE = re.compile(r"--([^=:]+)([=:]?)")


def _is_secret_name(name: str) -> bool:
    """True when a flag's or query parameter's own name says its value is a secret."""
    norm = _CAMEL_RE.sub("-", name).lower().replace("_", "-").replace(".", "-")
    return bool(_SECRET_NAME_RE.search(norm)) and not _NOT_A_VALUE_RE.search(norm)


def _mask_arg(arg: str) -> tuple[str, bool]:
    """``(shown, mask_next)`` for one argv entry."""
    if header := _SECRET_HEADER_ARG_RE.match(arg):
        return f"{header.group(1)}***", False
    if arg.startswith("--"):
        flag = _LONG_FLAG_RE.match(arg)
        if flag and _is_secret_name(flag.group(1)):
            return (f"{flag.group(0)}***", False) if flag.group(2) else (arg, True)
        # ``--env=GITHUB_TOKEN=…``: the assignment's own key names the secret.
        if flag and flag.group(2) and (inner := _ASSIGN_RE.match(arg[flag.end():])) \
                and _SECRET_ENV_RE.search(inner.group(2)):
            return f"{flag.group(0)}{inner.group(1)}{inner.group(2)}=***", False
        return arg, False
    assign = None if arg.startswith("-") else _ASSIGN_RE.match(arg)
    if assign and _SECRET_ENV_RE.search(assign.group(2)):
        return f"{assign.group(1)}{assign.group(2)}=***", False
    return arg, False


def _mask_flag_values(args: list[str]) -> list[str]:
    out: list[str] = []
    mask_next = False
    for arg in args:
        # A long flag right after a secret flag is the next flag, not its value (``--token --exec x``
        # fails to parse as a token): masking it would hide what runs.
        if mask_next and not arg.startswith("--"):
            out.append("***")
            mask_next = False
            continue
        shown, mask_next = _mask_arg(arg)
        out.append(shown)
    return out


_URL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")


def _mask_query_pair(pair: str) -> str:
    key, eq, _value = pair.partition("=")
    # ``?key=`` is an API key in a URL (``?key=AIza…``); as a flag, ``--key`` is often a key *file*.
    return f"{key}=***" if eq and (key.lower() == "key" or _is_secret_name(key)) else pair


def _mask_url(match: re.Match[str]) -> str:
    """A userinfo password and secret-named query values become ``***``. Host, path, other query
    values and the fragment stay: the path of a ``deno run https://…`` URL is the code that runs."""
    url = match.group(0)
    head, sep, rest = url.partition("://")
    rest, hash_, fragment = rest.partition("#")
    rest, qmark, query = rest.partition("?")
    authority, slash, path = rest.partition("/")
    userinfo, at, host = authority.rpartition("@")
    if at and ":" in userinfo:
        authority = f"{userinfo.partition(':')[0]}:***@{host}"
    out = f"{head}{sep}{authority}{slash}{path}"
    if qmark:
        out += "?" + "&".join(_mask_query_pair(pair) for pair in query.split("&"))
    return out + hash_ + fragment


# Tab and newline are the two controls the approval sanitizer keeps; the card draws one line.
_LINE_BREAKS = str.maketrans({"\t": "\\t", "\n": "\\n"})


def _escape_composing(ch: str) -> str:
    # A combining mark draws over the letter before it (``pk\u0336g`` reads as a struck ``k``) and
    # U+FFFC draws as a box. Other non-ASCII letters stay as written (homoglyphs are out of scope).
    if ch == "\ufffc" or unicodedata.category(ch) in ("Mn", "Me"):
        return ch.encode("unicode_escape").decode("ascii")
    return ch


def _visible(text: str) -> str:
    """Every control, format (bidi overrides/isolates, zero-width, BOM, soft hyphen), separator and
    combining code point as a visible escape such as ``\\u202e``: the line must read as the argv runs,
    and the user should see that an arg is unusual rather than a cleaned-up version of it."""
    shown = sanitize_command_for_display(text.translate(_LINE_BREAKS))
    return shown if shown.isascii() else "".join(map(_escape_composing, shown))


def scrub(text: str) -> str:
    """The whole line, never cut: every argument the user approves is on the card. The shape-based
    secret redactor is not run here: it masks positionals that merely look like keys (a package
    named ``sk-…``, a JWT-shaped script arg), and the card must not hide what runs."""
    return _URL_RE.sub(_mask_url, " ".join(_visible(text).split()))


def _env_keys(env: Any) -> list[str]:
    """The child's extra env names, never their values: the runtime merges them into the spawn."""
    if not isinstance(env, dict):
        return []
    return [scrub(str(k)) for k in sorted(env, key=str) if str(k).strip()]


def launch_target(url: Any, command: Any, args: Any, env: Any = None, cwd: Any = None) -> Optional[dict[str, Any]]:
    """``{"kind": "remote"|"stdio", "target": text}``, plus ``env`` (names only) and ``cwd`` when a stdio
    entry sets them; the URL wins, as the transport choice does."""
    if isinstance(url, str) and url.strip():
        return {"kind": "remote", "target": scrub(url.strip())}
    if not (isinstance(command, str) and command.strip()):
        return None
    # A bare string ``args`` is one argument, not a sequence of characters.
    argv = [args] if isinstance(args, str) else list(args or []) if isinstance(args, (list, tuple)) else []
    # An argument with whitespace is quoted, so `node "my file.js"` does not read as two arguments.
    words = [command.strip(), *(shlex.quote(a) if any(c.isspace() for c in a) else a
                                for a in _mask_flag_values([str(a) for a in argv if a is not None]))]
    out: dict[str, Any] = {"kind": "stdio", "target": scrub(" ".join(words))}
    if keys := _env_keys(env):
        out["env"] = keys
    if isinstance(cwd, str) and cwd.strip():
        out["cwd"] = scrub(cwd.strip())
    return out


def launch_for(action: str, name: str) -> Optional[dict[str, Any]]:
    """The line for one MCP row; None when the name resolves to nothing readable."""
    if action == "install":
        from hermes_cli.mcp_catalog import get_entry

        entry = get_entry(name)
        if entry is None:
            return None
        transport = entry.transport
        return launch_target(transport.url, transport.command, transport.args, transport.env)
    from hermes_cli.mcp_catalog import installed_servers

    cfg = installed_servers().get(name)
    if not isinstance(cfg, dict):
        return None
    return launch_target(cfg.get("url"), cfg.get("command"), cfg.get("args"), cfg.get("env"), cfg.get("cwd"))
