"""What an MCP consent row will run or contact, worded for the card before the user approves.

Install reads the catalog manifest's transport; enable and authorize read the configured
``mcp_servers`` entry. The line is display-only and never carries env values or headers.

The card exists so the user consents to what will run, so hiding execution is worse than showing a
secret. The line is never cut, and a value is masked only when its own name says it is secret: a long
flag such as ``--api-key=…`` / ``--token …`` / ``--token:…``, a ``GITHUB_TOKEN=…`` assignment, a
``docker``-style ``-e KEY=…`` / ``--env KEY=…``, an ``Authorization:`` / ``Cookie:`` / API-key header
arg, a URL userinfo password, a secret-named URL query value (``?token=``, ``?key=``). A named secret's
value is masked whatever it holds (shell metacharacters included) — execve argv never runs a shell.
Positional args, short flags (``-p``, ``-k``, ``-t``), packages, images, scripts and URL paths are
always shown, even when one of them is a key: the user approves that exact command. The program given
to an interpreter's inline-program flag (``sh -c``, ``python -c``, ``node -e``, ``perl -e``), found
past any wrapper (``env``, ``npx -y node``, ``sudo -u x bash``), is never masked, and a bare
``KEY=VALUE`` holding a line break or shell metacharacter is shown too, so a secret name inside code
cannot hide the code. An argument is quoted when it would not read back as exactly that one argument
(whitespace, a quote, backslash or shell metacharacter, an empty argument), so ``shlex.split`` of the
line is the argv shown; an argument with control or invisible code points is a bash ``$'…'`` word whose
escapes keep a real newline distinct from the literal text ``\n``.
"""

from __future__ import annotations

import re
import shlex
import unicodedata
from typing import Any, Optional

from agent.redact import sanitize_command_for_display
from agent.redact_headers import SECRET_HEADER_NAME_LIST

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
# A bare ``KEY=VALUE`` positional is ambiguous: a real env assignment, or code an interpreter runs
# (``sh -c 'TOKEN=x; curl …'``). Only when the value can only be data (no line break, ``;``, ``|``,
# ``&``, ``$(…)``, a backtick, brackets, quotes) is it masked; a value that can run or split is shown.
# The value of a recognised secret mechanism (a named flag, ``-e``/``--env``, a header, URL userinfo
# or query) is always masked instead, whatever it holds — that is never an interpreter's code.
_PLAIN_VALUE_RE = re.compile(r"[^\s;|&$`<>(){}'\"\\]+")
# A header value is masked whole; only a line break or control (header injection) keeps it visible.
_PLAIN_HEADER_VALUE_RE = re.compile(r"[^\x00-\x1f\x7f]+")
# A precise interpreter word (command or arg, ``/usr/bin/python3.12``, ``bash.exe``). Names are
# listed in full, never ``[a-z]*sh`` (that caught ``ssh`` and ``publish`` and hid the secret after).
_INTERPRETER_RE = re.compile(
    r"(?:sh|bash|zsh|dash|ksh|mksh|ash|fish|csh|tcsh|rc|python[\d.]*|pypy[\d.]*|node(?:js)?|deno|bun|"
    r"tsx|ts-node|ruby|perl[\d.]*|php[\d.]*|lua(?:jit)?|pwsh|powershell|osascript|tclsh|r|rscript|elixir)(?:\.exe)?",
    re.IGNORECASE)
# Commands that run a following command; the real command (and whether it is an interpreter) is found
# past the wrapper's options, so ``-e`` after ``npx -y node`` is node's eval, not a docker env var.
_WRAPPERS = frozenset({
    "env", "sudo", "doas", "nohup", "setsid", "nice", "ionice", "time", "timeout", "stdbuf",
    "command", "exec", "busybox", "chroot", "xargs", "uv", "uvx", "npx", "pnpm", "bunx", "pipx",
    "poetry", "yarn",
})
_INLINE_CODE_FLAG_RE = re.compile(r"(-[ceEpr]|--eval|--print|--?command)(=?)(.*)", re.IGNORECASE | re.DOTALL)
_SECRET_ENV_RE = re.compile(r"TOKEN|SECRET|PASSW(?:OR)?D|API_?KEY|AUTH(?!OR(?!IZ))|CREDENTIAL|PRIVATE_?KEY",
                            re.IGNORECASE)
# One header arg, alone or as ``--header=…`` / attached ``-H…``: the auth scheme stays, the value goes.
_HEADER_NAMES = "|".join(sorted({"cookie", *SECRET_HEADER_NAME_LIST}, key=len, reverse=True))
_SECRET_HEADER_ARG_RE = re.compile(
    rf"^((?:--header[=:]?|-H)?[ \t]*(?:(?:proxy-)?authorization[ \t]*:[ \t]*(?:(?:bearer|basic|token|digest)[ \t]+)?"
    rf"|(?:{_HEADER_NAMES})[ \t]*:[ \t]*))(\S.*)$",
    re.IGNORECASE | re.DOTALL,
)
_LONG_FLAG_RE = re.compile(r"--([^=:]+)([=:]?)")


def _is_secret_name(name: str) -> bool:
    """True when a flag's or query parameter's own name says its value is a secret."""
    norm = _CAMEL_RE.sub("-", name).lower().replace("_", "-").replace(".", "-")
    return bool(_SECRET_NAME_RE.search(norm)) and not _NOT_A_VALUE_RE.search(norm)


def _mask_arg(arg: str, interpreter: bool = False) -> tuple[str, bool]:
    """``(shown, mask_next)`` for one argv entry. ``interpreter`` is True while the command word (argv[0])
    is an interpreter and its own options are still being read: a named-secret flag then shows its value
    (it may be the script or its code) instead of masking it, but an attached ``--flag=`` secret is still
    masked, since that value is never the program."""
    if (header := _SECRET_HEADER_ARG_RE.match(arg)) and _PLAIN_HEADER_VALUE_RE.fullmatch(header.group(2)):
        return f"{header.group(1)}***", False
    if arg.startswith("--"):
        flag = _LONG_FLAG_RE.match(arg)
        if flag and _is_secret_name(flag.group(1)):
            if not flag.group(2):
                return arg, not interpreter
            return f"{flag.group(0)}***", False  # ``--api-key=VALUE``: the value is never the program
        # ``--env=GITHUB_TOKEN=…``: the assignment's own key names the secret.
        if flag and flag.group(2):
            return f"{flag.group(0)}{_mask_assignment(arg[flag.end():], force=True)}", False
        return arg, False
    # ``KEY=VALUE`` alone, or attached to a short flag (``-eGITHUB_TOKEN=…``, java ``-Dapi.token=…``).
    return _mask_assignment(arg), False


def _mask_assignment(text: str, force: bool = False) -> str:
    """A ``KEY=VALUE`` whose key names a secret, with the value masked. ``force`` masks it whatever the
    value holds (the value of ``-e``/``--env``/``--flag=``); otherwise only a plain data value is masked,
    so a bare ``KEY=VALUE`` that is really an interpreter's inline code is shown."""
    assign = _ASSIGN_RE.match(text)
    if assign and _SECRET_ENV_RE.search(assign.group(2)) and (force or _PLAIN_VALUE_RE.fullmatch(assign.group(3))):
        return f"{assign.group(1)}{assign.group(2)}=***"
    return text


def _is_interpreter(word: str) -> bool:
    return bool(_INTERPRETER_RE.fullmatch(re.split(r"[\\/]", word)[-1]))


def _effective_is_interpreter(command: str, args: list[str]) -> bool:
    """Whether the command that actually runs is an interpreter, looking past wrapper words. ``env …
    sh``, ``npx -y node`` and ``sudo -u x bash`` all resolve to an interpreter, so a later ``-e`` is its
    eval flag, not a ``docker``-style env var worth masking."""
    if _is_interpreter(command):
        return True
    if re.split(r"[\\/]", command)[-1] not in _WRAPPERS:
        return False
    return any(_is_interpreter(a) for a in args if not a.startswith("-") and "=" not in a)


def _mask_flag_values(command: str, args: list[str]) -> list[str]:
    out: list[str] = []
    interp = _is_interpreter(command)  # argv[0] is an interpreter; its own flags may carry inline code
    allow_env_force = not _effective_is_interpreter(command, args)
    code_next = mask_next = env_next = False
    for arg in args:
        # The program an interpreter runs inline (``sh -c 'GITHUB_TOKEN=x; curl …'``) is what the user
        # approves: shown whole, whatever names it mentions.
        if code_next:
            out.append(arg)
            code_next = interp = False
            continue
        if mask_next:
            mask_next = False
            if not arg.startswith("-"):  # a following flag is the next flag, not the secret's value
                out.append("***")
                continue
        if env_next:
            out.append(_mask_assignment(arg, force=True))
            env_next = False
            continue
        if interp and (inline := _INLINE_CODE_FLAG_RE.fullmatch(arg)):
            out.append(arg)
            code_next = not (inline.group(2) or inline.group(3))
            interp = code_next  # attached code (``-ccode``/``--eval=code``) ends the option region now
            continue
        if interp and not arg.startswith("-"):
            interp = False  # the script positional: its own args mask normally from here
        # ``docker``-style ``-e KEY=VALUE`` / ``--env KEY=VALUE`` (only when no interpreter runs).
        if allow_env_force and arg in ("-e", "--env"):
            out.append(arg)
            env_next = True
            continue
        shown, mask_next = _mask_arg(arg, interpreter=interp)
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
    if at and ":" in userinfo:  # a userinfo password is always secret, whatever it holds
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
    """One value as written (spaces kept), named secrets in URLs masked, never cut. The shape-based
    secret redactor is not run here: it masks positionals that merely look like keys (a package
    named ``sk-…``, a JWT-shaped script arg), and the card must not hide what runs."""
    return _visible(_URL_RE.sub(_mask_url, text))


# Characters that change how a shell or ``shlex.split`` reads a word: such a word, or an empty one,
# is quoted, so ``python -c 'X=print(1)'`` reads as the one argument it is.
_NEEDS_QUOTE_RE = re.compile(r"[\s'\"\\;|&$`<>(){}]")
# Common controls get a readable name; every other hidden code point gets its ``\\x``/``\\u`` escape.
_CONTROL_NAMES = {"\n": "\\n", "\t": "\\t", "\r": "\\r"}


def _is_hidden(ch: str) -> bool:
    """A control, format (bidi override/isolate, zero-width, BOM, soft hyphen), non-space separator or
    combining code point: shown as an escape so the line reads as the argv runs, never a hidden char."""
    return not ch.isprintable() or unicodedata.category(ch) in ("Mn", "Me", "Mc") or ch == "\ufffc"


def _escape_cp(ch: str) -> str:
    if ch in _CONTROL_NAMES:
        return _CONTROL_NAMES[ch]
    o = ord(ch)
    return f"\\x{o:02x}" if o <= 0xff else (f"\\u{o:04x}" if o <= 0xffff else f"\\U{o:08x}")


def _ansi_c_quote(text: str) -> str:
    """An arg holding hidden code points as a bash ``$'…'`` word: backslashes are doubled and each hidden
    code point is a visible escape, so a real newline (``$'a\\nb'``) never reads as the literal text
    ``a\\n b`` (``'a\\nb'``) — the card must show which one the shell runs."""
    body = "".join("\\\\" if c == "\\" else (_escape_cp(c) if _is_hidden(c) else c) for c in text)
    return "$'" + body.replace("'", "\\'") + "'"


def _word(arg: str) -> str:
    """One argv entry as one shell word: ``''`` for an empty arg, ``'review  server.py'`` with both
    spaces, so the line never reads as a different argv. An arg with control or invisible code points is a
    ``$'…'`` word (bash reads it back); a plain arg is ``shlex.quote``-d, so ``shlex.split`` of the line
    is the argv shown."""
    shown = _URL_RE.sub(_mask_url, arg)
    if any(_is_hidden(c) for c in shown):
        return _ansi_c_quote(shown)
    return shlex.quote(shown) if not shown or _NEEDS_QUOTE_RE.search(shown) else shown


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
    shown = _mask_flag_values(command, [str(a) for a in argv if a is not None])
    out: dict[str, Any] = {"kind": "stdio", "target": " ".join(map(_word, [command, *shown]))}
    if keys := _env_keys(env):
        out["env"] = keys
    if isinstance(cwd, str) and cwd.strip():
        out["cwd"] = _word(cwd)
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
