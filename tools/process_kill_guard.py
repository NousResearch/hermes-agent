"""Unconditional floor: refuse process-kill commands whose targets include Hermes' own runtime.

A misordered ``pkill -f PATTERN -P 1`` is read by BSD/macOS ``pkill`` as *three* regexes
(``PATTERN``, ``-P``, ``1``), so it signalled every process with a "1" in its command line --
including the dashboard, the gateway body and the managed workers hosting the agent. Prompt-only
detection cannot protect that: ``approvals.mode: off`` (kanban, cron and headless runs) returns
before the dangerous-pattern layer. This module therefore runs in ``tools.approval._floor_block``
beside the hardline blocklist.

It does not guess from the spelling of the command. For the forms whose target set can be
computed without side effects (``pkill``, ``killall``, ``kill <pid|-pgid>``, ``kill $(pgrep|pidof|
lsof ...)``, ``pgrep|pidof|lsof ... | xargs kill``) it resolves the match set read-only and blocks
only when that set contains a *protected* process: the caller's own ancestor chain, launchd, or a
Hermes service/worker process. Legitimate kills of test runners, dev servers, etc. are unaffected.
Simple read-only PID pipelines and locally captured PID variables are also resolved. Forms whose
targets cannot be resolved statically are left to the other approval checks.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess

logger = logging.getLogger("tools.approval")

# Command lines of processes an agent must never signal by accident. Matched with re.search against
# the full argv string. Extra patterns can be added via ``approvals.protected_processes`` (list of
# regexes) without a code change.
_DEFAULT_PROTECTED = (
    r"hermes_cli\.main\b.*\b(?:gateway\s+run|dashboard|relay\s+start|webapp)\b",
    r"\bhermes\s+(?:(?:--profile|-p)\s+\S+\s+|--?\S+\s+)*(?:gateway\s+run|dashboard|relay\s+start)\b",
    r"-m\s+agent\.managed_worker\b",
    r"-m\s+hermes_cli\.kanban_worker_client\b",
)

_PRODUCER_TIMEOUT_S = 4.0
_MAX_PIDS = 20000
_RESOLVERS = {"pgrep", "pidof", "lsof"}
# xargs options that consume the following word.
_XARGS_ARG_OPTS = {"-n", "-I", "-L", "-P", "-s", "-E", "-J", "-R", "-S", "-d", "-a"}
_SIGNAL_NAMES = {s.name[3:] for s in signal.Signals} | {s.name for s in signal.Signals}

_SEPARATORS = ";&|\n"


def _protected_regexes() -> list[re.Pattern]:
    patterns = list(_DEFAULT_PROTECTED)
    try:
        from tools import approval_context as _ctx
        extra = _ctx._get_approval_config().get("protected_processes") or []
        patterns += [p for p in extra if isinstance(p, str) and p.strip()]
    except Exception:
        logger.exception("Unable to load approvals.protected_processes")
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error:
            logger.warning("Ignoring invalid approvals.protected_processes regex: %r", p)
    return compiled


# ---- process table ------------------------------------------------------------------------------

def _process_table() -> dict[int, tuple[int, int, str]] | None:
    """pid -> (ppid, pgid, command) from one ``ps`` call, or None if unavailable."""
    try:
        out = subprocess.run(
            ["ps", "-ww", "-axo", "pid=,ppid=,pgid=,command="],
            capture_output=True, text=True, timeout=_PRODUCER_TIMEOUT_S, check=False, stdin=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    table: dict[int, tuple[int, int, str]] = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
            continue
        table[int(parts[0])] = (int(parts[1]), int(parts[2]), parts[3] if len(parts) > 3 else "")
    return table or None


def _protected_pids(table: dict[int, tuple[int, int, str]]) -> dict[int, str]:
    """pid -> reason, for every process in *table* that must not be signalled."""
    protected: dict[int, str] = {1: "launchd/init"}
    pid = os.getpid()
    seen = set()
    while pid and pid not in seen and pid in table:
        seen.add(pid)
        protected[pid] = "an ancestor of this agent process" if pid != os.getpid() else "this agent process"
        pid = table[pid][0]
    regexes = _protected_regexes()
    roots = {p for p, (_, _, cmd) in table.items() if any(r.search(cmd) for r in regexes)}
    for p in roots:
        protected.setdefault(p, "a Hermes service/worker process")
    # A service can supervise workers in its own process group: protect the group when a matched
    # service is its leader.
    root_pgids = {table[p][1] for p in roots if table[p][1] == p}
    for p, (_, pgid, _) in table.items():
        if pgid in root_pgids:
            protected.setdefault(p, "part of a Hermes service process group")
    return protected


# ---- target resolution --------------------------------------------------------------------------

_EXPANSION_START = re.compile(r"[A-Za-z_0-9{(@*#?!$-]")


def _shell_words(text: str) -> list[str] | None:
    """Split *text* like the shell, or None when a word's value depends on run-time expansion.

    ``shlex.split`` discards quoting, so a regex such as ``pkill -f 'hermes.*gateway'`` used to look
    like an unresolved glob and the kill was waved through. Quoting decides instead:

    * inside single quotes nothing expands; inside double quotes only ``$`` and backticks do;
    * ``$NAME``/``${...}``/``$(...)``/backticks outside single quotes make the word unresolvable;
    * unquoted brace expansion (``{a,b}``) and a leading ``~`` make the word unresolvable;
    * glob characters (``* ? [``) never do: quoted they are literal, and unquoted the shell passes
      the word unchanged unless it happens to name files in the cwd. pgrep/killall read the result
      as a regex/name, so the literal word is the value to resolve. The floor only refuses when
      that resolved set contains a protected process, so this cannot refuse a harmless kill on the
      strength of a filename match.
    """
    words: list[str] = []
    cur: list[str] = []
    in_word = False
    quote = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote == "'":
            if ch == "'":
                quote = None
            else:
                cur.append(ch)
            i += 1
            continue
        if ch == "\\":
            if i + 1 >= n:
                return None
            nxt = text[i + 1]
            if quote == '"' and nxt not in '$`"\\\n':
                cur.append(ch)
            if nxt != "\n":
                cur.append(nxt)
            in_word = True
            i += 2
            continue
        if ch == "`" or (ch == "$" and i + 1 < n and _EXPANSION_START.match(text[i + 1])):
            return None
        if quote == '"':
            if ch == '"':
                quote = None
            else:
                cur.append(ch)
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            in_word = True
            i += 1
            continue
        if ch.isspace():
            if in_word:
                words.append("".join(cur))
                cur, in_word = [], False
            i += 1
            continue
        if ch == "~" and not in_word:
            return None
        if ch == "{":
            close = text.find("}", i)
            if close != -1 and ("," in text[i:close] or ".." in text[i:close]):
                return None
        cur.append(ch)
        in_word = True
        i += 1
    if quote is not None:
        return None
    if in_word:
        words.append("".join(cur))
    return words


def _run_lines(argv: list[str]) -> list[str] | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=_PRODUCER_TIMEOUT_S, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.split() if out.returncode in (0, 1) else None


_PIPE_STAGES = {"ps", "pgrep", "pidof", "lsof", "grep", "egrep", "fgrep", "awk"}
_UNSAFE_STAGE_RE = re.compile(r"\$\(|`|[<>]|\bsystem\s*\(|\bgetline\b|\bENVIRON\b|\|")


def _resolve_pipeline(text: str) -> set[int] | None:
    """Run a read-only PID-producing pipeline (``ps | grep | awk``, ``pgrep``, ``lsof -t``) and parse PIDs.

    Only an allowlist of side-effect-free filters is executed, with no redirection, substitution,
    awk ``system``/``getline``, or nested pipes inside a stage. Anything else is unresolvable (None).
    """
    stages = list(_segments(text.strip()))
    if not stages or any(sep != "|" for _, sep in stages[:-1]) or stages[-1][1] != "":
        return None
    commands = []
    for stage, _ in stages:
        toks = _shell_words(stage)
        if not toks or _UNSAFE_STAGE_RE.search(stage):
            return None
        name = os.path.basename(toks[0])
        if name not in _PIPE_STAGES:
            return None
        # Only an inline awk filter is inspected; -f could load an arbitrary program from disk.
        if name == "awk" and (len(toks) != 2 or toks[1].startswith("-")):
            return None
        commands.append([name, *toks[1:]])
    if len(commands) == 1:
        if commands[0][0] not in _RESOLVERS:
            return None
        words = _run_lines(commands[0])
    else:
        # Pass output as data, never re-evaluate the supplied pipeline through a shell.
        output = ""
        try:
            for argv in commands:
                out = subprocess.run(argv, input=output, capture_output=True, text=True,
                                     timeout=_PRODUCER_TIMEOUT_S, check=False)
                if out.returncode not in (0, 1):
                    return None
                output = out.stdout
        except (OSError, subprocess.SubprocessError):
            return None
        words = output.split()
    if words is None:
        return None
    pids = {int(w) for w in words if w.isdigit()}
    return pids if len(pids) <= _MAX_PIDS else None


def _strip_signal(args: list[str]) -> list[str]:
    """Drop pkill's leading ``-SIGNAL`` / ``-NUM`` argument, which pgrep does not accept."""
    if args and re.fullmatch(r"-(?:\d+|[A-Za-z][A-Za-z0-9]*)", args[0]):
        name = args[0][1:]
        if name.isdigit() or name.upper() in _SIGNAL_NAMES:
            return args[1:]
    return args


def _resolve_pkill(args: list[str] | None) -> set[int] | None:
    """*args* come from ``_shell_words``: quoting is already resolved, None means unresolvable."""
    if args is None:
        return None
    rest = _strip_signal(args)
    if not rest:
        return None
    # The same argv through pgrep reproduces the host's option parsing exactly (BSD stops at the
    # first operand; procps permutes) -- the whole point of the audit.
    words = _run_lines(["pgrep", *rest])
    return None if words is None else {int(w) for w in words if w.isdigit()}


def _resolve_killall(args: list[str] | None) -> set[int] | None:
    """killall: match by process name (``-m`` BSD / ``-r`` procps: regex). ``-s``/``-l`` only report.

    *args* come from ``_shell_words`` (None: unresolvable), so a quoted regex is resolved as such.
    """
    if args is None:
        return None
    names: list[str] = []
    regex, user, i = False, None, 0
    while i < len(args):
        a = args[i]
        if a in ("-s", "-l"):
            return set()  # show-only / list-signals: nothing is signalled
        if a in ("-m", "-r", "--regexp"):
            regex = True
        elif a in ("-u", "-t", "-c") and i + 1 < len(args):
            user = args[i + 1] if a == "-u" else user
            i += 1
        elif a == "--":
            names += args[i + 1:]
            break
        elif not a.startswith("-"):
            names.append(a)
        i += 1
    if not names:
        return None
    try:
        out = subprocess.run(["ps", "-ww", "-axo", "pid=,user=,comm="], capture_output=True, text=True,
                             timeout=_PRODUCER_TIMEOUT_S, check=False, stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    owner_filter = user or os.environ.get("USER") or ""
    hits: set[int] = set()
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or (owner_filter and parts[1] != owner_filter):
            continue
        base = os.path.basename(parts[2])  # ps ``comm`` is the executable path; killall compares the name
        try:
            if any(re.search(n, base) if regex else base == n for n in names):
                hits.add(int(parts[0]))
        except re.error:
            return None
    return hits


_SUBST_RE = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")


def _resolve_kill(raw_tail: str) -> set[int] | None:
    """``kill [-s SIG | -SIG | -n N] [--] PID | -PGID | $(producer)...`` -> targeted pids/-pgids."""
    targets: set[int] = set()
    resolved_any = False
    # Command substitutions are opaque to shlex: resolve their read-only producers separately.
    for m in _SUBST_RE.finditer(raw_tail):
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        pids = _resolve_pipeline(inner)
        if pids is not None:
            targets |= pids
            resolved_any = True
    try:
        toks = shlex.split(_SUBST_RE.sub(" ", raw_tail))
    except ValueError:
        return targets if resolved_any else None
    signal_seen = operands_only = False
    i = 0
    while i < len(toks):
        t = toks[i]
        if not operands_only and t == "--":
            operands_only = True
        elif not operands_only and t in ("-s", "-n", "--signal"):
            signal_seen = True
            i += 1
        elif not operands_only and t in ("-l", "-L"):
            return set()
        elif not operands_only and not signal_seen and re.fullmatch(r"-(?:\d+|[A-Za-z][A-Za-z0-9]*)", t):
            signal_seen = True  # the one leading signal spec: -9, -KILL, -SIGTERM
        elif re.fullmatch(r"-?\d+", t):
            n = int(t)
            if n != -1:  # -1 (everything) is the hardline rule's job; 0 is the caller's own group
                targets.add(n)
                resolved_any = True
        i += 1
    return targets if resolved_any else None


# ---- command analysis ---------------------------------------------------------------------------

def _segments(command: str):
    """Yield ``(text, separator_after)`` for each top-level segment, keeping ``$(...)``/quotes intact."""
    from tools.approval_detection import _scan_shell
    start = 0
    for kind, i, j, quote in _scan_shell(command, subst="uq", comments=True):
        if kind == "comment" or (kind == "char" and quote is None and command[i] in _SEPARATORS):
            if start < i:
                yield command[start:i], (command[i] if kind == "char" else "\n")
            start = j
    if start < len(command):
        yield command[start:], ""


_WRAPPERS = {"sudo", "env", "nohup", "exec", "time", "command", "builtin", "setsid", "nice", "stdbuf"}
_KEYWORDS = {"then", "do", "else", "elif", "if", "while", "until", "!"}


def _split_command(segment: str) -> tuple[str, str] | None:
    """Return (command_name, raw_tail) for the segment, skipping env assignments and plain wrappers."""
    rest = segment.strip().lstrip("({ \t")
    while rest:
        m = re.match(r"(\S+)\s*(.*)\Z", rest, re.DOTALL)
        if not m:
            return None
        word, tail = m.group(1), m.group(2)
        base = os.path.basename(word.strip("\"'"))
        if re.fullmatch(r"\w+=\S*", word) or base in _WRAPPERS or word in _KEYWORDS:
            if base in ("sudo", "env", "nice", "stdbuf"):
                # drop wrapper options (-u root, -n 5 ...) conservatively: leading dash words only
                tail = re.sub(r"^(?:-\S+\s+)*", "", tail)
            rest = tail
            continue
        return base, tail
    return None


def _analyse(command: str, table, depth: int = 0) -> set[int] | None:
    """Union of resolvable kill targets in *command*; None when it contains no resolvable kill."""
    found: set[int] | None = None
    segs = list(_segments(command))
    variables: dict[str, set[int]] = {}  # ``pids=$(pgrep ...)`` captured for a later ``kill $pids``
    for idx, (text, sep) in enumerate(segs):
        assign = re.fullmatch(r"\s*(\w+)=(?:\$\(([^()]*)\)|`([^`]*)`)\s*", text)
        if assign:
            pids = _resolve_pipeline(assign.group(2) if assign.group(2) is not None else assign.group(3))
            if pids is not None:
                variables[assign.group(1)] = pids
            continue
        parsed = _split_command(text)
        if parsed is None:
            continue
        name, tail = parsed
        if name in ("sh", "bash", "zsh", "dash") and depth < 2:
            m = re.search(r"-\w*c\s+(?:'([^']*)'|\"([^\"]*)\")", tail)
            if m:
                inner = _analyse(m.group(1) or m.group(2) or "", table, depth + 1)
                if inner is not None:
                    found = (found or set()) | inner
            continue
        targets: set[int] | None = None
        if name == "pkill":
            targets = _resolve_pkill(_shell_words(tail))
        elif name == "killall":
            targets = _resolve_killall(_shell_words(tail))
        elif name == "kill":
            targets = _resolve_kill(tail)
            for var, pids in variables.items():
                if re.search(r"\$\{?" + re.escape(var) + r"\b", tail):
                    targets = (targets or set()) | pids
        elif name == "xargs" and idx > 0 and segs[idx - 1][1] == "|":
            try:
                toks = shlex.split(tail)
            except ValueError:
                toks = []
            j = 0
            while j < len(toks) and toks[j].startswith("-"):
                j += 2 if toks[j] in _XARGS_ARG_OPTS else 1
            if j < len(toks) and os.path.basename(toks[j]) == "kill":
                k = idx
                while k > 0 and segs[k - 1][1] == "|":
                    k -= 1
                targets = _resolve_pipeline("|".join(st for st, _ in segs[k:idx]))
        if targets is not None:
            found = (found or set()) | targets
    return found


def _expand_pgids(targets: set[int], table) -> set[int]:
    out = set()
    for t in targets:
        if t == 0:
            out |= {p for p, (_, pgid, _) in table.items() if pgid == os.getpgrp()}
        elif t < 0:
            out |= {p for p, (_, pgid, _) in table.items() if pgid == -t}
            out.add(-t)  # the group leader's pid equals the pgid
        else:
            out.add(t)
    return out


def detect_protected_process_kill(command: str, env_type: str = "local") -> str | None:
    """Description of a protected process the command would signal, or None.

    Only the local host is inspected (``env_type == "local"``): a remote/container backend runs the
    command against a process table this one knows nothing about.
    """
    if env_type != "local" or os.name != "posix" or "kill" not in command.lower():
        return None
    try:
        table = _process_table()
        if table is None:
            return None
        targets = _analyse(command, table)
        if not targets:
            return None
        pids = _expand_pgids(targets, table)
        protected = _protected_pids(table)
        # The kill utility itself and its own shell are not in the table yet; never count them.
        hit = sorted(p for p in pids if p in protected and p in table)
        if not hit:
            return None
        first = hit[0]
        cmd = table[first][2]
        cmd = cmd if len(cmd) <= 80 else cmd[:77] + "..."
        extra = f" (+{len(hit) - 1} more protected)" if len(hit) > 1 else ""
        return (f"kill would signal pid {first} ({protected[first]}: {cmd}){extra}; match set is "
                f"{len(pids)} process(es). Kill exact PIDs of your own processes instead"
                + ("; on macOS, options placed after a pkill pattern become extra patterns" if "pkill" in command else ""))
    except Exception:  # a guard bug must not turn into a tool outage; the other floors still apply
        logger.exception("process_kill_guard failed; allowing command")
        return None
