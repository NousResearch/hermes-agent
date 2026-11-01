"""Pre-spawn guard for regex tools aimed at spillover (persisted tool-result) files.

A spillover file is one escaped JSON line of 100 KB+. GNU grep expands a large bounded
interval such as ``.\\{0,26000\\}`` into a matcher state the size of the quantifier, so
``grep -o 'x.\\{0,26000\\}' spill.txt`` grew past 5 GB, outlived the terminal timeout (the
MSYS-handoff grep escaped the tree sweep) and starved the host. Banning unbounded ``*``/``+``
is not enough; this module refuses the regex engine itself before any subprocess exists and
points at bounded reads instead.

Pure functions over the command text, deterministic, no I/O. Used by ``terminal_tool``
(``_pre_exec_block``; ``execute_code`` reaches it because its ``terminal()`` stub dispatches
through the same tool), by ``search_files``, and by ``execute_code`` for scripts that shell
out to a regex tool. Command substitution, backticks, subshells and process substitution are
stripped before tokenising so the inner tool word is recognised; paths and tool names
assembled from shell variables stay unresolved — this is a static guard, not a sandbox. Two
bounds backstop what slips past it: a foreground spillover read is clamped to
SPILLOVER_READ_TIMEOUT_S, and a background spillover run is refused outright (no deadline
applies in the background). ``python -c "re.findall(...)"`` over a spill file is not refused:
CPython's engine backtracks rather than building state proportional to the quantifier, so the
clamp and the job memory cap bound it.
"""

import fnmatch
import re
import shlex
from typing import Optional

# Hard wall-clock cap for a foreground command that touches a spillover file. Ordinary reads
# (head/tail/wc/grep -F) finish in milliseconds; anything longer is the runaway class.
SPILLOVER_READ_TIMEOUT_S = 20

# Windows job-wide committed-memory cap for the same commands (applied in LocalEnvironment
# ._run_bash when the Windows containment job is present). A 100 KB file read needs a few MB; the
# incident process reached 5 GB.
SPILLOVER_JOB_MEMORY_BYTES = 512 * 1024 * 1024

# GNU grep/sed/awk blow up on large repetition counts independent of the input file, so an
# interval upper bound above this is refused for those tools everywhere (ordinary code
# search uses single-digit or low-hundreds bounds).
MAX_INTERVAL_BOUND = 1000

_SPILL_DIR = "spillover"
_TMP_RESULTS_DIR = "hermes-results"
# Spillover filenames are ``<tool_use_id>.txt`` (``call_<hex>``); matching the name catches a
# path reached via ``cd``/relative form with no directory component in the command.
_SPILL_FILE_RE = re.compile(r"\bcall_[0-9a-f]{16,}\.txt\b")
_CACHE_SEG_RE = re.compile(r"cache/([^/\s;|&<>()]+)")
_GLOB_CHARS = "*?["

# Tools that interpret a regex (or script) over their input.
_REGEX_TOOLS = frozenset({
    "grep", "egrep", "fgrep", "rgrep", "zgrep", "zegrep", "zfgrep", "pcregrep", "pcre2grep",
    "ugrep", "ug", "rg", "ripgrep", "ag", "ack", "sed", "gsed", "awk", "gawk", "mawk", "nawk",
    "perl", "ruby", "findstr", "select-string", "sls",
})
# Tools whose interval blow-up is independent of input size (POSIX/GNU regex compilers).
_INTERVAL_TOOLS = frozenset({
    "grep", "egrep", "fgrep", "rgrep", "zgrep", "zegrep", "sed", "gsed", "awk", "gawk", "mawk", "nawk",
})
_FIXED_FLAG_BY_TOOL = {"rg": "F", "ripgrep": "F", "ag": "Q", "ack": "Q"}
_ALWAYS_FIXED = frozenset({"fgrep", "zfgrep"})
_REGEX_MODE_LONG = frozenset({"--perl-regexp", "--extended-regexp", "--basic-regexp"})
_REGEX_MODE_SHORT = frozenset("EPG")
_SHELL_WRAPPERS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "busybox"})

_INTERVAL_RE = re.compile(r"\\*\{\s*(\d*)\s*,?\s*(\d*)\s*\\*\}")


def _normalize_path_text(text: str) -> str:
    """Fold quoting, escaping, separators and ``./`` hops so path variants compare equal."""
    text = text.lower()
    text = re.sub(r"[\'\"`^]", "", text)          # 'spill''over', "cache"/spillover, ^ in cmd.exe
    text = text.replace("\\", "/")                # C:\x\cache\spillover, JSON-escaped \\
    text = re.sub(r"/+", "/", text)               # cache//spillover
    return re.sub(r"(?:/\.)+(?=/)", "", text)     # cache/./spillover


def references_spillover(text: str) -> bool:
    """True when *text* names a spillover store, any profile, in any common spelling."""
    if not isinstance(text, str) or not text:
        return False
    norm = _normalize_path_text(text)
    if _SPILL_FILE_RE.search(norm) or _TMP_RESULTS_DIR in norm:
        return True
    for match in _CACHE_SEG_RE.finditer(norm):
        seg = match.group(1)
        if seg.startswith("spill"):
            return True
        if any(ch in seg for ch in _GLOB_CHARS) and fnmatch.fnmatchcase(_SPILL_DIR, seg):
            return True
    return False


def _split_stages(command: str) -> list:
    """Split on unquoted ``| ; & newline`` so a quoted ``|`` inside a regex stays one stage.

    An opening ``(`` of a subshell or process substitution also starts a new stage: bash runs
    ``x=$(grep ...)`` / ``(grep ...)`` / ``< <(grep ...)`` as a nested command whose first word
    is glued to the opener, so tokenising the stage as a whole would never see ``grep`` as a
    token. Splitting there exposes the inner tool word; unmatched ``(`` (regex groups such as
    ``grep -E 'a(bc)+'``) does no harm because the split boundary lands inside the quoted
    pattern, which re-joins in the same stage list.
    """
    stages, buf = [], []
    quote = None
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                i += 1
                buf.append(command[i])
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == "\\" and i + 1 < len(command):
            buf.append(ch)
            i += 1
            buf.append(command[i])
        elif ch == "(":
            if buf and buf[-1] == "$":       # $(cmd) command substitution: break so the inner
                buf.pop()                    # command starts a fresh stage at its tool word
                stages.append("".join(buf))
                buf = []
            elif i and command[i - 1] == "<":  # <(cmd) / >(cmd) process substitution: keep the
                buf.pop()                     # redirect so tokenising stays honest, break after
                stages.append("".join(buf))
                buf = []
            else:                             # (cmd) subshell — plain '(' is a regex group only
                stages.append("".join(buf))   # inside a pattern; unquoted here it opens a command
                buf = []
        elif ch in ")`":                      # closers end the nested command
            stages.append("".join(buf))
            buf = []
        elif ch in "|;&\n":
            stages.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    stages.append("".join(buf))
    return [s for s in (s.strip() for s in stages) if s]


def _tokens(stage: str) -> list:
    try:
        return shlex.split(stage, posix=True)
    except ValueError:
        return [t.strip("'\"") for t in stage.split()]


def _tool_name(token: str) -> str:
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def _is_fixed_string(tool: str, args: list) -> bool:
    """True when the stage uses its literal-string mode and no regex-mode flag overrides it."""
    flag = _FIXED_FLAG_BY_TOOL.get(tool, "F")
    fixed = tool in _ALWAYS_FIXED
    for arg in args:
        if arg == "--":
            break
        if arg in _REGEX_MODE_LONG:
            return False
        if arg == "--fixed-strings":
            fixed = True
        elif arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
            cluster = arg[1:]
            if set(cluster) & _REGEX_MODE_SHORT and tool in _INTERVAL_TOOLS | {"rg", "ugrep", "ug", "pcregrep"}:
                return False
            if flag in cluster:
                fixed = True
    return fixed


def _interval_bound_max(interval: str) -> int:
    """Max numeric bound in a ``{m,n}`` interval (1 when empty)."""
    values = [int(g) for g in re.findall(r"\d+", interval)]
    return max(values) if values else 1


def _interval_product(text: str) -> Optional[int]:
    """Largest product of nested interval bounds in *text*, else None.

    ``(.{0,999}){0,999}`` compiles ~10^6 states even though each bound is under
    MAX_INTERVAL_BOUND: an interval quantifier applied to a group whose body already carries
    an interval multiplies the bounds. Only nesting multiplies — separate intervals in one
    pattern (``a{0,999}b{0,999}``) are alternatives for the worst case, not a product.
    The interval braces and the group closer may each carry backslashes: BRE spells the
    same shape ``\\(\\.\\{0,999\\}\\)\\{0,999\\}`` and double-quoted shell text doubles them.
    """
    worst = None
    for m in re.finditer(r"(\\{0,2}\{[^{}]*\\{0,2}\})[^{}]*\\{0,2}\)\s*(\\{0,2}\{[^{}]*\\{0,2}\})", text):
        product = _interval_bound_max(m.group(1)) * _interval_bound_max(m.group(2))
        if product > MAX_INTERVAL_BOUND:
            worst = max(worst or 0, product)
    return worst


def _oversized_interval(text: str) -> Optional[int]:
    """Largest single interval bound above MAX_INTERVAL_BOUND found in *text*, else None."""
    worst = None
    for match in _INTERVAL_RE.finditer(text):
        for group in match.groups():
            if group and int(group) > MAX_INTERVAL_BOUND:
                worst = max(worst or 0, int(group))
    return worst


def _scan(command: str, *, spill_ref: bool, depth: int = 0):
    """Return ``(reason, detail)`` for the first offending stage, else None."""
    if depth > 3:
        return ("nesting", "shell nesting too deep to analyse")
    for stage in _split_stages(command):
        tokens = _tokens(stage)
        if not tokens:
            continue
        # Wrappers that carry a command string: analyse the inner command too.
        for idx, token in enumerate(tokens):
            name = _tool_name(token)
            if name in _SHELL_WRAPPERS and "-c" in tokens[idx + 1:]:
                inner = tokens[tokens.index("-c", idx + 1) + 1:]
                if inner:
                    found = _scan(" ".join(inner[:1]), spill_ref=spill_ref, depth=depth + 1)
                    if found:
                        return found
            elif name == "eval" and tokens[idx + 1:]:
                found = _scan(" ".join(tokens[idx + 1:]), spill_ref=spill_ref, depth=depth + 1)
                if found:
                    return found
        for idx, token in enumerate(tokens):
            tool = _tool_name(token)
            if tool not in _REGEX_TOOLS:
                continue
            args = tokens[idx + 1:]
            if tool in _INTERVAL_TOOLS:
                # Flat bound anywhere in the stage (quoted patterns included — the recorded
                # incident's bound was inside single quotes), plus the nested-product rule on
                # the stage and each argument (shlex keeps a quoted pattern as one token).
                bound = _oversized_interval(stage) or 0
                bound = max(bound, _interval_product(stage) or 0)
                bound = max(bound, max((_interval_product(a) or 0 for a in args), default=0))
                if bound:
                    return ("interval", f"{tool} interval bound {bound} exceeds {MAX_INTERVAL_BOUND}")
            if spill_ref and not _is_fixed_string(tool, args):
                return ("regex", f"{tool} regex over a spillover file")
    return None


_GUIDANCE = (
    "Read it in bounded pieces instead: read_file with offset/limit (use line limits; one-line "
    "JSON needs the byte window below), a byte window "
    "`tail -c +<start> <file> | head -c 20000`, or a literal (non-regex) search "
    "`grep -F -b -o -m 1 -- '<literal text>' <file> | head -c 2000` to get the byte offset of a "
    "match, then the byte window around that offset. Never pass the spillover file to a regex "
    "tool (grep -E/-P/BRE, rg, sed, awk, perl)."
)


def spillover_regex_block(command: str) -> Optional[str]:
    """Return a refusal message when *command* runs a regex engine over a spillover file,
    or any GNU regex tool with an oversized interval; else None. Runs before spawn."""
    if not isinstance(command, str) or not command:
        return None
    spill_ref = references_spillover(command)
    found = _scan(command, spill_ref=spill_ref)
    if found is None:
        return None
    reason, detail = found
    if reason == "interval":
        msg = (f"Blocked: {detail}. Large repetition counts make GNU grep/sed/awk expand a "
               "matcher state proportional to the count and can exhaust host memory "
               "(an overnight `.{0,26000}` grep reached 5 GB and outlived its timeout). ")
        if spill_ref:
            return msg + _GUIDANCE
        return msg + ("Use a literal search (`grep -F`) plus a byte window "
                      "(`tail -c +<start> <file> | head -c <n>`), or an interval bound of "
                      f"{MAX_INTERVAL_BOUND} or less.")
    return f"Blocked: {detail}. " + _GUIDANCE


_SPAWN_RE = re.compile(r"\b(?:subprocess|os\.system|os\.popen|os\.exec\w*|os\.spawn\w*|Popen|check_output)\b")
_REGEX_TOOL_WORD_RE = re.compile(
    r"(?<![\w-])(?:" + "|".join(sorted(re.escape(t) for t in _REGEX_TOOLS)) + r")(?:\.exe)?(?![\w-])", re.IGNORECASE)


def _literal_commands(code: str) -> list:
    """String constants and constant argv lists in *code*, each rendered as a shell command."""
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 20000:
            out.append(node.value)
        elif isinstance(node, (ast.List, ast.Tuple)) and node.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            out.append(shlex.join(e.value for e in node.elts))
    return out


def execute_code_spillover_block(code: str) -> Optional[str]:
    """Refusal for an ``execute_code`` script that shells out to a regex tool over a spillover
    file, or with an oversized GNU interval. The script's own ``terminal()`` calls are already
    covered by the terminal pre-exec guard; this covers direct ``subprocess``/``os.system``.
    Static only: commands assembled at runtime from variables are not resolved."""
    if not isinstance(code, str) or not code:
        return None
    for cmd in _literal_commands(code):
        msg = spillover_regex_block(cmd)
        if msg:
            return msg
    # Path and tool split across literals (``["grep", "-o", pat, path_var]`` with the spillover
    # path elsewhere in the script): refuse the combination when the script spawns processes.
    if references_spillover(code) and _SPAWN_RE.search(code) and _REGEX_TOOL_WORD_RE.search(code):
        return ("Blocked: this script spawns a regex tool in a script that names a spillover file. "
                + _GUIDANCE)
    return None


def spillover_search_block(pattern: str, path: str, file_glob: Optional[str], target: str) -> Optional[str]:
    """Refusal for ``search_files`` content search rooted at or globbed to a spillover store."""
    if target != "content":
        return None
    # The glob can complete a path that stops at ``cache`` (path=.../cache, glob=spillover/*.txt).
    joined = f"{path or ''}/{file_glob or ''}"
    if references_spillover(str(path or "")) or references_spillover(str(file_glob or "")) \
            or references_spillover(joined):
        return "Blocked: regex content search over a spillover file is not allowed. " + _GUIDANCE
    return None


def bounded_spillover_timeout(command: str, timeout: Optional[int]) -> Optional[int]:
    """Clamp a foreground timeout to SPILLOVER_READ_TIMEOUT_S when the command touches a
    spillover file; pass through otherwise."""
    if not references_spillover(command):
        return timeout
    return min(timeout, SPILLOVER_READ_TIMEOUT_S) if timeout else SPILLOVER_READ_TIMEOUT_S


def spillover_background_block(command: str, background: bool) -> Optional[str]:
    """Refusal for ``background=True`` commands that touch a spillover file.

    The background path (process_registry.spawn_local / spawn_via_env) has no call deadline:
    the incident class is precisely a grep that outlived its timeout. Run the read in the
    foreground instead, where the pre-exec guard and the hard clamp apply.
    """
    if background and isinstance(command, str) and references_spillover(command):
        return ("Blocked: background commands over a spillover file are not allowed — a "
                "background process has no call deadline, and the runaway this guard exists "
                "for outlived its timeout exactly that way. Run a bounded foreground read "
                "instead (it is hard-capped at "
                f"{SPILLOVER_READ_TIMEOUT_S}s). " + _GUIDANCE)
    return None
