"""Same-string download-then-run detection for the approval gate: a curl/wget writes FILE
(``-o``/``-O``/``--output``/``>``) and a later command in the same string runs that FILE (via a
shell, ``source``/``.``, ``./FILE`` or the bare path). Cross-call download then exec is out of
reach of a stateless detector.

Two linear steps instead of one backreferencing regex: a lazy ``[^\\n]*?`` between the target
and ``(?P=dl)`` retried from every fetch made 64 KB of ``curl -o f`` take tens of seconds. Step 1
collects targets per fetch segment, step 2 checks each later command start against that set."""

from __future__ import annotations

import hermes_re as re

from tools.approval_pipe_launchers import _PIPE_LAUNCHER_VALUE_FLAGS, _PIPE_PATH_PREFIX_RE
from tools.shell_heredoc import mask_heredoc_bodies_for_shell_scan

_FETCH_RE = re.compile(r"\b(?:curl|wget)\b", re.IGNORECASE)
_SEGMENT_END_RE = re.compile(r"[\n;&|]")
_TARGET_RE = re.compile(
    r"(?:\s-[a-z]*o\s*|\s--output(?:-document)?(?:=|\s+)|>[>|]?\s*)['\"]?([^\s;&|<>'\"`()]+)",
    re.IGNORECASE,
)
_TERMINATOR = r"(?=[\s;&|)`'\"]|$)"
# The run matcher never crosses a newline: `command -v bash` and `bash job.sh` on two lines are two commands.
_H = r"[^\S\n]"
# Syntax-check-only runs parse FILE without running it: every flag must be the check flag
# (`bash -n f`, `sh -xn f`, `node --check f`, `ruby -wc f`, `php -l f`). `perl -c` is not one: it runs
# BEGIN/use blocks.
_SHELL_CHECK_FLAG = r"-[vx]*n[nvx]*"
_CHECK_ONLY_FLAGS = {"node": r"-c|--check", "ruby": r"-w?c[cw]*", "php": r"-l"}
# Options whose next word is their value, not a script operand (`bash -euo pipefail`, `python3 -W ignore`).
_SHELL_VALUE_FLAG = r"[-+][a-zA-Z]*[oO]"
_PS_VALUE_FLAG = r"(?i)-(?:ep|ex\w*|w\w*|config\w*)"  # -ExecutionPolicy, -WindowStyle, -WorkingDirectory
_VALUE_FLAGS = {"python": r"-[WX]", "node": r"-r|--require|--import", "ruby": r"-[rI]",
                "pwsh": _PS_VALUE_FLAG, "powershell": _PS_VALUE_FLAG}
# cmd's switches are /x words; `call` only names the batch file after it (`cmd /c call j.bat`).
_OPTION_WORDS = {"cmd": r"/\S*|(?i:call)"}
_REDIRECT_RE = re.compile(r"\d*[<>]+&?\s*[^\s<>]*")
_ARGS_RE = re.compile(r"(?:[<>]&|[^\n;&|)`])*")  # `2>&1` is a redirection, not a list operator
_STDIN_RE = re.compile(r"(?<![<\S])0?<(?:\s+([^\s<>]+)|([^\s<>(&][^\s<>]*))")
# curl -O / --remote-name saves under the URL's basename, not an operand: `curl -fsSLO URL`.
# `command -v`/`-V` only looks the name up (`&& command -v tool` after installing tool).
_LOOKUP_ONLY_RE = re.compile(r"\bcommand(?:\s+-[a-uw-zA-UW-Z]+)*\s+-[a-zA-Z]*[vV]", re.IGNORECASE)
_CURL_REMOTE_NAME_RE = re.compile(r"\s(?:-([A-Za-z]+)|--remote-name(?:-all)?)(?=\s|$)")
_URL_RE = re.compile(r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://[^\s;&|<>'\"`()]+", re.IGNORECASE)
# PowerShell runs FILE's content through Invoke-Expression or a script block built from it
# (`iex (gc f -Raw)`, `Get-Content f | iex`, `[scriptblock]::Create((gc f -Raw))`), or dot-sources or
# calls FILE inside a -Command string (`pwsh -c ". ./f.ps1"`). Printing it (`Write-Output (gc f)`) runs nothing.
# An unquoted FILE never starts with `-` + a word character (that word is an option: `gc -Raw`), and a lone `-` is
# FILE only when no word character follows it ({1}; the piped form needs no test: a quote, a blank or `|` follows).
_PS_READ = (r"\b(?:gc|get-content|cat|type)(?:[^\S\n]+-\w+)*[^\S\n]+"
            r"(?:['\"](?P<{0}_quoted>[^\s;&|<>'\"`()]+)"
            r"|(?P<{0}>[^-\s;&|<>'\"`()][^\s;&|<>'\"`()]*|-[^\w\s;&|<>'\"`()][^\s;&|<>'\"`()]*|-{1}))")
_PS_LONE_DASH = r"(?!\w)"
_PS_READ_RUN_RE = re.compile(
    rf"\b(?:iex|invoke-expression)\b(?:[^\S\n]+-command\b)?(?:[($'\"]|[^\S\n])*{_PS_READ.format('iex', _PS_LONE_DASH)}"
    rf"|\[scriptblock\]::create\((?:[($'\"]|[^\S\n])*{_PS_READ.format('block', _PS_LONE_DASH)}"
    rf"|{_PS_READ.format('piped', '')}['\"]?(?:[^\S\n]+-\w+)*[^\S\n]*\|[^\S\n]*(?:iex|invoke-expression)\b"
    rf"|(?<=[\"'{{;(])[^\S\n]*[.&][^\S\n]+['\"]?(?P<sourced>[^\s;&|<>'\"`()]+)",
    re.IGNORECASE,
)


def _base(path: str) -> str:
    # Basename on both sides: a `cd` between write and run (`> /tmp/j; cd /tmp; sh j`) still matches.
    return path.rstrip("/").rsplit("/", 1)[-1]


# Launcher words in front of a command, read in steps: each step is a linear
# pattern that starts where the previous one ended, so an option or its value is never re-read as the command.
_RUN_LAUNCHER_RE = re.compile(
    rf"{_PIPE_PATH_PREFIX_RE}(?:{'|'.join(f'({name})' for name in _PIPE_LAUNCHER_VALUE_FLAGS)}){_H}+", re.IGNORECASE)
_RUN_BLANKS_RE = re.compile(rf"{_H}+")


def _launcher_arg_re(short: str, long_opts: tuple) -> re.Pattern:
    """One argument word of a launcher, tried in this order: a valued flag with its value, a bare flag,
    `VAR=value`, a duration. Blanks stay within a line."""
    valued = [rf"(?-i:-[^\W{short}\d-]*[{short}]){_H}+[^\s|;&<>]+"] if short else []
    valued += [rf"--(?:{'|'.join(long_opts)}){_H}+[^\s|;&<>]+"] if long_opts else []
    duration = r"(?:infinity|\d+(?:\.\d+)?(?:e\d+)?|\.\d+)[smhd]?(?=\s)"
    return re.compile("|".join([*valued, r"-[^\s|;&<>]+", r"\w+=[^\s|;&<>]*", duration]), re.IGNORECASE)


_RUN_LAUNCHER_ARG_RES = [_launcher_arg_re(*flags) for flags in _PIPE_LAUNCHER_VALUE_FLAGS.values()]


def _skip_launchers(text: str, pos: int) -> int:
    """End of the launcher words at *pos* (`sudo -u root`, `/usr/bin/env A=1`, `timeout 60`)."""
    while (launcher := _RUN_LAUNCHER_RE.match(text, pos)) is not None:
        end = launcher.end()
        arg_re = _RUN_LAUNCHER_ARG_RES[launcher.lastindex - 1]
        while (arg := arg_re.match(text, end)) is not None:
            blanks = _RUN_BLANKS_RE.match(text, arg.end())
            if blanks is None:
                return pos  # an argument that ends the command: this word is the command, not a launcher
            end = blanks.end()
        pos = end
    return pos


# The run of a command word, matched in steps like the launchers above, so that a flag or a `VAR=x` word is
# never re-read as FILE.
_RUN_HEAD_RE = re.compile(
    rf"[;&|\n(]\s*(?:(?:[{{!]|if|then|do|else|elif|while|until){_H}+)*(?:\w+=\S*{_H}+)*", re.IGNORECASE)
_RUN_SPACES_RE = re.compile(r"\s*")
_RUN_FILE = rf"['\"]?(?:\./)?(?P<file>[^\s;&|<>'\"`()]+){_TERMINATOR}"
_RUN_FILE_RE = re.compile(_RUN_FILE)
_RUN_SOURCED_RE = re.compile(rf"(?:(?:source|\.|call){_H}+)?{_RUN_FILE}", re.IGNORECASE)
_CONTENT_PIPED_RE = re.compile(rf"cat{_H}+['\"]?(?:\./)?(?P<piped>[^\s;&|<>'\"`()]+)['\"]?{_H}*\|&?\s*", re.IGNORECASE)
_CONTENT_ARGS_RE = re.compile(rf"(?:{_H}+[^\s;&|)`]+)*")


def _carrier_re(shell_names_re: str) -> re.Pattern:
    """A command string handed to a shell or su, up to its opening quote (`bash -c '`, `cmd /c "`)."""
    return re.compile(
        rf"['\"]?{_PIPE_PATH_PREFIX_RE}(?:{shell_names_re}|pwsh|powershell|cmd|su)(?:\.exe)?['\"]?"
        rf"(?:{_H}+[^\s'\";&|)`]+)*?{_H}+[-/](?:[a-z]*c|command){_H}+['\"]{_H}*", re.IGNORECASE)


def _tool_word_re(shell_names_re: str) -> str:
    # python's version digits and dots are never followed by `.exe`.
    return (rf"['\"]?{_PIPE_PATH_PREFIX_RE}(?:(?P<tool>{shell_names_re}|perl|ruby|node|php|pwsh|powershell|cmd)"
            rf"(?:\.exe)?|(?P<py>python[0-9.]*))['\"]?")


def _command_starts(carrier_re: re.Pattern, text: str, pos: int):
    """Where the command word may start: inside a `shell -c '` string at *pos*, then *pos* itself."""
    if (carrier := carrier_re.match(text, pos)) is not None:
        yield _skip_launchers(text, carrier.end())
    yield pos


def _runs(run_at, text: str, pos: int):
    """What *run_at(text, head)* finds at each command head from *pos* on."""
    while (head := _RUN_HEAD_RE.search(text, pos)) is not None:
        run = run_at(text, head)
        if run is not None:
            yield run
        # No run here: none either from a newline of the blank run just read (it reads on to the same word),
        # and stepping over that run keeps a long run of blank lines linear.
        pos = run.end() if run is not None else _RUN_SPACES_RE.match(text, head.start() + 1).end()


class _Run:
    """One run match: what `DownloadThenRunMatcher` reads from it."""

    def __init__(self, text: str, start: int, prog, file):
        self._start, self._end = start, file.end()
        self._prog_end = prog.end("prog") if prog else -1
        self._groups = {
            0: text[start:file.end()],
            "prog": prog and prog.group("prog"),
            "tool": prog and (prog.group("tool") or prog.group("py")),
            "flags": prog and prog.group("flags"),
            "file": file.group("file"),
        }

    def start(self) -> int:
        return self._start

    def end(self, group=0) -> int:
        return self._prog_end if group == "prog" else self._end

    def group(self, group=0):
        return self._groups[group]


class _RunFinder:
    """``finditer(text, pos)`` -> the commands that run a program or a file, as `_Run` matches."""

    def __init__(self, shell_names_re: str):
        self._carrier_re = _carrier_re(shell_names_re)
        self._prog_re = re.compile(
            rf"(?P<prog>{_tool_word_re(shell_names_re)}){_H}+(?P<flags>(?:-\S+{_H}+)*)(?:<{_H}*)?",
            re.IGNORECASE,
        )

    def _run_at(self, text: str, head) -> _Run | None:
        for pos in _command_starts(self._carrier_re, text, _skip_launchers(text, head.end())):
            if (prog := self._prog_re.match(text, pos)) is not None and (
                    file := _RUN_FILE_RE.match(text, prog.end())) is not None:
                return _Run(text, head.start(), prog, file)
            if (file := _RUN_SOURCED_RE.match(text, pos)) is not None:
                return _Run(text, head.start(), None, file)
        return None

    def finditer(self, text: str, pos: int = 0):
        return _runs(self._run_at, text, pos)


class _ContentRun:
    """One content-run match: what `DownloadThenRunMatcher` reads from it."""

    def __init__(self, start: int, end: int, groups: dict):
        self._start, self._end, self._groups = start, end, groups

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end

    def group(self, name: str):
        return self._groups.get(name)


class _ContentRunFinder:
    """``finditer(text, pos)`` -> the commands that run a FILE's content, as `_ContentRun` matches."""

    def __init__(self, shell_names_re: str):
        self._carrier_re = _carrier_re(shell_names_re)
        tool = _tool_word_re(shell_names_re)
        read = r"(?:\$\(\s*(?:cat\s+|<\s*)|`\s*cat\s+|<\(\s*cat\s+)['\"]?(?:\./)?(?P<file>[^\s;&|<>'\"`()]+)"
        self._read_re = re.compile(
            rf"(?:(?:eval|source|\.|{tool}(?P<rflags>(?:{_H}+[^\s;&|)`$<'\"][^\s;&|)`]*)*)){_H}+)?['\"]?{read}",
            re.IGNORECASE,
        )
        self._piped_tool_re = re.compile(rf"{tool}{_TERMINATOR}", re.IGNORECASE)

    def _run_at(self, text: str, head) -> _ContentRun | None:
        for pos in _command_starts(self._carrier_re, text, _skip_launchers(text, head.end())):
            if (read := self._read_re.match(text, pos)) is not None:
                return _ContentRun(head.start(), read.end(), {
                    "file": read.group("file"), "rtool": read.group("tool") or read.group("py"),
                    "rflags": read.group("rflags")})
            if (piped := _CONTENT_PIPED_RE.match(text, pos)) is not None and (
                    tool := self._piped_tool_re.match(text, _skip_launchers(text, piped.end()))) is not None:
                args = _CONTENT_ARGS_RE.match(text, tool.end())
                return _ContentRun(head.start(), args.end(), {
                    "piped": piped.group("piped"), "ptool": tool.group("tool") or tool.group("py"),
                    "pargs": args.group()})
        return None

    def finditer(self, text: str, pos: int = 0):
        return _runs(self._run_at, text, pos)


class DownloadThenRunMatcher:
    """``search(text)`` -> a match whose span covers the run of a just-downloaded file, or None."""

    def __init__(self, shell_names_re: str, writer_re: re.Pattern = _FETCH_RE):
        self._writer_re = writer_re
        # A command starts after a list/pipe operator or `(`, past `{`, `!`, compound
        # keywords and `VAR=value` words. Launchers (shared with the pipe sink: path-qualified, valued
        # options such as `sudo -u USER`, `timeout 60`), script interpreters and `shell < FILE` run FILE too.
        # A command string handed to a shell or su (`bash -c 'bash job.sh'`, `cmd /c "call j.bat"`) is a command line too.
        self._run_re = _RunFinder(shell_names_re)
        self._check_flags = {**{name: _SHELL_CHECK_FLAG for name in shell_names_re.split("|")}, **_CHECK_ONLY_FLAGS}
        # FILE's content run instead of FILE, at a command start: `$(cat f)`, `$(< f)`, `` `cat f` ``
        # or `<(cat f)`, bare or after eval, source or a program (`sh -c "$(cat f)"`), or `cat f | sh`.
        # A substitution used as an argument (`echo "$(cat f)"`) runs nothing.
        self._read_run_re = _ContentRunFinder(shell_names_re)

    def _syntax_check_only(self, run: _Run) -> bool:
        flag = self._check_flags.get((run.group("tool") or "").lower())
        flags = run.group("flags").split() if flag and run.group("flags") else ()
        return bool(flags) and all(re.fullmatch(flag, f) for f in flags)

    def _split_args(self, tool: str, words: list[str]) -> tuple[list[str], list[str]]:
        """(options, words from the operand on): option values are not operands."""
        shell = self._check_flags.get(tool) == _SHELL_CHECK_FLAG
        value = _SHELL_VALUE_FLAG if shell else _VALUE_FLAGS.get("python" if tool.startswith("python") else tool, r"[^\s\S]")
        flags, i = [], 0
        option = r"[-+]\S*" if shell else _OPTION_WORDS.get(tool, r"-\S*")
        while i < len(words) and re.fullmatch(option, words[i]):
            flags.append(words[i])
            i += 2 if re.fullmatch(value, words[i]) else 1
        return flags, words[i:]

    def _check_only(self, tool: str, flags: list[str]) -> bool:
        check = self._check_flags.get(tool)
        return bool(check and flags) and all(re.fullmatch(check, f) for f in flags)

    def _operand_ran(self, run: _Run, text: str, targets: dict[str, int]) -> bool:
        """`bash -o pipefail job.sh`, `powershell -ExecutionPolicy Bypass -File .\\j.ps1`: the real operand."""
        tool = run.group("tool").lower()
        args = _ARGS_RE.match(text, run.end("prog")).group()
        flags, rest = self._split_args(tool, _REDIRECT_RE.sub(" ", args).split())
        operand = rest[0] if rest else (stdin.group(1) or stdin.group(2) if (stdin := _STDIN_RE.search(args)) else None)
        if operand is None or self._check_only(tool, flags):
            return False
        return run.start() >= targets.get(_base(operand.strip("'\"").replace("\\", "/")), len(text))

    def _runs_content(self, read: _ContentRun) -> bool:
        """`bash -n <(cat f)`, `cat f | bash -n` and `cat f | python3 -m json.tool` don't run f."""
        tool = (read.group("rtool") or read.group("ptool") or "").lower()
        # redirections (`> run.log`, `2>&1`) are not operands
        flags, rest = self._split_args(tool, _REDIRECT_RE.sub(" ", read.group("rflags") or read.group("pargs") or "").split())
        if self._check_only(tool, flags):
            return False
        # A piped program reads its code from stdin unless a script or -c/-m operand comes first.
        shell = self._check_flags.get(tool) == _SHELL_CHECK_FLAG
        if read.group("piped") is None:
            return not rest  # `bash other.sh "$(cat f)"` hands f's content to other.sh as an argument
        return not rest or any(
            f == "-" or (shell and re.fullmatch(r"-[a-zA-Z]*s[a-zA-Z]*", f)) for f in flags)

    def search(self, text: str):
        targets: dict[str, int] = {}  # downloaded FILE -> where its download ends
        pos = 0
        while (fetch := self._writer_re.search(text, pos)) is not None:
            end_match = _SEGMENT_END_RE.search(text, fetch.end())
            seg_end = end_match.start() if end_match else len(text)
            for target in _TARGET_RE.finditer(text, fetch.end(), seg_end):
                targets.setdefault(_base(target.group(1)), target.end())
            if fetch.group().lower() == "curl" and any(
                    m.group(1) is None or "O" in m.group(1)
                    for m in _CURL_REMOTE_NAME_RE.finditer(text, fetch.end(), seg_end)):
                for url in _URL_RE.finditer(text, fetch.end(), seg_end):
                    name = url.group().split("?", 1)[0].split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1]
                    if name:
                        targets.setdefault(name, seg_end)
            pos = seg_end  # one segment scan per fetch segment, never per fetch word
        if not targets:
            return None
        for run in self._run_re.finditer(text, min(targets.values())):
            # A path-qualified program word can itself be the download (`./node.exe --version`).
            prog = (run.group("prog") or "").strip("'\"")
            prog_ran = "/" in prog and run.start() >= targets.get(_base(prog), len(text))
            file_ran = run.start() >= targets.get(_base(run.group("file")), len(text)) and not self._syntax_check_only(run)
            file_ran = file_ran or bool(run.group("tool")) and self._operand_ran(run, text, targets)
            if (prog_ran or file_ran) and not _LOOKUP_ONLY_RE.search(run.group()):
                return run
        for read in self._read_run_re.finditer(text, min(targets.values())):
            written = read.start() >= targets.get(_base(read.group("file") or read.group("piped")), len(text))
            if written and self._runs_content(read):
                return read
        for read in _PS_READ_RUN_RE.finditer(text, min(targets.values())):
            if read.start() >= targets.get(_base(read.group(read.lastgroup).replace("\\", "/")), len(text)):
                return read
        return None


_CAT_WRITERS: dict[str, DownloadThenRunMatcher] = {}


def heredoc_scan_sources(command: str, shell_names_re: str, *, floors: bool = False) -> tuple[str, ...]:
    """Body-dropped copies, plus the raw command when a dropped `cat` body's file runs later in it
    (`cat > job.sh <<'EOF'` ... `bash job.sh`: the body is code, not data).

    The hardline and deny floors (*floors*) always take the raw command as well, so they read every
    body as main does. What can execute a file written earlier in the string has no end (`mv` then
    run, a variable, `make`, git hooks, a bin dir, rc files), so recognizing runs cannot keep a body
    off the floors; only the dangerous-prompt layer, where a miss costs a prompt, drops it. The
    dropped copies still expose a command after the terminator that a body quote hid (#140704)."""
    data_dropped = mask_heredoc_bodies_for_shell_scan(command, code_consumers=False)
    if floors:
        code_dropped = mask_heredoc_bodies_for_shell_scan(command, code_consumers=True)
        return tuple(dict.fromkeys((data_dropped, code_dropped, command)))
    if shell_names_re not in _CAT_WRITERS:
        _CAT_WRITERS[shell_names_re] = DownloadThenRunMatcher(shell_names_re, writer_re=re.compile(r"\bcat\b"))
    runs_written = data_dropped != command and _CAT_WRITERS[shell_names_re].search(data_dropped)
    code_dropped = mask_heredoc_bodies_for_shell_scan(command, code_consumers=True)
    return tuple(dict.fromkeys((data_dropped, code_dropped, *((command,) if runs_written else ()))))
