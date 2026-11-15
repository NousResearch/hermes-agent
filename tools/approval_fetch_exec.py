"""Same-string download-then-run detection for the approval gate: a curl/wget writes FILE
(``-o``/``-O``/``--output``/``>``) and a later command in the same string runs that FILE (via a
shell, ``source``/``.``, ``./FILE`` or the bare path). Cross-call download then exec is out of
reach of a stateless detector.

Two linear steps instead of one backreferencing regex: a lazy ``[^\\n]*?`` between the target
and ``(?P=dl)`` retried from every fetch made 64 KB of ``curl -o f`` take tens of seconds. Step 1
collects targets per fetch segment, step 2 checks each later command start against that set."""

from __future__ import annotations

import re

from tools.approval_pipe_launchers import _PIPE_LAUNCHER_VALUE_FLAGS, _pipe_launcher_re
from tools.shell_heredoc import mask_heredoc_bodies_for_shell_scan

_FETCH_RE = re.compile(r"\b(?:curl|wget)\b", re.IGNORECASE)
_SEGMENT_END_RE = re.compile(r"[\n;&|]")
_TARGET_RE = re.compile(
    r"(?:\s-[a-z]*o\s*+|\s--output(?:-document)?(?:=|\s++)|>[>|]?\s*+)['\"]?([^\s;&|<>'\"`()]++)",
    re.IGNORECASE,
)
_TERMINATOR = r"(?=[\s;&|)`'\"]|$)"
_SCRIPT_INTERPRETERS_RE = r"python[0-9.]*+|perl|ruby|node|php|pwsh|powershell|cmd"
_PIPE_PATH_PREFIX_RE = r"(?>(?:[a-zA-Z]:)?(?:[.\w~-]*+[/\\])*+)"


def _launcher_wrappers_re(ws: str) -> str:
    launchers = "|".join(_pipe_launcher_re(n, *f, ws) for n, f in _PIPE_LAUNCHER_VALUE_FLAGS.items())
    return rf"(?>{_PIPE_PATH_PREFIX_RE}(?:{launchers}){ws}++)*+"


_PIPE_WRAPPERS_RE = _launcher_wrappers_re(r"\s")
# The run matcher never crosses a newline: `command -v bash` and `bash job.sh` on two lines are two commands.
_H = r"[^\S\n]"
_LINE_WRAPPERS_RE = _launcher_wrappers_re(_H)
# Syntax-check-only runs parse FILE without running it: every flag must be the check flag
# (`bash -n f`, `sh -xn f`, `node --check f`, `ruby -wc f`, `php -l f`). `perl -c` is not one: it runs
# BEGIN/use blocks.
_SHELL_CHECK_FLAG = r"-[vx]*+n[nvx]*+"
_CHECK_ONLY_FLAGS = {"node": r"-c|--check", "ruby": r"-w?c[cw]*+", "php": r"-l"}
# Options whose next word is their value, not a script operand (`bash -euo pipefail`, `python3 -W ignore`).
_SHELL_VALUE_FLAG = r"[-+][a-zA-Z]*[oO]"
_PS_VALUE_FLAG = r"(?i)-(?:ep|ex\w*|w\w*|config\w*)"  # -ExecutionPolicy, -WindowStyle, -WorkingDirectory
_VALUE_FLAGS = {"python": r"-[WX]", "node": r"-r|--require|--import", "ruby": r"-[rI]",
                "pwsh": _PS_VALUE_FLAG, "powershell": _PS_VALUE_FLAG}
# cmd's switches are /x words; `call` only names the batch file after it (`cmd /c call j.bat`).
_OPTION_WORDS = {"cmd": r"/\S*|(?i:call)"}
_REDIRECT_RE = re.compile(r"\d*+[<>]++&?+\s*+[^\s<>]*+")
_ARGS_RE = re.compile(r"(?:[^\n;&|)`]|(?<=[<>])&)*+")  # `2>&1` is a redirection, not a list operator
_STDIN_RE = re.compile(r"(?<![<\S])0?<(?![<(&])\s*+([^\s<>]++)")
# curl -O / --remote-name saves under the URL's basename, not an operand: `curl -fsSLO URL`.
# `command -v`/`-V` only looks the name up (`&& command -v tool` after installing tool).
_LOOKUP_ONLY_RE = re.compile(r"\bcommand(?:\s+-[a-uw-zA-UW-Z]+)*\s+-[a-zA-Z]*[vV]", re.IGNORECASE)
_CURL_REMOTE_NAME_RE = re.compile(r"\s(?:-([A-Za-z]++)|--remote-name(?:-all)?)(?=\s|$)")
_URL_RE = re.compile(r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*+://[^\s;&|<>'\"`()]++", re.IGNORECASE)


def _base(path: str) -> str:
    # Basename on both sides: a `cd` between write and run (`> /tmp/j; cd /tmp; sh j`) still matches.
    return path.rstrip("/").rsplit("/", 1)[-1]


class DownloadThenRunMatcher:
    """``search(text)`` -> a match whose span covers the run of a just-downloaded file, or None."""

    def __init__(self, shell_names_re: str, writer_re: re.Pattern = _FETCH_RE):
        self._writer_re = writer_re
        # A command starts after a list/pipe operator or `(`, past `{`, `!`, compound
        # keywords and `VAR=value` words. Launchers (shared with the pipe sink: path-qualified, valued
        # options such as `sudo -u USER`, `timeout 60`), script interpreters and `shell < FILE` run FILE too.
        self._run_re = re.compile(
            rf"[;&|\n(]\s*+(?:(?:[{{!]|if|then|do|else|elif|while|until){_H}++)*+(?:\w++=\S*+{_H}++)*+"
            rf"{_LINE_WRAPPERS_RE}(?:(?P<prog>['\"]?{_PIPE_PATH_PREFIX_RE}(?P<tool>{shell_names_re}|{_SCRIPT_INTERPRETERS_RE})(?:\.exe)?['\"]?){_H}++"
            rf"(?P<flags>(?:-\S++{_H}++)*+)(?:<{_H}*+)?|(?:source|\.){_H}++)?"
            rf"['\"]?(?:\./)?(?P<file>[^\s;&|<>'\"`()]++){_TERMINATOR}",
            re.IGNORECASE,
        )
        self._check_flags = {**{name: _SHELL_CHECK_FLAG for name in shell_names_re.split("|")}, **_CHECK_ONLY_FLAGS}
        # FILE's content run instead of FILE, at a command start: `$(cat f)`, `$(< f)`, `` `cat f` ``
        # or `<(cat f)`, bare or after eval, source or a program (`sh -c "$(cat f)"`), or `cat f | sh`.
        # A substitution used as an argument (`echo "$(cat f)"`) runs nothing.
        read = r"(?:\$\(\s*+(?:cat\s++|<\s*+)|`\s*+cat\s++|<\(\s*+cat\s++)['\"]?(?:\./)?(?P<file>[^\s;&|<>'\"`()]++)"
        tool = rf"['\"]?{_PIPE_PATH_PREFIX_RE}(?P<{{}}>{shell_names_re}|{_SCRIPT_INTERPRETERS_RE})(?:\.exe)?['\"]?"
        self._read_run_re = re.compile(
            rf"[;&|\n(]\s*+(?:(?:[{{!]|if|then|do|else|elif|while|until){_H}++)*+(?:\w++=\S*+{_H}++)*+"
            rf"{_LINE_WRAPPERS_RE}(?:(?:(?:eval|source|\.|{tool.format('rtool')}(?P<rflags>(?:{_H}++(?![$`<'\"])[^\s;&|)`]++)*?)){_H}++)?"
            rf"['\"]?{read}|cat{_H}++['\"]?(?:\./)?(?P<piped>[^\s;&|<>'\"`()]++)['\"]?{_H}*+\|(?!\|)&?\s*+"
            rf"{_LINE_WRAPPERS_RE}{tool.format('ptool')}{_TERMINATOR}(?P<pargs>(?:{_H}++[^\s;&|)`]++)*+))",
            re.IGNORECASE,
        )

    def _syntax_check_only(self, run: re.Match) -> bool:
        flag = self._check_flags.get((run.group("tool") or "").lower())
        flags = run.group("flags").split() if flag and run.group("flags") else ()
        return bool(flags) and all(re.fullmatch(flag, f) for f in flags)

    def _split_args(self, tool: str, words: list[str]) -> tuple[list[str], list[str]]:
        """(options, words from the operand on): option values are not operands."""
        shell = self._check_flags.get(tool) == _SHELL_CHECK_FLAG
        value = _SHELL_VALUE_FLAG if shell else _VALUE_FLAGS.get("python" if tool.startswith("python") else tool, "(?!)")
        flags, i = [], 0
        option = r"[-+]\S*" if shell else _OPTION_WORDS.get(tool, r"-\S*")
        while i < len(words) and re.fullmatch(option, words[i]):
            flags.append(words[i])
            i += 2 if re.fullmatch(value, words[i]) else 1
        return flags, words[i:]

    def _check_only(self, tool: str, flags: list[str]) -> bool:
        check = self._check_flags.get(tool)
        return bool(check and flags) and all(re.fullmatch(check, f) for f in flags)

    def _operand_ran(self, run: re.Match, text: str, targets: dict[str, int]) -> bool:
        """`bash -o pipefail job.sh`, `powershell -ExecutionPolicy Bypass -File .\\j.ps1`: the real operand."""
        tool = run.group("tool").lower()
        args = _ARGS_RE.match(text, run.end("prog")).group()
        flags, rest = self._split_args(tool, _REDIRECT_RE.sub(" ", args).split())
        operand = rest[0] if rest else (stdin.group(1) if (stdin := _STDIN_RE.search(args)) else None)
        if operand is None or self._check_only(tool, flags):
            return False
        return run.start() >= targets.get(_base(operand.strip("'\"").replace("\\", "/")), len(text))

    def _runs_content(self, read: re.Match) -> bool:
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
        return None


_CAT_WRITERS: dict[str, DownloadThenRunMatcher] = {}


def heredoc_scan_sources(command: str, shell_names_re: str) -> tuple[str, ...]:
    """Body-dropped copies, plus the raw command when a dropped `cat` body's file runs later in it
    (`cat > job.sh <<'EOF'` ... `bash job.sh`: the body is code, not data)."""
    data_dropped = mask_heredoc_bodies_for_shell_scan(command, code_consumers=False)
    if shell_names_re not in _CAT_WRITERS:
        _CAT_WRITERS[shell_names_re] = DownloadThenRunMatcher(shell_names_re, writer_re=re.compile(r"\bcat\b"))
    runs_written = data_dropped != command and _CAT_WRITERS[shell_names_re].search(data_dropped)
    code_dropped = mask_heredoc_bodies_for_shell_scan(command, code_consumers=True)
    return tuple(dict.fromkeys((data_dropped, code_dropped, *((command,) if runs_written else ()))))
