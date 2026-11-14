"""Same-string download-then-run detection for the approval gate: a curl/wget writes FILE
(``-o``/``-O``/``--output``/``>``) and a later command in the same string runs that FILE (via a
shell, ``source``/``.``, ``./FILE`` or the bare path). Cross-call download then exec is out of
reach of a stateless detector.

Two linear steps instead of one backreferencing regex: a lazy ``[^\\n]*?`` between the target
and ``(?P=dl)`` retried from every fetch made 64 KB of ``curl -o f`` take tens of seconds. Step 1
collects targets per fetch segment, step 2 checks each later command start against that set."""

from __future__ import annotations

import re

_FETCH_RE = re.compile(r"\b(?:curl|wget)\b", re.IGNORECASE)
_SEGMENT_END_RE = re.compile(r"[\n;&|]")
_TARGET_RE = re.compile(
    r"(?:\s-[a-z]*o\s*+|\s--output(?:-document)?(?:=|\s++)|>[>|]?\s*+)['\"]?([^\s;&|<>'\"`()]++)",
    re.IGNORECASE,
)
_TERMINATOR = r"(?=[\s;&|)`'\"]|$)"
_SCRIPT_INTERPRETERS_RE = r"python[0-9.]*+|perl|ruby|node|php"
# curl -O / --remote-name saves under the URL's basename, not an operand: `curl -fsSLO URL`.
_CURL_REMOTE_NAME_RE = re.compile(r"\s(?:-([A-Za-z]++)|--remote-name(?:-all)?)(?=\s|$)")
_URL_RE = re.compile(r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*+://[^\s;&|<>'\"`()]++", re.IGNORECASE)


def _base(path: str) -> str:
    # Basename on both sides: a `cd` between write and run (`> /tmp/j; cd /tmp; sh j`) still matches.
    return path.rstrip("/").rsplit("/", 1)[-1]


class DownloadThenRunMatcher:
    """``search(text)`` -> a match whose span covers the run of a just-downloaded file, or None."""

    def __init__(self, shell_names_re: str, writer_re: re.Pattern = _FETCH_RE):
        self._writer_re = writer_re
        # Launchers and script interpreters run FILE too (`env bash f`, `nohup sh f`, `python3 f`).
        self._run_re = re.compile(
            r"(?:;|&&|\|\||\n)\s*+(?:(?:sudo|env|exec|nohup|command)\s++(?:-\S++\s++|\w++=\S*+\s++)*+)*+"
            rf"(?:(?:{shell_names_re}|{_SCRIPT_INTERPRETERS_RE})\s++(?:-\S++\s++)*+|(?:source|\.)\s++)?"
            rf"['\"]?(?:\./)?([^\s;&|<>'\"`()]++){_TERMINATOR}",
            re.IGNORECASE,
        )

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
            if run.start() >= targets.get(_base(run.group(1)), len(text)):
                return run
        return None
