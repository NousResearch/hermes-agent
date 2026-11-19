"""`wmic ... shadowcopy ... delete` matcher for the approval gate.

Moved out of :mod:`tools.approval_detection` (size cap); ``DANGEROUS_PATTERNS_COMPILED`` swaps it in
for the wmic row.
"""

import re

_WMIC_DELETE_IN_SEGMENT_RE = re.compile(r"\sshadowcopy\b.*\sdelete\b", re.IGNORECASE | re.DOTALL)
# Leading switches of a carrier before its script: cmd /s /q /c, start /b "title", pwsh -NoProfile
# -ExecutionPolicy Bypass -Command, doas -u x, ssh -i k -p 22 host. Option values are taken only for
# the options known to take one, so `powershell -NoProfile wmic ...` keeps wmic as the script.
_CARRIER_SWITCHES_RE = re.compile(
    r"""(?:\s*+(?:/\w++(?::\S*+)?|-(?:ExecutionPolicy|ep|ex|WindowStyle|w|Version|v|WorkingDirectory|wd"""
    r"""|ConfigurationName|config|SettingsFile|settings|OutputFormat|of|InputFormat|inp|if|u|p|i|o|l|J|F|a)"""
    r"""\s++(?:"[^"\n]*+"|'[^'\n]*+'|[^\s;&|]++)|--?\w*+))*+""",
    re.IGNORECASE,
)
# `start "title" prog`: start alone takes a quoted window title before the program.
_START_TITLE_RE = re.compile(r"""\s*+"[^"\\\n]*+"(?=\s*+["\w\\/])""")


class WmicShadowDeleteMatcher:
    """`wmic ... shadowcopy ... delete` run as a command: the program word (bare, a full path, behind
    sudo/env/nohup...) is wmic and its own command carries shadowcopy and then delete. The script of a
    carrier (cmd /c, pwsh, powershell, start, call, doas, wsl, xargs, ssh host) is walked the same way,
    so `cmd /c "echo hi & wmic shadowcopy delete"` counts. Nesting is capped, so the scan stays linear."""

    _CARRIERS = frozenset({"cmd", "pwsh", "powershell", "start", "call", "doas", "wsl", "xargs", "ssh"})
    _MAX_DEPTH = 3

    def search(self, text: str, depth: int = 0):
        from tools.approval_detection import (
            _deobfuscate_shell_word_for_detection, _iter_shell_command_word_spans, _shell_command_segment,
        )

        if "wmic" not in text.lower():
            return None
        for _, end, word in _iter_shell_command_word_spans(text):
            # Split the path before deobfuscating: the shell reading drops the backslashes of C:\...\wmic.
            name = _deobfuscate_shell_word_for_detection(re.split(r"[\\/]", word)[-1]).strip("\"'").lower().removesuffix(".exe")
            if name == "wmic":
                if match := _WMIC_DELETE_IN_SEGMENT_RE.search(" " + _shell_command_segment(text, end)):
                    return match
            elif name in self._CARRIERS and depth < self._MAX_DEPTH:
                for script in self._carrier_script(name, _shell_command_segment(text, end)):
                    if match := self.search(script, depth + 1):
                        return match
        return None

    @staticmethod
    def _carrier_script(name: str, rest: str) -> tuple[str, ...]:
        rest = _CARRIER_SWITCHES_RE.sub("", rest, count=1)
        if name == "start":
            rest = _CARRIER_SWITCHES_RE.sub("", _START_TITLE_RE.sub("", rest, count=1), count=1)
        rest = rest.lstrip()
        if name == "ssh":  # the first word after the options is the host
            rest = re.sub(r"^[^\s;&|]+", "", rest, count=1)
            rest = _CARRIER_SWITCHES_RE.sub("", rest, count=1).lstrip()
        rest = re.sub(r"^-(?:Command|c)\b\s*", "", rest, flags=re.IGNORECASE)
        rest = rest.rstrip().replace('\\"', '"')
        if rest[:1] not in ("'", '"'):
            return (rest,)
        # A leading quote opens either the whole script (`"wmic shadowcopy delete"`) or just the program
        # word (`"wmic" shadowcopy delete`); the normalized variant drops `\"` pairs, so its closing quote
        # can be missing. Read it both ways.
        return (rest, rest[1:-1] if len(rest) > 1 and rest[-1] == rest[0] else rest[1:])
