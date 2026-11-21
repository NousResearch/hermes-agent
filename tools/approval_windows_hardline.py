"""Windows hardline rules (volume wipes, drive-root deletes, power-off), split out of approval_detection.

Regex text and two small matcher classes, built from the caller's command-position prefix so this module needs no
import back.
"""

import bisect
import hermes_re as re

# Windows drive root (`C:/`, `C:/*`, `C:/*.*`; backslashes are flattened to `/` by the
# Windows-path detection variant).
_HARDLINE_DRIVE_ROOT = r'[a-z]:/+(?:\*(?:\.\*)?)?'
# Quoted root that ENDS its shell word (past `./`, `..`, globs; a glued `>nul` ends it too): `"C:\"*.*` is the
# root, `"C:\"Users\me` one word naming a subdirectory. Bare roots carry each rule's own terminator.
_HARDLINE_DRIVE_QUOTED = r'["\'](?:' + _HARDLINE_DRIVE_ROOT + r')["\'](?:(?:\.\.?)?/)*(?:\.\.?)?(?:\*(?:\.\*)?)?(?=\s|$|[)`;|&<>])'
# PowerShell -Recurse and its unambiguous prefixes (same set as approval_detection._PS_RECURSE).
_PS_RECURSE_HL = r'-r(?:e(?:c(?:u(?:r(?:s(?:e)?)?)?)?)?)?\b'
# The drive-root rules pass over a PowerShell delete whose literal -Include narrows it (`*.tmp`): one that is
# not after cmd `/s` or GNU `--` (a file name there), not inside quotes/comments, not `*`/`*.*`/a list/`$x`,
# and never past a `)` that closes rm's own `$(...)`/subshell (`echo $(rm -rf C:/) -Include x` is echo's flag).
# RE2 has no lookahead to ask that inside a rule, so NarrowableDeleteMatcher reads it from these positions.
_DELETE_VERB = r'(?:remove-item|ri|rm|rd|rmdir|del|erase)'
_DELETE_VERB_RE = re.compile(_DELETE_VERB + r'\b', re.IGNORECASE)
_RM_WORD_RE = re.compile(r'rm\s+', re.IGNORECASE)
_RECURSE_SWITCH_RE = re.compile(r'(?<=\s)(?:/s\b|' + _PS_RECURSE_HL + r')', re.IGNORECASE)
# cmd `/s` or GNU `--` as a word of its own; group 1 holds the whitespace before it, if any.
_INCLUDE_VOID_RE = re.compile(r'(\s?)(?:/s\b|--(?=\s|$))', re.IGNORECASE)
_INCLUDE_VALUE_CHAR = r'[^\s,;&|"\'$`(){}@]'
# `-Include <value>` whose value holds a character that is no wildcard and that no comma follows (not a list).
_NARROWING_INCLUDE_RE = re.compile(
    r'[^\S\r\n]-inc(?=[a-z]*(?::|\s+)["\']?' + _INCLUDE_VALUE_CHAR + r'*[^\s,;&|"\'$`(){}@*?.]' + _INCLUDE_VALUE_CHAR
    + r'*["\']?(?:[;&|)]|\s+[^\s,]|\s*$))', re.IGNORECASE)
_STATEMENT_BREAK_RE = re.compile(r'[\r\n;&|]')
_INCLUDE_SCAN_STOPS = '\r\n;&|#`)'
# A cmdlet name ends its token: `Stop-Computer.ps1` / `Clear-Disk-Report` name other commands.
_CMDLET_END = r'(?![\w.\\/-])'
# The same end for a cmdlet that is the command word itself (no launcher before it): its word must not run on into
# a `/` either, `clear-disk+tools/run.sh` being a program path whose first directory merely starts like the cmdlet.
_CMDLET_WORD_END = r'(?=(?:[^\w.\\/\-\s)=<>][^\s/)=<>]*)?(?:[\s)=<>]|$))'
# What the -WhatIf exemption reads (see _mask_previewed_cmdlets), as positions instead of nested lookaheads.
_GUARDED_CMDLET_RE = re.compile(r'(\s?)(?:clear-disk|(?:restart|stop)-computer)' + _CMDLET_END, re.IGNORECASE)
_WHATIF_SWITCH_RE = re.compile(r'\s+-whatif(?=\s|["\']?$)', re.IGNORECASE)
_WHATIF_VALUE_RE = re.compile(r'-whatif:', re.IGNORECASE)
_STATEMENT_BREAKS = '\r\n;&|'
_NOT_PLAIN_ARGUMENT = _STATEMENT_BREAKS + '"\'`(){}$@#'
# A powershell/pwsh/cmd launcher with its switches (a switch may take one bare value, `-ExecutionPolicy Bypass`,
# when another switch or the script's opening quote follows it) and an opening quote around the script.
# `-File` is not skipped: its operand is a script path (`pwsh -File Stop-Computer.ps1`), not code. A switch word
# ends at a blank, so "not f / fi / fil / file before a blank or `:`" is spelled per prefix.
_WIN_SWITCH = (r'[^\S\r\n]+(?:-(?:[^\s"\'f]|f[^\s"\'i:]|fi[^\s"\'l:]|fil[^\s"\'e:]|file[^\s"\':])[^\s"\']*'
               r'|/[^\s"\']+)')
_WIN_SWITCH_VALUE = r'[^\S\r\n]+\w+'
_WIN_SWITCHES = _WIN_SWITCH + r'(?:(?:' + _WIN_SWITCH_VALUE + r')?' + _WIN_SWITCH + r')*'
_WIN_LAUNCHER = (r'(?:[^\s;&|"\'`]*[\\/])?(?:powershell|pwsh|cmd)(?:\.exe)?'
                 r'(?:' + _WIN_SWITCHES + _WIN_SWITCH_VALUE + r'[^\S\r\n]+["\']'
                 r'|(?:' + _WIN_SWITCHES + r')?[^\S\r\n]+["\']?)')
# A bare `-WhatIf` of the same invocation only previews; a quote/variable/splat/subexpression/comment before it, a
# comma right before it (`srv1, -WhatIf` is an array element), or any `-WhatIf:<value>`, voids that (a closing quote may only end it: `pwsh -c "… -WhatIf"`).
# A bare CR ends the statement too, and a guarded cmdlet named again later voids it: a quoted `pwsh -c` newline is
# masked to a space, so `-WhatIf\nClear-Disk 1` reaches this as `-WhatIf Clear-Disk 1`.
_PS_WHATIF_GUARDED = r'(?<![\w.\\/-])(?:clear-disk|(?:restart|stop)-computer)' + _CMDLET_END
# A guarded cmdlet with no real -WhatIf of its own, judged per raw statement (CR/LF quoted or not, `;`, `|`, `&`).
# RE2 has no nested lookahead, so the three conditions above (a bare -WhatIf of its own, no `-WhatIf:<value>`, no
# guarded cmdlet named again) are matched from the end of each cmdlet name.
_WHATIF_GUARDED_NAME_RE = re.compile(_PS_WHATIF_GUARDED, re.IGNORECASE)
_WHATIF_OWN_SWITCH_RE = re.compile(
    r'(?:[^\r\n;&|"\'`(){}$@#]*[^\r\n;&|"\'`(){}$@#\s,])?\s+-whatif(?:\s|["\']?$)', re.IGNORECASE)
_WHATIF_OWN_VALUE_RE = re.compile(r'[^\r\n;&|"\'`(){}$@#]*-whatif:', re.IGNORECASE)
_WHATIF_LATER_CMDLET_RE = re.compile(
    r'[^\r\n;&|]*\s(?:clear-disk|(?:restart|stop)-computer)' + _CMDLET_END, re.IGNORECASE)


def _statement_is_unguarded(statement: str) -> bool:
    """Whether ``statement`` names a guarded cmdlet that its own bare ``-WhatIf`` does not cover."""
    for name in _WHATIF_GUARDED_NAME_RE.finditer(statement):
        end = name.end()
        if (not _WHATIF_OWN_SWITCH_RE.match(statement, end) or _WHATIF_OWN_VALUE_RE.match(statement, end)
                or _WHATIF_LATER_CMDLET_RE.match(statement, end)):
            return True
    return False


# Any other destructive verb anywhere voids the preview: a masked quoted newline joins its statement onto the
# guarded one, so the `-WhatIf` of `pwsh -c "Clear-Disk 0<CR>Format-Volume C -WhatIf"` would cover Clear-Disk.
_WHATIF_OTHER_HARDLINE_RE = re.compile(
    r'(?<![\w.\\/$-])(?:format-volume|format(?:\.com)?|remove-item|ri|rm|rd|rmdir|del|erase|mkfs(?:\.\w+)?|dd|kill|'
    r'shutdown|reboot|halt|poweroff|init|telinit|systemctl)(?![\w-])', re.IGNORECASE)


def whatif_preview_is_real(command: str) -> bool:
    """Whether every guarded cmdlet in ``command`` previews: each statement naming one carries its own real
    ``-WhatIf`` and no other destructive verb appears anywhere (the -WhatIf exemption is per statement).
    PowerShell also ends a statement at NEL, U+2028 and U+2029, so those split too."""
    if _WHATIF_OTHER_HARDLINE_RE.search(command):
        return False
    return not any(_statement_is_unguarded(s) for s in re.split(r'[\r\n\x85\u2028\u2029;&|]', command))


def _first_from(positions: list[int], start: int, missing: int) -> int:
    index = bisect.bisect_left(positions, start)
    return positions[index] if index < len(positions) else missing


def _mask_previewed_cmdlets(text: str) -> str:
    """``text`` with the `-` of every guarded cmdlet name that only previews replaced by `+`.

    A name previews when a bare `-WhatIf` follows it in the same run of plain arguments (no quote, variable, splat,
    subexpression or comment before it, no comma right before it), no `-WhatIf:<value>` does, and no guarded cmdlet
    is named again before the statement ends. `+` is in every character class of the command-position prefix that
    `-` is in and the hyphen of a name never follows a blank, so the swap hides that one name and nothing else."""
    switches = [m.start() for m in _WHATIF_SWITCH_RE.finditer(text) if text[m.start() - 1:m.start()] != ',']
    cmdlets = list(_GUARDED_CMDLET_RE.finditer(text)) if switches else []
    if not cmdlets:
        return text
    values = [m.start() for m in _WHATIF_VALUE_RE.finditer(text)]
    named_again = [m.start() for m in cmdlets if m.group(1)]
    breaks = [i for i, char in enumerate(text) if char in _STATEMENT_BREAKS]
    not_plain = [i for i, char in enumerate(text) if char in _NOT_PLAIN_ARGUMENT]
    size, chars = len(text), list(text)
    for cmdlet in cmdlets:
        end = cmdlet.end()
        plain_end = _first_from(not_plain, end, size)
        if (_first_from(switches, end, size + 1) <= plain_end < _first_from(values, end, size + 1)
                and _first_from(named_again, end, size + 1) > _first_from(breaks, end, size)):
            chars[text.index('-', cmdlet.start())] = '+'
    return ''.join(chars)


class WhatIfExemptMatcher:
    """A -WhatIf-exempt hardline rule: its strict pattern, searched with the previewing cmdlets masked out."""

    def __init__(self, strict):
        self._strict = strict

    def search(self, text: str):
        return self._strict.search(_mask_previewed_cmdlets(text))


def windows_whatif_strict_patterns(cmdpos: str) -> dict[str, str]:
    """The -WhatIf-exempt rules without the exemption, keyed by description (see ``whatif_preview_is_real``)."""
    return {"system shutdown/reboot (PowerShell)": _win_cmdlet(cmdpos, r'(?:restart|stop)-computer'),
            "wipe disk (Clear-Disk)": _win_cmdlet(cmdpos, r'clear-disk')}


def _win_cmdlet(cmdpos: str, name: str) -> str:
    # `name` as a command: after a bare CR, behind a launcher, or as the command word itself.
    return (r'(?:\r[^\S\r]*(?:' + _WIN_LAUNCHER + r')?|' + cmdpos + _WIN_LAUNCHER + r')' + name + _CMDLET_END
            + r'|' + cmdpos + name + _CMDLET_WORD_END)


def _win_cmdpos(cmdpos: str) -> str:
    # cmdpos, optionally followed by a powershell/pwsh/cmd launcher (see _WIN_LAUNCHER).
    # PowerShell's tokenizer also reads a bare CR as a statement break (bash does not, so only here).
    return r'(?:\r[^\S\r]*|' + cmdpos + r')(?:' + _WIN_LAUNCHER + r')?'


def _include_reach(text: str, includes: list[int], lowest: int) -> bytearray:
    """Marks each position from ``lowest`` on whose plain arguments run on to a narrowing -Include: through
    unquoted characters and whole one-line quoted strings, never past a statement break, `#`, a backtick or a `)`."""
    last = includes[-1]
    reach = bytearray(last + 2)
    for start in includes:
        reach[start] = 1
    closing = {'"': -1, "'": -1}
    for index in range(last, lowest - 1, -1):
        char = text[index]
        if char == '\n':
            closing = {'"': -1, "'": -1}
        elif char in closing:
            if closing[char] >= 0:
                reach[index] = reach[closing[char] + 1]
            closing[char] = index
        elif not reach[index] and char not in _INCLUDE_SCAN_STOPS:
            reach[index] = reach[index + 1]
    return reach


def _mask_narrowed_deletes(text: str, verb, recursive: bool) -> str:
    """``text`` with the first letter of every delete verb that cannot be a drive-root wipe replaced by `0`.

    That is a verb whose arguments run on to a narrowing -Include with no `/s` or `--` word in its statement, and,
    for a ``recursive`` rule, a verb with no `/s` / -Recurse switch in its statement. A digit is in every character
    class of the rules that a letter is in and in none of their words, and two verbs never overlap, so the swap
    hides that one verb from the rule and nothing else."""
    verbs = [(found.start(), found.end()) for found in verb.finditer(text)]
    includes = [found.start() for found in _NARROWING_INCLUDE_RE.finditer(text)] if verbs else []
    if not verbs or not (recursive or includes):
        return text
    size = len(text)
    breaks = [found.start() for found in _STATEMENT_BREAK_RE.finditer(text)]
    switches = [found.start() for found in _RECURSE_SWITCH_RE.finditer(text)] if recursive else []
    voids = [(found.end(1), found.group(1)) for found in _INCLUDE_VOID_RE.finditer(text)] if includes else []
    void_here = {at for at, _ in voids}
    void_later = [at for at, blank in voids if blank]
    reach = _include_reach(text, includes, verbs[0][1]) if includes else b''
    chars = list(text)
    for start, end in verbs:
        # A switch word of this statement starts after whitespace, at the latest right after the break that ends it.
        latest = _first_from(breaks, end, size) + 1
        recursed = _first_from(switches, end + 1, size + 2) <= latest
        voided = end in void_here or _first_from(void_later, end + 1, size + 2) <= latest
        narrowed = end < len(reach) and reach[end] and not voided
        if narrowed or (recursive and not recursed):
            chars[start] = '0'
    return ''.join(chars)


class NarrowableDeleteMatcher:
    """A drive-root delete rule that passes over the deletes a literal -Include narrows (_mask_narrowed_deletes)."""

    def __init__(self, verb, rule: str, recursive: bool):
        self._verb, self._recursive = verb, recursive
        self._rule = re.compile(rule, re.IGNORECASE | re.DOTALL)

    def search(self, text: str):
        return self._rule.search(_mask_narrowed_deletes(text, self._verb, self._recursive))


def _win_root_delete(cmdpos: str) -> str:
    # A delete verb as a command (after a bare CR, behind a launcher, or as the command word itself) with a drive
    # root among its arguments. As the command word its word must not run on into a `/`: `./del/run.sh x C:/` is a
    # program path whose directory is merely named like the verb.
    root = r'(?:' + _HARDLINE_DRIVE_QUOTED + '|' + _HARDLINE_DRIVE_ROOT + r'(?:\s|$|[)`;|&"\'<>,]))'
    return (r'(?:\r[^\S\r]*(?:' + _WIN_LAUNCHER + r')?|' + cmdpos + _WIN_LAUNCHER + r')' + _DELETE_VERB
            + r'(?:[^\w\r\n;&|][^\r\n;&|]*)?[\s,]' + root
            + r'|' + cmdpos + _DELETE_VERB
            + r'(?:[^\w\s/)=<>;&|][^\s/)=<>;&|]*)?(?:\s|(?:[)=<>]|[^\S\r\n])[^\r\n;&|]*[\s,])' + root)


def windows_hardline_patterns(cmdpos: str) -> list[tuple[str | NarrowableDeleteMatcher, str]]:
    """The Windows analogues of the shutdown / mkfs / root-rm hardline rules, anchored at ``cmdpos``."""
    win_cmdpos = _win_cmdpos(cmdpos)
    strict = windows_whatif_strict_patterns(cmdpos)
    return [
        # Windows analogues of the shutdown / mkfs / dd / root-rm rules. Command-position anchored like
        # the POSIX rules (#93392), also behind a powershell/pwsh/cmd launcher because cmdlets are
        # only reachable that way from git-bash, so `git commit -m "never format c:"` stays runnable.
        # The two -WhatIf-exempt rules are listed strict: approval_detection compiles them behind WhatIfExemptMatcher.
        (strict["system shutdown/reboot (PowerShell)"], "system shutdown/reboot (PowerShell)"),
        (_win_cmdlet(cmdpos, r'format-volume'), "format filesystem (Format-Volume)"),
        (win_cmdpos + r'format(?:\.com)?\s+[a-z]:', "format drive (format.com)"),
        (strict["wipe disk (Clear-Disk)"], "wipe disk (Clear-Disk)"),
        # Recursive delete of a drive ROOT (`C:\`, `C:\*`, `C:/*.*`; a PowerShell/cmd `,` starts or ends it, as does a glued `>nul`). `C:\build`, `C:` (the drive's cwd)
        # and a literal PowerShell `-Include *.tmp` (only matching items; `*`, a list, `"*$x"`, cmd `/s`, after `--`/`#` still wipe) fall through.
        # The two drive-root rules are matcher objects: approval_detection keeps a non-str entry as it is.
        (NarrowableDeleteMatcher(_DELETE_VERB_RE, _win_root_delete(cmdpos), recursive=True), "recursive delete of drive root"),
        (NarrowableDeleteMatcher(_RM_WORD_RE, cmdpos + r'rm\s+(?:-[^\s]*\s+)*(?:' + _HARDLINE_DRIVE_QUOTED + '|' + _HARDLINE_DRIVE_ROOT + r'(?:\s|$|[)`;|&<>]))',
                                 recursive=False), "recursive delete of drive root"),
    ]
