"""Windows hardline rules (volume wipes, drive-root deletes, power-off), split out of approval_detection.

Pure regex text, built from the caller's command-position prefix so this module needs no import back.
"""

# Windows drive root (`C:/`, `C:/*`, `C:/*.*`; backslashes are flattened to `/` by the
# Windows-path detection variant).
_HARDLINE_DRIVE_ROOT = r'[a-z]:/+(?:\*(?:\.\*)?)?'
# Quoted root that ENDS its shell word (past `./`, `..`, globs; a glued `>nul` ends it too): `"C:\"*.*` is the
# root, `"C:\"Users\me` one word naming a subdirectory. Bare roots carry each rule's own terminator.
_HARDLINE_DRIVE_QUOTED = r'["\'](?:' + _HARDLINE_DRIVE_ROOT + r')["\'](?:(?:\.\.?)?/)*(?:\.\.?)?(?:\*(?:\.\*)?)?(?=\s|$|[)`;|&<>])'
# PowerShell -Recurse and its unambiguous prefixes (same set as approval_detection._PS_RECURSE).
_PS_RECURSE_HL = r'-r(?:e(?:c(?:u(?:r(?:s(?:e)?)?)?)?)?)?\b'
# Vetoes the drive-root rules when THIS PowerShell delete carries a literal -Include that narrows it (`*.tmp`):
# not after cmd `/s` or GNU `--` (a file name there), not inside quotes/comments, not `*`/`*.*`/a list/`$x`.
_PS_INCLUDE_NARROWS = r'(?!(?!(?:[^\n;&|]*\s)?(?:/s\b|--(?:\s|$)))(?:[^\n;&|#"\'`]|"[^"\n]*"|\'[^\'\n]*\')*\s-inc[a-z]*(?::|\s+)["\']?(?=[^\s,;&|"\'$`(){}@]*[^\s,;&|"\'$`(){}@*?.])[^\s,;&|"\'$`(){}@]+["\']?(?!\s*,)(?:[\s;&|)]|$))'
# A cmdlet name ends its token: `Stop-Computer.ps1` / `Clear-Disk-Report` name other commands.
_CMDLET_END = r'(?![\w.\\/-])'
# A bare `-WhatIf` of the same invocation only previews; a quote/variable/splat/subexpression/comment before it, a
# comma right before it (`srv1, -WhatIf` is an array element), or any `-WhatIf:<value>`, voids that (a closing quote may only end it: `pwsh -c "… -WhatIf"`).
# A bare CR ends the statement too, and a guarded cmdlet named again later voids it: a quoted `pwsh -c` newline is
# masked to a space, so `-WhatIf\nClear-Disk 1` reaches this as `-WhatIf Clear-Disk 1`.
_PS_NOT_WHATIF = r'(?!(?=(?:[^\r\n;&|"\'`(){}$@#]*[^\r\n;&|"\'`(){}$@#\s,])?\s+-whatif(?:\s|["\']?$))(?![^\r\n;&|"\'`(){}$@#]*-whatif:)(?![^\r\n;&|]*\s(?:clear-disk|(?:restart|stop)-computer)(?![\w.\\/-])))'


def windows_hardline_patterns(cmdpos: str) -> list[tuple[str, str]]:
    """The Windows analogues of the shutdown / mkfs / root-rm hardline rules, anchored at ``cmdpos``."""
    # cmdpos, optionally followed by a powershell/pwsh/cmd launcher and its switches (a switch may
    # take one bare value: `-ExecutionPolicy Bypass`) and an opening quote around the script.
    # `-File` is not skipped: its operand is a script path (`pwsh -File Stop-Computer.ps1`), not code.
    # PowerShell's tokenizer also reads a bare CR as a statement break (bash does not, so only here).
    win_cmdpos = (r'(?:\r\s*|' + cmdpos + r')(?:(?:[^\s;&|"\'`]*[\\/])?(?:powershell|pwsh|cmd)(?:\.exe)?'
                  r'(?:\s+(?:-(?!f(?:i(?:le?)?)?(?:[\s:]|$))|/)[^\s"\']+(?:\s+\w+(?=\s+[-/"\']))?)*\s+["\']?)?')
    return [
        # Windows analogues of the shutdown / mkfs / dd / root-rm rules. Command-position anchored like
        # the POSIX rules (#93392), also behind a powershell/pwsh/cmd launcher because cmdlets are
        # only reachable that way from git-bash, so `git commit -m "never format c:"` stays runnable.
        (win_cmdpos + r'(?:restart|stop)-computer' + _CMDLET_END + _PS_NOT_WHATIF, "system shutdown/reboot (PowerShell)"),
        (win_cmdpos + r'format-volume' + _CMDLET_END, "format filesystem (Format-Volume)"),
        (win_cmdpos + r'format(?:\.com)?\s+[a-z]:', "format drive (format.com)"),
        (win_cmdpos + r'clear-disk' + _CMDLET_END + _PS_NOT_WHATIF, "wipe disk (Clear-Disk)"),
        # Recursive delete of a drive ROOT (`C:\`, `C:\*`, `C:/*.*`; a glued `>nul` or PowerShell `,` ends it). `C:\build`, `C:` (the drive's cwd)
        # and a literal PowerShell `-Include *.tmp` (only matching items; `*`, a list, `"*$x"`, cmd `/s`, after `--`/`#` still wipe) fall through.
        (win_cmdpos + r'(?:remove-item|ri|rm|rd|rmdir|del|erase)\b(?=[^\n;&|]*\s(?:/s\b|' + _PS_RECURSE_HL + r'))' + _PS_INCLUDE_NARROWS +
         r'[^\n;&|]*\s(?:' + _HARDLINE_DRIVE_QUOTED + '|' + _HARDLINE_DRIVE_ROOT + r'(?:\s|$|[)`;|&"\'<>,]))', "recursive delete of drive root"),
        (cmdpos + r'rm\s+' + _PS_INCLUDE_NARROWS + r'(?:-[^\s]*\s+)*(?:' + _HARDLINE_DRIVE_QUOTED + '|' + _HARDLINE_DRIVE_ROOT + r'(?:\s|$|[)`;|&<>]))', "recursive delete of drive root"),
    ]
