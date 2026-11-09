"""Command-position anchor and host power-state (reboot/shutdown) hardline rules.

Split out of :mod:`tools.approval_detection` so the power-state opt-in tables live in one topical
module; callers import them from here directly. Pure data and regex fragments: no config reads.
"""

# Start-of-command position: start of string, newline, subshell opener ($( or backtick), optionally consuming
# sudo/env/exec/nohup/setsid/time wrappers. Keeps shutdown/reboot rules from firing on "echo reboot" / "grep
# 'shutdown' log". Real ;/&/| separators are converted to newlines by the quote-aware _mark_command_starts pass;
# keeping them here mistakes quoted data (grep '(safe|rm -rf /)') for commands.
#
# Every quantifier is possessive (``*+`` / ``++`` / ``?+``, Python 3.11+): the pieces are separated by
# whitespace, so there is only one way to split a prefix and backtracking into it can never produce a
# new match. Greedy quantifiers here let the engine retry every split of a long whitespace run
# between the adjacent ``\s`` runs (quadratic: ~10 s per rule set on a 4095-char separator-free input).
_CMDPOS = (
    r'(?:^|[\n`]|\$\()' r'\s*+'  # start position, optional whitespace
    r'(?:sudo\s++(?:-[^\s]++\s++)*+)?+' r'(?:env\s++(?:\w++=\S*+\s++)*+)?+'  # optional sudo + flags, env VAR=VAL
    r'(?:(?:exec|nohup|setsid|time)\s++)*+'  # optional wrapper commands
    r'(?:command\s++(?:-p\s++)?+)?+'  # `command [-p] NAME` runs NAME (`command -v NAME` only looks it up)
)

# Optional directory prefix on a command word: `/sbin/`, `/usr/local/sbin/`, `./`, `../bin/`, `~/bin/`.
# Path characters cannot include whitespace, quotes, separators or parentheses, so the prefix is
# confined to the single command word that _CMDPOS already anchored (`ls /sbin/reboot` stays
# an argument; `$(` / backtick openers are consumed by _CMDPOS itself).
# Each segment excludes `/` itself so the repetition is unambiguous (one way to split `////x`);
# allowing `/` inside the segment class made `(?:[^…]*/)+` backtrack exponentially on long slash runs.
_CMD_PATH_PREFIX = r'(?:[^\s\'"`;|&()<>\\/]*/)+'
# What may follow a path-prefixed command name: end of the command word. Stricter than `\b` so a
# path-prefixed script like `/opt/reboot-helper.sh` or `/usr/bin/ddrescue` is not mistaken for the
# real binary (bare names keep their historical `\b` boundary).
_CMD_WORD_END = r'(?=[\s;&|)`<>]|$)'


def _cmd_name(name_re: str) -> str:
    """Command-name matcher for _CMDPOS-anchored rules: the bare name (`reboot`) or the same name
    behind a directory prefix (`/sbin/reboot`, `/usr/bin/systemctl`). Without the prefix branch
    absolute-path spellings slipped past every command-name hardline rule."""
    return rf'(?:{_CMD_PATH_PREFIX}(?:{name_re}){_CMD_WORD_END}|(?:{name_re})\b)'


# ---- Host power-state (reboot / shutdown) rules -------------------------------------------
# `shutdown` reboots only with -r/--reboot; -h/-H/-P/--halt/--poweroff (or no flag) halt the host.
# Variants are fully lowercased before matching, so -[a-z]*[hp] also covers -H/-P. Real ;/&/|
# separators are already newlines here, so [^\n]* scans only this command's arguments.
_SHUTDOWN_REBOOT_FLAG = r'(?=[^\n]*\s(?:-[a-z]*r[a-z]*|--reboot)\b)'
_SHUTDOWN_HALT_FLAG = r'(?=[^\n]*\s(?:-[a-z]*[hp][a-z]*|--halt|--poweroff)\b)'
# systemctl global flags may precede the verb (`systemctl --force reboot`, `systemctl -i poweroff`).
_SYSTEMCTL_PREFIX = _CMDPOS + _cmd_name(r'systemctl') + r'\s+(?:-[^\s]+\s+)*'
_REBOOT_DESC = "system reboot"
_REBOOT_INIT_DESC = "init/telinit 6 (reboot)"
_REBOOT_SYSTEMCTL_DESC = "systemctl reboot/kexec"
_SHUTDOWN_DESC = "system shutdown/poweroff/halt"
_SHUTDOWN_INIT_DESC = "init/telinit 0 (shutdown)"
_SHUTDOWN_SYSTEMCTL_DESC = "systemctl poweroff/halt (shutdown)"
_POWER_STATE_PATTERNS = [
    # reboot category
    (_CMDPOS + _cmd_name(r'reboot'), _REBOOT_DESC),
    (_CMDPOS + _cmd_name(r'shutdown') + _SHUTDOWN_REBOOT_FLAG, _REBOOT_DESC),
    (_CMDPOS + _cmd_name(r'(?:halt|poweroff)') + r'[^\n]*\s--reboot\b', _REBOOT_DESC),
    (_CMDPOS + _cmd_name(r'(?:tel)?init') + r'\s+6\b', _REBOOT_INIT_DESC),
    (_SYSTEMCTL_PREFIX + r'(?:reboot|kexec)\b', _REBOOT_SYSTEMCTL_DESC),
    # shutdown / poweroff / halt category
    (_CMDPOS + _cmd_name(r'(?:halt|poweroff)'), _SHUTDOWN_DESC),
    # Plain `shutdown` (no -r) or any shutdown carrying a halt/poweroff flag.
    (_CMDPOS + _cmd_name(r'shutdown') + r'(?:(?!' + _SHUTDOWN_REBOOT_FLAG + r')|' + _SHUTDOWN_HALT_FLAG + r')', _SHUTDOWN_DESC),
    (_CMDPOS + _cmd_name(r'reboot') + r'[^\n]*\s(?:-[a-z]*p[a-z]*|--poweroff|--halt)\b', _SHUTDOWN_DESC),
    (_CMDPOS + _cmd_name(r'(?:tel)?init') + r'\s+0\b', _SHUTDOWN_INIT_DESC),
    (_SYSTEMCTL_PREFIX + r'(?:poweroff|halt)\b', _SHUTDOWN_SYSTEMCTL_DESC),
]
# Hardline descriptions an operator may opt out of the floor, keyed to their category. An opted-in
# category is NOT auto-run: it goes to a human-only gate (tools.approval._power_command_gate) that
# always asks a live person once — smart mode, allowlists, session approvals and yolo/off cannot
# approve it, and contexts with nobody to ask (cron, -q, unattended platforms) deny it.
HARDLINE_OPT_IN_CATEGORIES = {
    _REBOOT_DESC: "reboot", _REBOOT_INIT_DESC: "reboot", _REBOOT_SYSTEMCTL_DESC: "reboot",
    _SHUTDOWN_DESC: "shutdown", _SHUTDOWN_INIT_DESC: "shutdown", _SHUTDOWN_SYSTEMCTL_DESC: "shutdown",
}
# config.yaml key (under ``approvals``) that opts each category out of the hardline floor.
HARDLINE_OPT_IN_SETTINGS = {"reboot": "approvals.allow_reboot", "shutdown": "approvals.allow_shutdown"}
# Second, separate level per category: with BOTH approvals.allow_<cat> and approvals.<cat>_without_approval
# true, an interactive session runs that category without a prompt. The without_approval key alone has
# no effect (the category stays on the hardline floor), and unattended contexts still deny.
HARDLINE_NO_APPROVAL_SETTINGS = {
    "reboot": "approvals.reboot_without_approval", "shutdown": "approvals.shutdown_without_approval",
}
# Any of the four keys above, as a whole word (bare or ``approvals.``-prefixed). Commands that write
# one are a hardline rule of their own in tools.approval_detection (never opt-in).
_POWER_SETTING_KEY_RE = r'\b(?:allow_reboot|allow_shutdown|reboot_without_approval|shutdown_without_approval)\b'


def hardline_opt_in_setting(description: str) -> str | None:
    """Config key that would lift this hardline block (``None`` for non-opt-in rules)."""
    category = HARDLINE_OPT_IN_CATEGORIES.get(description)
    return HARDLINE_OPT_IN_SETTINGS.get(category) if category else None
