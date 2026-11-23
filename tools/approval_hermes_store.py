"""Shell-path patterns for the Hermes home, its protected stores (#109367) and the other sensitive write targets.

The terminal approval gate (tools.approval_detection) prompts before destructive verbs reach these
targets; the store list is HERMES_PROTECTED_SUBPATHS, shared with agent.file_safety's write deny.
The sensitive-path fragments (credential, SSH, shell-rc, Hermes config/env and system-config targets)
live here too: the detection tables compose them, and that module is at its size cap.
"""
import functools

import hermes_re as re

from agent.file_safety import SHELL_RC_FILES
from hermes_cli.home_data_layout import HERMES_PROTECTED_SUBPATHS

# The Hermes home itself and the application-owned stores the file tools refuse to write
# (HERMES_PROTECTED_SUBPATHS: state.db, sessions/, vault/, ...). Destroying them is recoverable only
# from backup, so destructive shell verbs on these targets prompt; reads and copies OUT stay clean.
_HERMES_HOME_ROOT = r'(?:~/\.hermes|(?:\$home|\$\{home\})/\.hermes|(?:\$hermes_home|\$\{hermes_home\}))'
_PATH_WORD_END = r'["\']?(?=[\s"\'`;|&<>()]|$)'
# A store, its SQLite sidecars (state.db-wal) or a path inside a directory store (sessions/x.jsonl).
# Neighbours such as state.db.bak or sessions-old.tar are not stores (agent.file_safety agrees).
_HERMES_STORE_FILE = (
    _HERMES_HOME_ROOT + r'/(?:' + '|'.join(
        re.escape(sub) + (r'(?:-wal|-shm|-journal)?' if sub.endswith('.db') else r'(?:/[^\s"\'`;|&<>()]*)?')
        for sub in HERMES_PROTECTED_SUBPATHS) + r')' + _PATH_WORD_END
)
# Overwrites inside credential directories must prompt (vault.key, MCP OAuth tokens, browser cookies).
# New session transcript files stay clean, as before; only the sessions/ root is an overwrite target.
_HERMES_STORE_ROOT_WORD = (
    _HERMES_HOME_ROOT + r'/(?:' + '|'.join(
        re.escape(sub) + (
            r'(?:-wal|-shm|-journal)?' if sub.endswith('.db')
            else r'/?' if sub == 'sessions' else r'(?:/[^\s"\'`;|&<>()]*)?'
        )
        for sub in HERMES_PROTECTED_SUBPATHS) + r')'
)
_HERMES_STORE_ROOT = _HERMES_STORE_ROOT_WORD + _PATH_WORD_END
_HERMES_HOME_BARE = rf'{_HERMES_HOME_ROOT}/?\*?{_PATH_WORD_END}'
_HERMES_STORE_PATH = rf'(?:{_HERMES_STORE_FILE}|{_HERMES_HOME_BARE})'


# Sensitive write targets, matched via ~ / $HOME / $HERMES_HOME spellings. The resolved absolute
# home is folded into these forms at detection time by _normalize_command_for_detection(), so no
# import-time path snapshot (stale once HERMES_HOME is set after import) lives in the patterns.
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|\n?\Z)'
# A Hermes home: the default root or a named profile under it (profiles/<name>/ has its own config.yaml/.env).
_HERMES_HOME_DIR = (
    r'(?:~\/\.hermes/|(?:\$home|\$\{home\})/\.hermes/|(?:\$hermes_home|\$\{hermes_home\})/)'
    r'(?:profiles/[^/\s"\'`]+/)?'
)
_HERMES_ENV_PATH = (
    _HERMES_HOME_DIR + r'(?:\.op)?\.env\b'
)
# ~/.hermes/config.yaml IS the security policy (approvals.mode, yolo, allowlist) and the config cache is mtime-keyed,
# so a write takes effect mid-session. Terminal-side coverage (sed -i, tee, >, cp) pairs the file_tools deny.
_HERMES_CONFIG_PATH = (
    _HERMES_HOME_DIR + r'config\.yaml\b'
)
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)?)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = rf"(?:~|\$home|\$\{{home\}})/(?:{'|'.join(re.escape(f) for f in SHELL_RC_FILES)})\b"
_CREDENTIAL_FILES = r'(?:~|\$home|\$\{home\})/\.' r'(?:netrc|pgpass|npmrc|pypirc)\b'
_HOME_PREFIX = r'(?:~|\$home|\$\{home\})/'
_SECRET_KEY_FILE = r'id_(?:rsa|ed25519|ecdsa|dsa)\b'
_SECRET_NAMED_PATH_FILE = (
    rf'{_HOME_PREFIX}\.(?:aws/credentials|git-credentials|config/gh/hosts\.yml|docker/config\.json|kube/config)\b'
    r'|/etc/(?:shadow|passwd)\b'
)
_SECRET_FILE_NAMED = rf'{_SECRET_NAMED_PATH_FILE}|\b{_SECRET_KEY_FILE}'
_SECRET_PATH_FILE = (
    rf'{_SSH_SENSITIVE_PATH}|{_HERMES_ENV_PATH}|{_CREDENTIAL_FILES}|{_PROJECT_ENV_PATH}(?![\w-])'
    rf'|{_SECRET_NAMED_PATH_FILE}'
)
_SECRET_VAR = r'\$\{?\w*(?:key|token|secret|passwd|password|credential)\w*\}?'
# macOS: /etc, /var, /tmp, /home are symlinks to /private/*, so /private/etc/sudoers would bypass a plain
# "/etc/" check. Match both forms.
_MACOS_PRIVATE_SYSTEM_PATH = r'/private/(?:etc|var|tmp|home)/'
_SYSTEM_CONFIG_PATH = rf'(?:/etc/|{_MACOS_PRIVATE_SYSTEM_PATH})'
# Two groups, for a rule that reads on past the target: a directory-like prefix takes any rest of its word, a
# named file ends at a word boundary.
_SENSITIVE_WRITE_DIR = rf'(?:{_SYSTEM_CONFIG_PATH}|/dev/sd|{_SSH_SENSITIVE_PATH})'
_SENSITIVE_WRITE_FILE = rf'(?:{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH}|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
_SENSITIVE_WRITE_TARGET = rf'(?:{_SENSITIVE_WRITE_DIR}|{_SENSITIVE_WRITE_FILE})'
_USER_SENSITIVE_WRITE_TARGET = rf'(?:{_SSH_SENSITIVE_PATH}|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
# touch/mkdir/ln: the home-side targets only; /etc is out of their scope (#85321) and admin commands
# like `mkdir -p /etc/apt/keyrings` would be mislabelled as credential/SSH/shell-rc writes.
_HOME_SENSITIVE_WRITE_TARGET = rf'(?:{_USER_SENSITIVE_WRITE_TARGET}|{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH})'
# Same set for touch/mkdir/ln, but a named FILE must end the shell word: `ln -s a ~/.bashrc.bak` /
# `touch ~/.bashrc.d/x` name a sibling, not the startup or credential file (`~/.ssh/` is a directory: any entry;
# a bare `~/.ssh` operand anywhere in the command names the directory itself, `~/.sshx` does not).
# A glued redirect (`touch ~/.bashrc>/dev/null`) ends the word too; the ln rule also allows trailing redirects.
_HOME_SENSITIVE_CREATE_FILE = (
    rf'(?:{_HOME_PREFIX}\.ssh|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES}|{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH})'
)
_HOME_SENSITIVE_CREATE_TARGET = rf'(?:{_SSH_SENSITIVE_PATH}|{_HOME_SENSITIVE_CREATE_FILE}' r'(?=["\'\s;&|)`<>]|$))'
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
# cp/mv/install: the sensitive path is a write target only as the LAST argument (destination), so
# `cp config.yaml backup.yaml` (config.yaml as SOURCE) stays out.
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'
# `>`/`>>`/tee: the path is ALWAYS a write target regardless of what follows, so only require a
# shell word boundary (_COMMAND_TAIL let `echo x > .env extra` / `echo x > .env # note` slip past).
# `#` is deliberately NOT a boundary: a glued `#` is part of the filename (`.env#backup`).
_WRITE_TARGET_BOUNDARY = r'(?=[\s;&|<>"\']|$)'

# `sqlite3` writing a store. The stdlib rule was `sqlite3\b(?=[^\n]*STORE)(?=[^\n]*WRITE)`: the command's line
# names a store file AND a write (a destructive SQL word, `.restore` or a heredoc). RE2 has no lookahead, so
# SqliteStoreWriteMatcher finds the command and runs each condition as its own search from where it ends.
_FLAGS = re.IGNORECASE | re.DOTALL
_SQLITE3_WORD_RE = re.compile(r'sqlite3\b', _FLAGS)
_STORE_FILE_RE = re.compile(_HERMES_STORE_FILE, _FLAGS)
# Write candidates: `vacuum` only rewrites the file when no `[schema] into` follows, and `<<` is a heredoc only as
# a run of exactly two `<` (the stdlib rule's `vacuum(?!\s+(?:\w+\s+)?into)` and `(?<!<)<<(?!<)`).
_SQL_WRITE_RE = re.compile(
    r'\b(?:delete\s+from|drop\s+\w+|update\s+(?:or\s+\w+\s+)?\S+\s+(?:as\s+\S+\s+)?set|insert\s+(?:or\s+\w+\s+)?into'
    r'|replace\s+into|alter\s+table|vacuum)\b|\.restore\b|<+', _FLAGS)
_VACUUM_INTO_RE = re.compile(r'\s+(?:\w+\s+)?into', _FLAGS)


def _sql_write_start(text: str, pos: int) -> int | None:
    for write in _SQL_WRITE_RE.finditer(text, pos):
        word = write.group().lower()
        if not (word == 'vacuum' and _VACUUM_INTO_RE.match(text, write.end())
                or word.startswith('<') and len(word) != 2):
            return write.start()
    return None


class _RepeatEnds:
    """Where a run of *step* matches (each starting where the previous one ended) that starts at a position ends.
    Every step boundary a walk passes is remembered, so the many candidates inside one long run (`ln a ~/.ssh/"1> ...`,
    `gc -cat -cat ...`) do not each walk the rest of it. Each step is matched alone, so none gives anything back:
    the run is the possessive `(?:step)*+` of the stdlib rules it stands in for."""

    def __init__(self, step, text: str):
        self._step, self._text, self._ends = step, text, {}

    def at(self, pos: int) -> int:
        walked = []
        while (end := self._ends.get(pos)) is None:
            walked.append(pos)
            if (step := self._step.match(self._text, pos)) is None or step.end() == pos:
                end = pos
                break
            pos = step.end()
        for at in walked:
            self._ends[at] = end
        return end


class _NextStart:
    """Where *find(text, pos)* next matches at or after pos; a found start serves every pos up to it."""

    def __init__(self, find, text: str):
        self._find, self._text, self._pos, self._start = find, text, -1, None

    def at(self, pos: int) -> int:
        if self._pos < 0 or pos < self._pos or (self._start is not None and self._start < pos):
            self._pos, self._start = pos, self._find(self._text, pos)
        return len(self._text) if self._start is None else self._start


# touch/mkdir naming a sensitive home target. `touch -r REF` / `--reference REF` only reads REF (its timestamps), so
# the stdlib rule vetoed the operand right after that flag with lookbehinds, unless a bare `--` came first (the flag
# word is then a file name). RE2 has no lookbehind: _is_create_operand reads the words before each operand.
_CREATE_TARGET_RE = re.compile(r'["\']?' + _HOME_SENSITIVE_CREATE_TARGET, _FLAGS)
_REFERENCE_FLAG_RE = re.compile(r'\s(?:-r|--reference)\s', _FLAGS)
_END_OF_OPTIONS_RE = re.compile(r'\s--', _FLAGS)
_BLANK_RE = re.compile(r'\s')
_SEGMENT_END_RE = re.compile(r'[\n;&|]')
# ln: the stdlib rule reads on from the target with possessive runs: the rest of its word, a closing quote, every
# redirect (`2>/dev/null`, `>&2`), the blanks, and then the command must end. Each run is matched on its own here, so
# none gives anything back to the next (`\s*+` keeps a newline, which then cannot end the command).
_LINK_WORD_REST_RE = re.compile(r'[^\s"\'<>;&|)`]*["\']?')
_LINK_REDIRECT_RE = re.compile(r'\s*\d*[<>]+&?\s*[^\s<>;&|)`]*')
_LINK_BLANKS_RE = re.compile(r'\s*')


def _is_create_operand(text: str, at: int) -> bool:
    if _CREATE_TARGET_RE.match(text, at) is None:
        return False
    for width in (4, 13):  # `\s-r\s`, `\s--reference\s` ending at *at*
        flag = at - width
        if flag >= 0 and _REFERENCE_FLAG_RE.fullmatch(text, flag, at):
            return flag >= 3 and _END_OF_OPTIONS_RE.fullmatch(text, flag - 3, flag) is not None
    return True


def _is_link_destination(redirects: _RepeatEnds, text: str, at: int) -> bool:
    if (target := _CREATE_TARGET_RE.match(text, at)) is None:
        return False
    end = _LINK_BLANKS_RE.match(text, redirects.at(_LINK_WORD_REST_RE.match(text, target.end()).end())).end()
    return end == len(text) or text[end] in ';&|)`'


class SegmentOperandMatcher:
    """``search(text)`` -> the command (``command_re``) with an operand that ``operand(text)(at)`` accepts: one that
    starts after any blank from the command word's end to where its segment ends (a newline blank included) or, with
    ``first``, right at the command word's end. ``operand(text)`` is built once per text, so it can keep state."""

    def __init__(self, command_re, operand, first: bool):
        self._command_re, self._operand_for, self._first = command_re, operand, first

    def search(self, text: str):
        pos, checked = 0, (0, 0)
        operand = self._operand_for(text)
        while (command := self._command_re.search(text, pos)) is not None:
            start = command.end()
            if self._first and operand(start):
                return command
            if not checked[0] <= start < checked[1]:  # a later command in a checked segment has no new operand
                end = _SEGMENT_END_RE.search(text, start)
                stop = len(text) if end is None else end.start() + (end.group() == '\n')
                if any(operand(blank.end()) for blank in _BLANK_RE.finditer(text, start, stop)):
                    return command
                checked = (start, stop)
            pos = command.start() + 1
        return None


def home_create_target_matcher(command_prefix: str) -> SegmentOperandMatcher:
    """touch/mkdir with a sensitive home target among its operands that is not a reference operand."""
    return SegmentOperandMatcher(re.compile(command_prefix + r'(?:touch|mkdir)\s', _FLAGS),
                                 lambda text: functools.partial(_is_create_operand, text), True)


def link_destination_matcher(command_prefix: str) -> SegmentOperandMatcher:
    """ln whose operand after a blank is a sensitive home target that ends the command (redirects aside)."""
    return SegmentOperandMatcher(re.compile(command_prefix + r'ln\s', _FLAGS),
                                 lambda text: functools.partial(_is_link_destination, _RepeatEnds(_LINK_REDIRECT_RE, text), text), False)


class SqliteStoreWriteMatcher:
    """``search(text)`` -> the `sqlite3` command whose line names a store and a write after the command word."""

    def __init__(self, command_prefix: str):
        self._command_re = re.compile(command_prefix + r'sqlite3\b', _FLAGS)

    def search(self, text: str):
        store = _NextStart(lambda t, p: (m.start() if (m := _STORE_FILE_RE.search(t, p)) else None), text)
        write = _NextStart(_sql_write_start, text)
        pos = 0
        while (command := self._command_re.search(text, pos)) is not None:
            # The regex rule could end the command word at any `sqlite3` its path names (`sqlite3.restore/sqlite3`);
            # an earlier end sees more of its line, so every one in the match is tried.
            for word in _SQLITE3_WORD_RE.finditer(text, command.start(), command.end()):
                line_end = text.find('\n', word.end())
                line_end = len(text) if line_end < 0 else line_end
                if store.at(word.end()) < line_end and write.at(word.end()) < line_end:
                    return command
            pos = command.start() + 1
        return None
