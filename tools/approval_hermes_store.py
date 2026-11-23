"""Shell-path patterns for the Hermes home, its protected stores (#109367) and the other sensitive write targets.

The terminal approval gate (tools.approval_detection) prompts before destructive verbs reach these
targets; the store list is HERMES_PROTECTED_SUBPATHS, shared with agent.file_safety's write deny.
The sensitive-path fragments (credential, SSH, shell-rc, Hermes config/env and system-config targets)
live here too: the detection tables compose them, and that module is at its size cap.
"""
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
# The ln rule's tail, without possessive quantifiers or a lookahead in the middle (RE2 has neither): any number
# of redirects, then the end of the command. A redirect is its operator plus `&` + target, blank + target or a
# glued target; a bare operator can only end the command (an `&` after it belongs to it, never a separator).
# Whitespace before the end may hold newlines, but a newline is not itself an end: more text may follow it.
_LN_REDIRECT = r'[<>]+(?:&\s*[^\s<>;&|)`]*|\s+[^\s<>;&|)`]*|[^\s<>;&|)`]+)'
_LN_BARE_REDIRECT_END = r'[<>]+(?:[;|)`]|\Z)'
_LN_COMMAND_END = r'\s*(?:[;&|)`]|\Z)'
_LN_TAIL = rf'(?:\s*\d*{_LN_REDIRECT})*(?:\s*\d*{_LN_BARE_REDIRECT_END}|{_LN_COMMAND_END})'
# A named file must end its shell word: a closing quote, a redirect not glued to it by fd digits, or the end.
_LN_FILE_TAIL = (rf'(?:["\']{_LN_TAIL}|(?:\s+\d*)?{_LN_REDIRECT}{_LN_TAIL}|(?:\s+\d*)?{_LN_BARE_REDIRECT_END}'
                 rf'|{_LN_COMMAND_END})')
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


class _NextStart:
    """Where *find(text, pos)* next matches at or after pos; a found start serves every pos up to it."""

    def __init__(self, find, text: str):
        self._find, self._text, self._pos, self._start = find, text, -1, None

    def at(self, pos: int) -> int:
        if self._pos < 0 or pos < self._pos or (self._start is not None and self._start < pos):
            self._pos, self._start = pos, self._find(self._text, pos)
        return len(self._text) if self._start is None else self._start


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
