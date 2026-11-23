"""Shell-path patterns for the Hermes home and its protected stores (#109367).

The terminal approval gate (tools.approval_detection) prompts before destructive verbs reach these
targets; the store list is HERMES_PROTECTED_SUBPATHS, shared with agent.file_safety's write deny.
"""
import hermes_re as re

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
