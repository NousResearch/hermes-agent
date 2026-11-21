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

# `sqlite3` writing a store: the command's line names a store file AND a write (a destructive SQL word, `.restore`
# or a heredoc), in either order. No engine lookahead may sit mid-pattern, so both orders are spelled out and only
# the last condition of each alternative is a lookahead. `_BLANK` is a blank that stays on the line.
_BLANK = r'[^\S\n]'
_HERMES_STORE_DB = _HERMES_HOME_ROOT + r'/(?:' + '|'.join(
    re.escape(sub) + r'(?:-wal|-shm|-journal)?' for sub in HERMES_PROTECTED_SUBPATHS if sub.endswith('.db')) + r')'
_HERMES_STORE_DIR = _HERMES_HOME_ROOT + r'/(?:' + '|'.join(
    re.escape(sub) for sub in HERMES_PROTECTED_SUBPATHS if not sub.endswith('.db')) + r')'
# The rest of sqlite3's own word (never ending in `/`: then sqlite3 was a directory), then the character after it.
_SQLITE3_WORD = r'(?:[^\w\s/)=<>][^\s/)=<>]*)?'
_SQLITE3_WORD_END = rf'(?:{_BLANK}|[)=<>])'
_SQL_WRITE_VERBS = (
    r'(?:delete\s+from|drop\s+\w+|update\s+(?:or\s+\w+\s+)?\S+\s+(?:as\s+\S+\s+)?set'
    r'|insert\s+(?:or\s+\w+\s+)?into|replace\s+into|alter\s+table)'
)
_SQL_WRITE_VERBS_ON_LINE = _SQL_WRITE_VERBS.replace(r'\s', _BLANK)
# `vacuum` rewrites the file, `vacuum [schema] into FILE` only copies it out. A word that does not begin `into`:
_NOT_INTO_WORD = r'(?:(?:[^\Wi]|i(?:[^\Wn]|n(?:[^\Wt]|t[^\Wo])))\w*|int?|i)'
_VACUUM_IN_PLACE = (
    r'vacuum(?=[^\w\s]|\s*\Z|\s+(?:[^\w\s]|' + _NOT_INTO_WORD
    + r'(?:[^\w\s]|\s*\Z|\s+(?:[^i\s]|i(?:[^n]|n(?:[^t]|t(?:[^o]|\Z)|\Z)|\Z)))))'
)
# The same test with a store still to come on the line: everything from `vacuum` up to where that store starts.
_VACUUM_BLANK_TO_STORE = (
    rf'{_BLANK}+(?:(?:[^\w\s][^\n]*)?|{_NOT_INTO_WORD}(?:(?:[^\w\s][^\n]*)?'
    rf'|{_BLANK}+(?:(?:[^i\s]|i(?:[^n\n]|n(?:[^t\n]|t[^o\n])))[^\n]*|(?:int?|i)?)))'
)
# Store first: a write that starts after the store word, or inside the path below a directory store.
_SQL_WRITE_AT_END = (
    rf'(?:(?:[^\n]*[^\w\n])?(?:{_SQL_WRITE_VERBS}(?!\w)|{_VACUUM_IN_PLACE})|[^\n]*\.restore(?!\w))'
)
_SQLITE3_STORE_THEN_WRITE = (
    rf'{_SQLITE3_WORD}{_SQLITE3_WORD_END}[^\n]*(?:'
    rf'{_HERMES_STORE_DIR}/(?:{_SQL_WRITE_AT_END}|(?:[^\n]*[^<\n])?<<(?!<))'
    rf'|(?:{_HERMES_STORE_DB}|{_HERMES_STORE_DIR})["\']?(?:(?:{_BLANK}|["\'`;|&<>()]){_SQL_WRITE_AT_END}'
    rf'|(?:(?:{_BLANK}|["\'`;|&>()])|(?:{_BLANK}|["\'`;|&<>()])[^\n]*[^<\n])?<<(?!<)))'
)
# Write first: its start is inside sqlite3's own word (`sqlite3.restore`), the character ending that word (`<<`), or later.
_SQLITE3_WORD_TO_SQL = (
    rf'(?:[^\w\s/)=<>](?:[^\s/)=<>]*[^\w\s/)=<>])?|{_SQLITE3_WORD}{_SQLITE3_WORD_END}(?:[^\n]*[^\w\n])?)'
)
_SQLITE3_WRITE_THEN_STORE = (
    rf'(?:{_SQLITE3_WORD_TO_SQL}{_SQL_WRITE_VERBS_ON_LINE}(?:[^\w\n][^\n]*)?'
    rf'|[^\w\s/)=<>](?:[^\s/)=<>]*[^\w\s/)=<>])?vacuum(?:[^\w\s/)=<>][^\s/)=<>]*{_SQLITE3_WORD_END}[^\n]*|[)=<>][^\n]*|{_VACUUM_BLANK_TO_STORE})'
    rf'|{_SQLITE3_WORD}{_SQLITE3_WORD_END}(?:[^\n]*[^\w\n])?vacuum(?:(?:[^\w\s][^\n]*)?|{_VACUUM_BLANK_TO_STORE})'
    rf'|{_SQLITE3_WORD}\.restore{_SQLITE3_WORD}{_SQLITE3_WORD_END}[^\n]*'
    rf'|{_SQLITE3_WORD}{_SQLITE3_WORD_END}[^\n]*\.restore(?:[^\w\n][^\n]*)?'
    rf'|{_SQLITE3_WORD}(?:(?:{_BLANK}|[)=>])?|{_SQLITE3_WORD_END}[^\n]*[^<\n])<<(?:[^<\n][^\n]*)?'
    rf'){_HERMES_STORE_FILE}'
)
# A store named inside the operand of `update ... set` (the statement may go on past the line).
_STORE_IN_SQL_WORD = rf'\S*(?:{_HERMES_STORE_DB}|{_HERMES_STORE_DIR}(?:/[^\s"\'`;|&<>()]*)?)(?:["\'`;|&<>()]\S*)?\s+'
_SQLITE3_UPDATE_OF_STORE = (
    rf'{_SQLITE3_WORD_TO_SQL}update{_BLANK}+(?:or{_BLANK}+\w+{_BLANK}+)?'
    rf'(?:{_STORE_IN_SQL_WORD}(?:as\s+\S+\s+)?|\S+{_BLANK}+as{_BLANK}+{_STORE_IN_SQL_WORD})set(?!\w)'
)
_SQLITE3_STORE_WRITE = rf'(?:{_SQLITE3_STORE_THEN_WRITE}|{_SQLITE3_WRITE_THEN_STORE}|{_SQLITE3_UPDATE_OF_STORE})'
