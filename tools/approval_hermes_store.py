"""Shell-path patterns for the Hermes home and its protected stores (#109367).

The terminal approval gate (tools.approval_detection) prompts before destructive verbs reach these
targets; the store list is HERMES_PROTECTED_SUBPATHS, shared with agent.file_safety's write deny.
"""
import re

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
# The store itself, for the overwrite rules: a new file written into sessions/ (`> sessions/x`) stays clean.
_HERMES_STORE_ROOT = (
    _HERMES_HOME_ROOT + r'/(?:' + '|'.join(
        re.escape(sub) + (r'(?:-wal|-shm|-journal)?' if sub.endswith('.db') else r'/?')
        for sub in HERMES_PROTECTED_SUBPATHS) + r')' + _PATH_WORD_END
)
_HERMES_HOME_BARE = rf'{_HERMES_HOME_ROOT}/?\*?{_PATH_WORD_END}'
_HERMES_STORE_PATH = rf'(?:{_HERMES_STORE_FILE}|{_HERMES_HOME_BARE})'
