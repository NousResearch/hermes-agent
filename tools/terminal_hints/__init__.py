"""Advisory lines the terminal tool adds to a foreground result; none of them block a command.

``reread`` notes a plain file view of a file this task already read in full through read_file, unchanged since.
"""

from tools.terminal_hints.reread import REREAD_HINT, reread_hint, viewed_path

__all__ = ["REREAD_HINT", "reread_hint", "viewed_path"]
