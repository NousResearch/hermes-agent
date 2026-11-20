"""Re-read hint for the terminal tool: a plain file view of a file this task already read in full through
read_file, unchanged since, gets one advisory line. Never blocks.

``hint`` parses the view command (``viewed_path``) and checks the read tracker (``reread_hint``).
"""

from tools.terminal_reread.hint import REREAD_HINT, reread_hint, viewed_path

__all__ = ["REREAD_HINT", "reread_hint", "viewed_path"]
