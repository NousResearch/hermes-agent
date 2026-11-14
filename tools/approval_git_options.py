"""Git global-option grammar for the approval rules (#150744).

Git's global options (`-C <path>`, `-c key=val`, `--git-dir=<path>`) sit between the binary and the
subcommand, so the destructive-git rules in approval_detection match the SUBCOMMAND past a run of
them rather than a literal `git <verb>` prefix.
"""

import re

# Git's global options sit between the binary and the subcommand: `git -C <path> push --force`,
# `git --git-dir=<path> reset --hard`, `git -c key=val clean -fd`. Each option is ONE shell word that
# may carry a JOINTED value (`-C'/tmp/my repo'`, `-c user.name='a b'`), so the value is matched whole —
# quoted spans included — rather than split at its first space, which left an unmatched fragment before
# the subcommand whenever the path held a space. A separate value is still at most one non-dash word.
# The git rules in approval_detection therefore match the SUBCOMMAND, which no literal `git <verb>` prefix ever is
# (#150744).
_GIT_OPTION_WORD = r"""(?:[^\s"'`$]++|\$\([^()]*+\)|`[^`]*+`|\$(?!\([^()]*+\))|"[^"]*"|'[^']*')+"""
_GIT_GLOBAL_FLAGS = rf'(?:-{_GIT_OPTION_WORD}(?:\s++(?!-\S){_GIT_OPTION_WORD})?\s++)*'


_ESCAPED_WORD_CHAR = re.compile(r"\\([ \t\"'`])|\\.", re.DOTALL)


def _join_escaped_word_chars(command: str) -> str:
    """Replace each backslash-escaped space, tab, quote or backtick with `_`, outside single quotes
    (where a backslash is literal); `\\` pairs are consumed first so an escaped backslash stays itself."""
    def join(span: re.Match) -> str:
        if span.group(0)[0] == "'":
            return span.group(0)
        return _ESCAPED_WORD_CHAR.sub(lambda e: "_" if e.group(1) else e.group(0), span.group(0))
    return re.sub(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"|\\.", join, command, flags=re.DOTALL)
