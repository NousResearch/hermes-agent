"""Git global-option grammar for the destructive-git approval rules (#150744, #155469).

The rules in :mod:`tools.approval_detection` match the SUBCOMMAND past git's global options
(`git -C <dir> reset --hard`); `_join_escaped_word_chars` feeds its escaped-separator variant.
"""

import re

# `git` plus the global options that may sit before the subcommand (`git -C <dir> reset --hard`).
# Value-taking options consume their separate value word; any other option is a single dash token.
# A word may contain quoted runs (`-C "/a b"`, `--git-dir='/a b/.git'`) or a command substitution
# (`-C $(git rev-parse --show-toplevel)`), which the shell passes as one argument. Each option is an
# atomic group with the value-taking form tried first: the two alternatives overlap (`-c x` is also
# two dash-or-word tokens), so a non-atomic run of option-like tokens (`git -c -c -c ... x`) is
# re-split exponentially on a failed match; words are possessive for the same reason.
# An option missing from that list still takes a separate value when another option follows it
# (`--opt val --flag`), unless that value is a gated verb or the option is a known value-less one.
# Every word ending in a `git` word (`-c git`, `alias.x=git`) starts its own search over the same rest
# of the command, so no option word may end in one: the run stops there instead of rescanning what
# that search covers, which keeps an uncapped run linear (`git -c git -c ... x`).
# A `$(...)` may nest (`-C $(realpath $(pwd))`), up to _GIT_SUBST_DEPTH levels; each level is an atomic
# run of non-paren characters or one deeper group, so an unclosed `$(` fails in a single pass.
_GIT_SUBST_DEPTH = 4
_GIT_PARENS = r'\([^()\n]*\)'
for _ in range(_GIT_SUBST_DEPTH - 1):
    _GIT_PARENS = r'\((?>[^()\n]|' + _GIT_PARENS + r')*+\)'
_GIT_WORD = r'''(?:"[^"]*"|'[^']*'|\$''' + _GIT_PARENS + r'''|`[^`\n]*`|[^\s;|&"'])++(?<!\bgit)'''
_GIT_NO_VALUE = (r'(?:--no-pager|--paginate|--bare|--no-replace-objects|--no-lazy-fetch|--no-optional-locks'
                 r'|--no-advice|--(?:literal|glob|noglob|icase)-pathspecs|--(?:html|man|info)-path|--version|--help|-[pPvh])')
_GIT = (r'\bgit(?>\s+(?:(?:-c|--git-dir|--work-tree|--namespace|--exec-path|--super-prefix|--config-env'
        rf'|--attr-source)\s+{_GIT_WORD}|(?!{_GIT_NO_VALUE}\s)(?:--[^\s=;|&"\'`$]++|-[^\s;|&"\'`$-])\s+'
        rf'(?!-|(?:push|reset|clean|branch)\s){_GIT_WORD}(?=\s+-)|-(?:{_GIT_WORD}|(?=\s))))*\s+')
_GIT_PUSH = rf'{_GIT}push\b'
# The push's own arguments, bounded to ONE command segment: a later command's flag is never the
# push's (`git push -q origin main && grep -f patterns.txt log` is not a force push). A quoted
# string is one atom, so a quoted `;`/`&`/`|` (`-o 'ci;skip'`) does not end the segment, and a
# flag inside quotes is not seen; a redirection's `&`/`|` (`2>&1`, `&>`, `>|`) stays inside too.
# Every alternative starts on a different character (or lookaround), so a failed match scans a
# run once instead of re-splitting it exponentially (`>&>&>&...`, `'a' 'a' 'a' ...`). The scan
# stops at a nested push, whose own search covers the rest, so `git push git push ...` is linear.
_GIT_PUSH_ARGS = (r'''(?:(?>'[^']*'|"(?:[^"\\]|\\[\s\S])*")|'(?![^']*')|"(?![^"]*")'''
                  r'''|\\[\s\S]|(?!''' + _GIT_PUSH + r''')[^;&|\n'"\\]|(?<=[<>])&|(?<![<>])&(?=>)|(?<=>)\|)*''')


def _join_escaped_word_chars(command: str) -> str:
    """Turn each escaped `;`/`&`/`|`, space, tab or quote into `_`, so it stays literal text inside its
    word after normalization strips the backslash. Inside single quotes a backslash is literal, so only
    the separators are joined there, as before."""
    def join(span: re.Match) -> str:
        chars = ';&|' if span.group(0)[0] == "'" else ';&| \t"\'`'
        return re.sub(r'\\(.)', lambda m: '_' if m.group(1) in chars else m.group(0), span.group(0))
    return re.sub(r''''[^']*'|"(?:[^"\\]|\\[\s\S])*"|\\[\s\S]''', join, command)
