"""Quoted grep PCRE operand spans for the approval detectors (inert data, not commands).

Moved out of :mod:`tools.approval_detection` (size cap); that module re-exports both names.
"""

import os


def _quoted_grep_pattern_spans(command: str) -> tuple[list[tuple[int, int]], bool]:
    """Structurally locate quoted grep PCRE operands -> (spans, malformed). On an ambiguous or
    malformed grep parse callers fail closed and use the original command: no text is hidden on
    an uncertain parse."""
    from tools.approval_detection import (
        _GREP_OPTIONS_WITH_ARG, _GREP_SHORT_OPTIONS_WITH_ARG, _deobfuscate_shell_word_for_detection,
        _iter_shell_command_word_spans, _iter_top_level_shell_segments,
    )
    from tools.approval_shell_tokens import _shell_tokens_with_spans

    spans: list[tuple[int, int]] = []
    offset = 0
    for segment in _iter_top_level_shell_segments(command):
        segment_at = command.find(segment, offset)
        offset = segment_at + len(segment)
        for start, _, word in _iter_shell_command_word_spans(segment):
            if os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower() not in {"grep", "egrep"}:
                continue
            tokens = _shell_tokens_with_spans(segment, start)
            if tokens is None:
                return [], True
            args, pattern_indexes = tokens[1:], []
            pcre = explicit_patterns = False
            operand_index, i, options = None, 0, True
            while i < len(args):
                token = args[i][0]
                if options and token == "--":
                    options = False
                elif options and token.startswith("--"):
                    option, equals, _ = token.partition("=")
                    pcre = pcre or option == "--perl-regexp"
                    explicit_patterns = explicit_patterns or option in {"--regexp", "--file"}
                    takes_next = option in _GREP_OPTIONS_WITH_ARG and not equals
                    if takes_next and i + 1 >= len(args):
                        return [], True
                    if option == "--regexp":
                        pattern_indexes.append(i + 1 if takes_next else i)
                    i += 1 if takes_next else 0
                elif options and token.startswith("-") and token != "-":
                    chars = token[1:]
                    for j, char in enumerate(chars):
                        pcre = pcre or char == "P"
                        explicit_patterns = explicit_patterns or char in {"e", "f"}
                        if char in _GREP_SHORT_OPTIONS_WITH_ARG:
                            # The first argument-taking short option owns the rest of the bundle,
                            # or the next token when it comes last.
                            attached = j + 1 < len(chars)
                            if not attached and i + 1 >= len(args):
                                return [], True
                            if char == "e":
                                pattern_indexes.append(i if attached else i + 1)
                            i += 0 if attached else 1
                            break
                elif operand_index is None:
                    operand_index = i
                i += 1
            if not explicit_patterns:
                if operand_index is None:
                    return [], pcre
                pattern_indexes.append(operand_index)
            if pcre:
                spans.extend(
                    (segment_at + token_start, segment_at + token_end)
                    for _, token_start, token_end, quoted in map(args.__getitem__, pattern_indexes) if quoted
                )
    return spans, False


def _grep_safe_detection_variant(command: str) -> tuple[str, bool]:
    from tools.approval_detection import _splice

    spans, malformed = _quoted_grep_pattern_spans(command)
    if malformed or not spans:
        return command, malformed
    return _splice(command, [(start, end, " " * (end - start)) for start, end in spans]), False
