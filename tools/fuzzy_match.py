#!/usr/bin/env python3
"""
Fuzzy Matching Module for File Operations

Implements a multi-strategy matching chain to robustly find and replace text,
accommodating variations in whitespace, indentation, and escaping common
in LLM-generated code.

The 8-strategy chain (inspired by OpenCode), tried in order:
1. Exact match - Direct string comparison
2. Line-trimmed - Strip leading/trailing whitespace per line
3. Whitespace normalized - Collapse multiple spaces/tabs to single space
4. Indentation flexible - Ignore indentation differences entirely
5. Escape normalized - Convert \\n literals to actual newlines
6. Trimmed boundary - Trim first/last line whitespace only
7. Block anchor - Match first+last lines, use similarity for middle
8. Context-aware - 50% line similarity threshold

Multi-occurrence matching is handled via the replace_all flag.

Usage:
    from tools.fuzzy_match import fuzzy_find_and_replace
    
    new_content, match_count, strategy, error = fuzzy_find_and_replace(
        content="def foo():\\n    pass",
        old_string="def foo():",
        new_string="def bar():",
        replace_all=False
    )
"""

import re
from typing import Tuple, Optional, List, Callable
from difflib import SequenceMatcher

UNICODE_MAP = {
    "\u201c": '"', "\u201d": '"',  # smart double quotes
    "\u2018": "'", "\u2019": "'",  # smart single quotes
    "\u2014": "--", "\u2013": "-", # em/en dashes
    "\u2026": "...", "\u00a0": " ", # ellipsis and non-breaking space
}

def _unicode_normalize(text: str) -> str:
    """Normalizes Unicode characters to their standard ASCII equivalents."""
    for char, repl in UNICODE_MAP.items():
        text = text.replace(char, repl)
    return text


def fuzzy_find_and_replace(content: str, old_string: str, new_string: str,
                            replace_all: bool = False) -> Tuple[str, int, Optional[str], Optional[str]]:
    """
    Find and replace text using a chain of increasingly fuzzy matching strategies.

    Args:
        content: The file content to search in
        old_string: The text to find
        new_string: The replacement text
        replace_all: If True, replace all occurrences; if False, require uniqueness

    Returns:
        Tuple of (new_content, match_count, strategy_name, error_message)
        - If successful: (modified_content, number_of_replacements, strategy_used, None)
        - If failed: (original_content, 0, None, error_description)
    """
    if not old_string:
        return content, 0, None, "old_string cannot be empty"

    if old_string == new_string:
        return content, 0, None, "old_string and new_string are identical"

    # Try each matching strategy in order
    strategies: List[Tuple[str, Callable]] = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("unicode_normalized", _strategy_unicode_normalized),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]

    for strategy_name, strategy_fn in strategies:
        matches = strategy_fn(content, old_string)

        if matches:
            # Found matches with this strategy
            if len(matches) > 1 and not replace_all:
                return content, 0, None, (
                    f"Found {len(matches)} matches for old_string. "
                    f"Provide more context to make it unique, or use replace_all=True."
                )

            # Escape-drift guard: when the matched strategy is NOT `exact`,
            # we matched via some form of normalization. If new_string
            # contains shell/JSON-style escape sequences (\' or \") that
            # would be written literally into the file but the matched
            # region of the file has no such sequences, this is almost
            # certainly tool-call serialization drift — the model typed
            # an apostrophe/quote and the transport added a stray
            # backslash. Writing new_string as-is would corrupt the file.
            # Block with a helpful error so the model re-reads and retries
            # instead of the caller silently persisting garbage (or not).
            if strategy_name != "exact":
                drift_err = _detect_escape_drift(content, matches, old_string, new_string)
                if drift_err:
                    return content, 0, None, drift_err

            # Perform replacement
            new_content = _apply_replacements(content, matches, new_string)
            return new_content, len(matches), strategy_name, None

    # No strategy found a match
    return content, 0, None, "Could not find a match for old_string in the file"


def _detect_escape_drift(content: str, matches: List[Tuple[int, int]],
                         old_string: str, new_string: str) -> Optional[str]:
    """Detect tool-call escape-drift artifacts in new_string.

    Looks for ``\\'`` or ``\\"`` sequences that are present in both
    old_string and new_string (i.e. the model copy-pasted them as "context"
    it intended to preserve) but don't exist in the matched region of the
    file. That pattern indicates the transport layer inserted spurious
    shell-style escapes around apostrophes or quotes — writing new_string
    verbatim would literally insert ``\\'`` into source code.

    Returns an error string if drift is detected, None otherwise.
    """
    # Cheap pre-check: bail out unless new_string actually contains a
    # suspect escape sequence. This keeps the guard free for all the
    # common, correct cases.
    if "\\'" not in new_string and '\\"' not in new_string:
        return None

    # Aggregate matched regions of the file — that's what new_string will
    # replace. If the suspect escapes are present there already, the
    # model is genuinely preserving them (valid for some languages /
    # escaped strings); accept the patch.
    matched_regions = "".join(content[start:end] for start, end in matches)

    for suspect in ("\\'", '\\"'):
        if suspect in new_string and suspect in old_string and suspect not in matched_regions:
            plain = suspect[1]  # "'" or '"'
            return (
                f"Escape-drift detected: old_string and new_string contain "
                f"the literal sequence {suspect!r} but the matched region of "
                f"the file does not. This is almost always a tool-call "
                f"serialization artifact where an apostrophe or quote got "
                f"prefixed with a spurious backslash. Re-read the file with "
                f"read_file and pass old_string/new_string without "
                f"backslash-escaping {plain!r} characters."
            )
    return None


def _apply_replacements(content: str, matches: List[Tuple[int, int]], new_string: str) -> str:
    """
    Apply replacements at the given positions.
    
    Args:
        content: Original content
        matches: List of (start, end) positions to replace
        new_string: Replacement text
    
    Returns:
        Content with replacements applied
    """
    # Sort matches by position (descending) to replace from end to start
    # This preserves positions of earlier matches
    sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)
    
    result = content
    for start, end in sorted_matches:
        result = result[:start] + new_string + result[end:]
    
    return result


# =============================================================================
# Matching Strategies
# =============================================================================

def _strategy_exact(content: str, pattern: str) -> List[Tuple[int, int]]:
    """Strategy 1: Exact string match."""
    matches = []
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(pattern)))
        start = pos + 1
    return matches


def _strategy_line_trimmed(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 2: Match with line-by-line whitespace trimming.
    
    Strips leading/trailing whitespace from each line before matching.
    """
    # Normalize pattern and content by trimming each line
    pattern_lines = [line.strip() for line in pattern.split('\n')]
    pattern_normalized = '\n'.join(pattern_lines)
    
    content_lines = content.split('\n')
    content_normalized_lines = [line.strip() for line in content_lines]
    
    # Build mapping from normalized positions back to original positions
    return _find_normalized_matches(
        content, content_lines, content_normalized_lines,
        pattern, pattern_normalized
    )


def _strategy_whitespace_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 3: Collapse multiple whitespace to single space.
    """
    def normalize(s):
        # Collapse multiple spaces/tabs to single space, preserve newlines
        return re.sub(r'[ \t]+', ' ', s)
    
    pattern_normalized = normalize(pattern)
    content_normalized = normalize(content)
    
    # Find in normalized, map back to original
    matches_in_normalized = _strategy_exact(content_normalized, pattern_normalized)
    
    if not matches_in_normalized:
        return []
    
    # Map positions back to original content
    return _map_normalized_positions(content, content_normalized, matches_in_normalized)


def _strategy_indentation_flexible(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 4: Ignore indentation differences entirely.
    
    Strips all leading whitespace from lines before matching.
    """
    content_lines = content.split('\n')
    content_stripped_lines = [line.lstrip() for line in content_lines]
    pattern_lines = [line.lstrip() for line in pattern.split('\n')]
    
    return _find_normalized_matches(
        content, content_lines, content_stripped_lines,
        pattern, '\n'.join(pattern_lines)
    )


def _strategy_escape_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 5: Convert escape sequences to actual characters.
    
    Handles \\n -> newline, \\t -> tab, etc.
    """
    def unescape(s):
        # Convert common escape sequences
        return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    
    pattern_unescaped = unescape(pattern)
    
    if pattern_unescaped == pattern:
        # No escapes to convert, skip this strategy
        return []
    
    return _strategy_exact(content, pattern_unescaped)


def _strategy_trimmed_boundary(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 6: Trim whitespace from first and last lines only.
    
    Useful when the pattern boundaries have whitespace differences.
    """
    pattern_lines = pattern.split('\n')
    if not pattern_lines:
        return []
    
    # Trim only first and last lines
    pattern_lines[0] = pattern_lines[0].strip()
    if len(pattern_lines) > 1:
        pattern_lines[-1] = pattern_lines[-1].strip()
    
    modified_pattern = '\n'.join(pattern_lines)
    
    content_lines = content.split('\n')
    
    # Search through content for matching block
    matches = []
    pattern_line_count = len(pattern_lines)
    
    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i:i + pattern_line_count]
        
        # Trim first and last of this block
        check_lines = block_lines.copy()
        check_lines[0] = check_lines[0].strip()
        if len(check_lines) > 1:
            check_lines[-1] = check_lines[-1].strip()
        
        if '\n'.join(check_lines) == modified_pattern:
            # Found match - calculate original positions
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


def _build_orig_to_norm_map(original: str) -> List[int]:
    """Build a list mapping each original character index to its normalized index.

    Because UNICODE_MAP replacements may expand characters (e.g. em-dash → '--',
    ellipsis → '...'), the normalised string can be longer than the original.
    This map lets us convert positions in the normalised string back to the
    corresponding positions in the original string.

    Returns a list of length ``len(original) + 1``; entry ``i`` is the
    normalised index that character ``i`` maps to.
    """
    result: List[int] = []
    norm_pos = 0
    for char in original:
        result.append(norm_pos)
        repl = UNICODE_MAP.get(char)
        norm_pos += len(repl) if repl is not None else 1
    result.append(norm_pos)  # sentinel: one past the last character
    return result


def _map_positions_norm_to_orig(
    orig_to_norm: List[int],
    norm_matches: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Convert (start, end) positions in the normalised string to original positions."""
    # Invert the map: norm_pos -> first original position with that norm_pos
    norm_to_orig_start: dict[int, int] = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm[:-1]):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos

    results: List[Tuple[int, int]] = []
    orig_len = len(orig_to_norm) - 1  # number of original characters

    for norm_start, norm_end in norm_matches:
        if norm_start not in norm_to_orig_start:
            continue
        orig_start = norm_to_orig_start[norm_start]

        # Walk forward until orig_to_norm[orig_end] >= norm_end
        orig_end = orig_start
        while orig_end < orig_len and orig_to_norm[orig_end] < norm_end:
            orig_end += 1

        results.append((orig_start, orig_end))

    return results


def _strategy_unicode_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    """Strategy 7: Unicode normalisation.

    Normalises smart quotes, em/en-dashes, ellipsis, and non-breaking spaces
    to their ASCII equivalents in both *content* and *pattern*, then runs
    exact and line_trimmed matching on the normalised copies.

    Positions are mapped back to the *original* string via
    ``_build_orig_to_norm_map`` — necessary because some UNICODE_MAP
    replacements expand a single character into multiple ASCII characters,
    making a naïve position copy incorrect.
    """
    # Normalize both sides. Either the content or the pattern (or both) may
    # carry unicode variants — e.g. content has an em-dash that should match
    # the LLM's ASCII '--', or vice-versa.  Skip only when neither changes.
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)
    if norm_content == content and norm_pattern == pattern:
        return []

    norm_matches = _strategy_exact(norm_content, norm_pattern)
    if not norm_matches:
        norm_matches = _strategy_line_trimmed(norm_content, norm_pattern)

    if not norm_matches:
        return []

    orig_to_norm = _build_orig_to_norm_map(content)
    return _map_positions_norm_to_orig(orig_to_norm, norm_matches)


def _strategy_block_anchor(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 8: Match by anchoring on first and last lines.
    Adjusted with permissive thresholds and unicode normalization.
    """
    # Normalize both strings for comparison while keeping original content for offset calculation
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)
    
    pattern_lines = norm_pattern.split('\n')
    if len(pattern_lines) < 2:
        return []
    
    first_line = pattern_lines[0].strip()
    last_line = pattern_lines[-1].strip()
    
    # Use normalized lines for matching logic
    norm_content_lines = norm_content.split('\n')
    # BUT use original lines for calculating start/end positions to prevent index shift
    orig_content_lines = content.split('\n')
    
    pattern_line_count = len(pattern_lines)
    
    potential_matches = []
    for i in range(len(norm_content_lines) - pattern_line_count + 1):
        if (norm_content_lines[i].strip() == first_line and 
            norm_content_lines[i + pattern_line_count - 1].strip() == last_line):
            potential_matches.append(i)
            
    matches = []
    candidate_count = len(potential_matches)
    
    # Thresholding logic: 0.50 for unique matches, 0.70 for multiple candidates.
    # Previous values (0.10 / 0.30) were dangerously loose — a 10% middle-section
    # similarity could match completely unrelated blocks.
    threshold = 0.50 if candidate_count == 1 else 0.70

    for i in potential_matches:
        if pattern_line_count <= 2:
            similarity = 1.0
        else:
            # Compare normalized middle sections
            content_middle = '\n'.join(norm_content_lines[i+1:i+pattern_line_count-1])
            pattern_middle = '\n'.join(pattern_lines[1:-1])
            similarity = SequenceMatcher(None, content_middle, pattern_middle).ratio()
        
        if similarity >= threshold:
            # Calculate positions using ORIGINAL lines to ensure correct character offsets in the file
            start_pos, end_pos = _calculate_line_positions(
                orig_content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


def _strategy_context_aware(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 9: Line-by-line similarity with 50% threshold.
    
    Finds blocks where at least 50% of lines have high similarity.
    """
    pattern_lines = pattern.split('\n')
    content_lines = content.split('\n')
    
    if not pattern_lines:
        return []
    
    matches = []
    pattern_line_count = len(pattern_lines)
    
    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i:i + pattern_line_count]
        
        # Calculate line-by-line similarity
        high_similarity_count = 0
        for p_line, c_line in zip(pattern_lines, block_lines):
            sim = SequenceMatcher(None, p_line.strip(), c_line.strip()).ratio()
            if sim >= 0.80:
                high_similarity_count += 1
        
        # Need at least 50% of lines to have high similarity
        if high_similarity_count >= len(pattern_lines) * 0.5:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


# =============================================================================
# Helper Functions
# =============================================================================

def _calculate_line_positions(content_lines: List[str], start_line: int,
                              end_line: int, content_length: int) -> Tuple[int, int]:
    """Calculate start and end character positions from line indices.

    Args:
        content_lines: List of lines (without newlines)
        start_line: Starting line index (0-based)
        end_line: Ending line index (exclusive, 0-based)
        content_length: Total length of the original content string

    Returns:
        Tuple of (start_pos, end_pos) in the original content
    """
    start_pos = sum(len(line) + 1 for line in content_lines[:start_line])
    end_pos = sum(len(line) + 1 for line in content_lines[:end_line]) - 1
    if end_pos >= content_length:
        end_pos = content_length
    return start_pos, end_pos


def _find_normalized_matches(content: str, content_lines: List[str],
                              content_normalized_lines: List[str],
                              pattern: str, pattern_normalized: str) -> List[Tuple[int, int]]:
    """
    Find matches in normalized content and map back to original positions.
    
    Args:
        content: Original content string
        content_lines: Original content split by lines
        content_normalized_lines: Normalized content lines
        pattern: Original pattern
        pattern_normalized: Normalized pattern
    
    Returns:
        List of (start, end) positions in the original content
    """
    pattern_norm_lines = pattern_normalized.split('\n')
    num_pattern_lines = len(pattern_norm_lines)
    
    matches = []
    
    for i in range(len(content_normalized_lines) - num_pattern_lines + 1):
        # Check if this block matches
        block = '\n'.join(content_normalized_lines[i:i + num_pattern_lines])
        
        if block == pattern_normalized:
            # Found a match - calculate original positions
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + num_pattern_lines, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


def _map_normalized_positions(original: str, normalized: str,
                               normalized_matches: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Map positions from normalized string back to original.
    
    This is a best-effort mapping that works for whitespace normalization.
    """
    if not normalized_matches:
        return []
    
    # Build character mapping from normalized to original
    orig_to_norm = []  # orig_to_norm[i] = position in normalized
    
    orig_idx = 0
    norm_idx = 0
    
    while orig_idx < len(original) and norm_idx < len(normalized):
        if original[orig_idx] == normalized[norm_idx]:
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            norm_idx += 1
        elif original[orig_idx] in ' \t' and normalized[norm_idx] == ' ':
            # Original has space/tab, normalized collapsed to space
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            # Don't advance norm_idx yet - wait until all whitespace consumed
            if orig_idx < len(original) and original[orig_idx] not in ' \t':
                norm_idx += 1
        elif original[orig_idx] in ' \t':
            # Extra whitespace in original
            orig_to_norm.append(norm_idx)
            orig_idx += 1
        else:
            # Mismatch - shouldn't happen with our normalization
            orig_to_norm.append(norm_idx)
            orig_idx += 1
    
    # Fill remaining
    while orig_idx < len(original):
        orig_to_norm.append(len(normalized))
        orig_idx += 1
    
    # Reverse mapping: for each normalized position, find original range
    norm_to_orig_start = {}
    norm_to_orig_end = {}
    
    for orig_pos, norm_pos in enumerate(orig_to_norm):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos
        norm_to_orig_end[norm_pos] = orig_pos
    
    # Map matches
    original_matches = []
    for norm_start, norm_end in normalized_matches:
        # Find original start
        if norm_start in norm_to_orig_start:
            orig_start = norm_to_orig_start[norm_start]
        else:
            # Find nearest
            orig_start = min(i for i, n in enumerate(orig_to_norm) if n >= norm_start)
        
        # Find original end
        if norm_end - 1 in norm_to_orig_end:
            orig_end = norm_to_orig_end[norm_end - 1] + 1
        else:
            orig_end = orig_start + (norm_end - norm_start)
        
        # Expand to include trailing whitespace that was normalized
        while orig_end < len(original) and original[orig_end] in ' \t':
            orig_end += 1
        
        original_matches.append((orig_start, min(orig_end, len(original))))
    
    return original_matches


def find_closest_lines(old_string: str, content: str, context_lines: int = 2, max_results: int = 3) -> str:
    """Find lines in content most similar to old_string for "did you mean?" feedback.

    Returns a formatted string showing the closest matching lines with context,
    or empty string if no useful match is found.
    """
    if not old_string or not content:
        return ""

    old_lines = old_string.splitlines()
    content_lines = content.splitlines()

    if not old_lines or not content_lines:
        return ""

    # Use first line of old_string as anchor for search
    anchor = old_lines[0].strip()
    if not anchor:
        # Try second line if first is blank
        candidates = [l.strip() for l in old_lines if l.strip()]
        if not candidates:
            return ""
        anchor = candidates[0]

    # Score each line in content by similarity to anchor
    scored = []
    for i, line in enumerate(content_lines):
        stripped = line.strip()
        if not stripped:
            continue
        ratio = SequenceMatcher(None, anchor, stripped).ratio()
        if ratio > 0.3:
            scored.append((ratio, i))

    if not scored:
        return ""

    # Take top matches
    scored.sort(key=lambda x: -x[0])
    top = scored[:max_results]

    parts = []
    seen_ranges = set()
    for _, line_idx in top:
        start = max(0, line_idx - context_lines)
        end = min(len(content_lines), line_idx + len(old_lines) + context_lines)
        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        snippet = "\n".join(
            f"{start + j + 1:4d}| {content_lines[start + j]}"
            for j in range(end - start)
        )
        parts.append(snippet)

    if not parts:
        return ""

    return "\n---\n".join(parts)
