#!/usr/bin/env python3
"""
Fuzzy Matching Module for File Operations

Implements a multi-strategy matching chain to robustly find and replace text,
accommodating variations in whitespace, indentation, and escaping common
in LLM-generated code.

The 9-strategy chain (inspired by OpenCode):
1. Exact match - Direct string comparison
2. Line-trimmed - Strip leading/trailing whitespace per line
3. Block anchor - Match first+last lines, use similarity for middle
4. Whitespace normalized - Collapse multiple spaces/tabs to single space
5. Indentation flexible - Ignore indentation differences entirely
6. Escape normalized - Convert \\n literals to actual newlines
7. Trimmed boundary - Trim first/last line whitespace only
8. Context-aware - 50% line similarity threshold
9. Multi-occurrence - For replace_all flag

Usage:
    from tools.fuzzy_match import fuzzy_find_and_replace
    
    new_content, match_count, error = fuzzy_find_and_replace(
        content="def foo():\\n    pass",
        old_string="def foo():",
        new_string="def bar():",
        replace_all=False
    )
"""

import re
from typing import Tuple, Optional, List, Callable
from difflib import SequenceMatcher


def fuzzy_find_and_replace(content: str, old_string: str, new_string: str,
                            replace_all: bool = False) -> Tuple[str, int, Optional[str]]:
    """
    Find and replace text using a chain of increasingly fuzzy matching strategies.
    
    Args:
        content: The file content to search in
        old_string: The text to find
        new_string: The replacement text
        replace_all: If True, replace all occurrences; if False, require uniqueness
    
    Returns:
        Tuple of (new_content, match_count, error_message)
        - If successful: (modified_content, number_of_replacements, None)
        - If failed: (original_content, 0, error_description)
    """
    if not old_string:
        return content, 0, "old_string cannot be empty"
    
    if old_string == new_string:
        return content, 0, "old_string and new_string are identical"
    
    # Try each matching strategy in order
    strategies: List[Tuple[str, Callable]] = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]
    
    for strategy_name, strategy_fn in strategies:
        matches = strategy_fn(content, old_string)
        
        if matches:
            # Found matches with this strategy
            if len(matches) > 1 and not replace_all:
                return content, 0, (
                    f"Found {len(matches)} matches for old_string. "
                    f"Provide more context to make it unique, or use replace_all=True."
                )
            
            # Perform replacement
            new_content = _apply_replacements(content, matches, new_string)
            return new_content, len(matches), None
    
    # No strategy found a match
    return content, 0, "Could not find a match for old_string in the file"


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
    def strip_indent(s):
        return '\n'.join(line.lstrip() for line in s.split('\n'))
    
    pattern_stripped = strip_indent(pattern)
    
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
            start_pos = sum(len(line) + 1 for line in content_lines[:i])
            end_pos = sum(len(line) + 1 for line in content_lines[:i + pattern_line_count]) - 1
            if end_pos >= len(content):
                end_pos = len(content)
            matches.append((start_pos, end_pos))
    
    return matches


def _strategy_block_anchor(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 7: Match by anchoring on first and last lines.
    
    If first and last lines match exactly, accept middle with 70% similarity.
    """
    pattern_lines = pattern.split('\n')
    if len(pattern_lines) < 2:
        return []  # Need at least 2 lines for anchoring
    
    first_line = pattern_lines[0].strip()
    last_line = pattern_lines[-1].strip()
    
    content_lines = content.split('\n')
    matches = []
    
    pattern_line_count = len(pattern_lines)
    
    for i in range(len(content_lines) - pattern_line_count + 1):
        # Check if first and last lines match
        if (content_lines[i].strip() == first_line and 
            content_lines[i + pattern_line_count - 1].strip() == last_line):
            
            # Check middle similarity
            if pattern_line_count <= 2:
                # Only first and last, they match
                similarity = 1.0
            else:
                content_middle = '\n'.join(content_lines[i+1:i+pattern_line_count-1])
                pattern_middle = '\n'.join(pattern_lines[1:-1])
                similarity = SequenceMatcher(None, content_middle, pattern_middle).ratio()
            
            if similarity >= 0.70:
                # Calculate positions
                start_pos = sum(len(line) + 1 for line in content_lines[:i])
                end_pos = sum(len(line) + 1 for line in content_lines[:i + pattern_line_count]) - 1
                if end_pos >= len(content):
                    end_pos = len(content)
                matches.append((start_pos, end_pos))
    
    return matches


def _strategy_context_aware(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 8: Line-by-line similarity with 50% threshold.
    
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
            start_pos = sum(len(line) + 1 for line in content_lines[:i])
            end_pos = sum(len(line) + 1 for line in content_lines[:i + pattern_line_count]) - 1
            if end_pos >= len(content):
                end_pos = len(content)
            matches.append((start_pos, end_pos))
    
    return matches


# =============================================================================
# Helper Functions
# =============================================================================

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
            start_pos = sum(len(line) + 1 for line in content_lines[:i])
            end_pos = sum(len(line) + 1 for line in content_lines[:i + num_pattern_lines]) - 1
            
            # Handle case where end is past content
            if end_pos >= len(content):
                end_pos = len(content)
            
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


# =============================================================================
# Utility Functions
# =============================================================================

def find_best_match(content: str, pattern: str) -> Optional[Tuple[int, int, str]]:
    """
    Find the best match for a pattern and return the strategy name.
    
    Returns:
        Tuple of (start, end, strategy_name) or None if no match
    """
    strategies = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]
    
    for strategy_name, strategy_fn in strategies:
        matches = strategy_fn(content, pattern)
        if matches:
            return (matches[0][0], matches[0][1], strategy_name)
    
    return None
