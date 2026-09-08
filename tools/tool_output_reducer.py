"""High-SNR (Signal-to-Noise Ratio) tool output reduction and pruning.

Intelligently reduces large tool outputs by preserving high-value diagnostic sections
(Stack traces, Error/Exception lines, Panics, Failure reports, Test summaries) with
surrounding context instead of blindly lopping off the middle between head and tail.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Patterns representing high-value diagnostic signal in terminal/code outputs
_ERROR_BLOCK_PATTERNS = [
    re.compile(r"(Traceback \(most recent call last\):[\s\S]*?(?:^\w*(?:Error|Exception|Exit|Interrupt):.*$))", re.M),
    re.compile(r"(thread '[^']+' panicked at [\s\S]*?(?:stack backtrace:[\s\S]*?(?=\n\n|\Z)))", re.M),
    re.compile(r"(panic: [\s\S]*?(?:goroutine \d+ \[[^\]]+\]:[\s\S]*?(?=\n\n|\Z)))", re.M),
    re.compile(r"((?:FAILED|FAIL|ERROR):?\s+[^\n]+[\s\S]*?(?:=+ short test summary info =+|FAILURES?|ERRORS?|===+[\s\S]*?===+))", re.M),
    re.compile(r"((?:FATAL|ERROR|CRITICAL)(?:[ :\]\n][^\n]*\n(?:[ \t]+at [^\n]+\n)*)+)", re.I),
]

_SINGLE_LINE_ERROR_RE = re.compile(
    r"^(?:.*?(?:error|fatal|panic|exception|traceback|undefined reference|cannot find module|failed to build|syntaxerror):.*)$",
    re.I | re.M,
)


def extract_high_snr_blocks(text: str, max_blocks: int = 5, context_lines: int = 4) -> List[str]:
    """Extract critical error/panic/traceback blocks from text with context."""
    if not text:
        return []

    # Fast head/tail slice guard on large text before regex to eliminate ReDoS risk
    if len(text) > 200_000:
        text = text[:100_000] + "\n\n" + text[-100_000:]

    extracted: List[Tuple[int, int, str]] = []  # (start_char, end_char, block_text)

    # 1. Match multi-line structured error blocks
    for pat in _ERROR_BLOCK_PATTERNS:
        for m in pat.finditer(text):
            block = m.group(0).strip()
            if block:
                extracted.append((m.start(), m.end(), block))

    # 2. Match line-based errors if few multi-line blocks found
    if len(extracted) < max_blocks:
        lines = text.splitlines(keepends=True)
        char_offsets = []
        curr = 0
        for line in lines:
            char_offsets.append(curr)
            curr += len(line)

        for i, line in enumerate(lines):
            if _SINGLE_LINE_ERROR_RE.match(line.strip()):
                start_line_idx = max(0, i - context_lines)
                end_line_idx = min(len(lines), i + context_lines + 1)
                block_start = char_offsets[start_line_idx]
                block_end = char_offsets[end_line_idx - 1] + len(lines[end_line_idx - 1])

                # Check overlap with existing extracted blocks
                overlapping = any(
                    (s <= block_start <= e) or (s <= block_end <= e) or (block_start <= s and block_end >= e)
                    for s, e, _ in extracted
                )
                if not overlapping:
                    block_text = "".join(lines[start_line_idx:end_line_idx]).strip()
                    extracted.append((block_start, block_end, block_text))
                    if len(extracted) >= max_blocks * 2:
                        break

    # Sort by appearance in original text and deduplicate
    extracted.sort(key=lambda x: x[0])
    
    unique_blocks: List[str] = []
    last_end = -1
    for s, e, blk in extracted:
        if s >= last_end:
            unique_blocks.append(blk)
            last_end = e
        if len(unique_blocks) >= max_blocks:
            break

    return unique_blocks


def _is_json_like(text: str) -> bool:
    """Check if text appears to be JSON (object or array) to avoid inserting disruptive banners."""
    stripped = text.strip()
    return (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    )


def reduce_tool_output(output: str, max_chars: int) -> str:
    """Intelligently reduce large tool output preserving high-SNR errors and head/tail context.
    
    If output fits within max_chars, returns unchanged.
    Otherwise, extracts diagnostic error blocks and stitches them between head and tail chunks.
    """
    if len(output) <= max_chars:
        return output

    # Find high SNR blocks (skip banner injection if JSON-like payload)
    is_json = _is_json_like(output)
    snr_blocks = [] if is_json else extract_high_snr_blocks(output)
    
    # Calculate budgets
    overhead = 250  # notices and separators
    available_chars = max(0, max_chars - overhead)

    if not snr_blocks:
        # Standard head/tail fallback
        head_chars = int(available_chars * 0.4)
        tail_chars = max(0, available_chars - head_chars)
        omitted = len(output) - head_chars - tail_chars
        notice = f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted out of {len(output)} total] ...\n\n"
        tail_slice = output[-tail_chars:] if tail_chars > 0 else ""
        return output[:head_chars] + notice + tail_slice

    # Allocate budget: 20% head, 25% tail, 55% high-SNR error diagnostics
    head_chars = int(available_chars * 0.20)
    tail_chars = int(available_chars * 0.25)
    snr_budget = max(0, available_chars - head_chars - tail_chars)

    # Format SNR blocks within snr_budget
    snr_sections: List[str] = []
    used_snr_chars = 0
    for blk in snr_blocks:
        if used_snr_chars + len(blk) + 30 <= snr_budget:
            snr_sections.append(blk)
            used_snr_chars += len(blk) + 30
        else:
            remaining = snr_budget - used_snr_chars - 40
            if remaining > 100:
                snr_sections.append(blk[:remaining] + "\n... [error block clipped]")
            break

    if snr_sections:
        snr_content = "\n\n--- [HIGH-SNR DIAGNOSTIC / ERROR BLOCKS EXTRACTED FROM OMITTED LOGS] ---\n" + "\n\n---\n".join(snr_sections)
    else:
        snr_content = ""
    
    omitted = max(0, len(output) - head_chars - tail_chars - len(snr_content))
    notice = f"\n\n... [OUTPUT REDUCED - {omitted} chars omitted out of {len(output)} total] ...{snr_content}\n\n... [RECENT LOG TAIL] ...\n\n"

    tail_slice = output[-tail_chars:] if tail_chars > 0 else ""
    return output[:head_chars] + notice + tail_slice
