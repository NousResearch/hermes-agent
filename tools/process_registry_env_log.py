"""Incremental sandbox log reads for ProcessRegistry's non-local poller."""


def log_delta_command(quoted_log_path: str, offset: int) -> str:
    """Shell command that reads only the log bytes written since ``offset``
    (``cat``-ing the whole file every poll re-sends all output over docker/SSH).

    Prints one header line ``"<size> <offset>"`` then the bytes in [offset, size).
    The size is read first and the tail cut at that same size, so a growing file
    never sends a byte twice; a file that shrank was rotated/truncated, so the
    offset drops to 0 and the reader starts over. The window end is pulled back
    to a UTF-8 character boundary (the backend decodes each ``execute()`` result
    on its own, so a straddling multibyte char would become U+FFFD and break watch
    patterns at the seam): up to 3 trailing continuation bytes are held for the
    next poll and the header reports the trimmed size."""
    return (
        f"O={offset}; "
        f"S=$({{ wc -c < {quoted_log_path}; }} 2>/dev/null | tr -dc '0-9'); "
        f"S=${{S:-0}}; "
        f'if [ "$S" -lt "$O" ]; then O=0; fi; '
        # Scan back up to 3 continuation bytes (octal 200-277) to the lead byte; if
        # the lead's declared length (3xx=2, 34x-35x=3, 36x-37x=4) exceeds the bytes
        # present, trim to before it. Complete sequences and ASCII tails untouched.
        f'N=0; P=$S; while [ "$P" -gt "$O" ] && [ "$N" -lt 3 ]; do '
        f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
        f'case "$B" in 2[0-7][0-7]) P=$((P-1)); N=$((N+1));; *) break;; esac; done; '
        f'if [ "$N" -gt 0 ] || [ "$P" -eq "$S" ]; then '
        f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
        f'case "$B" in 3[0-3][0-7]) L=2;; 3[4-5][0-7]) L=3;; 3[6-7][0-7]) L=4;; *) L=1;; esac; '
        f'if [ "$L" -gt $((N+1)) ]; then S=$((P-1)); fi; fi; '
        f'echo "$S $O"; '
        f'if [ "$S" -gt "$O" ]; then '
        f"tail -c +$((O+1)) {quoted_log_path} 2>/dev/null | head -c $((S-O)); fi"
    )
