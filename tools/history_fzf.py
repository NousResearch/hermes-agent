#!/usr/bin/env python3
"""fzf-based input history picker for Ctrl+R in the CLI.

Reads from the prompt_toolkit FileHistory file (~/.hermes/.hermes_history),
which is written synchronously on every input submission — no DB lag issues.

On selection, the chosen text is returned for insertion into the input buffer.
No relaunch needed — just a buffer.text assignment.

Uses subprocess.call() with stdin from temp file so fzf gets direct /dev/tty
access for its interactive TUI on all platforms.
"""

import os
import shutil
import subprocess
import tempfile


def _history_file_path():
    """Resolve the .hermes_history path for the current profile."""
    try:
        from hermes_constants import get_hermes_home
        return os.path.join(get_hermes_home(), ".hermes_history")
    except Exception:
        return os.path.expanduser("~/.hermes/.hermes_history")


def parse_history(filepath, limit=2000):
    """Parse prompt_toolkit FileHistory format into (timestamp, text) tuples.

    Format:
        # 2026-04-10 19:41:03.572755
        +first line of input
        +second line of input
        <blank line>

    Returns list of (timestamp_str, text) tuples, newest-first, deduplicated.
    """
    entries = []
    current_ts = ""
    current_lines = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("# "):
                    if current_lines:
                        entries.append((current_ts, "\n".join(current_lines)))
                    current_ts = line[2:]
                    current_lines = []
                elif line.startswith("+"):
                    current_lines.append(line[1:])
                elif line == "":
                    if current_lines:
                        entries.append((current_ts, "\n".join(current_lines)))
                    current_ts = ""
                    current_lines = []
            if current_lines:
                entries.append((current_ts, "\n".join(current_lines)))
    except FileNotFoundError:
        return []

    entries.reverse()

    seen = set()
    result = []
    for ts, text in entries:
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            result.append((ts, key))
        if len(result) >= limit:
            break

    return result


def _format_ts(raw_ts):
    """Format raw timestamp for display. '2026-04-10 19:41:03.572755' -> '04-10 19:41'"""
    if not raw_ts:
        return "??-?? ??:??"
    try:
        parts = raw_ts.split()
        if len(parts) >= 2:
            date_part = parts[0]
            time_part = parts[1]
            mm_dd = date_part[5:]
            hh_mm = time_part[:5]
            return f"{mm_dd} {hh_mm}"
        return raw_ts[:11]
    except Exception:
        return raw_ts[:11]


def fzf_history_picker(items):
    """Launch fzf to search through past user inputs.

    Parameters
    ----------
    items : list[(str, str)]
        (timestamp, text) tuples, newest-first.

    Returns
    -------
    str or None
        The selected text, or None if cancelled / fzf unavailable.
    """
    if not items:
        return None

    if not shutil.which("fzf"):
        return None

    # Write items to a temp file — fzf reads from this via stdin.
    # stdout/stderr are NOT redirected so fzf can access /dev/tty
    # directly for its interactive TUI.
    fd, input_path = tempfile.mkstemp(suffix=".fzf-in")
    os.close(fd)

    try:
        with open(input_path, "w", encoding="utf-8") as f:
            for ts, text in items:
                display = " ".join(text.split())
                if len(display) > 200:
                    display = display[:197] + "..."
                f.write(f"{_format_ts(ts)}  {display}\n")

        fzf_cmd = [
            "fzf",
            "--ansi", "--no-multi",
            "--height=60%", "--layout=reverse",
            "--prompt=History> ",
            "--preview-window=hidden",
            "--bind=ctrl-y:accept",
            "--header=Enter insert | Esc cancel",
            "--exact",
            "--tiebreak=begin,length",
        ]

        # fzf uses stderr for its interactive TUI (/dev/tty),
        # stdout for the selected result.  We must NOT redirect stderr.
        with open(input_path, "r", encoding="utf-8") as f_in:
            proc = subprocess.run(
                fzf_cmd,
                stdin=f_in,
                stdout=subprocess.PIPE,
                stderr=None,  # let fzf use the terminal for its UI
                text=True,
            )

        if proc.returncode != 0:
            import logging
            logging.getLogger("hermes.fzf").debug(
                "fzf_history_picker: fzf exited with returncode=%d, stdout=%r",
                proc.returncode, proc.stdout[:200] if proc.stdout else None
            )
            return None

        selected = proc.stdout.strip()
        if not selected:
            import logging
            logging.getLogger("hermes.fzf").debug(
                "fzf_history_picker: fzf stdout was empty after strip"
            )
            return None

        # Strip the "MM-DD HH:MM  " prefix (11 chars) to get the display text
        display_text = selected[11:].strip() if len(selected) > 11 else selected

        # Find the original full-text entry by prefix match
        for ts, text in items:
            collapsed = " ".join(text.split())
            if collapsed[:197] == display_text[:197]:
                return text

        import logging
        logging.getLogger("hermes.fzf").debug(
            "fzf_history_picker: no prefix match for display_text=%r (selected=%r)",
            display_text[:100], selected[:100]
        )
        return display_text

    except Exception as e:
        import logging
        logging.getLogger("hermes.fzf").debug(
            "fzf_history_picker: caught exception: %s: %s", type(e).__name__, e
        )
        return None

    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass
