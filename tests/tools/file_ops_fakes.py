"""Fakes for ``ShellFileOperations``' compound shell probes.

``read_file`` and ``write_file`` ask the shell everything in ONE command
whose stdout is split on a per-call random sentinel line. Test doubles that
script ``env.execute`` / ``_exec`` need to answer that command with exactly
the stream the shell would produce; these helpers build it. Match the
sentinel out of the command first (it is random), then compose:

    m = READ_SENTINEL_RE.search(command)
    if m:
        return {"output": compound_read_output(m.group(0), size=5, sample=b"hello",
                                               content="hello\\n", total_lines=1),
                "returncode": 0}
"""

import base64
import re
from typing import Optional

READ_SENTINEL_RE = re.compile(r"__HERMES_RF_[0-9a-f]{32}__")
WRITE_SENTINEL_RE = re.compile(r"__HERMES_WF_[0-9a-f]{32}__")


def compound_read_output(
    sentinel: str,
    *,
    size: int,
    sample: Optional[bytes],
    content: str,
    total_lines: int,
    trailing_newline: bool = True,
    sample_rc: int = 0,
    read_rc: int = 0,
) -> str:
    """Stdout of ``_read_probe_cmd`` for a regular file.

    ``content`` is the ``sed | cut`` page exactly as the shell prints it:
    every line newline-terminated (``cut`` always adds one), or ``""`` for a
    page past EOF. ``sample`` is the raw first-1000-bytes slice (``None``
    emits an empty base64 segment, e.g. a shell without ``base64``).
    """
    sample_seg = base64.b64encode(sample).decode() + "\n" if sample else ""
    return (
        f"{size}\n{sentinel}\n"
        f"{sample_seg}{sentinel}\n"
        f"{content}{sentinel}\n"
        f"{total_lines}\n{sentinel}\n"
        f"{1 if trailing_newline else 0}\n{sentinel}\n"
        f"{sample_rc} {read_rc}\n"
    )


def compound_write_probe_output(sentinel: str, *, head3: bytes, body: str) -> str:
    """Stdout of ``_write_probe_cmd`` for an existing file.

    ``head3`` is the first three bytes on disk (BOM detection); ``body`` is
    the second segment: the whole file when pre-content was wanted, else
    the 4 KB line-ending sample.
    """
    head_seg = base64.b64encode(head3).decode() + "\n" if head3 else ""
    return f"{head_seg}{sentinel}\n{body}"
