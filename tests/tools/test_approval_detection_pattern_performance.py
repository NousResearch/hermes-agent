"""Hard-timeout guard against quadratic patterns in the dangerous-command rule set.

The approval scan runs inline on the caller's thread, so a quadratic pattern
holds the GIL and stalls every other thread with no watchdog (see #104781 and
#108451). Time the individual rule this repo already had to bound — the
echo|tr|shell pipeline rule — against the shape that made it quadratic: long,
separator-rich, pipe-free input full of `echo` start tokens. The bounded rule
finishes in milliseconds; an unbounded `[^|]*` gap takes seconds and grows
quadratically with input size.
"""

import os
import pathlib
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

TR_RULE_DESCRIPTION = "pipe tr-transformed output to shell (possible command obfuscation)"

# 512k chars, separator-rich, echo-heavy, contains no `|`. The probe imports
# tools.approval_detection (stdlib-only import chain) and grabs the compiled
# rule from the module, so it cannot drift from what production runs.
PROBE = (
    "import sys, time\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from tools.approval_detection import DANGEROUS_PATTERNS_COMPILED\n"
    "rule = next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == sys.argv[2])\n"
    "s = ('echo hi; ' * 60000)[:512000]\n"
    "assert rule.search(\"echo 'eq -pe v/' | tr 'eqv' 'rmf' | bash\")\n"
    "t0 = time.monotonic()\n"
    "rule.search(s)\n"
    "print(f'{time.monotonic() - t0:.3f}')\n"
)

# The bounded rule scans once (~milliseconds at 512k). An unbounded gap
# rescans from every `echo` token and needs many seconds here.
MAX_RULE_SECONDS = 2.0

HARD_KILL_SECONDS = 30.0


def test_tr_pipeline_rule_stays_bounded_on_separator_rich_input():
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, str(REPO_ROOT), TR_RULE_DESCRIPTION],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=HARD_KILL_SECONDS,
    )
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, (
        f"rule probe failed rc={proc.returncode}: {proc.stderr[-2000:]}"
    )
    rule_seconds = float(proc.stdout.strip().splitlines()[-1])
    assert rule_seconds < MAX_RULE_SECONDS, (
        f"the echo|tr|shell rule took {rule_seconds:.2f}s to scan 512k "
        f"separator-rich input; its pipeline gaps must stay bounded "
        f"([^|;&\\n]*), otherwise the scan is quadratic and starves the GIL"
    )
    assert elapsed < HARD_KILL_SECONDS
