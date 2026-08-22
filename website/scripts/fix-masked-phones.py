#!/usr/bin/env python3
"""Replace masked example phone numbers with valid fictional E.164 numbers.

+155****4567 -> +15550123456  (555-01xx is the FCC-reserved fictional range)
+155****6543 -> +15550987654
+155****3456 -> +15550123456
+155****2222 -> +15550122222
+155****0000 -> +15550120000
+1555***4567 -> +15550123456
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "docs"

MASKED = {
    "+155****4567": "+15550123456",
    "+155****6543": "+15550987654",
    "+155****3456": "+15550123456",
    "+155****2222": "+15550122222",
    "+155****0000": "+15550120000",
    "+1555***4567": "+15550123456",
}

pat = re.compile("|".join(re.escape(k) for k in MASKED))

changed = []
for p in sorted(ROOT.rglob("*.md")):
    try:
        t = p.read_text(encoding="utf-8")
    except Exception:
        continue
    new = pat.sub(lambda m: MASKED[m.group(0)], t)
    if new != t:
        p.write_text(new, encoding="utf-8")
        changed.append(str(p.relative_to(ROOT)))

print(f"Fixed {len(changed)} files")
for c in changed:
    print(" ", c)
