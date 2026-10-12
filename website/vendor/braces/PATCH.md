# braces 3.0.3 + nesting-depth guard (vendored)

Upstream: braces@3.0.3 (npm). Advisory: CVE-2026-93687 / GHSA-vfj7-8cjw-p6xm
(stack exhaustion DoS through deeply nested patterns). No patched release exists on npm
as of 2026-10-13.

Change: identical to the security commit bd1cc77acd0e in the Im-Fran/braces fork (MAX_DEPTH=100),
limited to two files:
- lib/constants.js: adds MAX_DEPTH: 100
- lib/parse.js: throws SyntaxError when nesting depth exceeds MAX_DEPTH

Verified (scratch): original 3.0.3 throws RangeError (stack overflow) on a 9,999-char
nested pattern for compile and expand; patched throws SyntaxError. Normal patterns: 0 output
differences on a 16-pattern corpus and 3,000 random nested cases.

Remove this vendored copy once a patched braces is published on npm and the override is dropped.
