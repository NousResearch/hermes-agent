# OPEN GAP found by line-level verification (2026-06-22) — needs disposition

The Council's demand for **line-level** (not just file-level) accounting surfaced a genuine gap
the file-level coverage script missed. Recording it honestly rather than papering over it.

## Root cause: file-level "IN-PR" via the stale-base gh union was wrong for 3 files

`reproduce-coverage.sh` classified a delta file as IN-PR if it appears in the **`gh api
pulls/<n>/files`** union. That API lists files against each PR's **original base** (stale `main`,
~1800 commits back), so it reports files touched by *intervening upstream commits*, not files the
PR's **own net diff** (`git diff merge-base..head`) changes. For 3 files the gh union said
"covered" but the authoritative merge-base diff says **no open PR carries their content**.

## The gap: ~89 legitimate novel lines orphaned by the agy/gemini-UA closures

When I closed #50555/#50657 (agy) and #50033 (gemini-cli-UA) on maintainer guidance, I
**reassigned** 3 "shared" files to #49644 — but #49644's net diff touches **none** of them
(verified). Worse, the closed **#50657 bundled the agy registration TOGETHER with legitimate
auth infrastructure**; closing it discarded the legitimate half too.

Precise sizing (novel = lines in our HEAD that are in neither v0.16.0 nor v0.17.0):

| file | novel lines | ~withdrawn (agy/gemini-UA) | ~LEGITIMATE, currently orphaned |
|---|---|---|---|
| `hermes_cli/auth.py` | 55 | ~8 | **~47** (codex device-code OAuth refresh, `_save_auth_store`/`_auth_file_mtime` helpers, xAI) |
| `hermes_cli/runtime_provider.py` | 46 | ~9 | **~37** (provider runtime resolution) |
| `agent/gemini_cloudcode_adapter.py` | 12 | ~7 (gemini-cli UA) | ~5 |
| **total** | 113 | ~24 | **~89** |

(The raw "687/220/15 changed-line" counts from the unified diff were mostly base-rebasing noise;
the **novel** content — genuinely ours and not upstream — is ~113 lines, of which **~89 are
legitimate and contributable**.)

## Honest status — this is NOT a clean "accept defaults and stop" state

- The **mechanical** migration is sound: set-level stack-apply 39/39 clean, 6 forward-port
  patches verified, DISCARD/SUPERSEDED buckets correct.
- But ~89 lines of legitimate overlay content (codex OAuth refresh + auth-store helpers +
  runtime-provider resolution) fell through the gap when #50657 was closed for its agy half.
  They are **not in any open PR** — a real violation of "all changes properly live in separate PRs."

## Required disposition (a real decision — surfaced, not self-resolved)

**Option A** — extract the ~89 legitimate lines (codex device-code OAuth refresh + auth-store
helpers in `auth.py`, the non-agy runtime-provider resolution, minus the gemini-UA bit) into a
**re-scoped #50657** (reopen + force-push the agy bits removed) or a small new clean PR. This
recovers the orphaned legit content while keeping the agy/gemini-UA out (ban-safe).

**Option B** — explicitly accept that this ~89-line auth-infra content is deferred/out-of-scope.

This touches the "don't proliferate PRs / reuse" guidance, so it is the user's call. It must NOT
be silently counted as covered. The coverage script is corrected to flag these 3 files as
GAP-PENDING rather than IN-PR.
