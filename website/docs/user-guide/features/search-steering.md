---
sidebar_position: 96
title: "Search Steering"
description: "Zero-match search steering and multi-path recovery — actionable hints when a content search comes up empty"
---

# Search Steering

A content search that returns zero matches is a dead turn for an agent: the model knows the text exists somewhere, the search says it doesn't, and there is nothing in the result to explain the gap. **Search Steering** fixes that. When `search_files` finds nothing, Hermes runs a few cheap probes and attaches an actionable hint — wrong casing, regex metacharacters that need escaping, or matches hiding in hidden/gitignored files — so the model can correct course on the next call instead of retrying the same query.

It also recovers from sloppy multi-path inputs. Models routinely pass several search roots in one string (`"dir1 dir2"` or comma-separated). Instead of failing the whole call when one of them doesn't exist, Hermes searches every path that does exist, merges the results, and tells you what it skipped.

## Why zero-match steering matters

A bare `0 matches` result gives the model nothing to steer by. The most common causes of a false zero are cheap to check and cheap to fix:

- **Wrong casing** — the pattern is case-sensitive by default, and the real text differs in case.
- **Regex metacharacters** — the pattern is interpreted as a regex, so `.`, `*`, `(`, `[`, etc. silently change what is being searched for.
- **Hidden or gitignored files** — ripgrep skips dot-directories and `.gitignore`d files by default, so a match that lives only in `.hidden/` or a vendored directory is invisible to the normal search.

A hint costs one or two short commands and converts a dead turn into a corrected one. Without it, the model's only moves are guessing, re-searching the same pattern, or burning a read on a directory listing.

## How the probes work

When a content search returns exactly zero results, Hermes runs a bounded probe sequence — count-only, with a short timeout — and attaches the first finding as a warning on the result:

1. **Case-insensitive probe.** Re-runs the pattern with case-insensitive matching. If it hits, the hint explains that casing may be wrong.
2. **Hidden/ignored probe.** Re-runs with hidden files and ignore rules included. If the only matches live there, the hint says so explicitly.
3. **Literal probe.** Only when the pattern contains regex metacharacters, re-runs it as a fixed string. If that hits, the pattern was being interpreted as regex.

The probes are cheap by design: count-only output, capped output lines, short timeouts, and at most a handful of invocations per zero-match result. They only run when the search genuinely found nothing — a normal search with matches pays no probe cost.

:::info
Probes never change the search results themselves. They only add a `warning` field to the zero-match result; the model sees the same empty result it would have seen, plus the hint that explains it.
:::

## Multi-path recovery

Models often pass multiple search roots as a single `path` string — `"src tests"` or `"src,tests,docs"`. Hermes treats a path that doesn't exist as a potential multi-path input:

- The string is split on whitespace and commas.
- Every candidate that exists is searched, and the results are **merged** into a single result set.
- Missing candidates are skipped and reported in a note: `path contained 3 entries; searched 2 that exist; skipped missing: vendor`.
- When more than three paths are missing, the note caps the list: `skipped missing: a, b, c (+4 more)`.

This applies to both content searches and file-name searches, and the merged result respects the caller's `limit` across all searched paths.

If the input is a **single** path and it doesn't exist, multi-path recovery doesn't apply — Hermes keeps the `Path not found: <path>` error, and when the parent directory exists it adds similar-path suggestions so the model can spot a typo (`Similar paths: src/utils, src/utilities`).

## Engine support

Content search prefers **ripgrep** when it is available (faster, respects `.gitignore` and hidden-file defaults) and falls back to **grep** when it isn't. The steering pass runs after either engine: the probes are engine-agnostic and the hints are identical whether the search was executed by ripgrep or by the grep fallback. If neither engine is installed, the search returns a clear install hint for ripgrep.

## What you'll see

Zero-match content searches now carry a warning explaining the near-miss, instead of a bare zero:

```
0 exact matches, but 4 case-insensitive match(es) in 2 file(s) — the pattern's casing may be wrong.
```

```
0 matches in visible files, but 3 match(es) in 1 hidden or gitignored file(s) — these are excluded by default. Search the hidden path explicitly to include them.
```

```
0 regex matches, but 12 literal match(es) — the pattern contains regex metacharacters that likely need escaping (or pass a simpler substring).
```

Multi-path searches with a missing entry report the merge in the result's note:

```
path contained 3 entries; searched 2 that exist; skipped missing: vendor
```

A single nonexistent path keeps the familiar error, with a typo hint when one can be inferred:

```
Path not found: srce/utils. Similar paths: src/utils, src/utilities
```
