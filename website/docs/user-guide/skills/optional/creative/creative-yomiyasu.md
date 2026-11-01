---
title: "Yomiyasu — Fix AI-sounding Japanese into natural prose, upstream-kept"
sidebar_label: "Yomiyasu"
description: "Fix AI-sounding Japanese into natural prose, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Yomiyasu

Fix AI-sounding Japanese into natural prose, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/yomiyasu` |
| Path | `optional-skills/creative/yomiyasu` |
| Version | `1.0.5` |
| Author | nanaism (ALGO ARTIS) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `japanese`, `writing`, `editing`, `de-ai`, `proofreading`, `日本語`, `推敲`, `tech-writing` |
| Related skills | [`humanizer`](../../bundled/creative/creative-humanizer.md), [`simple-english`](../../optional/creative/creative-simple-english.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# yomiyasu（よみやす） (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [nanaism/yomiyasu](https://github.com/nanaism/yomiyasu): the project ships
> its own agent skill directory (`skills/yomiyasu/`, a Japanese `SKILL.md` plus
> domain references and two stdlib Python checkers). `hermes skills install
> official/creative/yomiyasu` pulls the current tree live from that repo
> (quarantined and scanned like any hub install) — this directory holds only
> the catalog metadata, so the rules can never lag the upstream corpus work.

yomiyasu rewrites AI-generated Japanese into prose a person would write:
readable, information-dense, and faithful to the original. Its first rule is
meaning preservation — claim, emphasis, strength of assertion and the
sentence's function (evaluation, explanation, request, plan) must survive the
rewrite — and its second is adding nothing. Within that frame it dismantles
inanimate subjects ("the architecture decides…"), replaces metaphorical verbs
(効く, 壊れる, 倒す) with what actually happens, removes emoji and trailing
colons, drops decorative parentheticals, strips the stray half-width spaces
around Latin words, and flattens excess bold, bullet lists and "not A but B"
contrasts into plain sentences that say who did what. Domain references tune
it for tech articles, business documents (specs, PR descriptions, reports) and
essays; a stance setting (勧め / 決まり / 説明) keeps sentence endings
consistent with the document's voice instead of policing them one by one.

The upstream author built it at ALGO ARTIS after finding that banned-word
lists only swap one vague word for another; the write-up (Zenn:
"AI臭い日本語を脱臭するAgent Skill『yomiyasu』を作った話") and the benchmark
corpus in the repo document the approach.

## Prerequisites

- Python 3 on `PATH` for the two optional checkers that ship with the skill:
  `scripts/yomiyasu_lint.py <file>` flags the slop patterns in a draft, and
  `scripts/yomiyasu_diff.py <before> <after> --stance=<勧め|決まり|説明>`
  checks that a rewrite preserved meaning and stance. Both are standard
  library only (`re`, `unicodedata`, `difflib`, `json`) and make no network
  calls.
- The upstream skill is a Japanese `SKILL.md` (~30 KB) + `references/`
  (`gemini-syntax.md` conversion rules, `slop-catalog.md`, `domains/{tech,
  business,essay}.md`) + `scripts/` (~110 KB total); the fetch is pinned to
  one tree SHA, recorded in the bundle metadata. The body is in Japanese on
  purpose: the task is Japanese prose, and the rules quote the forms they fix.
- Upstream warns that other Japanese style or proofreading skills active in
  the same session interfere with its instructions; keep one active.

## When to prefer it

- A draft in Japanese reads as machine-written: 技術記事, 設計書・仕様書,
  PR 説明文, 社内レポート, note/essay posts. Requests like 「読みやすくして」
  「AIっぽさをなくして」「自然な日本語にして」「文章を脱臭して」.
- For English text use the bundled `humanizer`; for controlled technical
  English use `simple-english`. yomiyasu is the Japanese counterpart, not a
  translator — it does not move text between languages.

Full documentation: https://github.com/nanaism/yomiyasu#readme
