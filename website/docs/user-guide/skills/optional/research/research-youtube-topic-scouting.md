---
title: "Youtube Topic Scouting — Scout sourced AI/security video topics for approval"
sidebar_label: "Youtube Topic Scouting"
description: "Scout sourced AI/security video topics for approval"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Youtube Topic Scouting

Scout sourced AI/security video topics for approval.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/youtube-topic-scouting` |
| Path | `optional-skills/research/youtube-topic-scouting` |
| Version | `0.1.0` |
| Author | Hugo Gomes (Ukrawave), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `YouTube`, `Research`, `AI`, `Security`, `Content` |
| Related skills | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), [`grounded-citations`](/docs/user-guide/skills/bundled/research/research-grounded-citations), [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# YouTube Topic Scouting Skill

Create a short, ranked slate of evidence-backed topic briefs for a faceless AI, engineering, or security YouTube channel. Use it for research and selection, not scripting, production, publishing, or posting.

## When to Use

- A user wants fresh YouTube topic ideas with citations.
- A channel operator needs a daily or weekly scouting slate.
- A research workflow must avoid recycled, stale, sponsored, or weakly sourced stories.
- Do not use for writing the approved video script; use the channel's script-writing workflow after greenlight.

## Prerequisites

- Ask for or discover the channel's project root if deduplication against past videos is required.
- Use `web_search` to discover leads and `web_extract` or `browser` to open canonical sources.
- Use `search_files` and `read_file` to examine the channel's prior scripts when the project root is available.
- Load `arxiv` for fresh research papers, `grounded-citations` for deeper source work, and `youtube-content` only when an existing source video needs analysis.

## How to Run

1. Confirm the requested channel lane, audience, and time window; default to AI agents, AI engineering, and agent security, with a 72-hour news window.
2. If the channel root exists, use `search_files` to list prior projects and identify relevant `script.md` files. Completion: serious candidate incidents have a dedup search target before source research begins.
3. Gather leads from reputable reporting, official disclosures, release notes, advisories, repositories, and fresh papers. Completion: each candidate has at least one canonical source URL.
4. Use `web_extract` or `browser` to read source bodies, record the underlying event date, and resolve secondary reporting to the disclosing party where possible. Completion: every intended claim has direct supporting text.
5. Run the source-quality and content-dedup gates below, score the survivors, and return 0–3 briefs. Completion: every returned brief passes every gate.

## Brief Format

```text
<N>. TOPIC: <one line>
HOOK: <why the viewer stops scrolling>
WHY NOW: <underlying news/release/research peg with exact date>
KEY CLAIMS:
- <claim> — <canonical URL> [Primary|Editorial|Attributed]
- <claim> — <canonical URL> [Primary|Editorial|Attributed]
- <claim> — <canonical URL> [Primary|Editorial|Attributed]
SUGGESTED VISUALS: <faceless diagrams, animations, or screen recordings>
TARGET LENGTH: <minutes>
ANGLE CONFIDENCE: <high|med|low — rationale and material caveats>
SOURCE CHECK: <sources opened; sponsorship and lead-gen result>
DEDUP CHECK: <underlying-incident check result>
SCORE: <N>/20
```

Use 3–6 claims only when evidence supports them. `Attributed` denotes a self-reported or otherwise unverified claim; it cannot be the title or thumbnail spine.

## Quality Gates

A candidate that fails any gate is disqualified.

1. **Real why-now:** assess the date of the underlying event, release, or paper—not a fresh reaction or syndication. Default limit: 72 hours for news and 7 days for a newly released paper or product. Evergreen topics need user approval.
2. **Claim-level evidence:** every key claim needs a canonical URL read in this run whose body supports the claim. Search pages, RSS feeds, HTTP status codes, and guessed URLs are leads, not evidence.
3. **Source hierarchy:** prefer original disclosures, advisories, papers, release notes, and repositories. Use independent editorial sources for corroboration. Label social or self-reported claims as attributed.
4. **Source quality:** reject sponsored articles, advertorials, vendor lead magnets, and pages dominated by a trial/demo pitch. Do not reuse a statistic unless its primary study/report was opened. Original vendor technical research is allowed, but disclose a relevant commercial interest.
5. **Content-level dedup:** search previous scripts using the incident entities and distinctive mechanism, then read any matches. A new nickname, an outlet's reaction story, or a shared product/model name does not make an old incident new. Reject exact recycles; disclose and distinguish genuine follow-ups or adjacent mechanisms.
6. **Channel fit and visual clarity:** favor stories that can be explained with faceless visuals and have a concrete technical mechanism, consequence, or decision—not generic hype.
7. **No quota padding:** return fewer briefs when necessary. A zero-topic result is better than weak material.

## Ranking

Score each qualifying candidate out of 20:

- Dated, verifiable why-now: 0–5
- Fit with the channel lane: 0–5
- Source strength: 0–4
- Visual and narrative clarity: 0–3
- Novelty versus prior coverage: 0–3

Normally return only candidates scoring at least 14/20. Group leads into distinct themes and select the strongest verified candidate per theme rather than producing several versions of the same video.

## Outcomes

- **One or more qualifying candidates:** return ranked briefs and ask the user which number to greenlight.
- **Notable but unsourceable event:** add one short “On the radar” note explaining what source access or primary evidence is missing; do not make it a full brief.
- **No viable material:** return exactly `[SILENT]` for scheduled runs, or plainly explain that no candidate passed the gates in interactive work.

## Pitfalls

- Fresh reporting can repackage old research. Verify the original event date before treating it as news.
- A page that returns HTTP 200 may be a bot wall, empty shell, wrong fuzzy match, or generic landing page. Read the body.
- A headline number can be self-reported, secondhand, or commercial marketing even when the underlying story is real. Keep it out of the main hook unless independently verified.
- A vendor disclosure may be technically legitimate while the vendor sells the mitigation. State that conflict rather than discarding original research automatically.
- A dedup search hit is a prompt to read the prior script, not proof of duplication; shared names can appear in unrelated incidents.

## Verification

- [ ] Each brief contains every field in the brief format.
- [ ] Each key claim is supported by a canonical source read during this run.
- [ ] The source body passed sponsorship/lead-gen review.
- [ ] The underlying event date is stated and within the requested window.
- [ ] Prior-channel deduplication was checked when a project root was available.
- [ ] Every candidate passed all gates and scored at least 14/20.
- [ ] The slate is distinct by theme and not padded.
