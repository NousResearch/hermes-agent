---
title: "Apa7 References — APA 7 refs: X posts, GitHub PRs, software, data sets"
sidebar_label: "Apa7 References"
description: "APA 7 refs: X posts, GitHub PRs, software, data sets"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apa7 References

APA 7 refs: X posts, GitHub PRs, software, data sets.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/social-media\apa7-references` |
| Version | `1.0.0` |
| Author | Axl Ibiza, MBA |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `apa`, `references`, `citation`, `writing`, `academic`, `apa7` |
| Related skills | `axl-voice-and-public-writing`, `ai-agents-public-writing` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# APA 7th Edition References — Citation Map

Use when drafting a References list that must be APA 7th edition correct —
especially for X/Twitter posts, GitHub issues and PRs, software, data sets,
webpages, and long-form articles. This is the mapping Axl uses for public
writing (e.g. the "I KILLED THE GODFILE" article's reference comment).

## Core APA 7 rules (the spine)

1. **Alphabetical order** by first author's surname, then initials, then date
   (earliest first). "Nothing precedes something": a work with no author
   comes before works with authors.
2. **Same author, multiple works**: chronological by year. Same author+year:
   suffix letters a, b, c by title order, and cite them as 2026a, 2026b.
3. **The reference entry format**: Author. (Date). *Title* [Work type].
   Site/Publisher. URL
4. **Italics** go on the title of standalone works (books, reports,
   articles-as-standalone, software names, data sets). No italics on article
   titles inside periodicals.
5. **Each entry ends with a URL.** No trailing period after a URL.
6. **Hanging indent** in a formatted document (0.5 in); on X, use line breaks
   between entries.
7. **Retrieval dates**: only include when the content is designed to change
   (e.g. "Retrieved August 3, 2026, from ...") — do NOT add for stable pages.
8. **Authors**: up to 20 authors listed; last author preceded by "&".
   Use the username in parentheses for social media when real name differs:
   `Axl Ibiza, MBA [@andrexibiza]. (2026, August 3). ...`
9. **Group authors** (companies, orgs, repos): cite the organization as the
   author — e.g. `Nous Research.` — spelled out, not abbreviated.

## Exact forms (with the rules that produce them)

### X / Twitter posts
```
Author, A. A. [@username]. (Year, Month Day). *First 20 words of the post*
[Post]. X (formerly Twitter). URL
```
- Use the real name if known, plus handle in brackets.
- If only the handle is known, the handle becomes the author: `@username. (Year, Month Day). *...* [Post]. X. URL`
- Title = the first 20 words of the post, in italics, sentence case.
- The date is the post date; use `(n.d.)` if no date.

### X Articles (long-form)
```
Author. (Year, Month Day). *Title of the article* [X Article]. X (formerly
Twitter). URL
```
- The Article has a real title — use it (not "first 20 words").
- Same bracketed-username rule when real name differs from handle.

### GitHub issues
```
RepositoryAuthor. (Year, Month Day). *Issue title* [Issue]. GitHub. URL
```
- Repository author = the org/user that owns the repo (e.g. Nous Research).
- Date = the issue's `created_at` (real, from the API — never guessed).

### GitHub pull requests
```
RepositoryAuthor. (Year, Month Day). *PR title* [Pull request]. GitHub. URL
```
- Date = the PR's `created_at` (real, from the API).
- PR title = the PR's title (not the branch name).

### Software / repositories
```
Author or Org. (n.d.). *Software name* [Computer software]. Publisher. URL
```
- For a repo: `Nous Research. (n.d.). *Hermes Agent* [Computer software]. GitHub. https://github.com/NousResearch/hermes-agent`

### Source files inside a repo (docs, code, baselines)
```
Org. (n.d.). *Path/to/file.md* [Computer software source file]. GitHub. URL
```
- Cite the file path as the title when the file is the specific cited object
  (a spec, a generated baseline, a test).

### Data sets
```
Author. (Year, Month Day). *Title of the data set* [Data set]. Publisher/
host site. URL
```

### Webpages / documentation
```
Author or Org. (n.d.). *Page title*. Site name. URL
```

### Academic / periodical articles (when needed)
```
Author, A. A., & Author, B. B. (Year). Title of article. *Journal Name,
Volume*(Issue), pages. https://doi.org/xxx
```

## Worked example — the GodFile reference comment

The exact output that validated this mapping (35 GitHub items with real
creation timestamps, X posts, software, data set — all alphabetized):

```
Nous Research. (2026, June 29). *Issue #54962: ...* [Issue] (created
15:00:37 UTC). GitHub. https://github.com/NousResearch/hermes-agent/issues/54962
...
Teknium. (2026, August 3). *Deepseek v4 flash 0731* [Post]. X (Twitter). URL
Axl Ibiza, MBA. (2026, August 1). *Stop fixing Hermes. Start using it.*
[X Article]. X (Twitter). URL
```

## Real-date discipline (critical)

- **Never guess dates.** For GitHub issues/PRs, fetch `created_at` via the
  API (`gh api repos/{owner}/{repo}/issues/{n} --jq .created_at`).
- For X posts, use the post's timestamp.
- When a source has no retrievable date, use `(n.d.)` — APA 7 prefers honest
  "no date" over a fabricated date.
- Times are optional in APA (dates only are standard), but when Axl wants
  the full record, append `(created HH:MM:SS UTC)` to the GitHub entries.

## Verification gate

Before delivering a References list:
- [ ] Every URL in the source document appears in the reference list (and
      vice versa — no orphan references).
- [ ] Every GitHub date matches the API `created_at` value.
- [ ] Alphabetical by author surname, then date.
- [ ] No trailing period after URLs.
- [ ] Italics on standalone-work titles only.
- [ ] Work types bracketed correctly: [Post], [X Article], [Issue],
      [Pull request], [Computer software], [Data set].
- [ ] No inline APA citations in the body — final list only (Axl's rule).
