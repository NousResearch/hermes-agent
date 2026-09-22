---
title: "Citation Integrity — Use when verifying citation ledgers, quotes, and evidence"
sidebar_label: "Citation Integrity"
description: "Use when verifying citation ledgers, quotes, and evidence"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Citation Integrity

Use when verifying citation ledgers, quotes, and evidence.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/citation-integrity` |
| Path | `optional-skills/research/citation-integrity` |
| Version | `1.0.0` |
| Author | Hermes Agent contributors |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `research`, `citations`, `evidence`, `verification`, `provenance` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Citation Ledger Field Notes

A standalone field guide for making citation-ledger work reliable. Use it when
an answer or document must preserve the identity of its sources, attach quotes
that can be checked verbatim, detect duplicate citations, and pass a final
evidence review.

This skill is tool-agnostic. It does not require a particular retrieval service,
ledger implementation, shell environment, or private workflow. Apply the rules
manually or implement them in any citation-ledger tool that records sources,
archived evidence, and draft citations.

## Core model

Treat these as separate objects:

- **Source identity**: the canonical identity of a resource, normally its URL
  plus stable metadata such as title, publisher, author, and publication date.
- **Retrieval record**: one fetch of that source, including retrieval time,
  final URL after redirects, response metadata, and the captured text.
- **Evidence**: an immutable text capture or document snapshot from which a
  quote was copied.
- **Quote**: an exact substring of one evidence capture, linked to both the
  source identity and the evidence record.
- **Citation**: a draft reference to a ledger identifier. A citation is not
  evidence until its source has a matching quote or an explicitly recorded
  reason that no quote is available.

Never use a citation number as the source's identity. Numbers are presentation
labels and may change; the canonical source key and evidence record must not.

## When to use

Use this guide when:

- a draft quotes, paraphrases, or compares external sources;
- multiple searches or fetches may return the same page;
- a source changes, redirects, is mirrored, or is available in several URL
  encodings;
- the reader needs to audit the path from claim to source to exact text; or
- a final review must fail closed on missing, mismatched, or unverifiable
  evidence.

## Procedure

### 1. Register sources at retrieval time

Record each source immediately after fetching it, before drafting claims. Store
at least:

- the original URL and final URL;
- a canonical URL used for identity comparison;
- title, publisher, author, and publication date when available;
- retrieval timestamp;
- the evidence file or snapshot identifier; and
- a content digest of the captured bytes or normalized text.

Keep the original URL for auditability, but compare sources using a canonical
form. Canonicalization may remove an explicitly approved set of tracking
parameters, normalize the hostname and default port, and normalize equivalent
percent-encoding. Do not discard meaningful query parameters, fragments used
for content identity, or path segments merely because they look unusual.

If the site exposes a stable document identifier, record it and use it as an
additional identity key. A redirect target alone does not prove that two
records are duplicates: retain the redirect chain and inspect the resulting
content.

### 2. Detect duplicates before assigning new IDs

Before minting a new ledger ID, compare the candidate with existing records in
this order:

1. exact canonical URL;
2. final URL and redirect chain;
3. stable publisher/document identifier;
4. matching content digest for the same retrieval representation; and
5. title, author, date, and high-similarity text as a review signal only.

URL spelling differences such as equivalent percent-encoding, a trailing slash,
case differences in a hostname, or known tracking parameters must not silently
create a second source identity. Conversely, similar titles or syndicated
articles must not be merged automatically when they have different publishers,
dates, or text.

When a possible duplicate is found:

- keep one canonical source identity;
- attach all observed URLs and redirect chains as aliases;
- retain separate retrieval records when the content or retrieval date differs;
- move quotes to the surviving source identity only after checking that the
  evidence text is the same; and
- record the merge decision so a later reviewer can reproduce it.

A duplicate warning is not proof of identity. Require a human or deterministic
comparison rule for ambiguous cases.

### 3. Capture evidence before quoting

Archive the full text or a stable document snapshot used for drafting. Give the
capture a unique ID and digest, and keep a source-to-evidence map. Search-result
snippets, summaries, and model recollection are discovery aids, not page
evidence.

For every quote:

1. locate the wording in the archived evidence;
2. copy the exact matched substring from that evidence;
3. store the evidence ID, byte/text offsets when available, and capture digest;
4. store the quote exactly as captured, including punctuation, capitalization,
   diacritics, and visible markup; and
5. link the quote to the claim or draft passage it supports.

Do not retype a quote. Extract it programmatically or copy it directly from the
archived capture. If the capture contains Markdown or HTML artifacts, preserve
the captured form and optionally store a reader-facing rendering separately.
The evidence check may use a documented normalization for whitespace or markup,
but the original quote and original evidence must remain available for audit.

A paraphrase is a claim, not a quote. Label it as a paraphrase and verify that
its meaning is supported by the evidence; never put a paraphrase in a verbatim
quote field merely to make a check pass.

### 4. Preserve evidence identity

Evidence belongs to a particular retrieval representation. Do not attach a
quote from one version of a page to another version just because both have the
same URL. If content changes, create a new retrieval record and digest.

At minimum, a quote record should contain:

```text
source_id       canonical source identity
evidence_id     immutable capture or snapshot identity
quote_text      exact text copied from the capture
capture_digest  digest of the capture used for verification
location        offset, page, section, or other locator when available
retrieved_at    time the evidence was captured
```

If an evidence file is moved or renamed, update its metadata without changing
its identity. If it is regenerated, compare its digest and create a new
identity when the bytes or normalized text differ. Never repair a failed quote
match by editing the quote until it matches; recapture the source or correct the
quote from the evidence.

### 5. Verify the draft and ledger together

Run a final verification that fails on any of the following:

- a citation ID is absent from the ledger;
- a citation points to a source identity marked as a duplicate or unresolved;
- a source block disagrees with the ledger's canonical URLs or metadata;
- a quote is not an exact substring of its recorded evidence after only the
  documented comparison normalization;
- a quote's evidence digest, source identity, or retrieval record is missing;
- a claim is presented as supported but has neither a citation nor an explicit
  `unverified`/`not found` declaration; or
- a draft cites a source whose evidence cannot be located or read.

Also report warnings for registered-but-uncited sources, duplicate candidates,
quotes without locators, and claims supported only by low-confidence signals.
Warnings do not replace the fail-closed checks.

For a strict review, verify both directions:

- every citation in the draft resolves to exactly one ledger identity; and
- every cited ledger identity has the evidence and quote records required by
  the review policy.

Inspect the actual claim-to-quote mapping, not only aggregate coverage. A
coverage percentage can look healthy while a central claim has no evidence.

## Practical checks

Use a small script or ledger command to perform these checks rather than relying
on a status line alone:

- canonicalize candidate URLs and show aliases side by side;
- compare content digests before merging records;
- search each quote in its own archived evidence file;
- report missing evidence IDs and unreadable captures;
- compare the draft's citation IDs with the rendered source list; and
- rerun verification after any source merge, URL change, quote edit, or draft
  rewrite.

When a source ID must be migrated, copy the surviving evidence mapping, recheck
every quote against its recorded capture, regenerate the source list, and run
the strict verifier again. Never hand-edit only the number in the draft.

## Common failure modes

- **Equivalent URLs mint duplicate IDs.** Normalize approved URL differences and
  retain aliases; do not use a raw URL string as the only identity key.
- **A plain citation check passes without evidence.** Make evidence presence and
  quote-to-capture matching explicit strict-mode requirements.
- **Quotes are retyped from memory.** Copy or extract the exact substring from
  the archived capture, including typography and markup artifacts.
- **Search snippets become evidence.** Fetch and archive the source page before
  recording a supporting quote.
- **A changed page reuses an old capture.** Bind every quote to an evidence ID
  and digest, and create a new retrieval record when content changes.
- **Aggregate statistics hide a missing key claim.** Audit the claim-to-quote
  mapping and treat coverage as a diagnostic, not proof.
- **A hand-edited source list goes stale.** Render source metadata from the
  ledger and verify the rendered list against the draft's citations.

## Delivery checklist

Before publishing a grounded document, confirm:

- [ ] Every external claim has a citation or an explicit provenance marker.
- [ ] Every cited source resolves to one canonical identity.
- [ ] URL aliases and duplicate decisions are recorded.
- [ ] Every required quote was copied from archived evidence, not retyped.
- [ ] Every quote has an evidence ID and matching capture digest.
- [ ] The source list was generated from the ledger.
- [ ] Strict verification passed after the final draft edit.
- [ ] Any unresolved source, missing capture, or unverified claim is disclosed.
