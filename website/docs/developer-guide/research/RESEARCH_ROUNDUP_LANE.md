# Research Roundup Lane (Approach A)

Owner: Octacon (Coding Lead)
Approved editorial direction: Approach A (curated research-roundup)
Status: engine-implemented; pending downstream SahilBlog schema update

## What this is

A fourth SahilBlog content stream, `research`, that emits a **weekly curated
research roundup** instead of the essay shape used by the `ai`, `pm`, and
`builder` streams. It is DAIR.AI-inspired in information architecture:
numbered, scannable entries with a recurring tightly scoped container,
plain-English technical translation, cross-source synthesis, and explicit
judgement plus honest limitations.

The lane is **evidence-first** — a single-source roundup is invalid, and every
claim carries a named source, link, and date where available.

## Roundup shape (what the writer emits)

1. One `# Title` that names the week's through-line (specific, not clickbait).
2. A 2–3 sentence thesis paragraph stating that through-line in plain English.
3. Exactly **five** numbered, scannable entries (`## 01` … `## 05`), each
   titled with the specific finding — not a generic category like "Models".
4. Every entry follows the same tightly scoped container:
   - **(a)** the finding in plain English (one sentence, no jargon without definition)
   - **(b)** the evidence — named source, link, date
   - **(c)** the mechanism or context — one paragraph, jargon defined first use
   - **(d)** why a builder or PM should care — concrete
   - **(e)** honest limitations — what this does not show / what could be wrong
5. `## Takeaways` — three to five concrete moves plus one explicit
   "no action needed" verdict.
6. `## What I'd try next` — the open question the roundup surfaced.

Article-plus-social packaging: the thesis + the first numbered entry must be
publishable as a self-contained X/LinkedIn post without the rest of the body.

## Engine contract

| Field | Value | Notes |
|-------|-------|-------|
| `tier` | `research` | **NEW enum** — clamped to `pm` downstream (see below) |
| `source` | `curated-roundup` | **NEW enum** — clamped to `manual` downstream |
| `format` | `roundup` | **NEW enum** — clamped to `essay` downstream |
| `word_target` | `1500` | |
| `section_target` | `7` | thesis + 5 entries + takeaways + try-next |
| `entries_target` | `5` | numbered entries, independent of section_target |
| `require_two_distinct_sources` | `true` | cross-source synthesis floor |
| `article_plus_social` | `true` | deck + first entry self-contained |

### Files that implement the lane

- `content_engine/blog/blog_streams.py` — the `research` stream config.
- `content_engine/blog/blog_generator.py` — `_build_research_roundup_prompt`
  builds a roundup-shaped system/user prompt distinct from the essay skeleton.
- `content_engine/blog/blog_router.py` — `_gather_candidates` reads
  `blog_topics/research.jsonl`; `_read_manual_queue` lets the research queue
  fall through to the stream's `curated-roundup` source instead of the
  historical `manual_queue` override.
- `content_engine/blog/blog_gate.py` — `adhoc_check` hard-requires an external
  link (and dead-link detection) for the research stream.
- `content_engine/blog/schema_contract.py` — `_TIER_ALIASES`, `_SOURCE_ALIASES`,
  and `_FORMAT_ALIASES` clamp the new enum values to build-safe defaults.
- `content_engine/blog/blog_pipeline.py` — CLI `--stream` accepts `research`.

### Feeding the lane

Populate `blog_topics/research.jsonl` (one JSON object per line), matching the
AI/PM manual queue shape:

```json
{"topic_id": "r1", "title_hint": "Agent memory roundup", "priority": 7}
```

Optional `source_override` on a queue entry still wins; otherwise the lane
falls through to `curated-roundup` provenance.

## Downstream schema clamp (IMPORTANT)

The SahilBlog Astro schema (`src/content.config.ts`) does **not** yet enumerate
`tier: research`, `source: curated-roundup`, or `format: roundup`. Until it
does, `normalise_frontmatter` silently clamps:

| Engine value | Clamped to |
|--------------|------------|
| `tier: research` | `tier: pm` |
| `source: curated-roundup` | `source: manual` |
| `format: roundup` | `format: essay` |

This means the lane is fully available in the engine contract while production
ingestion stays unaffected. The clamp lives in
`content_engine/blog/schema_contract.py`.

> **Do not edit** `/home/kensei/repos/SahilBlog/src/content.config.ts` as part
> of the engine work. The schema update is a separate, coordinated downstream
> change. Until then the clamps above are load-bearing.

## Testing

Focused coverage lives in the blog test files:

- `content_engine/tests/test_blog_streams.py` — research stream shape + knobs.
- `content_engine/tests/test_blog_generator.py` — roundup prompt shape,
  word/section targets, verification warning, retry feedback, frontmatter.
- `content_engine/tests/test_schema_contract.py` — tier/source/format clamps.
- `content_engine/tests/test_blog_router.py` — `curated-roundup` provenance
  and the `manual_queue` regression guard for AI/PM.
- `content_engine/tests/test_blog_gate.py` — research external-link gate and
  dead-link detection.
