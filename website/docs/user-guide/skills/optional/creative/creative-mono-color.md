---
title: "Mono Color — Generate one- or two-ink editorial print poster images"
sidebar_label: "Mono Color"
description: "Generate one- or two-ink editorial print poster images"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Mono Color

Generate one- or two-ink editorial print poster images.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/mono-color` |
| Path | `optional-skills/creative/mono-color` |
| Version | `1.0.0` |
| Author | Yan Liu (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `design`, `poster`, `print`, `duotone`, `risograph`, `editorial`, `image-generation` |
| Related skills | [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic), [`meme-generation`](/docs/user-guide/skills/optional/creative/creative-meme-generation), [`pixel-art`](/docs/user-guide/skills/optional/creative/creative-pixel-art) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Mono-Color Editorial Print Skill

Turn any user theme, sentence, or reference photo into an original printed editorial artifact with one stable visual language: adaptive neutral substrate + one or two inks + mechanically reproduced image + typographic tension + concise human voice.

This skill designs and generates the image; it does not imitate any one reference, copy a source composition, wording, logo, or artwork, and it never uses more than two printing inks.

## When to Use

The user asks for a monochrome editorial poster, duotone print, risograph/zine poster, halftone photo treatment, one-ink or two-ink cover, or names the mono-color style. Chinese trigger vocabulary includes 单色海报、双色印刷、单色调视觉、蓝色/绿色孔版印刷、网点照片、复古或当代编辑排版. Do not trigger merely because a request mentions a color.

## Prerequisites

- The Hermes `image_generate` tool (search/describe it via the deferred-tool catalog if not loaded). If image generation is unavailable, deliver prompt-only and say so.
- The `design-system/` catalogs bundled with this skill (see Quick Reference).

## Quick Reference

Print modes:

| Mode | When |
|---|---|
| Pure one-ink | User explicitly requests one ink, monochrome, or one named ink without a second color |
| Chromatic ink + black | Quiet, observational, natural, architectural, long-form subjects; chromatic plate carries the image, carbon/charcoal carries text |
| Complementary duotone | General default; dominant plate 70–85%, accent 15–30% with a specific role; fallback pair Cobalt + Terracotta `#2148B8` + `#C65F38` |
| Overprint duotone | Two plates deliberately overlap; the darker mixed zone is not a third ink |

Catalogs (source of truth — **when an exact value differs, the catalog wins over any prose**; read only the catalog relevant to the current decision):

| File | Provides |
|---|---|
| `design-system/colors.json` | Substrate IDs + exact hex, one-ink palette, approved two-ink pairs |
| `design-system/compositions.json` | Layout-family IDs and geometry |
| `design-system/typography.json` | Type-hierarchy role IDs |
| `design-system/rhythm.json` | Visual tension profiles, focal events, unresolved edges |
| `design-system/imperfections.json` | Controlled print-imperfection effect IDs and ranges |
| `design-system/carriers.json` | Carrier signals (poster, journal page, cover, etc.) |

References:

- `references/visual-language.md` — full color/space/image-treatment/typography/tone rules
- `references/composition.md` — layout decision flow, layout families, composition grammar, rhythm
- `references/quality-gate.md` — originality firewall, hard avoids, inspection checklist

## Procedure

1. **Read the input.** Extract five things:
   - **Subject:** the one person, object, scene, or idea that must remain recognizable.
   - **Intent:** poetic observation, announcement, field note, personal statement, cultural poster, or specimen page.
   - **Words:** preserve exact supplied text verbatim in its original language — never translate or rewrite it. If no text is supplied, invent one English display phrase of 2–8 words and keep it stable across retries. Omit text only on explicit request.
   - **Image role:** hero photograph, isolated specimen, cropped fragment, texture source, or none.
   - **Representation:** faithful reproduction (default) or abstract symbol extraction (when the user asks for abstract, artistic, loose, experimental, less realistic, or less photographic treatment).

   For a complex topic, pick one concrete visual metaphor; do not illustrate every point. If the user supplies an image, preserve its identity and factual content — crop/isolate/halftone it, never replace the subject or invent branded details.

2. **Resolve the recipe manifest.** Fill every field; do not skip any and do not expose the manifest unless the user asks for process details. Look up IDs and exact values in `design-system/`.

   ````yaml
   subject: <one recognizable subject>
   intent: <one intent from Input Reading>
   exact_text: <user text, generated 2-8 word phrase, or none>
   text_language: <language of supplied text, otherwise English>
   representation: <faithful reproduction or abstract symbol extraction>
   ratio: <explicit ratio or 3:4>
   carrier: <one carrier ID from design-system/carriers.json or none>
   substrate: <one substrate ID and exact hex from design-system/colors.json>
   mode: <pure one-ink, chromatic + black, complementary duotone, or overprint duotone>
   palette: <one palette ID from design-system/colors.json>
   inks: <the palette's named ink or approved pair with exact hex values>
   plate_roles: <one explicit role per ink plate>
   layout: <one composition ID from design-system/compositions.json>
   empty_paper: <explicit percentage>
   visual_tension: <relaxed, balanced, or assertive from design-system/rhythm.json>
   focal_event: <one strong visual event from design-system/rhythm.json>
   release_zone: <one deliberately quiet region that gives the focal event room>
   unresolved_edge: <one optional edge behavior from design-system/rhythm.json or none>
   image_treatment: <one mechanical reproduction process>
   type_hierarchy: <one role ID from design-system/typography.json>
   disruption: <one deliberate disruption>
   imperfection_seed: <stable hash derived from the resolved recipe>
   imperfections: <0-2 restrained effect IDs for contemporary work, or 2-3 for tactile/vintage work>
   ````

   Defaults when the user hasn't chosen: ratio `3:4`; substrate Neutral White `#FAFAF7` (Cool Gray `#E9E9E5` for architecture/tech/restrained; Pale Beige `#F5F1E8` only for tactile/archival/nostalgic subjects — never assume beige merely because the work uses halftone or risograph language); mode complementary duotone with Cobalt + Terracotta; empty paper `35%`; tension `relaxed` for reflective/leisure/unspecified cultural subjects, `balanced` for editorial information, `assertive` only for forceful declarations; disruption = one off-center image crop, or one oversized word when there is no image. Explicit user choices override defaults unless they violate the two-ink limit or the originality firewall. Identical inputs must resolve to the identical manifest — never vary palette, layout, percentages, or process for novelty.

   Resolve generic color words consistently: blue→Cobalt, green→Botanical Green, orange→Terracotta Orange, red→Signal Red, purple→Aubergine, black→Charcoal; green+black→Mint Green + Charcoal; blue+orange→Cobalt + Terracotta. Exact named inks always take precedence.

3. **Choose the layout.** Walk the decision flow in `references/composition.md` top to bottom and take the first match (events→ruled information poster; botanical→archival plate; repeated object→object field; crossing layers→overprint collage; supplied photo→image field or editorial cover; isolated objects→specimen annotation; phrase-as-subject→type-led declaration; essay-like→editorial journal; otherwise editorial cover).

4. **Compile the prompt** in five compact paragraphs, in order:
   1. **Canvas and ink:** ratio, exact substrate hex and reason, exact one/two-ink palette hexes, print mode, plate roles, flat front-facing page (no mockup, frame, desk, or shadow).
   2. **Original composition:** layout family, tension profile, one focal event, one release zone, margins (5–9%), empty-paper percentage (25–55%), grid, dominant object scale (45–80% of page) and edge crop, optional unresolved edge, one manual gesture.
   3. **Subject:** what appears; for faithful reproduction, preservation/crop/halftone/paper exposure; for abstract extraction, the 2–4 identity anchors, dominant mass, structural contour, repeated rhythm, and where exposed paper cuts through.
   4. **Typography and words:** hierarchy, type voices, exact short display text, and the explicit overlap/crossing/split/tight alignment between headline and dominant object.
   5. **Material and avoids:** dots, fibers, bleed, misregistration, plus the hard negative constraints from `references/quality-gate.md`.

   Describe only visible outcomes. Never mention reference artists, studios, sample posters, or "in the style of."

5. **Generate and inspect.** Call `image_generate` with the compiled prompt. Inspect at full and thumbnail size against the checklist in `references/quality-gate.md`; regenerate once on failure (extra ink, missing plate roles, empty paper outside 25–55%, unrecognizable subject, no ≥5x type scale jump, garbled text, composition copying a reference, no identifiable focal event). If exact text still renders wrong after one retry, generate a text-light base image and state that typography should be overlaid in a layout tool — never pretend distorted text is correct.

6. **Deliver.** Save outputs under `./mono-color-output/` in the user's working directory (create it if needed), or another location the user names. Present:
   1. the generated image (path or rendered);
   2. the final prompt in a fenced `text` block;
   3. a short recipe note: Mode, Ink (exact hexes), Layout, Type (editorial + utility voice), Process, and one Originality sentence naming the structural departures from any supplied reference.

   Stop at prompt-only only when the user explicitly asks or image generation is unavailable.

## Pitfalls

- **Never more than two printing inks.** The substrate is not an ink; overprint mixing and density variation are not extra inks. Gradients, rainbow accents, and full-color photography are always out.
- **Catalog wins over prose.** When a hex, ID, range, or geometry in `design-system/` differs from any prose description, use the catalog value.
- **Preserve supplied text verbatim** — original language, exact wording, no translation unless asked. Never distort microcopy or factual text with imperfection effects.
- **Never copy a source composition, wording, logo, or artwork.** Change at least four structural features from any supplied reference (see the originality firewall). No fake signatures, mastheads, sponsors, URLs, or invented branding.
- **Contemporary by default.** Do not add yellowed paper, sepia, distressed borders, or retro props merely because the work uses halftone or limited inks — only when the user asks for vintage/archival mood.
- **One focal event, one release zone.** Never center everything, never distribute elements evenly like a template, never fill the quiet zone with decoration.

## Verification

- Manifest fully resolved, all IDs present in the `design-system/` catalogs.
- Result uses one intentional white/gray/pale-beige substrate and ≤2 inks with clear plate roles.
- 25–55% visibly empty paper; one dominant object at 45–80%; headline visibly crosses or locks to it.
- Type hierarchy shows a 5–12x scale jump with ≤3 type voices.
- Supplied subject and text preserved exactly; ≥4 structural features differ from every supplied reference.
- An image was generated (unless prompt-only was requested) and saved under the output directory, and the recipe note was delivered.

## Notice

Upstream example artwork is not included: `examples/` in the source repo is all-rights-reserved (see upstream ASSET-LICENSE.md); only MIT-licensed text and design-system catalogs are vendored here. Code and text are MIT (see `LICENSE.txt`).
