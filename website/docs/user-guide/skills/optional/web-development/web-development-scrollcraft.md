---
title: "Scrollcraft — Premium scroll-driven landing pages; scroll = timeline"
sidebar_label: "Scrollcraft"
description: "Premium scroll-driven landing pages; scroll = timeline"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Scrollcraft

Premium scroll-driven landing pages; scroll = timeline.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/scrollcraft` |
| Path | `optional-skills/web-development/scrollcraft` |
| Version | `1.0.0` |
| Author | nateherkai (upstream scroll-craft), ported by Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `web-development`, `landing-page`, `scrollytelling`, `animation`, `design`, `frontend` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# scrollcraft

Scroll is the only input every visitor already knows. This skill treats it as a
timeline: the wheel is a scrubber, the page is a film with real text on top,
and each section behaves differently enough that the visitor keeps going.

**What you produce:** an interview brief, a page grammar, a customer-journey
map, a feeling curve with one engineered peak, a scroll score, one signature
move, assets, one real HTML page on a token-driven design floor, and a strip of
screenshots proving it holds up at every scroll position.

Use for: "scrollytelling", "scroll animation site", "a site where scrolling
plays a video", "Apple-style landing page", "3D scroll world", "make my brand a
scroll experience", "this looks like a template", or any request for a site
that should feel like an experience rather than a document.

## What this is not

It is not "generate a flythrough and drop text on it." That produces one device
applied to a whole page, recognisable at a glance. Four spine rules:

1. **Variety is the product.** At least four device families, never the same
   device twice in a row. Read [references/devices.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/devices.md).
2. **The world is photographic** unless the brand is genuinely illustrated.
   Clay/low-poly diorama is banned as a default. Read [references/worlds.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/worlds.md).
3. **No continuous chain** unless the brief is literally "one continuous
   journey" (then see [references/worldflight.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/worldflight.md)).
4. **A different world is not a different page.** Structure is a separate axis;
   decide it deliberately. Read [references/uniqueness.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/uniqueness.md).

## Step 0: The interview

**Always ask the user in chat before building anything.** Real questions, asked
and answered in the conversation, written down — not a brief inferred from the
brand name. Eight questions in one pass:

1. **Vibe in three to five words**, plus up to three references from any medium
   (film, album cover, shop, magazine, game — not "sites you like").
2. **The scroll journey, section by section, in their words.**
3. **The energy curve** — where calm, where intense.
4. **How should someone feel while scrolling, stage by stage, and what is the
   ONE moment they should remember?** Becomes the feeling curve and the peak.
   See [references/feel.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/feel.md).
5. **One thing this site should do that no site they have seen does** — the
   seed of the signature move.
6. **How far from premium-minimal?** Offer the range in
   [references/uniqueness.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/uniqueness.md) §5: brutalist,
   maximalist, playful, retro, dense, editorial, premium-minimal.
7. **One unbroken world, or distinct scenes?** The biggest structural fork, and
   it is their call.
8. **What assets do they already have?** Footage, photos, product shots, brand
   kit. "Nothing" is fine and means a fully generated world.

Write the answers verbatim into `<workspace>/builds/<name>/BRIEF.md` (use
write_file) before any act planning. BRIEF.md must contain the eight answers,
the feeling curve (one line per act: emotion, then cause), the peak (as the
sentence a visitor would say to a friend), the completed "It's the site where
___" sentence, and any authored silence. If the user is genuinely unreachable
in a fully autonomous run, self-author BRIEF.md, mark it
`Self-authored, not interviewed`, and say so in the report.

## Bootstrap

Run the preflight rather than checking by hand (it catches a stripped ffmpeg
that reports missing filters as syntax errors):

```bash
node <skill>/scripts/doctor.mjs
node <skill>/scripts/workspace.mjs --ensure   # prints workspace, seeds registry
```

Workspace resolution order: `SCROLLCRAFT_HOME` env var; nearest
`.scrollcraft.json` (`{ "workspace": "..." }`) walking up from cwd;
`<project root>/scrollcraft`. Builds live at `<workspace>/builds/<name>/`, the
fingerprint registry at `<workspace>/FINGERPRINTS.md` (seeded from
[templates/FINGERPRINTS.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/templates/FINGERPRINTS.md), starts empty — the gate
stops you repeating *yourself*).

Copy `engine/scrollcraft.js` and `engine/scrollcraft.css` into the build
folder. **Never edit the engine per-project.** Theme with tokens; write your
own markup. Bespoke behaviour is bespoke JS in the page, driven off `--sc-p`
and your own `data-sc-*` attributes.

## Step 1: The brief, journey first

Ask the subject open, in plain prose. Then ask only what Step 0 did not cover:
what is this and who is it for; the one sentence the page installs; the one
next action (one label, used everywhere); what they already have; art
direction from [references/worlds.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/worlds.md). Then write the
**journey**: four to seven beats, each a shift in what the visitor knows or
feels. Beats are the spine; a section serving no beat is cut. Confirm the
journey with the user before generating assets — assets are the expensive part.

## Step 2: Grammar, gate, then score

Full detail in [references/uniqueness.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/uniqueness.md).

- **Pick a grammar.** Eight, mutually exclusive. Choosing filmic one-shot means
  saying in the report why the other seven lost. Nav, hero and close follow
  from the grammar.
- **Invent the signature move.** One bespoke interaction coded in the page, not
  a parameter change to a kit device. Interview question 5 is the seed.
- **Run the fingerprint gate.** The planned build must differ from every row in
  `<workspace>/FINGERPRINTS.md` on at least 4 of 6 dimensions: grammar, nav
  treatment, hero device, act-sequence shape, close pattern, signature move.
  If it fails, change the plan, not the log.
- **Write the feeling curve before the score table** (method:
  [references/feel.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/feel.md)). Then assign each beat a device in
  a written table (beat / device / why).

Checks before building: grammar bans hold; 4+ device families; no device
twice in a row; at most two `scrub` acts; no two adjacent acts with the same
feeling; one peak with the largest span; total page length 8–14
viewport-heights.

## Step 3: Assets

Full pipeline, prompt scaffolds and model notes: [references/assets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/assets.md).

**Hermes-native paths first:**

- **User-supplied footage and photos** — no key, no spend, a first-class route.
  Grade and encode them.
- **The `image_generate` tool** for stills: one style preamble reused verbatim
  in every prompt is what makes six images look like one shoot. Inspect every
  asset (vision_analyze) before use; rerolling beats shipping a bad frame.

**Optional upstream path — kie.ai** (vendored verbatim as
[scripts/kie.mjs](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/scripts/kie.mjs)): photoreal stills and camera-move clips.
Requires the `KIE_AI_API_KEY` environment variable (export it in your shell;
there is no bundled env file in this port). Check balance with
`node <skill>/scripts/kie.mjs probe`; a still costs cents, a 5s clip more.

```bash
node <skill>/scripts/kie.mjs still "<style preamble>\n\n<scene>" out/01-hero.png --ar 16:9
node <skill>/scripts/kie.mjs shot  "<camera move>" out/01-hero.png out/01.mp4 --dur 5
bash  <skill>/scripts/encode.sh out/01.mp4 assets/01.mp4
bash  <skill>/scripts/encode.sh out/01.mp4 assets/01-m.mp4 mobile
```

**Encode for scrubbing, not playback.** `encode.sh` sets a dense GOP because
seeking walks from the previous keyframe; a normal web encode scrubs like mud.
It also strips audio.

## Step 4: Build the page

Write real HTML — real `<h1>`, real `<p>`, real reading order. The engine reads
`data-sc-*` attributes off your markup and drives it; it never generates DOM.
Start from [references/template.html](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/template.html). Device
patterns: [references/devices.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/devices.md). Spacing, type, depth,
colour: [references/taste.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/taste.md) — read it before writing
markup. Theme by overriding tokens, six values and two fonts:

```css
:root {
  --sc-canvas: #0A0806;  --sc-surface: #16110E;
  --sc-ink:    #F5EBDD;  --sc-ink-soft: #A2968A;
  --sc-accent: #FF5A3D;  --sc-accent-ink: #15110F;
  --sc-font-display: "Archivo", system-ui, sans-serif;
  --sc-font-text:    "Geist", system-ui, sans-serif;
}
```

## Step 5: Verify by scrolling it

Not optional. Every scroll position is a different frame; failures live between
the two you looked at. Full procedure: [references/verify.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/verify.md).

```bash
cd <build project> && npm i playwright-core     # once
node <skill>/scripts/serve.mjs --root . --port 4500 &
node <skill>/scripts/shoot.mjs --url http://localhost:4500 --out lab/shots
node <skill>/scripts/shoot.mjs --url http://localhost:4500 --out lab/mobile --width 390 --height 844
node <skill>/scripts/shoot.mjs --url http://localhost:4500 --out lab/reduced --reduced-motion
```

The harness walks each act at six positions, waits for scrub video to settle,
reports dead scroll, cues that never reach full opacity, and composited
contrast; it writes a contact sheet. Then read `sheet.png` yourself
(vision_analyze) — the harness proves a clip advances, not that the page means
anything. Run the feel check ([references/feel.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/feel.md) §6):
scroll cold, one word per act, diff against BRIEF.md. Where they disagree the
page is wrong, not the brief.

A green run does not cover a real phone (video decoder, autoplay policy, Low
Power Mode). On any reported mobile defect, deploy
[references/device-diag.html](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/scrollcraft/references/device-diag.html) beside the site on
the first round and let the device answer.

## Hard rules (ship-blockers)

No clay diorama default; no "scroll to explore" cues or animated mouse icons;
no `01 / 06` section counters; at most one eyebrow per three sections; no
visible em dashes; vary the copy anchor; no device twice in a row; never build
before the interview; one engineered peak, not zero or three; the close
resolves instead of fading to a footer; curve before devices; one bespoke
signature move; 4-of-6 fingerprint clearance against every row; never edit the
engine; no full-frame dark overlay for contrast (scrim only where text sits);
no text baked into images; no invented statistics; no `transition: all` or
animating width/height/top/left (`transform`/`opacity`; `clip-path` for
wipes); no gradient text or neon glow; no audio on scrub clips; never ship
without Step 5.

## Output

The build folder including BRIEF.md, then a short report: grammar and why the
other seven lost, signature move, fingerprint gate result per row, journey,
feeling curve and peak, feel-check diff, score table, what you generated, what
you verified with screenshots, and what you could not verify. Append the
build's row to `<workspace>/FINGERPRINTS.md`.

## Pitfalls

- `scripts/shoot.mjs` needs Playwright (`npm install playwright` or
  `playwright-core` plus a Chrome install). Hermes' `browser_exec` tool is the
  lighter alternative for scroll-screenshot verification: serve the build,
  scroll in steps, capture screenshots, and inspect them yourself.
- `scripts/kie.mjs` needs `KIE_AI_API_KEY` and paid credit; prefer
  `image_generate` or user assets when the budget is unclear.
- `encode.sh` and `doctor.mjs` expect a full ffmpeg build; distro-stripped
  ffmpeg reports missing filters as command syntax errors — run
  `scripts/doctor.mjs` first.
- Upstream script invocations above are copied from upstream docs and
  unverified by this port beyond `node --check` syntax validation — trust
  `--help`/source if drifted.
- The upstream repo ships worked examples and a change log that are not
  vendored in this port; see the upstream repository if you want them.
