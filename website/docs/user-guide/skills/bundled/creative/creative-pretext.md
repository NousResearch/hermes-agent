---
title: "Pretext"
sidebar_label: "Pretext"
description: "Use when building creative browser demos with @chenglou/pretext — DOM-free text layout for ASCII art, typographic flow around obstacles, text-as-geometry gam..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pretext

Use when building creative browser demos with @chenglou/pretext — DOM-free text layout for ASCII art, typographic flow around obstacles, text-as-geometry games, kinetic typography, and text-powered generative art. Produces single-file HTML demos by default.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/creative/pretext` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `creative-coding`, `typography`, `pretext`, `ascii-art`, `canvas`, `generative`, `text-layout`, `kinetic-typography` |
| Related skills | [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Pretext Creative Demos

## Overview

[`@chenglou/pretext`](https://github.com/chenglou/pretext) is a 15KB zero-dependency TypeScript library by Cheng Lou (React core, ReasonML, Midjourney) for **DOM-free multiline text measurement and layout**. It does one thing: given `(text, font, width)`, return the line breaks, per-line widths, per-grapheme positions, and total height — all via canvas measurement, no reflow.

That sounds like plumbing. It is not. Because it is fast and geometric, it is a **creative primitive**: you can reflow paragraphs around a moving sprite at 60fps, build games whose level geometry is made of real words, drive ASCII logos through prose, shatter text into particles with exact per-grapheme starting positions, or pack shrink-wrapped multiline UI without any `getBoundingClientRect` thrash.

This skill exists so Hermes can make **cool demos** with it — the kind people post to X. See `pretext.cool` and `chenglou.me/pretext` for the community demo corpus.

## When to Use

Use when the user asks for:
- A "pretext demo" / "cool pretext thing" / "text-as-X"
- Text flowing around a moving shape (hero sections, editorial layouts, animated long-form pages)
- ASCII-art effects using **real words or prose**, not monospace rasters
- Games where the playfield / obstacles / bricks are made of text (Tetris-from-letters, Breakout-of-prose)
- Kinetic typography with per-glyph physics (shatter, scatter, flock, flow)
- Typographic generative art, especially with non-Latin scripts or mixed scripts
- Multiline "shrink-wrap" UI (smallest container width that still fits the text)
- Anything that would require knowing line breaks *before* rendering

Don't use for:
- Static SVG/HTML pages where CSS already solves layout — just use CSS
- Rich text editors, general inline formatting engines (pretext is intentionally narrow)
- Image → text (use `ascii-art` / `ascii-video` skills)
- Pure canvas generative art with no text role — use `p5js`

## Creative Standard

This is visual art rendered in a browser. Pretext returns numbers; **you** draw the thing.

- **Don't ship a "hello world" demo.** The `hello-orb-flow.html` template is the *starting* point. Every delivered demo must add intentional color, motion, composition, and one visual detail the user didn't ask for but will appreciate.
- **Dark backgrounds, warm cores, considered palette.** Classic amber-on-black (CRT / terminal) works, but so do cold-white-on-charcoal (editorial) and desaturated pastels (risograph). Pick one and commit.
- **Proportional fonts are the point.** Pretext's whole vibe is "not monospaced" — lean into it. Use Iowan Old Style, Inter, JetBrains Mono, Helvetica Neue, or a variable font. Never default sans.
- **Real source/text, not lorem ipsum.** The corpus should mean something. Short manifestos, poetry, real source code, a found text, the library's own README — never `lorem ipsum`.
- **First-paint excellence.** No loading states, no blank frames. The demo must look shippable the instant it opens.

## Stack

Single self-contained HTML file per demo. No build step.

| Layer | Tool | Purpose |
|-------|------|---------|
| Core | `@chenglou/pretext` via `esm.sh` CDN | Text measurement + line layout |
| Render | HTML5 Canvas 2D | Glyph rendering, per-frame composition |
| Segmentation | `Intl.Segmenter` (built-in) | Grapheme splitting for emoji / CJK / combining marks |
| Interaction | Raw DOM events | Mouse / touch / wheel — no framework |

```html
<script type="module">
import {
  prepare, layout,                   // use-case 1: simple height
  prepareWithSegments, layoutWithLines,  // use-case 2a: fixed-width lines
  layoutNextLineRange, materializeLineRange, // use-case 2b: streaming / variable width
  measureLineStats, walkLineRanges,  // stats without string allocation
} from "https://esm.sh/@chenglou/pretext@0.0.6";
</script>
```

Pin the version. `@0.0.6` at time of writing — check [npm](https://www.npmjs.com/package/@chenglou/pretext) for the latest if demo behavior is off.

## The Two Use Cases

Almost everything reduces to one of these two shapes. Learn both.

### Use-case 1 — measure, then render with CSS/DOM

```js
const prepared = prepare(text, "16px Inter");
const { height, lineCount } = layout(prepared, 320, 20);
```

You still let the browser draw the text. Pretext just tells you how tall the box will be at a given width, **without** a DOM read. Use for:
- Virtualized lists where rows contain wrapping text
- Masonry with precise card heights
- "Does this label fit?" dev-time checks
- Preventing layout shift when remote text loads

**Keep `font` and `letterSpacing` exactly in sync with your CSS.** The canvas `ctx.font` format (e.g. `"16px Inter"`, `"500 17px 'JetBrains Mono'"`) must match the rendered CSS, or measurements drift.

### Use-case 2 — measure *and* render yourself

```js
const prepared = prepareWithSegments(text, FONT);
const { lines } = layoutWithLines(prepared, 320, 26);
for (let i = 0; i < lines.length; i++) {
  ctx.fillText(lines[i].text, 0, i * 26);
}
```

This is where the creative work lives. You own the drawing, so you can:
- Render to canvas, SVG, WebGL, or any coordinate system
- Substitute per-glyph transforms (rotation, jitter, scale, opacity)
- Use line metadata (width, grapheme positions) as geometry

For **variable-width-per-line** flow (text around a shape, text in a donut band, text in a non-rectangular column):

```js
let cursor = { segmentIndex: 0, graphemeIndex: 0 };
let y = 0;
while (true) {
  const lineWidth = widthAtY(y);  // your function: how wide is the corridor at this y?
  const range = layoutNextLineRange(prepared, cursor, lineWidth);
  if (!range) break;
  const line = materializeLineRange(prepared, range);
  ctx.fillText(line.text, leftEdgeAtY(y), y);
  cursor = range.end;
  y += lineHeight;
}
```

This is the most important pattern in the whole library. It's what unlocks "text flowing around a dragged sprite" — the demo that went viral on X.

### Helpers worth knowing

- `measureLineStats(prepared, maxWidth)` → `{ lineCount, maxLineWidth }` — the widest line, i.e. multiline shrink-wrap width.
- `walkLineRanges(prepared, maxWidth, callback)` — iterate lines without allocating strings. Use for stats/physics over graphemes when you don't need the characters.
- `@chenglou/pretext/rich-inline` — the same system but for paragraphs mixing fonts / chips / mentions. Import from the subpath.

## Demo Recipe Patterns

The community corpus (see `references/patterns.md`) clusters into a handful of strong patterns. Pick one and riff — don't invent a new category unless asked.

| Pattern | Key API | Example idea |
|---|---|---|
| **Reflow around obstacle** | `layoutNextLineRange` + per-row width function | Editorial paragraph that parts around a dragged cursor sprite |
| **Text-as-geometry game** | `layoutWithLines` + per-line collision rects | Breakout where each brick is a measured word |
| **Shatter / particles** | `walkLineRanges` → per-grapheme (x,y) → physics | Sentence that explodes into letters on click |
| **ASCII obstacle typography** | `layoutNextLineRange` + measured per-row obstacle spans | Bitmap ASCII logo, shape morphs, and draggable wire objects that make text open around their actual geometry |
| **Editorial multi-column** | `layoutNextLineRange` per column + shared cursor | Animated magazine spread with pull quotes |
| **Kinetic type** | `layoutWithLines` + per-line transform over time | Star Wars crawl, wave, bounce, glitch |
| **Multiline shrink-wrap** | `measureLineStats` | Quote card that auto-sizes to its tightest container |

See `templates/donut-orbit.html` and `templates/hello-orb-flow.html` for working single-file starters.

## Workflow

1. **Pick a pattern** from the table above based on the user's brief.
2. **Start from a template**:
   - `templates/hello-orb-flow.html` — text reflowing around a moving orb (reflow-around-obstacle pattern)
   - `templates/donut-orbit.html` — advanced example: measured ASCII logo obstacles, draggable wire sphere/cube, morphing shape fields, selectable DOM text, and dev-only controls
   - `write_file` to a new `.html` in `/tmp/` or the user's workspace.
3. **Swap the corpus** for something intentional to the brief. Real prose, 10-100 sentences, no lorem.
4. **Tune the aesthetic** — font, palette, composition, interaction. This is the work; don't skip it.
5. **Verify locally**:
   ```sh
   cd <dir-with-html> && python3 -m http.server 8765
   # then open http://localhost:8765/<file>.html
   ```
6. **Check the console** — pretext will throw if `prepareWithSegments` is called with a bad font string; `Intl.Segmenter` is available in every modern browser.
7. **Show the user the file path**, not just the code — they want to open it.

## Performance Notes

- `prepare()` / `prepareWithSegments()` is the expensive call. Do it **once** per text+font pair. Cache the handle.
- On resize, only rerun `layout()` / `layoutWithLines()` — never re-prepare.
- For per-frame animations where text doesn't change but geometry does, `layoutNextLineRange` in a tight loop is cheap enough to do every frame at 60fps for normal-length paragraphs.
- When rendering ASCII masks per frame, keep a cell buffer (`Uint8Array`/typed arrays), derive measured per-row obstacle spans from the cells or projected geometry, merge spans, then feed those spans into `layoutNextLineRange` before drawing text.
- Keep visual animation and layout animation coupled. If a sphere morphs into a cube, tween both the rendered cell buffer and the obstacle spans with the same value; otherwise the demo looks painted-on instead of physically reflowed.
- For fades, prefer layer opacity over changing glyph intensity or obstacle scale. Put transient ASCII sprites on their own canvas and fade the canvas with CSS/GSAP opacity so geometry does not appear to shrink.
- Canvas `ctx.font` setting is surprisingly slow; set it **once** per frame if font doesn't vary, not per `fillText` call.

## Common Pitfalls

1. **Drifting CSS/canvas font strings.** `ctx.font = "16px Inter"` measured, but CSS says `font-family: Inter, sans-serif; font-size: 16px`. Fine *if* Inter loads. If Inter 404s, CSS falls back to sans-serif and measurements drift by 5-20%. Always `preload` the font or use a web-safe family.

2. **Re-preparing inside the animation loop.** Only `layout*` is cheap. Re-calling `prepare` every frame will tank perf. Keep the prepared handle in module scope.

3. **Forgetting `Intl.Segmenter` for grapheme splits.** Emoji, combining marks, CJK — `"é".split("")` gives you two chars. Use `new Intl.Segmenter(undefined, { granularity: "grapheme" })` when sampling individual visible glyphs.

4. **`break: 'never'` chips without `extraWidth`.** In `rich-inline`, if you use `break: 'never'` for an atomic chip/mention, you must also supply `extraWidth` for the pill padding — otherwise chip chrome overflows the container.

5. **Using `@chenglou/pretext` from `unpkg` with TypeScript-only entry.** Use `esm.sh` — it compiles the TS exports to browser-ready ESM automatically. `unpkg` will 404 or serve raw TS.

6. **Monospace fallbacks silently erasing the whole point.** Users seeing monospace-looking output often have a CSS `font-family` that fell through to `monospace`. Verify the actual rendered font via DevTools.

7. **Skipping rows vs adjusting width** when flowing around a shape. If the corridor on this row is too narrow to fit a line, *skip the row* (`y += lineHeight; continue;`) rather than passing a tiny maxWidth to `layoutNextLineRange` — pretext will return one-grapheme lines that look broken.

8. **Shipping a cold demo.** The default first-paint looks tutorial-grade. Add: vignette, subtle scanline, idle auto-motion, one carefully chosen interactive response (drag, hover, scroll, click). Without these, "cool pretext demo" lands as "intern repro of the README."

## Verification Checklist

- [ ] Demo is a single self-contained `.html` file — opens by double-click or `python3 -m http.server`
- [ ] `@chenglou/pretext` imported via `esm.sh` with pinned version
- [ ] Corpus is real prose, not lorem ipsum, and matches the demo's concept
- [ ] Font string passed to `prepare` matches the CSS font exactly
- [ ] `prepare()` / `prepareWithSegments()` called once, not per frame
- [ ] Dark background + considered palette — not the default white canvas
- [ ] At least one interactive response (drag / hover / scroll / click) or idle auto-motion
- [ ] Tested locally with `python3 -m http.server` and confirmed no console errors
- [ ] 60fps on a mid-tier laptop (or graceful degradation documented)
- [ ] One "extra mile" detail the user didn't ask for

## Reference: Community Demos

Clone these for inspiration / patterns (all MIT-ish, linked from [pretext.cool](https://www.pretext.cool/)):

- **Pretext Breaker** — breakout with word-bricks — `github.com/rinesh/pretext-breaker`
- **Tetris × Pretext** — `github.com/shinichimochizuki/tetris-pretext`
- **Dragon animation** — `github.com/qtakmalay/PreTextExperiments`
- **Somnai editorial engine** — `github.com/somnai-dreams/pretext-demos`
- **Bad Apple!! ASCII** — `github.com/frmlinn/bad-apple-pretext`
- **Drag-sprite reflow** — `github.com/dokobot/pretext-demo`
- **Alarmy editorial clock** — `github.com/SmisLee/alarmy-pretext-demo`

Official playground: [chenglou.me/pretext](https://chenglou.me/pretext/) — accordion, bubbles, dynamic-layout, editorial-engine, justification-comparison, masonry, markdown-chat, rich-note.
