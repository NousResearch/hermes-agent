---
title: "Visual Artifact Delivery — Render authored HTML or SVG to a verified image file"
sidebar_label: "Visual Artifact Delivery"
description: "Render authored HTML or SVG to a verified image file"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Visual Artifact Delivery

Render authored HTML or SVG to a verified image file.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/visual-artifact-delivery` |
| Path | `optional-skills/creative/visual-artifact-delivery` |
| Version | `1.0.0` |
| Author | Pedro Relvas (@PRELVAS94), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `diagram`, `rendering`, `screenshot`, `svg`, `verification`, `delivery` |
| Related skills | [`architecture-diagram`](../../bundled/creative/creative-architecture-diagram.md), [`baoyu-infographic`](../../bundled/creative/creative-baoyu-infographic.md), [`claude-design`](../../bundled/creative/creative-claude-design.md), [`p5js`](../../bundled/creative/creative-p5js.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Visual Artifact Delivery Skill

An authored diagram, chart, infographic or one-page mock is not delivered until it is an
image the user can see. This skill covers the last mile: turn the HTML/SVG the authoring
skills produce into a rendered picture, verify both the source geometry and the pixels,
and hand over a file path that lands in the user's chat. It complements the authoring
skills (`architecture-diagram`, `baoyu-infographic`, `claude-design`, `p5js`) rather than
competing with them.

## When to Use

Use when the deliverable is a *picture*:

- A diagram, infographic, floor plan, flowchart or one-page HTML mock the user asked for.
- The user asks to "see it", "show me", "send me the image", or is on a messaging platform
  where an HTML file is useless to them.
- Labels, legends, toolbars or callouts have collided in an earlier revision — the source
  geometry is worth checking before rendering again.

Don't use for: video output (`manim-video`), decks (`powerpoint`), PDFs (`pdf`), or a live
page the user opens themselves (plain browser tooling is enough there).

## Prerequisites

- **A browser tool** — one of `browser_exec` (browser-use backend), `browser_navigate` +
  `browser_vision`, or `browser_cdp` on a CDP-connected session. Required for the render
  step only.
- **`vision_analyze`** (and `browser_vision`) for the pixel check. Both need vision
  capability: the auxiliary `vision` task defaults to your main chat model, so on a
  text-only model lane they fail with a "no endpoints found that support image input" style
  error unless `auxiliary.vision.provider`/`model` points at a vision-capable model. When
  that happens, the pixel step is simply unavailable — do the geometric checks, state that
  the image was rendered and checked geometrically, and never imply a visual review you
  could not perform.
- **`write_file`** — the artifact must exist on disk before it is rendered.
- Python 3.8+ for the geometry checker (stdlib only, no network, no packages).

## How to Run

The checker is a real script in this skill — do not re-derive the arithmetic inline:

```bash
SKILL_DIR="${HERMES_SKILL_DIR}"
python3 "$SKILL_DIR/scripts/svg_geometry_check.py" diagram.svg          # human report
python3 "$SKILL_DIR/scripts/svg_geometry_check.py" page.html --json     # machine report
python3 "$SKILL_DIR/scripts/svg_geometry_check.py" - < diagram.svg      # stdin
```

It exits `0` clean, `1` for findings that need a fix (exit `2` for unusable input), so it
can gate the render step. `references/rendering-and-verification.md` has the render routes,
the flags that matter, and how to read a finding.

## Quick Reference

| Goal | Do |
|---|---|
| Check the source geometry | `python3 "$SKILL_DIR/scripts/svg_geometry_check.py" artifact.svg` |
| Open the artifact in the browser | `browser_navigate` to the `file://` URL, or a `data:text/html,…` URL when the browser cannot read the local filesystem — then confirm the artifact is in the DOM |
| Render a screenshot | `browser_vision` (returns a saved `screenshot_path`) |
| Full-page / explicit-size capture | `browser_cdp` → `Page.captureScreenshot` (`captureBeyondViewport`, `clip` with `scale`) |
| Read small text in the render | `vision_analyze` with a `region` crop |
| Deliver the image | `MEDIA:/absolute/path.png` on attachment platforms, plus the bare path |

## Procedure

1. **Author the artifact with a declared, fixed canvas.** Give the `<svg>` a `viewBox`, or
   the page an explicit pixel `width`/`height`. Viewport-relative units (`vw`, `vh`, `%` on
   the root) mean the render changes shape with whatever window it lands in.

2. **Check the source geometry before rendering.** Run the checker from *How to Run*. Fix
   every `collision`, `duplicate`, `outside-canvas` and `stacked` finding in the source —
   it is a text edit, and re-rendering to discover the same overlap costs a full render
   cycle. `contained` is normal nesting; `boundary` is a dashed rule overlapping a filled
   shape; both are reported, neither is a failure. Read the `note` lines: they name the
   shapes the checker could *not* measure (rotated groups, `path`/`text` elements) so you
   never mistake "not measured" for "verified clean".

3. **Render it.** Write the artifact to a path first, then open it and confirm it really
   loaded — a `file://` URL on a sandboxed or cloud browser lands on an error page
   (`chrome-error://chromewebdata/`) and every capture after that is a picture of nothing.
   Check `document.querySelector('svg')` (via `js` on `browser_exec` or
   `Runtime.evaluate` on `browser_cdp`) before capturing; when the browser cannot read the
   local filesystem, serve the page or reopen it as a `data:text/html,…` URL, which
   self-contained artifacts handle well.

   - `browser_navigate` to the opened artifact, then `browser_vision` with a question about
     what should be visible. Its result carries a `screenshot_path` that is already saved to
     disk and is exactly what `MEDIA:` wants. This is the one-browser-tool route and it works
     on every backend.
   - For a full-page or larger capture, use `browser_cdp` with `Page.captureScreenshot` and
     set `captureBeyondViewport` / a `clip` that carries `scale`. The result is base64
     `data`, so decode it to a file with `terminal` before delivering it. `clip.scale`
     multiplies the display's device pixel ratio, so on a 2× display `scale: 1` already
     produces 2× pixels — ask for the size you need and check the PNG's own dimensions
     rather than assuming, and keep the capture near 2× because a very large one can exceed
     the browser tool's response timeout.
   - On the browser-use backend, `browser_exec` runs Python with `new_tab`, `goto_url`,
     `js`, `cdp`, `page_info` and `capture_screenshot` pre-imported, so the open, check,
     wait and capture steps can be one call. `capture_screenshot()` takes the viewport and
     `capture_screenshot(full=True)` the whole page, neither with clip/scale control.

   Wait for the page to be ready before capturing: `wait_for_load()`, then
   `document.fonts.ready` — a capture that beats font loading ships fallback-font metrics
   and shifted labels.

4. **Verify the pixels.** Run `vision_analyze` on the rendered file and ask what the labels
   *say* and whether anything overlaps or is clipped. `vision_analyze` downscales large
   images, so if you cannot read the small text, call it again with a `region` crop — the
   crop keeps full resolution, so zoom in rather than re-rendering. If vision is unavailable
   on this install, stop at the geometric evidence and say which checks you actually ran.

5. **Deliver.** Put the image path in your reply as `MEDIA:/absolute/path.png` on platforms
   with an attachment channel (Telegram, desktop, email) and state the plain path as well.
   Include the source file path too when the user may want to edit it.

6. **Iterate on the source, not on the picture.** Every fix belongs in the HTML/SVG: change
   the box, reduce the font size, move the legend, re-run the checker, re-render.

## Pitfalls

- **Handing over the HTML and telling the user to open it.** That is not delivery; a user on
  a phone cannot open an HTML fragment. Render first, and never suggest the user screenshot
  it themselves.
- **Rendering a path you have not written yet.** A `file://` URL that does not match the
  file on disk renders the previous revision, or nothing, and the failure looks like a CSS
  bug rather than a typo in the path.
- **Capturing a page that never loaded.** A `file://` URL on a sandboxed or cloud browser
  lands on an error page, and the capture is a picture of the error. Confirm the artifact is
  in the DOM before capturing, and reopen it as a `data:` URL or over a served address when
  the browser cannot read the filesystem.
- **A capture that is cut to the window.** With no explicit canvas or an unset viewport, a
  screenshot silently takes only the visible area. Set the window/canvas to the artifact
  size, or capture beyond the viewport.
- **Assuming the capture scale is the pixel ratio you asked for.** `clip.scale` compounds
  with the display's device pixel ratio, so a 2× display turns `scale: 1` into 2× pixels and
  `scale: 2` into 4×. Check the file's own dimensions, and stay near 2× — an oversized
  capture can time out.
- **1× captures of dense diagrams.** Small text is unreadable after the platform downsizes
  the image. Capture at 2× and keep the original file.
- **Reading a downscaled image for fine detail.** If the labels are illegible to
  `vision_analyze`, crop with `region` instead of concluding the render is broken.
- **Geometry clean, picture broken.** The checker sees boxes, not rendered text: a line
  whose glyphs are wider than its box, three `<tspan>` lines in a two-line box, or a chart
  drawn entirely with `path` will pass the arithmetic and fail in the image. The `note`
  lines tell you which parts were not measured — look at those in the render.
- **Fonts and emoji that differ on the rendering host.** A glyph that rendered in the
  authoring environment can fall back to a missing-glyph box. Prefer widely available
  families, or embed the font, and confirm legibility with `vision_analyze`.
- **Claiming a visual check you could not run.** With a text-only model lane,
  `vision_analyze` fails instead of describing the image. Say the render exists and passed
  the geometric checks; do not report it as "looks correct" when nothing looked at it.
- **Delivering from a path the gateway refuses.** `MEDIA:` delivery is denied for system and
  credential directories (`/etc`, `/root`, `/var/log`, `~/.ssh`, `~/.config`, …). Render into
  a normal output directory, and deliver an absolute path — a relative path is not resolved.

## Verification

- [ ] The browser confirmed the artifact was in the DOM before the capture (not an error tab).
- [ ] `svg_geometry_check.py` exits `0`, or every remaining finding is one you can name as
      deliberate (`contained`, an intentional `boundary`).
- [ ] The `note` lines that name unmeasured shapes are ones you checked by eye in the render.
- [ ] The rendered file exists, is non-trivial in size, and is a `.png`/`.jpg`/`.webp`.
- [ ] `vision_analyze` read the labels back correctly (crop with `region` for the small ones).
- [ ] The delivered path is absolute, outside a denied system/credential directory, and the
      reply states the path in text as well as `MEDIA:`.
- [ ] The user receives the image in the same turn as the answer — not a promise to send it.
