---
title: "Halftone Art — Build halftone dot art for web: CSS, canvas, SVG, React"
sidebar_label: "Halftone Art"
description: "Build halftone dot art for web: CSS, canvas, SVG, React"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Halftone Art

Build halftone dot art for web: CSS, canvas, SVG, React.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/halftone-art` |
| Path | `optional-skills/creative/halftone-art` |
| Version | `1.0.3` |
| Author | Sakib Dalal, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `halftone`, `print`, `risograph`, `css`, `canvas`, `react`, `generative`, `svg` |
| Related skills | [`p5js`](../../bundled/creative/creative-p5js.md), [`mono-color`](../../optional/creative/creative-mono-color.md), [`popular-web-designs`](../../bundled/creative/creative-popular-web-designs.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Halftone Art

Halftone is tonal reproduction by **dot area**: a fixed grid of ink dots whose diameter — never
their colour or opacity — encodes lightness. This skill ships four production techniques (CSS
filter, CSS mask, SVG pattern, canvas engine), React bindings, die-cut cutouts on transparent
stock, three runnable pages, and the print-design rules that decide whether the output reads as
ink on paper or as a Photoshop filter.

Measured: 1440×900 hero renders in **13 ms** (mono) / **36 ms** (CMYK, 4 plates). Zero
dependencies, no build step required.

## When to Use

- Building halftone / dot-screen visuals for web: hero images, poster sections, editorial plates,
  album art, product cards, zine-style landing pages.
- Simulating risograph, screenprint, newsprint, pop-art (Lichtenstein), or process-colour proof.
- Masking display type into a dot screen.
- Replacing a flat photo with a printed-looking treatment that still carries the subject.
- Creating dot textures that feel like paper, not like a UI grid.
- Cutting a halftone subject out onto a transparent background — stickers, patches, merch drops,
  decals, collage elements that must sit on any surface.

**Don't use for** — photo editing where you can send a real file to Photoshop; dithering/
1-bit pixel art (different algorithm: error diffusion, not AM screening); film-grain or noise
overlays (use an `feTurbulence` filter); or a quick background gradient (plain CSS wins).

## The One Idea

```
tone t ∈ [0,1]  →  dot area = t × cell²   →  radius = √(area/π)
```

Fixed grid spacing, varying dot diameter, one ink per plate, plates overprinted with `multiply`.
Everything else — angles, gain, curves — is refinement on top of that line. If a technique cannot
vary the dot's **area** per cell, it is a texture, not a halftone; label it as such and use it only
decoratively.

## Choose the Technique

| need | technique | why |
|---|---|---|
| a photo as halftone, tone-accurate, animatable, exportable | **canvas engine** (`templates/halftone-core.js`) | real separations, per-plate angles, 13 ms |
| photo → dots with no JS at all | **CSS filter** (§B in techniques.md) | works, but needs per-image calibration |
| dot texture / gradient field / tint behind text | **CSS mask** (§C) | sharp at any DPR, 6 lines of CSS |
| logo-scale detail, print, infinite zoom, rolling screen | **SVG pattern** (§D) | vector, crisp, `patternTransform` animates |
| display type cut out of dots | **CSS mask on glyphs** (§E) | 4 lines; see the no-background rule |
| React app | `templates/Halftone.tsx` | typed, ResizeObserver, rAF-coalesced |
| two-plate riso look | canvas `mode: 'duotone'` + `misregister` | real second separation, not a copy |
| die-cut sticker / transparent cut-out | canvas `cutout: true` (+ `cutoutOutline`) | clips to the subject, keeps alpha |
| cut-out with no pre-cut asset | `cutoutMode: 'key'` | floods the backdrop from the corners — draft only |

Full code, maths and verified parameters for every row: `references/techniques.md`.
Deeper design rules: `references/design-language.md`.

## Quick Reference — canvas engine options

```js
renderHalftone(canvas, image, {
  cell: 8,              // CSS px per dot cell — the master control (see dot-scale table)
  angle: 45,            // screen angle; 45 mono, 27/62 duo, 15/75/0/45 CMYK
  shape: 'round',       // round | square | diamond | ellipse | line | cross | ring
  ink: '#171717', paper: '#f2eee2',
  gain: 0.95,           // dot gain: 0.8 light · 1 neutral · 1.15 ink spread
  contrast: 1.15, gamma: 1, brightness: 0, levels: 0,   // tone transfer
  autoLevels: true,     // stretch the source's real tonal range (do this for scans)
  mode: 'mono',         // mono | duotone | cmyk
  inks: ['#0f4bd8', '#ff5f8f'], plateAngles: [27, 62],
  plateCurves: ['full', 'midtone'],   // full | shadows | midtone | highlights | liner
  misregister: 2,       // px offset between plates
  grain: 0.4,           // per-dot coverage jitter — kills the "generated" sheen
  focal: [0.5, 0.2], zoom: 1.3,       // crop control
  width, height, pixelRatio,          // explicit size for off-screen renders

  // cutout / sticker
  cutout: true,                 // transparent background, subject only
  cutoutMode: 'alpha',          // alpha (exact) | key/flood (draft) | auto
  cutoutOutline: 8,             // px of paper bleed = the die-cut rim
  cutoutOutlineColor: '#ffffff',
  cutoutFill: 'paper',          // 'paper' = stock inside the shape, 'none' = clear vinyl
  cutoutTolerance: 0.22,        // key mode: 0..1 colour distance that counts as backdrop
  cutoutClose: 2,               // key mode: morphological closing, seals leaks into fur
});
```

Procedural fields (hero bands, blobs, animated reveals):

```js
renderHalftoneField(canvas, (u, v) => 1 - Math.hypot(u - .5, v - .5) * 1.7, { cell: 8 });
```

## Procedure

0. **Study references before choosing parameters.** Fetch 2–4 real artefacts (a macro of a
   printed screen, a pre-1970 colour litho poster, a comic/pop panel, a riso print) and inspect
   them with `vision_analyze` before touching code — *what* to look at, and the lesson each one
   teaches, is in `references/design-language.md` §0. Guessing angles and dot scale without
   looking is how halftone work ends up looking synthetic. *Done when* you can name the screen
   angle, the plate count and the paper tint you are imitating.
1. **Pick the ink model before the technique.** Choose paper + 1–3 inks from the palette table in
   `references/design-language.md`. *Done when* you can name the paper hex and the plate count.
2. **Choose the cell from the final display width**, using the dot-scale table
   (cell ÷ width ≈ 1/70–1/90 for legible photographic). *Done when* the ratio is in a named band
   and you are not in the 1/100–1/150 dead zone.
3. **Copy the template.** `templates/starter.html` (or `subjects.html` for a preset gallery) plus
   `halftone-core.js` into a folder, and add source photos to `img/` — `portrait.jpg`, `cat-tabby.jpg`,
   `color_poster.jpg` are what the pages look for; missing files render a generated stand-in. Serve over HTTP —
   `python3 -m http.server 8000` — because `getImageData` on `file://` is blocked as cross-origin.
4. **Render, then look at it.** Screenshot the page and inspect at three scales: 100% (dot
   integrity), fit-to-screen (subject legibility), and thumbnail (does it read as a grey field?).
   *Done when* all three pass the self-critique list in `design-language.md` §8.
5. **Fix in this order:** flat/foggy → `autoLevels: true`, then `contrast`; crushed shadows →
   `gain` down; moiré in flat areas → change `angle` by 15–30°, then raise `cell`; sparkles in
   shadows → confirm the flood threshold (~0.95) is active; dead highlights → `gain` down to
   0.8–0.9. *Done when* re-shot renders show no pinholes, no cross-hatch, and paper in the
   lightest area.
6. **Ship it properly.** Canvas sized by `devicePixelRatio` (cap at 2), re-render on
   `ResizeObserver` (not `window.resize`), `role="img"` + `aria-label` on meaningful canvases,
   `aria-hidden` on decorative fields, and no motion under `prefers-reduced-motion`.
   *Done when* the page survives a resize, a 2× display, a 390 px viewport and a screen reader
   pass. At 390 px a three-item flex masthead and a two-column caption/filter strip both collapse
   into narrow stacked columns — drop the middle masthead item and stack the strip instead of
   shrinking the type.

## Design Rules (the short version)

- **Never `#000` on `#fff`.** Warm paper (`#f2eee2`) + near-black ink (`#171717`) is already most
  of the print illusion.
- **One dotted element per viewport.** Halftone everything and there is no figure/ground.
- **Keep the dot grid spacing constant.** Only diameter varies. That is the signature.
- **Two plates = 30–45° apart.** Equal angles produce clumping, not overprint.
- **Misregistration 1–3 px.** Zero is a tell; 8 px is a broken 3D effect.
- **Shadows go solid, highlights go to paper.** If neither end is clean, the screen is wrong.
- **Photographic panels inside flat fields**, mono plate captions, hairline rules — the vintage
  poster structure still beats a full-page dot treatment.
- **Type:** heavy display (Archivo Black / Anton / Druk) at tight tracking, mono captions at
  9.5–11 px with 0.14–0.2em tracking. Never body copy in dots; never halftone type under 40 px.

## Pitfalls

1. **Detached canvas renders zoomed.** `getBoundingClientRect()` returns 0×0 off-screen, so the
   engine falls back to a default size and you get the top-left crop at 2×. Pass explicit
   `width`/`height` for exports and contact sheets.
2. **Nearest-neighbour tone sampling creates moiré.** Half-resolution tone maps read with
   nearest taps beat against the dot grid. Bilinear + taps rotated with the screen angle. This one
   change removed the cross-hatch in every test render.
3. **Skipping `autoLevels` on scans.** A photo using luma 0.25–0.8 collapses to two dot sizes and
   reads as fog. Stretch the histogram first, then adjust contrast.
4. **Halftone type with a background = solid bar.** The mask eats the background box as well as
   the glyphs. Colour the text, mask the element, leave `background` unset.
5. **CSS `contrast()` threshold blows out highlights.** Below ~5 nothing snaps, above ~12 the
   image collapses to paper. Work at 7–9 and compensate on the `<img>` itself
   (`grayscale(1) contrast(1.35)`).
6. **CMYK from a greyscale source is mud.** Naive separation sends a B/W photo almost entirely to
   the K plate. Process colour needs a colour source, or accept a K-only proof deliberately.
7. **Pinholes in shadows.** Circles that overlap still leave white corners. Flood the cell above
   ~95% coverage (the engine does this; hand-rolled loops usually don't).
8. **`file://` taints the canvas.** Reading pixels from a local image throws a security error.
   Serve over HTTP or bundle the image as a data URL.
9. **Canvas rendered at CSS size.** On a 2× display the dots blur into grey. Always
   `canvas.width = cssWidth * min(dpr, 2)`.
10. **Re-rendering every animation frame on a large canvas.** 13 ms per frame is fine for a
    300 px field and not fine for a full-bleed CMYK hero. Animate the tone function on a small
    canvas, or pre-render frames.
11. **Rotating a CSS dot overlay without over-sizing it.** A `::after` rotated 45° leaves empty
    triangles; use `inset: -32%` (or `-50%` for safety) to cover.
12. **Text over the dot field.** Dots at high contrast vibrate behind type. Drop the field's
    contrast below ~40% under text, or set the type on a paper-coloured panel.
13. **Painting paper first kills the cutout.** A transparent result needs the halftone rendered on
    its own layer and then clipped to the silhouette — fill the background and you get a sticker
    with a white box around it. The engine's `cutout` path does the layer dance; hand-rolled code
    usually does not.
14. **Automatic cutouts fail on fur, hair and busy backdrops.** `cutoutMode: 'key'` floods inward
    from the frame corners, so a light subject against a light backdrop leaks. It is a draft tool:
    for production, cut the subject to a transparent PNG and use `cutoutMode: 'alpha'`. When keying,
    tune `cutoutTolerance` 0.1–0.3 and `cutoutClose` 2–6 — the two interact.
15. **Clear vinyl disappears on dark surfaces.** With `cutoutFill: 'none'` only ink remains, so
    dark ink on a navy page is invisible. Choose a stock (`cutoutFill: 'paper'`) or an ink that
    reads on both light and dark surfaces.
16. **Rim fills enclosed cut-outs.** A naive dilated rim seals eye holes and handle gaps. Grow the
    *hole-filled* silhouette, then punch the holes back out (the engine does this) — otherwise a
    die-cut cat stares at you with solid white eyes.

## Files

```
SKILL.md
references/techniques.md        full code + maths + verified parameters per technique
references/design-language.md   dot-scale table, palettes, angles, composition, anti-patterns
templates/halftone-core.js      zero-dependency canvas engine (renderHalftone, renderHalftoneField)
templates/halftone-core.d.ts    TypeScript declarations for the engine
templates/Halftone.tsx          React: <HalftoneImage>, <HalftoneField>, <HalftoneType>, useExportHalftone
templates/starter.html          complete editorial page (hero plate, band, cards, specimen grid)
templates/subjects.html         six-subject preset gallery (cats, dog, portraits, CMYK)
templates/stickers.html         die-cut cutouts: transparent stock, paper rim, PNG download
templates/css-techniques.html   six verified pure-CSS techniques side by side
```

## Verification

- Serve `templates/starter.html` and screenshot it; the page must render with **and without**
  `img/` present (generated fallback).
- Check the render at 100%: dots crisp, shadows solid, no pinholes, no cross-hatch in flat areas.
- Check at thumbnail: the subject still reads as a silhouette.
- Check at 1.5× and 2× DPR: no shimmer, no half-dot seams at the right/bottom edge.
- Resize the window: the canvas re-renders at the new size, still sharp.
- For cutouts: save the PNG, open it on a dark background, and confirm the area outside the
  silhouette is genuinely transparent and the rim hugs the shape.
- Tab through with a screen reader: meaningful plates announce once; decorative fields are hidden.
- `node ./node_modules/typescript/bin/tsc -p tsconfig.json` on `Halftone.tsx` against the shipped
  `.d.ts` — must be clean under `strict`. Two setup notes: invoke the compiler through `node` on
  the installed bin (the `typescript` package ships both `tsc` and `tsserver`, so `npx typescript
  tsc` fails with "could not determine executable to run"), and add a local `react-stub.d.ts`
  declaring the hooks/props used plus `react/jsx-runtime` — no React install needed to typecheck.
