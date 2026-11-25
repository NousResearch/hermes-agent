---
title: "Anatomy — Explain an idea as an interactive isometric machine figure"
sidebar_label: "Anatomy"
description: "Explain an idea as an interactive isometric machine figure"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Anatomy

Explain an idea as an interactive isometric machine figure.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/anatomy` |
| Path | `optional-skills/creative/anatomy` |
| Version | `1.0.0` |
| Author | Ryan (wheresryan22), ported to Hermes by Nous Research |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `isometric`, `svg`, `webgl`, `3d`, `diagram`, `explainer`, `illustration`, `docs`, `html`, `react` |
| Related skills | [`concept-diagrams`](../../optional/creative/creative-concept-diagrams.md), [`system-atlas`](../../optional/creative/creative-system-atlas.md), [`architecture-diagram`](../../bundled/creative/creative-architecture-diagram.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Anatomy

This skill builds figures that work like a small instrument on a bench: a heavy base on feet, rails with tick rulers, a carriage on bushings, knurled knobs, coil springs, a dial gauge whose needle moves, and screws drawn as tiny dots. The reader understands the idea because they recognise the machine. Every number the figure shows comes from the same model that moves its parts.

Output is a standalone HTML page (SVG plus optional WebGL layers), a lone `.svg`, or React/Next.js components. The kit uses only Node built-ins, so there is nothing to `npm install`.

## Running it in Hermes

**Skill directory.** Hermes expands `${HERMES_SKILL_DIR}` to this skill's installed directory. Shell variables don't carry over between terminal calls, so repeat the assignment in every call that uses it:

```bash
SKILL_DIR="${HERMES_SKILL_DIR}"
```

**Never build inside the skill directory.** Every `build.mjs` writes its HTML next to itself, and the examples import `../../kit/` and `../../scripts/`. Lay out a workspace with the same shape: the user's project if they have one, otherwise a folder they name.

```bash
SKILL_DIR="${HERMES_SKILL_DIR}"
WORK="$HOME/figures/pin-lock"
mkdir -p "$WORK/figures"
cp -r "$SKILL_DIR/kit" "$SKILL_DIR/scripts" "$WORK/"
cp -r "$SKILL_DIR/examples/dial-indicator" "$WORK/figures/"
cd "$WORK/figures/dial-indicator" && node build.mjs && node build.mjs --light
```

Start a new figure by copying the closest example into `$WORK/figures/<name>/` and renaming it. Leaving the installed skill untouched keeps `hermes skills update` from flagging it as locally modified.

**The examples ship as source only.** Upstream's built pages are 9 MB of generated HTML. Build an example (as above) to open or capture it, since it takes a few seconds.

**Looking at your work.** The upstream workflow says "read every PNG". In Hermes that means `vision_analyze` on each capture, and the 4× close-ups use its `region` crop on a high-DPR capture or a `--zoom` capture. Ask specific questions ("does the pipe at the flange end on the flange face?"), not "does this look good".

**Browser.** The capture, drive, line-end and 3D checks start a headless Chrome over a loopback DevTools port. They use `CHROME_PATH` if set, then a Playwright headless shell if one is installed (`chromium_headless_shell-*` in the Playwright cache, e.g. from `npx playwright install chromium`), then a system Chrome or Chromium. Prefer the headless shell: it renders the WebGL layers through SwiftShader and makes no outbound connections. A full desktop Chrome opens background connections to Google services even with a throwaway profile, and it fails to start ("Socket path too long", then "Chrome did not start") when `TMPDIR` is a long path, as Hermes' per-profile scratch dir can be. If you must use one, set `TMPDIR` to a short directory for that command. Building a figure needs no browser at all.

**Long runs.** `node build.mjs --audit` takes about two minutes on a dense Hero, and a full 3D orbit audit takes 15 to 20 minutes for 30 parts. Run those with `terminal(background=true, notify=true)` and keep working on something else.

**Asking about scope.** The workflow's sizing step decides whether to build or ask. For an Epic, put the estimate and the Hero alternative to the user with `clarify`, and wait for their choice.

**Delivering.** Hand over the absolute path of the built HTML (both themes) plus the PNG captures you verified. On a messaging platform, attach the desktop and phone PNGs, since a chat can't run the page.

## The examples

Open the closest one before you start. Each framework-free example builds with `node build.mjs` in its directory; add `--light` for the light theme.

- `examples/dial-indicator/` is the smallest: a dial gauge on a stand, about 300 lines in all.
- `examples/desk-computer/` is framework-free SVG: a desk computer with a keyboard you can type on. The typed line appears on a CRT screen drawn in the plane of the case front, and the picture collapses to a line and a dot when you switch it off. It has about 85 solids and 850 paths.
- `examples/arcade-cabinet/` is an upright arcade cabinet with its side panel off, a Hero of about 240 solids and 15 cables. A quarter flies into the slot, rolls down the coin chute at the speed a rolling disc reaches on a 30° slope, trips the coin-switch wire and drops into the cash box. The switch's pulse runs chunk by chunk along its cable to the board, and the CRT warms up, self-tests and runs an attract mode drawn in the plane of the tilted monitor. The stick tilts and closes a microswitch under the panel. It shows how to place a camera so a cutaway's mechanism stays in view, a concave board drawn as a prism along any axis, moving parts rebuilt in the page from the same module the build uses, and live coins and a wire drawn in reserved slots of the settled order. `node build.mjs --audit` proves the geometry.
- `examples/raptor-engine/` is a rocket engine hanging in a test stand. Its round parts come from `kit/lathe.mjs` and its pipes and harness from `kit/tube.mjs`. It comes apart into 11 assemblies with staggered springs and per-frame depth re-sorting. Every part records its true shape, and `node build.mjs --audit` (`kit/audit.mjs`) proves several things at rest, apart and in every replayed frame of the explode: nothing passes through anything, every end sits on a real mount, no bend pinches, folds or lands on another line, no curved profile's slope wobbles, and every overlap is drawn in depth order. A WebGL plume, shock, splash and steam render between the back and front SVG. It is the reference for dense pipework, smooth shading and fades.
- `examples/ripple-tank/` is a school ripple tank on tall legs over a paper screen, seen from 44° up. Two WebGL layers (lamp light, shadows and caustics on the paper, and the water's reflections and glass) are registered to the isometric planes and driven by a pure dispersion and optics model (`model.mjs`). A frequency change travels out from the dippers through a history texture.
- `examples/test-rig/` is a production figure from a real site (React/Next): a glass button on a spring-mounted carriage, a finger probe on a rail that presses and pulls, and a dial gauge reading the lean. It has 53 solids and about 470 paths.
- `examples/spot-plate/` is a second production figure from the same site: six pairs of glass drops on a spot plate, each pair drawn from one distance field, so a pair closer than `merge` runs together into one piece of glass. A fan of feeler blades sets `merge`, one blade per gap. It is the reference for shapes that morph into each other and for one control driving many parts.
- `examples/turning-dial/` is the optional 3D mode: a combination dial that turns in true 3D with a small robot hand on it. It has about 90 live parts in metal materials, a second turn group, a hinge, a three-hinge finger that lifts and a pod that breaks off as a free body. A WebGL LED, its light pool and sparks follow the moving parts. `stress.mjs` beside it is a dense hand of about 380 parts in a cluttered workshop, used to measure performance.

The test rig sets the quality bar for the drawing itself. The Raptor sets the bar for shader work: a WebGL effect has to match its world-space, no-pop, seamless rendering.

## The one idea that matters

**The object is the explanation.** Don't draw boxes, arrows and labels. Invent a real, well-made physical thing whose working *is* the concept. Here is how earlier figures mapped:

| Concept | Object |
|---|---|
| One shared GPU context serving many canvases | A gantry plotter: one print head visiting 16 wells |
| A cache key that skips work when nothing changed | A pin-tumbler lock: when every pin matches, the plug turns and nothing is redrawn |
| Seven render targets with different bytes per texel | A chest of seven drawers, each as deep as its bytes |
| A separable blur (49 × 49 taps vs 49 + 49) | A bed-of-nails stamp beside two combs on rails |
| A spring that presses and leans | A test rig with springs, a probe and a dial gauge |
| A height profile | A profile (contour) gauge pressed against the edge |
| Light that depends on facing | A small photo studio: key and fill lamps around a turntable |

Two ways this fails, both seen in practice:

- **Too abstract.** Stacked plates, floating slabs or labelled boxes don't read as anything.
- **Too flat and bland.** One big slab with a few lines on it has no volume, no parts and nothing to discover.

The fix for both is the same: a recognisable machine, built from many small, precise, purposeful parts.

**Every creation is highly detailed.** This holds for every figure, however small the idea or the request: a quick explainer is still a dense, real machine, never a few slabs. What scales with the size (workflow step 1) is how far you push it: a Figure gets the 4× sweep over its subject and mechanism, a Hero or an Epic over the whole drawing.
- **Read the real thing.** Build it the way the real object is built: every screw, seam, bevel, bushing, label plate, cable with its connector, scale with its ticks, and the parts that hold the parts.
- **Two looks.** At 1× it reads as one clear machine. At 4× every region still has something true to find; a bare face or an empty corner is unfinished work, not calm.
- **Purposeful, never noise.** Detail comes from parts that belong to the mechanism. Each one sits on another part, ends on an edge, and fades rather than stops.
- **Check it last.** Before you finish, zoom to 4× (over the subject and mechanism for a Figure, the whole drawing for a Hero or an Epic) and add the real parts that are missing wherever a region is plain.

## Workflow

Work in this order. Write the concept and the parts list before any geometry. Most of the quality is decided there.

1. **Size it before you start.** Tell the user which size the request is and roughly how long it will take, before any other work.
   - **Figure**: one mechanism, 40–80 parts, SVG with at most one WebGL layer. About 30–90 minutes. Build it without asking.
   - **Hero**: a dense object, several WebGL layers or one 3D subject, 80–150 parts. About 2–4 hours. Say so, then build it unless the user asked for something quick.
   - **Epic**: a whole vehicle, building or plant with many moving systems, or the 3D mode on more than one subject. 6 hours or more. Do not start it. Give the estimate, propose a Hero version that keeps the core of the idea, and list what the full version would add (ask with `clarify`). Build the Epic only when the user chooses it.
   - Whatever the size, have a working page that opens in the browser within the first hour, and deepen it from there. Run `date` at the start and at each step, and when a step runs past twice its share of the estimate, tell the user where it stands and what is left.
2. **Truth first.** Collect the real numbers and behaviour of the thing being explained: sizes, counts, thresholds, curves. The readout, motion and caption must all be true. If you can, write the model as a small pure module, like `rig.ts` in the test-rig example, and drive everything from it.
3. **Invent the object.** Pick a machine, tool, instrument or piece of furniture whose mechanism maps one-to-one onto the concept. Say in one sentence what moves and what that motion means. Never copy a reference image: take its finish, not its object.
4. **Parts list.** Write at least 40 parts (40–80 for a Figure, 80–150 for a Hero, more for an Epic), grouped: base (plate, feet, groove, corner screws, label plate, ruler strip), structure (posts, rails, frame bars, end stops), mechanism (carriage, bushings, springs, plunger, needle), controls (knurled knob, thumb screw, crown) and the subject itself, which is the one lit part. Each part sits on another part. `references/craft.md` has a vocabulary of parts with recipes.
5. **Plan the world.** Use world units of about 1 px of the real thing: x and y on the floor, z up. Lay out plans as rounded rectangles `{x, y, w, d, r}` and circles. Choose the camera: azimuth 45° by default, 35–60° to show the long side, elevation 30° (higher, like the ripple tank's 44°, to look into something open). Then fit with `fitProjection(points, width, height, { pad, azimuth })`.
6. **Build geometry once.** Precompute every static path in a geometry module with the kit: `slabOf`, `cylinder`, `extrude`, `planOutline`, `topTicks`, `knurl`, `coil`, `radialTicks` and so on. Round parts on any axis (bells, flanges, valves, actuators) come from `kit/lathe.mjs`, and pipes and cables from `kit/tube.mjs`. Moving parts are built once at a rest position and moved with `translateAlong`. Only parts whose shape changes are rebuilt per frame.
7. **Paint back to front, bottom to top.** Sort repeated parts with `depthOf`. Split anything that wraps another part into a back half and a front half: the springs around a rod, the bars of a frame around a carriage. A moving part is drawn in the slot where it physically sits. See "Painter's order" in `references/craft.md`. Once a figure has pipes, cables and round parts, stop tuning keys by eye: record each item's shape as you put it into `A.recorder(P, { order, spring })`, let `A.settle(R, P)` measure true depth wherever two items overlap and set the keys, and end the build with `A.auditOrExit(R, P)`. Route pipes before you draw them: real ports at both ends, legs along the structure, crossings on straights, collars gapping the pipe (`references/craft.md`, section 8). Record the true shape of what you draw: a stand-in shape that silences the audit is a bug, and the audit fails one. A rod that slides out of a gland is one rod from `G.rodOut`, recorded with `A.slide`.
8. **Frame it.** Put the figure in a card: `Fig n` at top left, a short title at top right, the interaction hint at bottom left and the live readout at bottom right. Legend keys and a true caption go under the card, and words stay out of the drawing apart from at most one tiny engraved label.
9. **Make it live.** Use calm time-based motion: exponential follow or real springs. An idle tour loops when nobody touches it. The figure responds to pointer and keyboard, stops when offscreen, and goes still under reduced motion. Write attributes on refs and don't re-render per frame. See `references/motion.md`.
10. **Verify like a critic.** The owner finds faults by zooming in, so you zoom in first. Read every PNG you make with `vision_analyze`.
   - Capture desktop, phone (390 wide), hover and several moments of motion with `scripts/capture.mjs`; use `scripts/drive.mjs` for sequences, the apart pose and the `--no-webgl` fallback.
   - **Run the audit to zero.** With pipes, cables or round parts, `node build.mjs --audit` must end `audit passed`, every failing line at 0: clearance at rest, apart and in mid-flight, terminals, loose ends, folds, self-overlap, crowding (against slender solids too), stand-in shapes, support, wobble, saddles and depth order. Fix an interlock in the geometry, never with a key bias.
   - **Run the line check to zero.** The audit never sees line markup. `node scripts/lines.mjs page.html --out shots/lines.png` (and on the `-light` page, and in every resting pose with `drive.mjs`'s `["lines"]` action) must print `line ends: 0`: every seam, rule, rib, tick and pointer ends on a stroke, a dot or an outline. Read the ringed PNG.
   - **4× close-ups of every junction in both themes** (build `--light` too), assembled and apart: every flange, tee, elbow, clamp, valve, bellows, band, saddle, ball joint and crossing, every curved body (no tone staircase), every place a texture ends (on an edge or fading), flat box faces with their seams, every textured hose at a chunk joint, every sliding rod at rest and at full stroke, and every window with text or shader content in it.
   - **Shaders:** a contact sheet slowed with `["slow", 0.05]` across every transition; nothing pops, no colour switch, edges read 0. In light, also log ink per frame through each transition (it must follow the light monotonically) and crop where the ink meets strokes at 8–16× (`references/verify.md`, "Light-theme shaders").
   - Put your PNG next to the examples and ask: is it instantly a real object, rich in small parts with volume, with every line ending on something? Is it highly detailed at 1× and still rewarding at 4×, with no bare face or empty corner? Iterate until yes. Checklist: `references/verify.md`.

## Setup

The kit is in `kit/`:

| File | Use it for |
|---|---|
| `kit/iso-kit.ts` | The whole core in one file: camera, solids, a parts library, an SVG string renderer, card and page builders, and the CSS (`ISO_CSS`). Copy it into a TypeScript project. |
| `kit/iso-kit.mjs` | The same core compiled to plain ESM, for Node scripts. Import it directly. |
| `kit/iso.css` | `ISO_CSS` written out. Import it once and put `class="iso"` (plus `data-theme="light"` for light) on a wrapper or `<body>`. |
| `kit/react/draw.tsx` | Server-safe React components: `Solid`, `Line`, `Wire`, `Dots`, `FaceText`, `IsoFigure`, `Plate`. |
| `kit/react/live.tsx` | `"use client"` hooks: `useReadout`, `useLoop`, `useEased`, `useInView`, `useStillness`, plus `spring` and `follow`. |
| `kit/lathe.mjs` | Round parts along any axis: `frameAlong`, `lathe` (smooth shading switches on by itself for any profile with a smooth knot), `solidOf`, `disc`, `ringBand`, `sphereOf`, `saddleOf`, `bandsOf` (one curved body cut into depth-sortable bands that share one surface, with one outline from `sidesOf`), visible arcs, ribs and bolt circles on round surfaces (`ribsOf` and `dotsOf` take `fade`, `ribsOf` takes `seams`), `rodOut` for a rod that slides out of a gland, and `fillet` for pipe routes. |
| `kit/tube.mjs` | `tubePieces` splits a 3D pipe or cable into depth-sortable chunks (with `breaks`, `gaps` under collars and clamps, elliptical end caps, and `rings` for a braid, corrugation or wrap that follows the cross-section through every joint), and `tubeSvg` draws one; `crossingsOf` finds where a route passes through a clamp, `runsOf` the drawn runs. Its `.tb-*` fills are in `ISO_CSS`. |
| `kit/audit.mjs` | Proves the geometry. `recorder(P, { order, spring })` collects items, shapes (turned bodies, slabs, prisms, boxes, balls, tubes, pads cut by their host) and routes as the helpers draw them; `settle(R, P)` ray-casts every overlapping pair for true depth, at rest, apart and in the replayed explode, and sets the painter's keys (`overlapsOf`, `orderKeys`); `auditOrExit(R, P)` is the `--audit` hook that runs every check (`clearances`, `solidClearances`, `terminals`, `looseEnds`, `supports`, `folds`, `selfOverlaps`, `crowding` with slender solids as lines, `coverage` for stand-in shapes, `wobble`, `seats`, `depthOrder`) and exits non-zero on any problem. `A.slide` records a rod that slides out of a gland. `references/kit.md`, section 12. |
| `kit/gl.mjs` | The WebGL layer for shaders inside the drawing: `GLSL_ISO` (the camera in GLSL, including `rayOf` and `pixelOf`), `glCamera(P)`, `glLayer(...)`, and the JS envelopes `smooth`, `burst` and `settle`. For 3D figures, `GLSL_TURN` and `glslPose(name)` bring world points into a moving group and mask a front canvas with `coverOf`. |
| `kit/turn.mjs`, `kit/turn-build.mjs`, `kit/turn-audit.mjs` | The optional 3D mode. `turn-build.mjs` has `turning(P, { recorder })`: groups (turn, slide, hinge, free), live layers, convex parts with materials, and `T.build`. `turn.mjs` is the runtime: `TURN.mount`, the controller, `TURN.cover`, `TURN.poseUniforms` and `TURN_CSS`. `turn-audit.mjs` has `orbitOrExit`, the orbit audit over every turn angle and motion clip. `references/3d.md`. |
| `scripts/capture.mjs` | Headless Chrome screenshots, close-ups and console checks, with WebGL on. Run with `node`. |
| `scripts/drive.mjs` | Scripted headless Chrome with WebGL: move, click, key, slow motion, wait-until, eval, shots and contact sheets that catch pops; `--no-webgl` and `--preload`. |
| `scripts/lines.mjs` | The line-end check on a built page, in headless Chrome: every texture line must end on a stroke, a dot or an outline. Rings the failures with `--out`; exits 1 on any. |
| `scripts/inline-kit.mjs` | `kitScript()`, `glScript()` and `turnScript()` return the kit, the WebGL layer and the 3D runtime as source to inline into a standalone page. |
| `scripts/turn-check.mjs`, `scripts/turn-bench.mjs` | 3D figures: fidelity against the kit builders, a z-buffer order check, pixel pops, line ends at 24 angles and frame timing in the browser; Node CPU time per part kind. |

Pick the path that fits the project:

- **Standalone HTML, a static SVG, a blog post or any non-React site.** Write a Node build script like `examples/dial-indicator/build.mjs` in the workspace laid out under "Running it in Hermes". It imports `kit/iso-kit.mjs`, assembles strings with `solidSvg`, `lineSvg`, `dotsSvg`, `wireSvg` and `faceTextSvg`, wraps them with `figureSvg` and `plateHtml`, and writes the page with `pageHtml({ script: kitScript() + live })`, or `kitScript() + glScript() + live` when it has a shader. For a lone `.svg` file use `figureSvg({ standalone: "dark" | "light" })`, which embeds the styles.
- **React or Next.js.** Copy `kit/` into the project (for example `components/iso/`). Keep geometry in plain `.ts` modules, build the static art in a server component with the `draw.tsx` components, and pass it as element props to one `"use client"` live component that moves things. `references/react.md` explains the split and a trap with large pages.

## Craft rules (short form)

These are the rules that separate a crafted figure from a diagram. `references/craft.md` gives the full reasoning and recipes.

- **Solids, not wireframes.** Every part is filled in paper colour and painted in order, so things in front hide things behind. No see-through edges and no dashed hidden lines.
- **Let the kit shade.** `extrude` returns four side tones by facing plus a lighter top. Keep it for anything with height, and use `flat` only for floors and inlays thinner than 1 unit.
- **Tone hierarchy.** Base and feet are `lo`, structure is `mid` and the subject is `hi`. Exactly one accent is `lit` (white in dark, black in light): the part the reader should watch. Grooves are `faint`, minor ticks `lo`, major ticks `mid` and the zero mark `hi`. Use colour only when colour is the subject, such as red, green and blue channels.
- **Small precise detail, never noise.** Use bevels of 0.4–1.6, grooves inset 3–5 from an edge, screws as 0.4–0.55 dots with a ring at the corners of plates, knurls on knobs, ruler ticks every 4–5 units with a major every 5th, and a label plate with three rule lines. Each detail belongs to a part.
- **Lines resolve.** Every line ends on something: a face, another line or a terminal dot. Connectors are `Wire`s whose terminal dots land on objects. There are no leader lines, balloons, dimension arrows or labels floating in the air.
- **No pipe or cable passes through anything.** Not through another pipe, cable or solid, at rest, apart or in any frame of the explode. Keep about 1.5 radii of clearance to everything but a line's own connection, run lines beside the structure rather than across the subject's face, and gap a pipe wherever a flange, band, bellows, clamp or valve wraps it. Parts meet flush, tangent or rim-on-surface, never sunk in: a flange's back face lands on its port (on a saddle pad, `G.saddleOf`, when the port is curved), a branch leaves from a ball tee, and every end, apart too, finishes on a flange, coupling, connector shell or terminal dot.
- **Every crossing is drawn in true depth order.** Where two items overlap on screen, the nearer is drawn on top, proved by `A.settle` and the audit, not tuned by eye. Lines cross on straights, well apart in depth: a bend, end or clamp that lands on another line's outline reads as a joint (a rod, tie rod, post or rail counts as a line: a pipe's last leg never runs along one on screen), a leg that points at the viewer and then turns pinches to a V (leave such a port through an elbow ball), and a route never doubles back on screen. A pair that can't be ordered interlocks, and the fix is a cut or a tangent contact, never a bias.
- **Curved surfaces shade smoothly, never in a staircase.** Mark curved knots smooth (`[s, r, 1]`) and `lathe` shades them from one monotone curve that also draws the outline; give any curved body larger than a fitting an analytic profile with `smooth: { slope }`, cut long bodies with `G.bandsOf`, and use `G.sphereOf` for balls. Tone bands taper to nothing instead of ending on a straight cut, and seams go under the outline (`inner`).
- **A texture ends on a real edge or fades.** Ribs, bolt circles, rule lines, seams and grooves never stop in the middle of a surface, flat or curved: `ribsOf` and `dotsOf` take `fade` (draw faded ribs with `fadedSvg`), a rib row ends on its seams (`ribsOf(…, { seams: true })`), a seam or weld on a box wraps its corners (`k.sideSeam`, as the slab's `inner`), a line on one face runs edge to edge, vents are closed slots, ticks hang from a ruler strip's edge, a single line that must stop ramps out with `fadeLineSvg`, and a row whose features face up (holes on top of a ring) is seen all round (`all: true`). A texture on a pipe (braid, corrugation, wrap) follows its cross-section and runs through every chunk joint: `tubePieces(…, { rings })`, never a dashed stroke down the centreline. A pipe's shine and shade run onto its end caps and through every chunk joint. Only rule lines that stand for printed text, a needle's tail and a pointer's tip may stop on a bare face, marked `free`.
- **Shaders never pop or cut off.** Every uniform rides an envelope with an attack and a decay, phase changes are crossfades driven by the physical quantity, every term reaches zero before any bound or canvas edge, and the effect lives in world space in the figure's own camera (see below).
- **Things on faces fit them.** Text a machine displays fits its window at its widest string and is clipped to it; content seen through an opening is clipped by the opening as well as by its own plane, with the recess walls drawn.
- **Volume.** Give things real thickness, make them stand on other things, and put a soft halo under the base and under anything lifted.
- **Composition.** Centre the figure and let it fill about 55–65% of the card. Keep a consistent scale across a set of figures, and make sure it still reads at 390 px wide.

## WebGL inside the drawing

Use a shader only for what lines can't draw: fire and exhaust, water and caustics, real lamp light, glow, steam. The parts stay SVG. The core rules come from the first shader figure, whose colour change between phases looked cut rather than crossfaded:

- **Three layers.** A back SVG holds what the effect covers, the canvas holds the effect (`pointer-events: none`), and a front SVG holds what covers it. Split a part, or the effect, when it is on both sides.
- **World space, same camera.** Build every shader point from `onFloor`, `onWallX` and `onWallY`, or march the pixel's ray `rayOf(vb) + t·towardViewer()`. Size everything in world units times `uK`. Anything lying on a plane registers exactly with that plane's SVG lines.
- **Nothing pops.** Every uniform follows an envelope with an attack and a decay (the kit's `burst` for events, `settle` for levels). Phase changes are crossfades where colour shifts continuously with the physical quantity, and a fade-out only gets darker.
- **No hard cut-offs.** Every term reaches zero before any `if`, bound or canvas edge, and before any cell edge of a cell noise.
- **Seen through an opening, masked twice.** Content behind a window or a recess is masked in its own plane and by the opening in the face plane, multiplied (`references/webgl.md`, section 5).
- **Blending.** Use `tonemap()` and dither. In dark, a canvas that only adds light uses premultiplied output with `mix-blend-mode: screen`, and one that must also darken (the ripple tank's water) stays normal-blended.
- **Light theme ships, and its shaders are designed for paper.** Use normal premultiplied alpha, never screen. Write the colour the paper should show and solve for the ink. A glow is a saturated body with a pale core, a coloured rim and an alpha floor, never a darkened dark-theme colour: that gave the Raptor's "red ink smoke".
  - What the light falls on takes a tint, and the cavity that holds it gets a darker physical ground (soot, refractory, water depth).
  - Hot bodies follow one blackbody ink table shared with the fallback.
  - Ink only grows with light: fade the finished ink once, never one term at a time.
  - Ink ends at each stroke's inner edge.
  - Measure ink per frame. `references/webgl.md`, section 8, has the recipes and the faults reviewers found.
- **Restraint.** Cap the device ratio at 1.5 and render only while the effect is active and onscreen. Reduced motion gets one still frame, and the SVG must still read without WebGL.

Read `references/webgl.md` before writing any shader.

## 3D: figures that turn (beta)

3D is an option like WebGL. Use it when a part moving in depth is the idea: a dial or turntable that turns, fingers that flex, a piece that breaks off and falls. The rest of the figure stays static kit art.

It is in beta: the engine is proved on its example and one test figure, not yet on a production figure. Turn one subject at a time and keep the clutter around it static. A small figure runs at 60 fps; a dense 380-part flexing hand measured about 20 fps at 1440 and 13–15 fps on a phone profile (390 px, 4× CPU throttle) on a heavily loaded machine.

- **Live parts are the kit's own picture.** Each frame, convex kit primitives (prisms, rounds, domes, balls with flats, lathes, tubes) are re-projected and re-shaded by the kit's rules. A turn by θ is the kit drawing at azimuth `a − θ`.
- **Order comes from separating planes.** Same-group pairs get a plane fixed in the group, pairs on different links get one fixed in their common frame, and the rest are solved each frame. The audit proves there are no cycles at any angle or in any clip frame.
- **Contacts are flush, tangent or rim-on-flat.** Joints are coaxial knuckle discs on the child group that touch the parent's link, or the audit calls the whole child group loose. No live line spans two moving groups: a cable crosses a joint through a clip on each side, or reaches the turntable through a slip ring at the hub.
- **Live tubes bend in near-horizontal planes or inside fittings, and never point along the view.** A bend needs R ≥ 2r / sin²(e − θmax), where θmax includes the pitch range of every hinge it rides: 8r is the bound when flat, and on a ±12° hinge it is 21r. Each route runs from mount surface to mount surface, with its ports in `gaps`. That holds at every angle and in every pose.
- **Materials remap the ramp.** Gold, chrome, steel, gunmetal, rubber, brass and copper each have a light and a dark set; tones still come from facing.
- **Free groups break off without a jump.** Coaxial parts are fixed pictures with turning textures.
- **Shaders follow the motion.** Anchor them to a group with `controller.anchor` or `toGroup`, occlude a front canvas with `coverOf`, and keep world effects (smoke, falling sparks, oil on the bench) out of group space.
- **Verify with `--audit`** (`turn audit passed`) and `scripts/turn-check.mjs` (`--fidelity`, `--order`, `--pops`, `--lines`, `--perf`). Iterate with `--quick --only=…`. Only a full run proves the figure, and it takes 15–20 minutes for 30 parts, so run it with `terminal(background=true, notify=true)`.

Read `references/3d.md` first.

## References

Read these as you need them:

- `references/craft.md`: the look in depth, including the parts vocabulary with code, tone tables, painter's order patterns, round parts, pipes and harnesses, exploded views, screens on faces, and the anti-patterns that got figures rejected. Read it before designing.
- `references/kit.md`: the API, with signatures and examples.
- `references/motion.md`: live behaviour (loops, springs, tours, pointer, keyboard, accessibility, reduced motion), plus envelopes for one-shot events, exploded views, depth re-sorting and arming, in vanilla JS and React.
- `references/react.md`: the React/Next.js server/client split, element props and the large-payload key-warning trap.
- `references/verify.md`: the `capture.mjs` and `drive.mjs` commands, the review checklist (including shaders) and common faults.
- `references/webgl.md`: shaders inside the drawing: when one earns its place, the layer stack, the camera contract, ray-marching in the iso camera, the full API with a minimal page, the craft rules with the bugs that taught them, the light theme (inks on paper, per-phenomenon recipes, the reviewers' faults), and how to verify them.
- `references/walkthrough.md`: how the test rig was designed, step by step, from concept to code.
- `references/3d.md`: the optional 3D mode: groups and poses, live layers, convex parts and their rules, materials, free groups, hinge chains, painter's order, static clutter around a turntable, shaders that follow the turn, the orbit audit, browser checks, performance and limits.
