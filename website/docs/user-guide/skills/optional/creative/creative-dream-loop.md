---
title: "Dream Loop — Build stunning 3D scenes via a concept-art fidelity loop"
sidebar_label: "Dream Loop"
description: "Build stunning 3D scenes via a concept-art fidelity loop"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dream Loop

Build stunning 3D scenes via a concept-art fidelity loop.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/dream-loop` |
| Path | `optional-skills/creative/dream-loop` |
| Version | `1.0.0` |
| Author | Anshu Chimala (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `3d`, `games`, `webgl`, `threejs`, `image-generation`, `visual-fidelity`, `creative` |
| Related skills | [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Dream Loop Skill

An autonomous process for building extremely impressive visuals, especially 3D
scenes (games, apps, usually browser three.js/WebGL): generate photorealistic
concept art of the ideal result, build it, screenshot the live build, have a
judge score screenshot vs concept against a gated ladder, and iterate until
convergence. Goal: the most visually stunning result at an acceptable frame
rate for the target platform (e.g. 60 fps browser, 120 fps modern mobile).

This skill does NOT cover general web-app functionality, 2D UI design, or
non-visual quality — only the visual-fidelity loop.

## When to Use

- User says "dream loop" or asks for a game/scene/app built to a very high
  level of graphical fidelity.
- Follow-up refinement passes on an existing visual product (see Follow-up
  loops below).

## Prerequisites

- **Concept art**: the `image_generate` tool. If unavailable, stop and ask the
  user for a concept image (or an image-generation API to connect to).
- **Screenshots**: `browser_exec` — serve the build locally
  (`python3 -m http.server` for static builds), then `new_tab(url)`,
  `wait_for_load()`, `capture_screenshot()`.
- **Judging**: `vision_analyze` (see Judge section for the one-image-per-call
  workaround).
- **Optional**: Blender for asset modeling — see the `blender-3d-automation`
  skill. `delegate_task` for parallel asset work and fresh-context judging.

If you don't have the tools needed for the full loop, flag that to the user
early and stop.

## Quick Reference

| Stage | What happens | Artifacts (`.dream-loop/`) |
|---|---|---|
| 1. Target | Get/confirm the user's description | notes |
| 2. Concept | Generate "in-engine screenshot" concept art | `concept.png` |
| 3. Budget | Record time budget & start time (if given) | notes |
| 4. Build | Implement the concept as well as possible in one go | source, plans |
| 5. Screenshot | Capture live build at concept resolution | `round-N.png` |
| 6. Self-check | Rigorous side-by-side audit before judging | assessment log |
| 7. Judge | Ladder-scored comparison, actionable directives | verdict log |
| 8. Iterate/Exit | Address directives or exit per criteria | — |

## Procedure

### 1. The target concept

If the user provided a description of a game, scene, or app, proceed — don't
ask for clarification unless it's too vague to generate concept art from. If
they didn't, ask for it.

Put working context/files in `.dream-loop/` and gitignore it (unless told
otherwise).

### 2. Concept art

The concept is a realistic, high-quality, impressive target: the look of a
current AAA game running in real time. Physically plausible materials (wet
stone, brushed metal, cloth, glass) with real roughness and normal detail,
correct proportions, atmosphere (fog, haze, rain, dust, volumetric light),
cinematic lighting with a clear key and rich shadows. It should NOT be
stylized or an artistic rendition — it should look like a true screenshot of
the ideal result.

Avoid these failure modes when generating with `image_generate`:

- **Overbaked**: photographic clutter, film grain, hundreds of unique small
  objects, excessive detail on every surface that reads as noise. A real-time
  build with modeled assets won't match this, and it won't even look good.
- **Oversimplified**: cartoon or toy look, flat shading, blobby primitive
  shapes, empty surfaces. Boring; will not impress the user.

Aim for the middle ground: beautiful surfaces and materials that shaders
render well, strong atmosphere and lighting, an interesting palette, and
focused hero elements with fine detail that draws the eye (not every element
fighting for attention).

Prompt for "in-engine screenshot" more than "concept art" and discourage the
noisy/grainy look. Review the image with `vision_analyze`; if it hits a
failure mode, pass it back to `image_generate` in edit mode and ask it to fix
the issue. Save it as `.dream-loop/concept.png`.

If you generated the art (the user didn't supply it), pause and confirm it
matches the user's vision before starting the build loop.

### 3. Time budget

If the user gives a time budget, record the start time and check the clock
between rounds. Don't degrade visual fidelity to hit the budget — strive for
the absolute best result, and don't rush work to the judge. Parallelize or
distribute work (e.g. `delegate_task` for independent assets) to hit the
time goal, but no shortcuts: it's better to hit the time limit with
meaningful, beautiful progress than with something broadly complete but ugly.

If no time budget is given, run until an exit criterion — but warn upfront
that this may consume a lot of tokens.

### 4. Build loop

Look at the concept art and implement it in one go, making that first pass
count across every tier of the score ladder: composition, textures, lighting,
details. Sculpt and model assets carefully (or use external ones if allowed);
don't settle for basic procedural elements and flat surfaces unless the art
style calls for it. Write intermediate files/plans to `.dream-loop/`.

- If Blender is installed and the concept involves 3D assets, prefer modeling
  in Blender (see the `blender-3d-automation` skill). For complex assets,
  delegate to subagents via `delegate_task`.
- If the user allows external assets, prefer them over modeling unless the
  asset is simple. If unspecified, assume NO external assets from the web.
- Do not be lazy with key environmental details (scenery, flooring,
  buildings): simple shapes look blocky, shiny, flat, and fake. Tiny details
  and texturing matter and need custom sculpting.
- Use `image_generate` for textures, normal maps, skyboxes, etc. — better
  looking and faster than procedural ones.

### 5. Screenshot

Serve the build (e.g. `python3 -m http.server` in the build dir), then via
`browser_exec`: `new_tab('http://localhost:8000')`, `wait_for_load()`, allow
the scene to settle, `capture_screenshot()`. Target the same resolution and
aspect ratio as the concept art so the comparison is fair. Save as
`.dream-loop/round-N.png`.

### 6. Self-check before judging

Each time, review the candidate screenshot yourself before submitting. Do not
submit half-baked work. Compare screenshot and concept side by side and log an
honest assessment of judge-readiness; only submit if confident you've
significantly improved the score. Be rigorous and audit every pixel: big stuff
(missing/incorrect objects, wrong scale, perspective, positioning) and small
stuff (rendering glitches, flat untextured surfaces, ugly lighting, poor
contrast, washed-out or oversaturated color, speckles, ugly shadows). Scan
surface by surface, object by object, and list findings.

### 7. Judge

Judging should ideally be done by a fresh subagent with a clean context each
round (`delegate_task`), to keep it objective and cheap. Give the judge the
latest screenshot, the concept, and (from round 2 on) the previous round's
screenshot and verdict.

Mechanics: `vision_analyze` takes one image per call. Either have the judge
make sequential calls (concept, then screenshot, then compare from memory of
its own descriptions), or — better — stitch a labeled side-by-side composite
with ImageMagick (`convert concept.png shot.png +append compare.png`) or PIL
and analyze that single image.

Judge prompt:

> You are an art director reviewing a real-time render against its concept
> art. Compare the screenshot to the concept and score it 0-10 using this
> ladder. The ladder is gated: a frame cannot score above a tier's cap until
> every requirement of the tiers below it is fully met. Be strict about the
> gates.
>
> - **Tier 1, shape (0-3):** camera, framing, composition, and the position
>   and rough scale of every major object match the concept. Layout, not
>   finish: every major element present, in the right region of the frame
>   (within ~10% of frame width/height), at roughly the right size (within
>   ~25%). Right place and vaguely correct outline passes even if edges and
>   surface are wrong; save precision nitpicks for Tier 4. Cap 3 until true.
> - **Tier 2, light and color (3-5):** key light direction and color, overall
>   exposure (no clipping to black or white), shadow depth, palette,
>   contrast, atmosphere. Attend to reflections, glows, etc. Ensure the scene
>   is not too bright or dark relative to the concept. Judge the whole frame,
>   not tiny details (Tier 4). Cap 5 until lighting/reflections/color/
>   contrast are generally right.
> - **Tier 3, materials and surfaces (5-7):** every surface reads as the
>   right material at a glance: textures, roughness, translucency, wetness,
>   reflections. Assets must not look procedural, blocky, smooth/plastic;
>   frame-dominating elements should be properly sculpted and detailed. Cap 7
>   until true.
> - **Tier 4, fine detail (7-9):** the small things. Nitpick relentlessly;
>   inspect every little object up close. Layout aligns near-perfectly;
>   materials extremely convincing. Cap 9 until right.
> - **Tier 5, indistinguishable (9-10):** holds up side by side and zoomed
>   in. Nitpick every pixel.
>
> If a previous verdict and screenshot are provided: you are one reviewer in
> a sequence, not the first. Maintain consistency. First mark each previous
> directive LANDED, PARTIAL, or NOT DONE against the new screenshot; carry
> forward anything PARTIAL or NOT DONE. Don't reverse a prior directive
> unless the result is clearly worse — and if you do, say so and why.
>
> Output format:
> 1. Score on the first line; "Tier N" (highest fully-passed gate) on the
>    second.
> 1b. If given a previous verdict: the LANDED / PARTIAL / NOT DONE list.
> 2. "Blocking:" the specific failures of the *next* tier's gate. The builder
>    must clear these before anything else counts. Name the element and the
>    change, with magnitudes: "Rocks: replace the stacked ovoid boulders with
>    one continuous fractured slab; cracks 2-5cm wide, dark interiors, add
>    surface texture so they don't look flat/plastic" — not "the rocks look
>    artificial".
> 3. Then at most 4 further directives from higher tiers, same style, ordered
>    by points recoverable.
>
> No non-actionable feedback ("this looks synthetic") — name the specific
> causes. Every directive must be actionable this round. Don't round up: if a
> gate isn't fully passed, the cap holds.

### 8. Exit criteria

- **Score >= 8 and target FPS acceptable**: done. Show the user the latest
  screenshot; ask if they want more iterations.
- **Score >= 8 but FPS unacceptable**: optimize — lossless wins first, then
  minimal-visual-impact ones. Re-judge afterwards to confirm no regression.
- **Stall approaching** (best score hasn't improved a full point in 2 rounds,
  or the judge named the same gap 3 times): stop incremental tweaks. Step
  back and ask what about the *approach* is capping the score. Make one big
  structural change in a round: swap asset strategy (sculpt in Blender, pull
  real models/textures/HDRIs if allowed), rewrite the lighting model, rebuild
  the composition, change the camera. Self-check carefully — big changes
  break things. Only repeat parameter tuning if you can articulate why it
  would work this time.
- **Stalled** (already tried a big structural change, score flat 3 rounds,
  judge is nitpicking or demanding intractable things like raytracing on a
  GPU-less machine): stop, tell the user why you're blocked, give options.
- **Otherwise**: address all or most heavy-hitting gaps this round, not just
  the top one — rounds are expensive. Prioritize gaps that move the needle
  most (lighting, textures, mesh detail). Only revert if the score dropped a
  full point or more; small dips are judge noise, and reverting a whole round
  throws out good changes with bad. If one change clearly regressed, undo
  just that change. Loop.

## Follow-up loops

When building on an existing product (or the user re-invokes the skill for
refinements), don't create new concept art in a vacuum — it may diverge from
what exists. Instead capture a live screenshot of the current product and
prompt `image_generate` to render the best possible version of it (current
screenshot → AAA-graphics version of the same shot), then use that as the
target. Multiple screens can run parallel judge loops if asked, at higher
token cost.

## Pitfalls

- Overbaked or oversimplified concept art (see failure modes) — fix the
  concept before building against it.
- Submitting half-baked screenshots to the judge; the self-check gate exists
  for a reason.
- Screenshot at a different resolution/aspect than the concept — unfair
  comparison, noisy verdicts.
- Tunnel-visioning on incremental tweaks when the judge says you're off base.
- Judging both images in one `vision_analyze` call — it takes one image;
  composite them first.
- Screenshotting before the scene loads/settles — add a wait after
  `wait_for_load()` for asset streaming and animation warm-up.

## Verification

- `.dream-loop/concept.png` exists and passed the failure-mode review (and
  user confirmation, if generated).
- Each round has a screenshot, a logged self-assessment, and a judge verdict
  with score + tier + directives.
- Exit only via an explicit exit criterion; final screenshot shown to the
  user with the final score and FPS measurement.

---
Adapted from [dream-loop](https://github.com/achimala/dream-loop) by Anshu
Chimala (MIT). Upstream license vendored as `LICENSE.txt`.
