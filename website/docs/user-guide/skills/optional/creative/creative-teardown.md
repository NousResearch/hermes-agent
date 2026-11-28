---
title: "Teardown — Build take-apart 3D machine simulators and films"
sidebar_label: "Teardown"
description: "Build take-apart 3D machine simulators and films"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Teardown

Build take-apart 3D machine simulators and films.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/teardown` |
| Path | `optional-skills/creative/teardown` |
| Version | `1.0.0` |
| Author | elvarofficial (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `3d`, `three.js`, `simulator`, `explainer`, `cutaway`, `exploded-view`, `video`, `webgl`, `mechanism`, `education` |
| Related skills | [`anatomy`](../../optional/creative/creative-anatomy.md), [`hyperframes`](../../optional/creative/creative-hyperframes.md), [`manim-video`](../../bundled/creative/creative-manim-video.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Teardown Skill

One machine, one page. The visitor sees the thing run, presses a button, and the part doing the work opens up while the gauges around it read the real numbers. Left alone, the page tours itself with a cursor. The same code records a film of the tour, frame by frame, with a score built on the same timeline.

`SKILL_DIR` below is the folder this file lives in (`${HERMES_SKILL_DIR}` in Hermes). Builds and renders go to `./teardown-out` of the folder you run from.

## First run

The skill ships its pinned `package.json` and `package-lock.json`, but never install into the skill directory itself (Hermes would flag the installed skill as locally modified). Install the dependencies once into a sibling folder and point `TEARDOWN_DEPS` at it:

```bash
SKILL_DIR="${HERMES_SKILL_DIR}"
DEPS="$HOME/.local/share/teardown"
mkdir -p "$DEPS" && cp "$SKILL_DIR/package.json" "$SKILL_DIR/package-lock.json" "$DEPS/" && cd "$DEPS" && npm ci --no-audit --no-fund
```

Every later `node` command needs `TEARDOWN_DEPS="$HOME/.local/share/teardown/node_modules"` in front of it. The pages need nothing else. The video needs `ffmpeg` on PATH and the score needs Python 3 (standard library only). Set `FFMPEG`, `PYTHON` or `CHROMIUM` to point at other binaries.

## Running it in Hermes

- **Skill directory.** Shell variables don't carry over between `terminal` calls, so start every call that uses the kit with `SKILL_DIR="${HERMES_SKILL_DIR}"` and `export TEARDOWN_DEPS="$HOME/.local/share/teardown/node_modules"`.
- **Work outside the skill.** Run builds from the user's project or a folder they name; output lands in `./teardown-out/` of that folder. A new machine lives in its own folder there and imports the kit as `teardown-kit/...` (see "How a machine is built"). Copy an example's `<name>-shared/` next to it and rewrite its `../kit/` imports to `teardown-kit/` when starting from one.
- **Browser.** `capture.mjs` uses Playwright's Chromium. Where `npx playwright install chromium` is unsupported (new distros), set `CHROMIUM=/usr/bin/google-chrome` or another system Chrome. `capture.mjs` prints the GPU it got: on a headless server that is SwiftShader, which renders correctly but slowly (about 3 s per 640x360 frame). Plan a full 60 fps film as a long job: run it with `terminal(background=true, notify=true)`, or record a short `--from/--to` slice at `--fps 15` to check a change.
- **Looking at your work.** Read every beats sheet and still with `vision_analyze`, asking concrete questions ("does piston 3 pass through the head?"), and use its `region` crop to zoom.
- **Delivering.** Hand over the absolute path of the built HTML and the final MP4. On a messaging platform, attach the MP4 or a still, since a chat cannot run the page.

## The 3 examples

Build one, open the HTML, press buttons. Copy the closest one when you start a new machine.

| | The machine | What drives it | Worth stealing |
|---|---|---|---|
| `v8-*` | A cross-plane V8 with glass liners: the charge burns inside the cylinder you track, valves lift on their cams, 8 pistons move from the exact slider-crank. | Crank angle. Firing order 1-8-4-3-6-5-7-2, a chamber pressure model for the indicator diagram. | Cutaway modes (glass, section, solid), the firing order as a cylinder picker, live charts synced to the crank. |
| `cessna-*` | The Cessna 337 main landing gear folding into the airframe on takeoff and dropping on landing. | Stroke 0..1: a 201 mm rack turns a 32-tooth pinion (module 4 mm) through exactly 180°. | One degree of freedom posing a whole mechanism, flight phases as functions of time, a runway that rolls under the model. |
| `strandbeest-*` | Theo Jansen's Strandbeest, 20 legs on one crankshaft, walking broadside over sand. | Wind. The crank speed is integrated from it; every joint follows by circle intersection from Jansen's 11 link lengths. | An exact linkage solver in 67 lines, instanced tubes, foot traces, a contact chart of all 20 feet. |

Each comes in 3 folders: `<name>-shared/` holds the machine, `<name>-film/` the recorded tour, `<name>-live/` the page you drive.

```
node "$SKILL_DIR/sims/build.mjs" v8-live          # → teardown-out/dist/v8-live.html, offline, ~800 KB
node "$SKILL_DIR/sims/build.mjs"                  # all 6
```

## How a machine is built

One folder per machine, split by job. The examples follow this exactly:

| File | Job |
|---|---|
| `kin.js` / `gear.js` / `jansen.js` | The math. Real dimensions in real units, one input (crank angle, stroke, wind) and every pose and readout as a function of it. No three.js in here. |
| `engine.js` / `gear.js` / `beast.js` | The geometry, built from code: lathed parts, tubes, instanced links. `update(input)` poses it. |
| `stage.js` | Studio, lights, shadows, post (GTAO, bloom). `frame(state, cam)` draws one frame. |
| `console.js` | The overlay: fixed 1600 x 900 CSS layout of panels, charts, toggles, callouts with leaders. `update(state)` redraws it from the frame's state. |
| `film.js` | A `SCRIPT` of operator moves (views, isolate, cutaway, throttle) and `render(t)` as a pure function of time. |
| `live.js` | The same machine in real time: buttons, sliders, orbit, zoom, keys. With no input it plays `film.js`'s script as an auto tour with a cursor; any click, drag, wheel or key hands control over, 25 s of quiet hands it back. |
| `score.py` | The sound on the film's timeline. |

For a machine outside `SKILL_DIR`, make a folder with a `main.js` and import the kit by name: `import {runSim} from 'teardown-kit/runtime.js'`. Build it with `node "$SKILL_DIR/sims/build.mjs" ./my-machine`.

## Rules that decide whether it works

1. **Start from a reference, never from imagination.** A photo, a cutaway drawing from a manual, a clip of the real thing. Build against it and compare frame by frame. Machines invented from a sentence come out generic every time; the same model rebuilds a real one closely when it has something to match.
2. **Real numbers, one source.** Get the real dimensions first (bore, stroke, tooth count, link lengths). Every gauge, chart and label reads from the same function that poses the parts, so nothing on screen can disagree with the motion. Check one by hand: in the Cessna, 32 teeth × 4 mm module is a 128 mm pitch diameter, and half its circumference is the 201 mm stroke.
3. **One input drives everything.** Pose from the crank angle, the stroke or the wind. Nothing is keyframed by hand, so the film, the live page and the score never drift.
4. **The part doing the work opens up.** Isolate it, cut it, x-ray it. A camera that goes inside is worth more than ten orbits.
5. **Light studio, one accent.** Pale backgrounds read in a feed; dark ones lose the first second. The accent marks the thing that is changing. Watch near-white parts on a pale backdrop: give them an ink outline or a darker ground.
6. **Pick the camera for the mechanism.** A planar linkage seen face-on collapses into a lattice; a 3/4 view along the spine keeps every leg readable.
7. **Fix collisions with a number.** When a part pokes through another, scale or move it in the parameter block. Never hide it with the camera.

## Live page rules

- **Test on an integrated GPU.** A page that runs at 60 fps on a desktop card can drop to 15 fps on a laptop's built-in one. The examples carry quality tiers (`TIERS` in `live.js`): no AO, lower pixel ratio, smaller shadow maps, panels redrawn every other frame. Find the expensive shader first: a displacement computed 5 times per vertex costs more than every tier together.
- **Warm the shaders.** `kit/warm.js` compiles every program in parallel before the first frame, so the page never freezes on first draw.
- **Keep the page under 1 MB.** Link crawlers (X, Slack) read the share card from the HTML and give up on a 15 MB page. Big models go out as files next to the page: `build.json` → `{"external": {"heavy.glb": "light.glb"}}`.

## render(t) contract

```js
import {runSim} from '../kit/runtime.js';

const DEFAULTS = {duration: 20, fps: 60, hud: {enabled: false}, director: {enabled: false}};

runSim(DEFAULTS, async ({cfg, canvas, capture}) => {
  // build the scene, precompute anything stateful
  return {render(t) { /* draw frame t from scratch */ return {}; }, resize() {}};
});
```

- `render(t)` is pure: no `Math.random` (use `rng(seed)` from `kit/util.js` at setup), no `Date`, no state carried between frames. Capture calls it at arbitrary times.
- Anything that moves by its own rules (agents, particles) is simulated once at load into typed arrays; `render` only reads them. A quantity integrated over time (the Strandbeest's crank under a changing wind) is integrated in closed form or tabulated at load.
- Every beat goes into `<name>-film/timeline.json` in seconds. The score reads the same file.
- Frame 0 is the thumbnail. It must already show the machine.

## Commands

```
node "$SKILL_DIR/sims/build.mjs" v8-film
node "$SKILL_DIR/sims/capture.mjs" v8-film --size 1600x900 --dpr 1.2 --stills beats --cols 4 --out teardown-out/v8-beats.png
node "$SKILL_DIR/sims/capture.mjs" v8-film --size 1600x900 --dpr 1.2 --still 12.5 --out teardown-out/v8-12.5.png
node "$SKILL_DIR/sims/capture.mjs" v8-film --size 1600x900 --dpr 1.2 --out teardown-out/v8-silent.mp4
python "$SKILL_DIR/run_sfx.py" "$SKILL_DIR/sims/v8-shared/score.py"
ffmpeg -y -i teardown-out/v8-silent.mp4 -i teardown-out/v8-score.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k -shortest -movflags +faststart teardown-out/v8.mp4
node "$SKILL_DIR/wave.mjs" teardown-out/v8.mp4 --timeline "$SKILL_DIR/sims/v8-film/timeline.json"
```

`--size` is the CSS layout and `--dpr` multiplies it: 1600 x 900 at 1.2 gives 1920 x 1080. Capture prints the GPU it got; a software renderer means the GPU flags in `paths.mjs` did not take.

## Review loop

You cannot watch the video; you can look at frames with `vision_analyze`. Never report a machine as done without reading sheets of the current render.

1. `--stills beats`: one tile per timeline beat, 0.3 s after it, each labelled `t=`.
2. Read every tile against the checklist below.
3. Zoom in: `--stills every:0.2 --from A --to B` for a transition, `--still T` for one full frame.
4. Fix, rebuild, go back to 1. Stop when a full pass finds nothing.
5. Final: `node "$SKILL_DIR/sheet.mjs" --video <final.mp4> --timeline <timeline.json>` and read that too: it shows what the encoder wrote.

Checklist:

- **The mechanism.** Every part that should touch touches; nothing passes through anything; the numbers on the panels match the pose.
- **Composition.** One focal point per frame. Nothing important within 60 px of an edge. Labels never cover the part they name.
- **Type.** Nothing clipped. The key readout reads at phone width.
- **3D.** Bloom eats detail: keep emissive layers under 1.0. Near-white surfaces blow out. Glossy metal under a low light streaks: roughness ≥ 0.45 and envMapIntensity ≈ 0.5 on big surfaces.
- **Timing.** Each state holds long enough to read: 1 s for a short phrase plus 0.25 s per word beyond three.

## three.js traps the examples already hit

- `smoothstep(a, b, x)` with `a > b` returns 0 on Windows (ANGLE/D3D). Write `1.0 - smoothstep(b, a, x)`.
- A `ShadowMaterial` or a decal a fraction of a millimetre above the floor loses the depth test inside a post-processing render target. Give it `polygonOffset`.
- A quad laid into the ground with a mirrored mapping gets culled as a back face. Use `DoubleSide`.
- One NaN pixel blacks out the whole frame through bloom. Clamp every `pow()` base in a shader.
- GTAO on a clipped (cutaway) model needs the same clipping planes on `gtao.normalMaterial`, or the cut half returns as a pale ghost.

## Sound

`sfx/` synthesises every sound in pure Python (48 kHz, reproducible) and reads the film's timeline, so sound cannot drift out of sync. `python "$SKILL_DIR/run_sfx.py" --list` shows the generators. `sample()` adds the recorded CC0 sounds in `sfx/samples/`. `sfx/score_kit.py` has the beds and accents the 3 scores use: a chord bed, a soft thud per power stroke, a motor, air under a drag.

- Every time comes from the timeline, never a typed number.
- Mastering is automatic: −14 LUFS, true peak ≤ −2 dBFS.
- Accent, never narrate: continuous motion stays silent. One key for every tonal sound.

## Taste, with numbers

- Easing: entrances `cubic-bezier(0.16, 1, 0.3, 1)`, exits `cubic-bezier(0.6, 0, 0.9, 0.35)` at half the entrance's length, state to state `cubic-bezier(0.65, 0, 0.35, 1)`. Linear only for continuous motion.
- The machine fills 60-70% of its viewport. Labels never under 34 px on a 1080 canvas.
- One accent, and it means something: the part that is working right now.
- Nothing dead-still: one element always alive.

## Credits

The kit, frame capture, sfx engine and review rules come from [whaleyxbt/claude-motion](https://github.com/whaleyxbt/claude-motion) (MIT). The recorded samples are CC0 from Freesound, trimmed by [Barty-Bart/motion-graphics](https://github.com/Barty-Bart/motion-graphics) (MIT); sources in `sfx/samples/SOURCES.md`.

Ported to Hermes from [elvarofficial/teardown](https://github.com/elvarofficial/teardown) at `a91262a` (MIT, `LICENSE.txt`). Changes: dependency install moved out of the skill directory, `sfx/synth.py` reads `FFMPEG` from the environment instead of a hardcoded Windows path, the two ffmpeg decode calls gained `stdin=DEVNULL` and a 300 s timeout, a few ruff autofixes (the rendered score is byte-identical), and the Hermes section above. The README's demo GIFs (14 MB) are not vendored.
