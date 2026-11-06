---
title: "Puppeteer Animation — Animate a rigged GLB from a driving video on a local GPU"
sidebar_label: "Puppeteer Animation"
description: "Animate a rigged GLB from a driving video on a local GPU"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Puppeteer Animation

Animate a rigged GLB from a driving video on a local GPU.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/gaming/puppeteer-animation` |
| Path | `optional-skills/gaming/puppeteer-animation` |
| Version | `0.1.0` |
| Author | Jonathan Korstad (@Jpalmer95), Hermes Agent |
| License | MIT |
| Platforms | linux |
| Tags | `3d`, `animation`, `video-guided`, `game-dev`, `glb`, `skeleton`, `puppeteer`, `blender` |
| Related skills | [`comfyui`](../../optional/creative/creative-comfyui.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Puppeteer Video-Guided Animation Skill

Drives `Seed3D/Puppeteer` (Apache-2.0, NeurIPS 2025) to fit a short video onto an
already-rigged mesh: it optimises per-frame joint rotations and root translation
until the skinned render matches the clip, then bakes the result onto the GLB's
armature in Blender. Works for **any** skeleton topology — quadrupeds, insects,
dragons, articulated props — unlike SMPL-only text-to-motion models.
It does not rig, generate the mesh, or retarget between skeletons — the
sibling `skintokens-rigging` and `trellis-image-to-3d` skills produce its input,
and `game-asset-pipeline` chains them.

This is the heaviest skill of the set: ~12 GB of checkpoints, a patched upstream
clone, a CUDA build of PyTorch, and a 12 GB GPU. Budget an hour for setup.

## When to Use

- A rigged GLB needs motion and no animation clip exists for its skeleton
- "Make this creature walk / attack / fly", driven by a reference clip
- Non-humanoid or custom-topology rigs where a retarget library has nothing
- Don't use for: humanoid-only retargeting from Mixamo-style libraries (direct
  retarget in Blender/Godot is cheaper), rigging, or mesh generation

## Prerequisites

- Linux, an NVIDIA GPU with 12 GB (24 GB removes most of the tuning), Blender
  3.6+, ffmpeg.
- A clone of `Seed3D/Puppeteer` and a working venv (below). `PUPPETEER_DIR`
  points at it, or pass `--puppeteer` per run.
- Checkpoints: `animation/download.py` (CoTracker3 + Video-Depth-Anything-Large).
  The `skeleton/` and `skinning/` checkpoints are only needed for Puppeteer's own
  rigging, which this skill does not use.

Build the environment (order matters — `setuptools<81` and `cython` must exist
before `requirements.txt`, or the `tetgen`/`flash-attn` builds fail):

```bash
git clone https://github.com/Seed3D/Puppeteer.git --recursive "$PUPPETEER_DIR"
cd "$PUPPETEER_DIR"
uv venv --python 3.10 .venv && . .venv/bin/activate
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install "setuptools<81" wheel cython==0.29.36
uv pip install -r requirements.txt
uv pip install flash-attn==2.6.3 --no-build-isolation
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
uv pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html
```

Two upstream patches make a 12 GB card viable (they live in the reference fork;
reapply on a fresh clone): skip the depth module when `depth/depth_gt_raw.pt`
already exists, and resize loaded frames to `--img-size`.

## How to Run

```bash
python3 <skill>/scripts/puppeteer_animate.py goblin_rigged.glb --video goblin_walk.mp4 \
    --puppeteer "$PUPPETEER_DIR" -o working/goblin/ --iter 500 --img-size 512 --decimate 20000
```

Then bake the fitted motion onto the rig:

```bash
blender -b --factory-startup -noaudio --python <skill>/scripts/bake_animation.py -- \
    goblin_rigged.glb --rig-txt working/goblin/objs/rig.txt \
    --results-dir working/goblin/results/goblin_anim/raw/ --fps 10 -o goblin_animated.glb
```

`<skill>` is the `skill_dir` from `skill_view(name='puppeteer-animation')`.
A 500-iteration run is minutes-per-second-of-video, not real time — launch it
with `terminal(background=true, notify_on_complete=true)` and batch overnight.

## Quick Reference

| Flag | Effect |
|---|---|
| `--iter 200` / `--iter 500` | draft (~13 min on a 4070 Ti) / final |
| `--img-size 512` | 512 keeps peak VRAM near 8 GB; 960 needs >12 GB |
| `--decimate 20000` | face target during GLB→RigNet conversion; required for TRELLIS meshes |
| `--skip-convert` / `--skip-precompute` | resume from prepared `objs/` or `imgs/` |
| `--main-renderer front_left` `--additional-renderers "right,back"` | which views get rendered for review |
| `--smooth-weight 1` | steadier motion for organic creatures |
| `--fps 10` | frame rate for extraction and bake; keep ≤4 s of video (40 frames) |

Outputs: `objs/rig.txt` + `objs/mesh.obj`, `imgs/`, flow/depth/track caches,
`results/<seq>_anim/raw/{local_quats,root_pos}.pt` (convert to `.npy` for the
bake — Blender ships numpy, not torch).

## Procedure

1. **Start from a rigged mesh.** Full-body, sensible rest pose, textures intact.
   Completion: the GLB opens with an armature and skin weights.
2. **Convert to RigNet layout.** `glb_to_rignet.py` runs inside Blender and bakes
   world transforms so joints and vertices share one frame. Completion:
   `objs/rig.txt` and `objs/mesh.obj` exist and the OBJ opens with the same
   silhouette.
3. **Produce the driving video.** Render first-frame views
   (`animation/utils/render_first_frame.py`), pick the view that shows joints
   best, and animate it with an image-to-video model (local ComfyUI, or a hosted
   one — a public URL is needed for local stills). Keep it ≤4 s, one dominant
   viewpoint, whole asset in frame. Completion: `input.mp4` looks like the motion
   you want, from an angle close to the rest pose camera.
4. **Fit.** Run `puppeteer_animate.py` at `--img-size 512`. Completion:
   `results/<seq>_anim/raw/local_quats.pt` and `root_pos.pt` exist and the
   rendered review clips show the motion tracking the video.
5. **Bake and import.** Run `bake_animation.py`, then open the animated GLB in
   Blender's timeline or Godot's AnimationPlayer. Completion: the clip plays and
   the mesh does not tear.

## Pitfalls

1. **OOM at `--img-size 960`.** Use 512 and precompute depth standalone so the
   1.5 GB depth model is freed before optimization.
2. **Resolution mismatch crashes.** Flow/track/depth caches take the resolution
   of `imgs/*.png` and upstream never resizes them. Change `--img-size` and you
   must delete `imgs flow flow_vis track_2d_joints track_2d_verts depth` and
   re-run.
3. **`ModuleNotFoundError: third_partys`.** Run from `animation/` with
   `PYTHONPATH=.` — the wrapper does this; hand-run commands must too.
4. **`hf-xet` silently drops large downloads.** Verify the checkpoint files
   exist; if not, re-run with `HF_HUB_DISABLE_XET=1`.
5. **`tetgen`/`flash-attn` build failures.** Install `setuptools<81`, `wheel`,
   `cython==0.29.36` before `requirements.txt`.
6. **Vertex count, not image size, is the memory wall.** Measured on 12 GB:
   ~204k verts OOMs during tracking, ~68k during rasterization, ~31k runs at
   ~11.6 GB peak. Decimate TRELLIS output to ~20k faces during conversion.
7. **Never decimate the OBJ after `rig.txt` is written.** Skin weights index
   vertices positionally; decimate inside `glb_to_rignet.py` (Blender keeps the
   vertex groups) or redo the conversion.
8. **Baking needs `.npy`.** Convert the `.pt` tensors in the Puppeteer venv
   first; Blender's Python has numpy but not torch.
9. **Long clips destabilize tracking.** 40 frames at 512 px is the reliable
   envelope; 50 frames pushed CoTracker3 off the edge in testing.
10. **Garbage in, garbage motion out.** A clip that loses the subject, changes
    camera angle mid-shot, or shows two competing motions will fit badly.

## Verification

- [ ] `results/<seq>_anim/raw/{local_quats,root_pos}` exist (`.npy` for the bake)
- [ ] Review renders exist for the main and additional views, no OOM
- [ ] Rendered motion visibly tracks the driving video, watched side by side
- [ ] Baked GLB plays in Blender or Godot and the mesh does not tear
