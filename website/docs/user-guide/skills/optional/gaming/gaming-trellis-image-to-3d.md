---
title: "Trellis Image To 3D — Generate textured GLB models from images on HF Spaces"
sidebar_label: "Trellis Image To 3D"
description: "Generate textured GLB models from images on HF Spaces"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Trellis Image To 3D

Generate textured GLB models from images on HF Spaces.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/gaming/trellis-image-to-3d` |
| Path | `optional-skills/gaming/trellis-image-to-3d` |
| Version | `0.1.0` |
| Author | Jonathan Korstad (@Jpalmer95), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `3d`, `game-dev`, `glb`, `image-to-3d`, `gradio`, `huggingface`, `trellis` |
| Related skills | [`skintokens-rigging`](../../optional/gaming/gaming-skintokens-rigging.md), [`game-asset-pipeline`](../../optional/gaming/gaming-game-asset-pipeline.md), [`comfyui`](../../optional/creative/creative-comfyui.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# TRELLIS.2 Image-to-3D Skill

Turns a 2D image into a textured GLB through the `microsoft/TRELLIS.2` Hugging
Face Space, spending the caller's `HF_TOKEN` ZeroGPU quota instead of local VRAM.
Ships `scripts/trellis_gen.py` (single and batch modes, quality gates, resumable
manifests) and `scripts/pipeline_qgates.py` (the gate helpers).
It does not rig, animate, or import into an engine — see `skintokens-rigging`
for skeletons and `game-asset-pipeline` for the whole chain.

## When to Use

- "Turn this concept art into a 3D model" / "make these images into GLBs"
- Batch asset runs: "10 elemental ant enemies as 3D models"
- Any pipeline that feeds Godot, Unity, or Blender with generated GLBs
- Don't use for: rigging or skinning (that is `skintokens-rigging`), text-to-3D
  with no source image (generate the image first), or retopo/UV work on
  existing meshes (do that in Blender)

## Prerequisites

- `HF_TOKEN` in the environment Hermes runs with (create one at
  https://huggingface.co/settings/tokens). Without it the Space runs anonymously
  and long generations usually fail in the queue.
- `pip install "gradio_client>=1.0"` — the only hard dependency. Use a Python
  that has it; a Hermes release venv often does not.
- Optional: `pip install rembg` for local background removal (first run
  downloads the u2net model, ~170 MB) and Pillow for the alpha gate.

## How to Run

Single image:

```bash
python3 <skill>/scripts/trellis_gen.py input.png -o assets/goblin.glb --name goblin --preprocess
```

Batch, run as a background job because each asset takes 3–8 minutes:

```bash
python3 <skill>/scripts/trellis_gen.py --batch images_masked/ --out-root assets --prompts-file prompts.txt --keep-image --target godot --resume
```

`<skill>` is the `skill_dir` that `skill_view(name='trellis-image-to-3d')`
reports. Invoke both through `terminal(command="…", timeout=…)`, and for batch
runs `terminal(background=true, notify_on_complete=true)`.

## Quick Reference

| Flag | Effect |
|---|---|
| `--preprocess` | alpha-mask server-side before generation (skip if inputs are already RGBA-masked) |
| `--keep-image` | copy the source image into `<out-root>/images/` |
| `--target godot-mobile\|godot\|unity\|blender\|film` | sets decimation + texture size (100k/1024, 100k/2048, 100k/2048, 300k/2048, 500k/4096) |
| `--resolution {512,1024,1536}` | generation resolution; 512 is the Space default |
| `--decimation N` / `--texture-size N` | explicit override; the Space rejects decimation under 100k |
| `--seed N` | fixed seed (default random); reuse it to retry extraction on cached generation |
| `--resume` | batch mode: skip assets whose `.glb` already exists |
| `--skip-gates` | disable the alpha and GLB gates |

Space endpoints as verified against `/gradio_api/info` (re-check if the Space
redeploys): `/preprocess_image` (image → masked image), `/image_to_3d`
(image, seed, resolution, 3 stages × guidance/rescale/steps/rescale_t →
preview), `/extract_glb` (decimation, texture size → GLB file).

## Procedure

1. **Write the prompt list.** One line per asset in `prompts.txt`; `#` lines and
   blanks are skipped. Anchors that survive every phase: `full body, T-pose
   front view, centered with margin, dark background`. Completion: the line
   count matches the images you intend to produce.
2. **Produce images** as `images/NN_slug.png` — local ComfyUI (`comfyui` skill)
   or the `image_generate` tool. Completion: every image is full-body and
   uncropped, checked with `vision_analyze` before any GPU time is spent.
3. **Alpha-mask.** Either pass `--preprocess` or run rembg through its Python
   API in a project venv (`from rembg import remove; remove(png_bytes)`); the
   CLI eagerly imports server dependencies and fails on a minimal install.
   Completion: the subject is isolated and thin props (antennae, weapons) are
   intact.
4. **Generate.** Run the batch command above. Completion: `manifest.csv` has one
   row per input with `status=ok`, and each `.glb` is >50 KB.
5. **Gate and hand off.** Open one GLB in Blender (File ▸ Import ▸ glTF 2.0) or
   Godot 4 to confirm texture, scale, and orientation. Completion: the sample
   opens textured in the target engine.

## Pitfalls

1. **Missing token.** `HF_TOKEN` exported in another shell is invisible to
   Hermes; put it in the environment Hermes loads, or prefix the command.
2. **`gradio_client` 1.x renamed the auth kwarg** from `hf_token=` to `token=`.
   The script already uses `token=`; old snippets fail with `unexpected keyword
   argument 'hf_token'`.
3. **Unmasked backgrounds become geometry.** TRELLIS reconstructs whatever the
   image shows, including the backdrop. Mask first.
4. **Cropped silhouettes produce incomplete meshes** — and cannot be rigged
   later, because a skeleton needs legs that exist. Prompt for full body with
   margin.
5. **Parallel batch calls queue-fight.** ZeroGPU serializes per account; run one
   sequential loop.
6. **`extract_glb` is the fragile step.** Generation is cached per session, so
   re-running with the same `--seed` retries extraction without re-generating.
7. **Gates are advisory by design.** They catch empty masks, untextured meshes,
   and busted vertex counts; they cannot judge whether the shape is *good*.
   Vision-gate the source images instead.

## Verification

- [ ] `HF_TOKEN` present; script exits with a clear error when it is not
- [ ] Inputs RGBA-masked (checked or `--preprocess` used)
- [ ] Output GLB opens textured in Blender or Godot
- [ ] `manifest.csv` row per asset, `status=ok` for the ones you keep
- [ ] Batch executed sequentially, through `--resume`
