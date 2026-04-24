---
title: Seedance reference-to-video notes
description: Practical notes for using Seedance 2.0 reference-to-video with ordered image references, timed beat prompts, and run artifacts.
sidebar_label: Seedance reference-to-video
---

# Seedance reference-to-video notes

These notes come from a real multi-agent review of a Seedance 2.0 run that tried to turn a symbolic fantasy sequence into a readable short cinematic unit. The useful lesson was not just that the output improved. The useful lesson was where it improved, where it still failed, and how to split future video-generation tasks by control type.

The short version: use `reference-to-video` when you need macro story flow across a compressed sequence. Use narrower `image-to-video` or start/end canaries when you need exact body mechanics, spatial continuity, or a single physical transition.

## When to use this pattern

Use Seedance `reference-to-video` for:

- trailer-proof sequences that need a beginning, turn, climax, and aftermath in one clip
- story-ordering tests where the question is "does the viewer understand the causal chain?"
- reference-governed visual language across several related beats
- audio-bearing canaries where rough music and sound design matter
- quick comparisons between a stitched edit and a single model-generated sequence

Avoid relying on a broad `reference-to-video` pass for:

- precise fall, injury, collapse, transformation, or hand-object mechanics
- locked scene geography across many cuts
- exact identity continuity across multiple characters
- object-specific proof where the audience must read one concrete prop or inscription

For those cases, make a smaller event canary first, then stitch it into the broader sequence.

## Working FAL request shape

A successful run used the FAL queue endpoint for `bytedance/seedance-2.0/reference-to-video` with uploaded local reference images.

```python
import fal_client

reference_paths = [
    "refs/testimony.png",
    "refs/archive-proof.png",
    "refs/ritual-geometry.png",
    "refs/corruption-backlash.png",
    "refs/failed-impact.png",
    "refs/proof-sealed.png",
]

image_urls = [fal_client.upload_file(path) for path in reference_paths]

payload = {
    "prompt": prompt_text,
    "image_urls": image_urls,
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "duration": "12",
    "generate_audio": True,
    "seed": 914,
}

handle = fal_client.submit("bytedance/seedance-2.0/reference-to-video", arguments=payload)
result = handle.get()
```

Important details:

- `image_urls` order maps directly to prompt handles: first URL is `@Image1`, second URL is `@Image2`, and so on.
- The API does not receive separate role metadata for each image. Put the role map in the prompt.
- `duration` is a string enum; `"12"` worked for a 12 second macro pass.
- `generate_audio: true` produced a video with an audio stream in the reviewed run.
- Keep provider responses and signed media URLs out of public docs and PR notes unless you have explicitly sanitized them.

## Reference packet pattern

The run improved when references were promoted from "nice images" into role-separated inputs:

```text
@Image1 = testimony / witness pressure
@Image2 = concrete archive proof / accusation record
@Image3 = ritual geometry / ascension stage
@Image4 = corruption backlash / failed power line
@Image5 = collapse impact / body failure endpoint
@Image6 = aftermath / proof sealed / witness record
```

What worked:

- native 16:9 references at the target framing
- one cinematic function per reference
- cleaned images with non-diegetic labels, panel numbers, and contact-sheet badges removed
- explicit prompt references to `@Image1`, `@Image2`, etc.
- a small enough reference set to stay coherent, but broad enough to cover the full sequence

What did not work as well:

- uploading storyboard pages or contact sheets as direct references
- relying on negative prompts to create story causality
- expecting a single broad prompt to solve exact body performance
- mixing portrait references with a 16:9 generation request when identity or headroom matters

## Prompt structure

The strongest prompt format combined a role map, timed beat blocks, camera logic, pacing, and sound design.

```text
REFERENCE ROLES
@Image1: witness testimony pressure, restrained and observational.
@Image2: archive proof object, readable evidence, no attack gesture.
@Image3: ritual geometry, upward ascension line, cold celestial palette.
@Image4: black-gold corruption interrupts the ascension line.
@Image5: failed collapse, body folds downward, no victory pose.
@Image6: aftermath record, proof sealed, witness remains present.

TIMED BEATS
0.0-2.0s: Establish witness relation and moral pressure.
2.0-4.0s: Cut to evidence object / archive proof. Make the accusation concrete.
4.0-7.0s: Ritual activates. The accused figure tries to ascend despite the proof.
7.0-10.0s: Corruption chokes the ascent. Upward movement breaks downward.
10.0-12.0s: Collapse aftermath. The witness does not attack; the record is sealed.

CAMERA AND EDITING
Use motivated cuts, not random spectacle. Keep witness/testifier and accused roles distinct. The camera should move from testimony to proof to ritual failure to witnessed aftermath.

NEGATIVE CONSTRAINTS
No healing glow. No white purification burst. No triumphant power-up. No spell blast from the witness. No full-frame explosion that hides body failure.

AUDIO
Low ritual drone under testimony. Rising pressure during the ascent attempt. Choked drop and muted impact at failure. No heroic swell.
```

## What went well

- **Reference-to-video improved macro causality.** The output read much more like a designed sequence than a stitched set of isolated canaries.
- **Role-separated references gave the model better visual memory.** The sequence had testimony, proof, ritual activation, corruption backlash, collapse, and aftermath instead of one generic confrontation tableau.
- **Trailer-board metadata helped.** Runtime segment, pacing, acceleration, cut density, sonic bed, visual language, and camera logic all translated into stronger film grammar.
- **Reference cleanup mattered.** Removing badges and layout text reduced non-diegetic contamination.
- **The run generated audio.** That made it possible to review the clip as a rough cinematic unit rather than silent motion only.
- **QC artifacts made the result reviewable.** A 24-frame flow sheet and old-vs-new comparison exposed sequence-level gains and remaining gaps quickly.

## What went wrong

- **Body mechanics were still weak.** The failure was implied by spectacle and rupture more than by readable physical causality.
- **A failed ascent can look like a power-up.** Bright cores, vertical beams, and upright poses can accidentally signal victory unless the prompt forces broken posture and downward failure.
- **Proof was still too symbolic.** Archive diagrams and celestial writing can imply evidence, but a viewer may not read a concrete accusation without a specific prop, seal, shard, oath tablet, or record.
- **Montage grammar was stronger than spatial grammar.** The model followed the beat order better than it preserved exact geography.
- **Identity and role can drift.** If the references do not lock character roles hard enough, the sequence can become mythically coherent while losing who is testifying, who is accused, and who collapses.

## What we learned

Treat each generation as a different control problem:

- **Macro story ordering:** use `reference-to-video`, 5-8 role-separated references, 10-12 seconds, timed beat blocks, and trailer-board metadata.
- **Exact physical transition:** use a 5-7 second canary, 2-3 references, and one event only.
- **Concrete evidence beat:** design the prop or accusation object before generation; do not leave "proof" as abstract light or glyphs.
- **Sequence quality:** always review the generated clip in context. Two individually passing clips can still fail as a stitched film unit.
- **Positive blocking beats negative prompts.** Negative constraints prevent drift, but story clarity comes from explicit action, camera relation, and causal order.

## Next improvement pattern

For a precise failed-body event, use a narrower canary instead of another broad macro pass.

```text
Name: C9_CHOKED_ASCENSION_BODY_FAILURE_V1
Duration: 5-7 seconds
References: 2-3 maximum
Goal: North tries to rise, corruption constricts the body, the ascent breaks, the body collapses.

0.0-1.5s: Upward lift begins. Robe, hair, dust, and debris pull slightly upward.
1.5-3.5s: Black-gold corruption tightens around chest, throat, spine, and core. The body loses vertical line.
3.5-5.5s: Ascent snaps downward. Knee and hand hit stone. The witness remains outside the action.
5.5-7.0s: Aftermath. No heroic glow, no purification, no attack from the witness.
```

Use this canary as an insert inside the broader reference-to-video sequence once the physical event reads clearly.

## Artifact hygiene

Save these artifacts for every run:

- `provider-request.json`
- uploaded reference manifest
- provider submit response
- status log
- provider result
- downloaded output video
- `ffprobe` metadata
- frame samples
- contact sheets
- old-vs-new or before-vs-after comparison sheets
- written review with pass/fail gates

The goal is to make each generation run auditable. The notes that matter are not only "this looked good"; they are the request shape, reference set, prompt structure, output behavior, failure mode, and next canary.
