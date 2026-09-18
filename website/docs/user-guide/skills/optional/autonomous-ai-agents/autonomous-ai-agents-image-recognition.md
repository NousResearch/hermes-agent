---
title: "Image Recognition — View images natively and diagnose vision path issues"
sidebar_label: "Image Recognition"
description: "View images natively and diagnose vision path issues"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Image Recognition

View images natively and diagnose vision path issues.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/image-recognition` |
| Path | `optional-skills/autonomous-ai-agents/image-recognition` |
| Version | `0.2.1` |
| Author | moqiecuican, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `vision`, `images`, `multimodal`, `diagnostics`, `config` |
| Related skills | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Image Recognition Skill

> **Scope: Hermes Agent only.** This skill is built exclusively for
> [Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous
> Research). It hooks into Hermes-internal APIs
> (`tools/vision_tools.py`, `agent/image_routing.py`,
> `agent/auxiliary_client.py`, `hermes_cli.config`) and Hermes-managed
> state (`config.yaml`, the models.dev cache). It will **not** work with
> other agent frameworks (Claude Code, Codex, OpenClaw, ...).

Hermes vision has two channels plus an automatic fallback chain: a natively
multimodal main model gets pixels pushed straight into its context (native
fast path); anything else falls back to an `auxiliary.vision` model that
describes images in text (lossy). This skill covers everyday image viewing,
path diagnosis, first-run setup, and model-switch self-checks. It does NOT
handle video (`video_analyze` is a separate tool) or OCR-heavy document
parsing (use an OCR skill if one is available).

## When to Use
- Viewing any image: local path, URL, browser / computer-use screenshots
- Symptom: "my main model is multimodal, but image details are lost / it
  reads like a paraphrase" → check which path is active first
- After switching provider/model, verify vision still runs natively
- New or self-hosted models absent from models.dev → declare vision manually
- Don't use for: video; OCR-style document extraction

## Prerequisites

`<hermes-home>` below means your Hermes data directory: `~/.hermes` by
default, or the value of `HERMES_HOME` if you set one. Run the diagnostic
script through the Hermes `terminal` tool; inspect `config.yaml` with
`read_file` when the checklist points there. All commands work on Linux,
macOS, and Windows (Windows uses `venv\Scripts\python.exe` instead of
`venv/bin/python`).

**Step 0 — Run the diagnostic script** (read-only, never mutates anything):

```bash
# Linux / macOS
<hermes-home>/hermes-agent/venv/bin/python \
  <skill-dir>/scripts/check_vision_path.py
# Windows
<hermes-home>\hermes-agent\venv\Scripts\python.exe <skill-dir>\scripts\check_vision_path.py
```

Completion criterion: a four-line verdict (provider/model, image input
route, tool-result images, model vision verdict, final path). Do not guess
your setup — read the output and fix line by line:

**Step 1 — `image input route` is `auto` or `text`** (most common trap):
In `auto` mode, an explicitly configured `auxiliary.vision` (any of
provider/model/base_url set) SUPPRESSES the main model's native vision —
every attached image gets paraphrased by the aux model even when the main
model is multimodal. Fix:

```bash
hermes config set agent.image_input_mode native
```

Rerun the script; expect `native`. (Alternative: remove the explicit
`auxiliary.vision` item instead — but keeping it is recommended, see
Pitfalls. If `text` was set deliberately, leave it.)

**Step 2 — `model vision verdict` is `None`** (fresh installs and
custom models): vision routing never touches the network, so a brand-new
install has no models.dev disk cache yet and known-multimodal models can
read as `None`. Two sub-cases:
- Fresh install → run Hermes normally once (any conversation) to populate
  `<hermes-home>/models_dev_cache.json`, then rerun the script.
- Model is genuinely multimodal but absent from the database (self-hosted,
  renamed, brand-new) → declare it explicitly:

```bash
hermes config set model.supports_vision true
```

Rerun; expect `True`. The config override has the highest priority and does
NOT depend on the cache — it is the most reliable channel for custom models.

**Step 3 — `model vision verdict` is `False`**: the model is classified as
text-only. If it is actually multimodal (new name, stale catalog), apply the
same `model.supports_vision true` override. If it is truly a text model,
change nothing — the auxiliary fallback is the designed behavior.

**Step 4 — Real-image acceptance test**: call
`vision_analyze(image_url=<a local image>, question=...)` and check the tool
result's first line:
- `Image loaded into your context — you can see it natively now.` → native
  path active; pixels are in context.
- `{"success": true, "analysis": "..."}` → still on the aux-fallback path;
  revisit Steps 1–2.

**Step 5 — Rollback / cost awareness**: undo with
`hermes config unset agent.image_input_mode`. Native images live in the
immutable conversation history and are re-sent every turn — long sessions
cost more tokens (that is the tradeoff for zero information loss).

## Core mechanism: two paths (identify by tool-result first line)

1. **Native fast path**: first line
   `Image loaded into your context — you can see it natively now.`
   The image rides back as a `_multimodal` envelope straight into the main
   model's context — no aux call, no information loss.
2. **Aux fallback path**: returns `{"success": true, "analysis": "..."}`
   — the `auxiliary.vision` model describes the image; pixels never enter
   the main-model context (lossy: layout / small text / exact colors).

The switch is `tools/vision_tools.py::_should_use_native_vision_fast_path()`:

- `agent.image_input_mode` resolves to `native`, AND either:
  - the provider accepts images inside tool results (static allowlist:
    anthropic / openai / openai-codex / azure-openai / gemini-3 /
    aggregators — many direct providers are NOT on it), OR
  - models.dev metadata says the model supports vision
    (e.g. `glm-5.3-flash` passes this way despite its provider being
    off-list).

## How to Run
Standard image-viewing moves:
1. `vision_analyze(image_url=<absolute path or URL>, question=...)`
2. Read the tool-result first line to know which path fired (see above)
3. Oversized images are auto-downscaled to ≤1568px with a coordinate
   multiplier note; for small text / fine detail re-view a crop via
   `region=[x1,y1,x2,y2]` (original-image pixel coordinates) to keep
   full resolution

User-attached images (CLI / TUI / messaging platforms) follow
`agent.image_input_mode` too: `native` = pixels enter context directly.

## Quick Reference
```bash
# No args = current main provider/model; two args = any provider/model pair
<hermes-home>/hermes-agent/venv/bin/python <skill-dir>/scripts/check_vision_path.py
<hermes-home>/hermes-agent/venv/bin/python <skill-dir>/scripts/check_vision_path.py zai glm-5.3-flash
```

## Procedure: first-run setup

**Step 0 — Run the diagnostic script** (read-only, never mutates anything):

```bash
# Linux / macOS
<hermes-home>/hermes-agent/venv/bin/python \
  <skill-dir>/scripts/check_vision_path.py
# Windows
<hermes-home>\hermes-agent\venv\Scripts\python.exe <skill-dir>\scripts\check_vision_path.py
```

Completion criterion: a four-line verdict (provider/model, image input
route, tool-result images, model vision verdict, final path). Do not guess
your setup — read the output and fix line by line:

**Step 1 — `image input route` is `auto` or `text`** (most common trap):
In `auto` mode, an explicitly configured `auxiliary.vision` (any of
provider/model/base_url set) SUPPRESSES the main model's native vision —
every attached image gets paraphrased by the aux model even when the main
model is multimodal. Fix:

```bash
hermes config set agent.image_input_mode native
```

Rerun the script; expect `native`. (Alternative: remove the explicit
`auxiliary.vision` item instead — but keeping it is recommended, see
Pitfalls. If `text` was set deliberately, leave it.)

**Step 2 — `model vision verdict` is `None`** (fresh installs and
custom models): vision routing never touches the network, so a brand-new
install has no models.dev disk cache yet and known-multimodal models can
read as `None`. Two sub-cases:
- Fresh install → run Hermes normally once (any conversation) to populate
  `<hermes-home>/models_dev_cache.json`, then rerun the script.
- Model is genuinely multimodal but absent from the database (self-hosted,
  renamed, brand-new) → declare it explicitly:

```bash
hermes config set model.supports_vision true
```

Rerun; expect `True`. The config override has the highest priority and does
NOT depend on the cache — it is the most reliable channel for custom models.

**Step 3 — `model vision verdict` is `False`**: the model is classified as
text-only. If it is actually multimodal (new name, stale catalog), apply the
same `model.supports_vision true` override. If it is truly a text model,
change nothing — the auxiliary fallback is the designed behavior.

**Step 4 — Real-image acceptance test**: call
`vision_analyze(image_url=<a local image>, question=...)` and check the tool
result's first line:
- `Image loaded into your context — you can see it natively now.` → native
  path active; pixels are in context.
- `{"success": true, "analysis": "..."}` → still on the aux-fallback path;
  revisit Steps 1–2.

**Step 5 — Rollback / cost awareness**: undo with
`hermes config unset agent.image_input_mode`. Native images live in the
immutable conversation history and are re-sent every turn — long sessions
cost more tokens (that is the tradeoff for zero information loss).

## Procedure: self-check after switching models

1. Run the diagnostic script → completion criterion: the four-line verdict
2. Verdict native + model actually has vision → done; confirm once with a
   real image (Step 4)
3. Verdict fallback with `model vision verdict: None` while the model IS
   multimodal → `hermes config set model.supports_vision true` → rerun
   script = native
4. Verdict fallback and the model is text-only → nothing to do; confirm
   `auxiliary.vision.model` is non-empty so the fallback has a backend

## Pitfalls
- **`auto` mode + explicit `auxiliary.vision`** (any of provider/model/
  base_url set) overrides native vision — a maintainer decision recorded in
  `agent/image_routing.py` ("a user who named a dedicated vision model
  wants it used"). Fix with `image_input_mode: native` or by removing the
  explicit aux item.
- **Do not delete `auxiliary.vision`**: it backs the fallback paraphrase
  for text-only models, the legacy path when the fast path does not fire,
  and auto-describing images already in history
  (`run_agent.py::_describe_image_for_anthropic_fallback`). It is a
  safety net, not redundancy.
- The verdict data is cached on disk at `<hermes-home>/models_dev_cache.json`
  (usable at any age; hot paths never hit the network) → verdicts survive
  new sessions and offline use. Cache deleted AND offline → graceful
  fallback to aux paraphrase (no crash).
- Providers off the tool-result allowlist rely entirely on the models.dev
  verdict (or a manual `model.supports_vision` override).
- Browser / computer-use screenshots travel through their own `_multimodal`
  envelope paths (`tools/browser_use_cli.py`,
  `tools/computer_use/vision_routing.py`) — same
  "pixels into the main model" mechanism. The `vision_analyze` tool itself
  stays available in every `image_input_mode`.
- Agent file tools (patch / write_file) are blocked from editing
  `<hermes-home>/config.yaml` by a security guard — configure via
  `hermes config set/unset` instead; manual editors: keep a backup and
  2-space YAML indentation.
- Profiles are isolated: each Hermes profile has its own `config.yaml`
  (e.g. a bot profile) — apply these settings per profile as needed.
- The session's skill loader is cached at startup: a newly installed skill
  becomes visible in the NEXT session, not the current one.
- `native` with a text-only main model does not crash: API-time
  preprocessing strips/converts image parts (last-chance text fallback in
  `run_agent.py::_preprocess_anthropic_content` /
  `_strip_images_from_messages`).

## Troubleshooting
- **`Could not import Hermes internals (ModuleNotFoundError: ...)`** → the
  script could not load Hermes APIs. Two causes, in likelihood order:
  wrong interpreter — use the venv that belongs to your Hermes install
  (`venv/bin/python` on Linux/macOS, `venv\Scripts\python.exe` on
  Windows); or the Hermes data dir was not found — set
  `HERMES_HOME=/path/to/.hermes` and retry (the script falls back to
  `~/.hermes` when `$HERMES_HOME` does not exist).
- **`HERMES_HOME does not exist - falling back to ~/.hermes`** → you set
  `HERMES_HOME` to a path that is not there. Mind the asymmetry: the
  Hermes core uses such a path as-is (empty config), while this checker
  falls back and keeps diagnosing — fix the variable so the verdict
  matches the runtime.
- **Config edits not taking effect** → settings are re-read every turn, but
  mid-session capability overrides like `model.supports_vision` only apply
  on the next model resolution; rerun the script to confirm.
- **Provider rejects image content (HTTP 400) after declaring
  `supports_vision`** → the declaration does not match reality; undo with
  `hermes config unset model.supports_vision`.
- **Still seeing paraphrase-style results after native setup** → rerun the
  script; if it says native, the aux result you are looking at is from
  before the change — take a fresh screenshot/call.
- **Token cost spike in long vision-heavy sessions** → expected: native
  images re-send every turn. Downscale inputs or drop back to the fallback
  path when fidelity does not matter.

## Verification
- Diagnostic script output matches the decision matrix (multimodal = native;
  text-only or unknown = fallback)
- Real-image test: native first line present, no aux round-trip
- Config changes apply without restart (config is re-read every turn)
