---
sidebar_position: 11
title: "Pets (Petdex Mascots)"
description: "Adopt an animated mascot that reacts to agent activity across the CLI, TUI, and desktop app"
---

# Pets

Hermes can show an animated **pet** — a small mascot sprite that reacts to what
the agent is doing (idle, running a tool, thinking, finishing, failing) across
the **CLI**, **TUI**, and **desktop app**. Pets come from the public
[petdex](https://github.com/crafter-station/petdex) gallery.

Pet animation is cosmetic. It has **no effect on prompt caching, tokens, or the
agent's behavior** — the sprite is a display concern only. In the desktop app,
the popped-out pet can also show an optional action center. That surface sends a
control only after you explicitly click or type; it never approves, replies, or
interrupts work on its own. The feature is **off by default** and stays dormant
until you install and select a pet.

## How it works

- Pets are installed into your profile's `pets/` directory
  (`<HERMES_HOME>/pets/<slug>/`), so each [profile](/docs/user-guide/profiles) keeps its
  own set.
- Selecting a pet writes `display.pet.slug` and `display.pet.enabled` to
  `config.yaml` — nothing is stored as a secret or env var.
- Each surface watches the activity it already tracks and maps it to one of six
  animation states. The mapping lives in one place so every surface behaves the
  same:

  | Agent activity | Pet state |
  | --- | --- |
  | A tool/turn just failed | `failed` |
  | A plan finished (all todos done) | `jump` (celebrate) |
  | A turn finished cleanly | `wave` |
  | A tool is executing | `run` |
  | The model is thinking/reading | `review` |
  | Turn in flight (unspecified) | `run` |
  | Blocked on you (a clarify/approval prompt is open) | `waiting` (falls back to `idle` on legacy 8-row sheets) |
  | Nothing happening | `idle` |

## Rendering

In the terminal (CLI/TUI), Hermes renders the sprite at full fidelity when your
terminal supports a graphics protocol (**kitty**, **Ghostty**, **WezTerm**,
**iTerm2**, or **sixel**). Otherwise it falls back automatically to a truecolor
Unicode **half-block** rendering. Inside a pipe or redirect (no TTY), terminal
rendering is disabled by design.

The desktop app draws the pet as a floating sprite on a canvas and toggles it
from **Settings → Appearance**.

## Quick start (CLI)

```bash
# Browse the gallery (filter by substring)
hermes pets list
hermes pets list cat

# Install a pet and make it active in one step
hermes pets install boba --select

# Preview / animate it in your terminal (Ctrl+C to stop)
hermes pets show

# Check your setup
hermes pets doctor
```

## `hermes pets` commands

| Goal | Command |
| --- | --- |
| Browse the gallery | `hermes pets list [query] [--limit N]` |
| List installed pets | `hermes pets list --installed` |
| Install a pet | `hermes pets install <slug> [--select] [--force]` |
| Set the active pet | `hermes pets select [slug]` (omit slug for a picker) |
| Resize the pet everywhere | `hermes pets scale <factor>` (e.g. `0.5`, clamped 0.1–3.0) |
| Preview/animate | `hermes pets show [slug] [--state <s>] [--cycle] [--once] [--mode <m>] [--scale <f>]` |
| Disable the pet | `hermes pets off` |
| Remove an installed pet | `hermes pets remove <slug>` |
| Diagnose setup | `hermes pets doctor` |

`hermes pets show` flags:

- `--state` — play a single state (`idle`, `wave`, `run`, `failed`, `review`,
  `jump`).
- `--cycle` — cycle through every state.
- `--once` — play once instead of looping.
- `--mode` — override the render protocol (`kitty`, `iterm`, `sixel`,
  `unicode`, `auto`).
- `--scale` — override the on-screen scale (`0` = use config).

## `/pet` slash command

Inside the CLI and TUI you can manage the pet without leaving the session:

- `/pet` — toggle the pet on/off (adopts the first installed pet if none is
  active).
- `/pet list` — browse the gallery.
- `/pet scale <factor>` — resize the pet everywhere (e.g. `/pet scale 0.5`).
- `/pet <slug>` — adopt a specific pet.
- `/pet off` — disable the pet.

In the TUI, `/pet list` opens an interactive picker overlay; in the desktop app
it opens the Cmd+K pet palette.

## Generating a pet (`/hatch`)

Beyond installing pre-made pets from the gallery, Hermes can **generate a brand-new pet** from a text description — its own AI sprite-generation pipeline.

- CLI/TUI: `/hatch <description>` (alias `/generate-pet`), or `hermes pets` → the generate flow.
- Desktop app: the Pokédex-style **generate** UI — an animated egg, hatch FX, and a draft picker.

How generation works (a two-step, cost-bounded flow):

1. **Base drafts** — a handful of cheap, prompt-only "what should this pet look like" variants are generated. You pick one, or remix/retry for a fresh round.
2. **Hatch** — the chosen base is used as a reference image to generate one grounded animation row per Hermes state (idle, thinking, tool use, etc.), which are deterministically sliced into frames and packed into a standard petdex/Codex atlas (8×9 grid of 192×208 cells). The result is a valid spritesheet you keep — and could `petdex submit`.

### Image backend

Generation uses the active [image-generation provider](/user-guide/features/image-generation), but it requires **reference-image grounding** so each animation row stays the same character as the base. Reference-capable backends: **Nous Portal**, **OpenRouter**, **OpenAI** (`gpt-image-2`), and **Krea**. OpenRouter/Nous run a quality-first model chain by default.

- Resolution order prefers Nous Portal → OpenAI → OpenRouter.
- If no reference-capable backend is configured, generation surfaces an actionable error pointing you to `hermes tools` → Image Generation. (Installing/adopting existing gallery pets needs no image backend.)
- Override the backend with the `HERMES_PET_IMAGE_PROVIDER` env var (e.g. `HERMES_PET_IMAGE_PROVIDER=openrouter`).

## Desktop app

In the desktop app you can manage the pet two ways:

- **Cmd+K → "Pets…"** — browse, search, adopt, and toggle pets without leaving
  the keyboard (mirrors the theme picker).
- **Settings → Appearance** — the same gallery plus a **size slider** that
  resizes the floating mascot live as you drag.

Both adopt/toggle/resize the floating mascot in place — size changes apply
instantly; adopting a new pet lights it up within a moment.

### Pop-out overlay

**Shift-click** the floating pet to pop it out into its own transparent,
always-on-top desktop window. Out there it stays visible while Hermes is
minimized (Codex-style), so a glance tells you what the agent is doing.

Gestures once it's popped out:

| Gesture | Action |
| --- | --- |
| **Drag** | Move the pet anywhere on screen, even outside the app. Its spot and in/out state persist across restarts. |
| **Single-click** | Open a mini composer to send a prompt to the most recent session — without surfacing the app. |
| **Double-click** | Toggle the app window: minimize it if it's up front, restore it if it's hidden. |
| **Shift-click** | Pop the pet back into the window. |
| **Mail icon** | Appears only when a turn finished while you were away; click to raise the app on the most recent thread (and mark it read). |
| **Action center button** | Open pending approvals, questions, and live-session controls without raising the app. It appears only when there is something it can show. |

Only the popped-out pet shows a **speech bubble** (`working…`, `thinking…`,
`your turn`, …) — in-window the app itself is the surface, so the pet stays
quiet there.

The overlay is a pure puppet of the in-app pet — it carries no separate gateway
connection and never appears in the dock or app switcher. The main app owns all
gateway connections, validates every action-center intent again, and routes it
to the exact profile and session that produced the item.

### Action center

The action center is a compact, keyboard-accessible control surface for work
that is already visible to the desktop app. Incoming work can make its button
appear, but **never opens the panel or steals focus**. Click the button to enter
the panel.

Each item shows its profile and session label plus a high-level live status:

| Status | Meaning |
| --- | --- |
| **Idle** | The selected session is ready for a new prompt. |
| **Working** | A turn is running. |
| **Reviewing** | The agent is reasoning or reviewing; private reasoning text is not shown. |
| **Waiting** | The turn is blocked on an approval or a question for you. |
| **Done** | The turn completed and its notification can be dismissed. |
| **Failed** | The turn ended with an error; open the app for full context. |

The strip may also show elapsed time, a safe high-level tool name, queued-message
count, and whether that profile's connection is connecting, offline, or in
error. It does not show tool arguments, tool output, transcripts, reasoning
content, credentials, tokens, or logs.

#### Approvals and questions

- **Run once** approves only the pending operation.
- **Allow this session** applies only to that session.
- **Always allow** is shown only when the backend says permanent approval is
  supported. It requires a second confirmation.
- **Reject** can include an optional reason. The reason is returned to the
  blocked turn; it is not a private note.
- Clarify questions support backend-provided choices, free text, and **Skip**
  when skipping is allowed.
- Password, sudo, and secret-entry prompts are intentionally excluded. The
  action center tells you to finish those secure inputs inside Hermes.

An approval or answer is always bound to the profile and runtime session that
created it. If that request has already resolved, the action center reports it
as stale instead of applying the response somewhere else.

#### Reply, steer, queue, and stop

Available buttons depend on the selected item's current capabilities:

- **Send** starts a new prompt in an idle session. If the runtime session went
  stale, Hermes may safely resume the same verified stored session in the same
  profile and retry once.
- **Steer** tries to alter a running turn while the backend still has a safe
  steering window.
- **Queue** stores the message for that exact profile/session. The shared
  composer sends queued messages in FIFO order when that session is active and
  ready. Queue remains available while that profile is temporarily offline.
- A rejected **Steer** never silently becomes a queued message. Your draft is
  kept and the UI offers an explicit **Queue** button.
- **Stop** interrupts only the exact currently running runtime session. Hermes
  does not resume a stale session merely to stop it.
- **Open in app** raises Hermes on the verified stored session when one exists.
- **Dismiss** acknowledges only the selected completion; unresolved approvals
  and questions remain in the action center.

Two profiles may use identical session IDs without sharing prompts, queues,
status, or controls. Identity is carried internally as a profile/session pair;
the overlay sends only an opaque item ID back to the main app.

#### Keyboard and focus

- **Tab / Shift+Tab** move through the open panel and stay inside its focus
  trap.
- **Enter** submits an idle prompt or the explicitly labelled default live
  action. **Shift+Enter** inserts a newline.
- **Escape** closes the current inline confirmation first, then the action
  center. Focus returns to the button that opened it.
- New background events do not open the panel, move focus, or navigate the app.

The native overlay grows to fit the open panel and returns to its compact size
when it closes, keeping the pet's feet anchored. Bounds are clamped to the
nearest display work area after a monitor is disconnected or its resolution
changes.

## Configuration

All settings live under `display.pet` in `config.yaml`:

```yaml
display:
  pet:
    enabled: false        # master on/off (true once you select a pet)
    slug: ""              # active pet; empty = first installed
    render_mode: auto      # auto | kitty | iterm | sixel | unicode | off
    scale: 0.33           # master size knob (relative to native 192x208 frames)
    unicode_cols: 0       # hard override for terminal width (0 = derive from scale)
```

- **`scale`** is the single master size knob. One number shrinks every surface:
  the desktop canvas scales its pixels by it, and the CLI/TUI derive their
  terminal column width from it. The half-block fallback clamps to a legibility
  floor — it can't shrink as far as true-pixel kitty/GUI rendering without
  turning to mush, so the same `scale` looks crisp under kitty but is floored in
  half-blocks.
- **`render_mode: auto`** detects kitty/iTerm2/sixel and falls back to unicode
  half-blocks. Set it explicitly to force a protocol or `off` to disable
  terminal rendering while keeping the pet on the desktop.
- **`unicode_cols`** pins the terminal column width independently of `scale`;
  leave it at `0` to derive width from `scale`.

## Troubleshooting

Run `hermes pets doctor` — it reports:

- the pets directory and which pets are installed,
- `display.pet.enabled`, `display.pet.slug`, and the resolved active pet,
- the configured `render_mode`, the detected terminal graphics protocol, and the
  effective mode for a TTY,
- whether Pillow (used for sprite decoding) is importable.

It prints `✓ ready` once a pet is installed, selected, enabled, and Pillow is
available.

Common gotchas:

- A pet only shows once one is **installed AND selected** (`enabled: true`).
- Inside a pipe/redirect (no TTY), terminal rendering is disabled by design.
- The petdex npm CLI installs to `~/.codex/pets`; Hermes uses its own
  profile-scoped `<HERMES_HOME>/pets/` instead — install through `hermes pets`.
- If the action center says a request is stale, it was already resolved or its
  live runtime ended. Open the exact session rather than retrying the old
  control.
- **Offline** disables actions that require the live gateway (Send, Steer,
  Stop). Local **Queue** and completion dismissal remain available.
- Secure password, sudo, and secret prompts never render their contents in the
  pet; open Hermes to complete them.
- Near a screen edge, Hermes expands the panel upward around the pet and clamps
  it to the nearest monitor's usable work area.

## See also

- The [`petdex` skill](../skills/bundled/productivity/productivity-petdex.md)
  lets the agent install and switch pets for you on request.
