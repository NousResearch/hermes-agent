---
title: "Tldraw Offline — Drive and script tldraw offline canvases with an agent"
sidebar_label: "Tldraw Offline"
description: "Drive and script tldraw offline canvases with an agent"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Tldraw Offline

Drive and script tldraw offline canvases with an agent.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/tldraw-offline` |
| Path | `optional-skills/creative/tldraw-offline` |
| Version | `1.0.0` |
| Author | Teknium + Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `tldraw`, `canvas`, `whiteboard`, `document-script`, `diagramming` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# tldraw offline Skill

Work with the tldraw offline desktop app (offline.tldraw.com): read the open
canvas, make edits, and write **document scripts** — JavaScript embedded in a
`.tldraw` file that runs on load and gives the file durable behavior. The app
runs a **local HTTP API** (default `localhost:7236`) that a coding agent drives
with plain `curl` from its terminal — this is exactly how the app's own homepage
demo (Codex editing a canvas live) works. The agent does NOT use computer-use /
GUI clicking, and does NOT hand-edit the `.tldraw` file directly. Keep tldraw
offline open while you work.

## When to Use

- The user has tldraw offline open and asks you to build or modify a canvas
  (diagrams, wireframes, layouts).
- You want to add durable behavior to a drawing (reactive shapes, interactive
  buttons, animation, connection logic) via an embedded document script.

Do NOT hand-place shapes to imitate a drawing — write the code that generates
them. Agents are far better at scripting the canvas than at drawing on it.

## Prerequisites

- **tldraw offline installed and running**, with a document open. Releases:
  https://github.com/tldraw/tldraw-offline/releases/latest (macOS DMG, Windows
  x64/Arm64, Linux `x86_64`/`arm64` AppImage or amd64/arm64 `.deb`).
- **Agent skills installed in the app**: `Develop → Install Agent Skills`. The
  app writes its own tldraw skill into `~/.codex/skills/`, `~/.claude/skills/`,
  `~/.cursor/skills/`, and `~/.gemini/skills/` — teaching that agent the `curl`
  recipes below. (This Hermes skill mirrors that guidance for Hermes.)
- **The local control API.** On launch the app writes `server.json` to its config
  dir (Linux `~/.config/tldraw/`, macOS `~/Library/Application Support/tldraw/`,
  Windows `%APPDATA%\tldraw\`) with `port` (default `7236`), a bearer `token`,
  `pid`, and `startedAt`. Every request except `GET /` needs
  `Authorization: Bearer <token>`. A clean quit removes `server.json`; if it's
  present but the port doesn't answer, the app quit uncleanly — treat as not
  running.
- **Re-read port + token on EVERY shell call.** Each terminal call is a fresh
  shell, so an `export`ed token does not persist — "export once and reuse" sends
  an empty token and 401s. Read both inline at the top of each call:
  `PORT=$(jq -r .port <server.json>); TOKEN=$(jq -r .token <server.json>)`.
- No account or network needed for local editing.

## How to Run

Two distinct workflows. Pick by whether the change must survive a reload.

**A. One-off canvas edits (`/exec`)** — layout, generating shapes, cleanup. This
is a live edit, not saved script:

```bash
BASE=http://localhost:7236
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.config/tldraw/server.json'))['token'])")
# find the focused document id
DOC=$(curl -s "$BASE/api/search" -X POST -H 'content-type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"code":"return (await api.getFocusedDoc()).id"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['result'])")
# run code with the live `editor` + `helpers` in scope
curl -s "$BASE/api/doc/$DOC/exec" -X POST -H 'content-type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"code":"const {createShapeId,toRichText}=await import(\"tldraw\"); editor.createShape({id:createShapeId(),type:\"geo\",x:0,y:0,props:{geo:\"rectangle\",w:200,h:100,color:\"blue\",fill:\"solid\",richText:toRichText(\"hello\")}}); return editor.getCurrentPageShapes().length"}'
```

**B. Durable behavior (`script/main.js`)** — reactive/interactive logic that must
survive reload. Edit the file on disk; the app's watcher applies it:

```bash
# get the live script file path for the doc
curl -s "$BASE/api/doc/$DOC/script-workspace" -X POST \
  -H "Authorization: Bearer $TOKEN"          # -> result.mainJsPath, result.isDefaultScript
# edit result.mainJsPath with read_file / patch / write_file (see scripts/main.js)
# then confirm the watcher applied it:
curl -s "$BASE/api/doc/$DOC/script-status" -H "Authorization: Bearer $TOKEN"
```

The ready-to-adapt document script is `scripts/main.js`.

## Quick Reference

The document-script contract (verified against the app's bundled
`script-context.d.ts`):

```js
import { createShapeId, toRichText } from 'tldraw'   // primitives: import, not globals

export default function ({ editor, helpers, signal }) {
  editor.run(() => {                                 // batch = one undo step
    helpers.createShapeIfMissing({                   // idempotent furniture
      id: createShapeId('node-1'), type: 'geo', x: 0, y: 0,
      props: { geo: 'rectangle', w: 200, h: 100, richText: toRichText('hi') },
    })
  })

  const stop = editor.store.listen(() => { /* react */ })  // fires the tick AFTER a commit
  signal.addEventListener('abort', () => stop())           // REQUIRED cleanup on rerun/close
}
```

- `ctx.editor` — the live `Editor` (`createShape`, `updateShape`, `deleteShapes`,
  `getCurrentPageShapes`, `getShape`, `getBindingsFromShape`, `zoomToFit`,
  `on('tick'|'event', fn)`, `run(fn, { history: 'ignore' })`).
- `ctx.helpers` — `createShapeIfMissing`, `createShapesIfMissing`,
  `createArrowBetweenShapes(from, to, { arrowheadEnd })`, `translateShapes`,
  `onShapeTranslate(id, fn, { signal })`, `richTextToPlainText`, `boxShapes`,
  `getLints`.
- `ctx.signal` — `AbortSignal`; attach every listener/interval teardown to it.
- `config.js` (separate file) registers custom shape/tool/component utils and
  runs before mount; `main.js` runs against the mounted editor and reruns on save.

## Interactive UI (clickable buttons that drive state)

Drawn shapes can behave like a real app — the thing a static whiteboard can't do.
Full example: `scripts/counter.js` (a number display + MINUS/RESET/PLUS buttons).

Verification boundary — read this before claiming interaction works or doesn't.
The app's OWN agent playbook says to verify a clickable-UI script with "one
simulated click and one state read" via `/exec` (`editor.dispatch` a pointer
event, await a tick, read the shape's state) — NOT by driving a real mouse. By
that standard the counter is verified: dispatched clicks stepped it
`0 → 1 → 2 → 1 → 0`. Two caveats worth writing down:
- **The script only runs once the app's file-watcher applies it.** On Linux that
  watcher uses inotify; a host with an exhausted `fs.inotify.max_user_instances`
  logs `inotify_add_watch ... No space left on device`, `script-status` shows
  `state: "not-watching"` / `hasEntry: false`, and the script never executes.
  This is a host limit, not a script bug — a normal desktop is unaffected.
- **Driving the canvas with computer-use needs FOREGROUND delivery, not
  background.** cua-driver's default `background` delivery returns
  `background_unavailable` against Electron's occluded renderer — but that is the
  first rung, not a wall. cua-driver returns an `escalation: "foreground"` hint;
  set `delivery_mode: "foreground"` (pair with `bring_to_front`) and its X11 XTest
  path (`x11_xtest_fg`) clicks land — this dismisses the "Run Script" consent
  dialog and clicks the canvas. Verified by the Cua team on this exact v1.11.0
  AppImage (Linux/X11). Do NOT conclude "Electron rejects synthetic clicks" and
  give up in background mode — climb to foreground. (The real product path is
  still `/exec`, not clicks; this note is for computer-use-driven testing.)

The pattern:

```js
export default function ({ editor, helpers, signal }) {
  // 1. Build buttons idempotently; tag each with meta so the handler finds them.
  //    Give buttons a visible label AND a meta.action.
  // 2. Hit-test pointer_down in PAGE coordinates against the button bounds:
  const inside = (b, p) => p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h
  function onEvent(info) {
    if (!info || info.name !== 'pointer_down') return
    let p = null
    try { if (info.point && editor.screenToPage) p = editor.screenToPage(info.point) } catch {}
    p = p ?? editor.inputs?.currentPagePoint
    if (!p) return
    const hit = editor.getCurrentPageShapes().find(
      (s) => s.meta?.ui === 'button' &&
        inside({ x: s.x, y: s.y, w: s.props.w, h: s.props.h }, p)
    )
    if (hit) runAction(hit.meta.action)   // mutate state; store it in a shape's meta
  }
  editor.on('event', onEvent)
  signal.addEventListener('abort', () => editor.off('event', onEvent))  // REQUIRED
}
```

- Find buttons by `meta` (or visible label via `helpers.richTextToPlainText`),
  not by hard-coded coordinates.
- **One script owns both build and read.** If the shapes are created by one code
  path (with `meta.action: 'inc'`) and the handler reads another convention
  (`meta.action === 'PLUS'`), clicks silently do nothing. Ship the buttons built
  by the same script that handles them, or ship an empty canvas so the script
  builds them fresh — never pre-bake mismatched shapes into the file's db.
- Keep app state in a shape's `meta` (e.g. `meta.count`) and render it as that
  shape's `richText` label, so it survives save and is readable for verification.
- **Detach the listener on `signal` abort.** Skipping this is not cosmetic: on
  the next save the old `onEvent` stays attached alongside the new one, so every
  click fires twice and a counter jumps by 2 instead of 1.
- For continuous motion use `editor.on('tick', fn)`; for a moving anchor with
  attached pieces use `helpers.onShapeTranslate(id, fn, { signal })`.

### Shipping a self-running scripted `.tldraw`

A `.tldraw` is a zip of `metadata.json` + `session.json` + `db.sqlite` + `assets/`
+ `script/` (only those entries are packable). For the script to auto-run without
the "This document contains a script → Run Script" consent dialog:

- `metadata.json` must carry a `script` manifest: `{ "sha256": "<digest>" }`, where
  the digest is `sha256` over each sorted `script/` path as `` `${path}\0${sha256hex(bytes)}\n` ``.
  A mismatch is rejected as tampered.
- Pre-trust the digest by adding it to `~/.tldraw/script-trust.json`
  (`{ "trusted": ["<digest>"] }`, or `$TLDRAW_SCRIPT_TRUST`). The app skips consent
  when `isScriptTrusted(digest)` is true.

## Procedure

1. Read the current token/port from `server.json`. Find the target doc with
   `api.getFocusedDoc()` (or `api.getDocs()`); name it explicitly if several are
   open.
2. For layout/generation, use `/exec`. For durable behavior, edit
   `script/main.js` via `/script-workspace`.
3. Make scripts idempotent: create durable shapes with `helpers.createShapeIfMissing`
   and stable `createShapeId('name')` ids. Scripts rerun on every load.
4. Keep script-owned writes out of the user's undo stack:
   `editor.run(fn, { history: 'ignore' })` (or `helpers.translateShapes`, which
   already does).
5. For reactivity, `editor.store.listen(cb)` and tear it down on `signal` abort.
   For interaction, `editor.on('event', h)` (hit-test `pointer_down` in page
   coords); for animation, `editor.on('tick', h)`.
6. For a single moving anchor + attached internals, prefer
   `helpers.onShapeTranslate(anchorId, fn, { signal })` over a broad store
   listener — a broad listener can turn your own writes into feedback loops.

## Shape props (validated against tldraw SDK v5 schema)

`editor.createShape` / `createShapeIfMissing` accept partial props (shape utils
fill defaults). When building **raw records** for a file snapshot, every prop
below is required (run `scripts/validate_shapes.mjs`):

| Shape | Required props |
|-------|----------------|
| `note`  | `richText`, `color`, `labelColor`, `size`, `font`, `align`, `verticalAlign`, `growY`, `fontSizeAdjustment`, `url`, `scale`, `textLastEditedBy` |
| `text`  | `richText`, `color`, `size`, `font`, `textAlign`, `w`, `scale`, `autoSize` |
| `frame` | `w`, `h`, `name`, `color` |
| `geo`   | `geo`, `w`, `h`, `color`, `fill`, `richText` (+ dash/size/etc. defaulted) |

`richText` must be `toRichText('...')` — a bare string is rejected. `color` enum:
`black grey light-violet violet blue light-blue yellow orange green light-green
light-red red white`. `font` enum: `draw sans serif mono`.

## Pitfalls

- **`store.listen` fires on the tick AFTER a commit, not synchronously.** If you
  write a shape and immediately read state expecting the listener to have run, it
  hasn't. Verified live: an in-turn read shows 0 fires; after one `setTimeout`
  tick it shows 1. Same reason the app notes `editor.dispatch` is async — await a
  tick before verifying.
- **`ctx`, not globals.** The entry is `export default function ({ editor,
  helpers, signal })`. There is no bare `editor` global in a document script.
  `createShapeId` / `toRichText` / `Vec` come from `import ... from 'tldraw'`.
- **`richText`, not `text`.** Text/note/geo labels use `richText: toRichText(s)`.
- **Raw records need every prop; `createShape` does not.** In-app pass only the
  props you care about; a hand-built `.tldraw` snapshot needs the full set (table).
- **Scripts rerun on every load — be idempotent.** Use `createShapeIfMissing`
  with stable ids or you duplicate content and clobber user edits.
- **Clean up on `signal`.** `signal.addEventListener('abort', () => stop())` for
  every `store.listen` / `editor.on` / `setInterval`; the signal fires before
  rerun and on close.
- **Keep script writes out of undo:** `editor.run(fn, { history: 'ignore' })`.
- **`editor.on('tick')` pauses when the window is hidden** (it is a RAF loop);
  `setInterval` keeps firing but Electron throttles it to ~1/s in the background.
- **The API needs the bearer token** from `server.json`; the port can be non-default
  (`server.listen(0)` picks one) — always read the file, don't hardcode `7236`.
- **Only `tldraw` / `react` / `react-dom` import** — not a Node project.

## Verification

- **Shape schema (offline, no app):** `node scripts/validate_shapes.mjs` — builds
  the real tldraw schema and validates note/text/frame. Passing prints `3/3`.
- **Live canvas edits:** after `/exec`, read back with `/api/search` →
  `api.getShapes(docId)` (returns `{ page, viewport, shapes }`) and
  `api.getBindings(docId)` (array). Confirm expected shapes/bindings exist. Grab
  `api.getScreenshot(docId)` (returns `{ filePath, ... }`) and inspect the PNG/JPEG
  with `vision_analyze`.
- **Durable script applied:** `GET /api/doc/:id/script-status`. Success is
  `state: "applied"` (`currentDiskDigest === lastAppliedDigest === manifestSha256`,
  `pendingApply === false`, `lastApplyError === null`). If it stays `"pending"`
  after a short retry, report that instead of claiming success; `"error"` means
  the apply failed — read `errorLogPath`.
