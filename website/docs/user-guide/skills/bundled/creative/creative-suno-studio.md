---
title: "Suno Studio — Automate Suno song and DJ-set generation in the browser"
sidebar_label: "Suno Studio"
description: "Automate Suno song and DJ-set generation in the browser"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Suno Studio

Automate Suno song and DJ-set generation in the browser.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/creative/suno-studio` |
| Version | `1.0.0` |
| Author | Jonathan Korstad (Jpalmer95), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `suno`, `music`, `generation`, `browser`, `automation`, `dj`, `prompt-engineering` |
| Related skills | [`songwriting-and-ai-music`](../../bundled/creative/creative-songwriting-and-ai-music.md), [`heartmula`](../../optional/creative/creative-heartmula.md), [`comfyui`](../../optional/creative/creative-comfyui.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Suno Studio Skill

Automates single-song and multi-track DJ-set generation on `suno.com/create` using Hermes browser automation (`browser_exec`) and a persistent Chromium CDP session — no Suno API key required. This skill owns the cross-platform CDP setup, React controlled-input batch filling, parallel DJ queue execution, and track link harvesting; load `songwriting-and-ai-music` for lyric craft, `[metatags]`, and phonetic rules.

## When to Use

- User asks to "make me a song about..." or "generate music on Suno for..."
- User describes a mood, scenario, or vibe arc and wants playable AI-generated tracks or a 4–8 track DJ set
- User wants to batch-fill Suno's Advanced tab, queue back-to-back variations, or extend/remix an existing Suno song
- Don't use for: offline/local GPU music generation without a browser (use `heartmula`) or lyric-only drafting when no Suno browser session is requested (use `songwriting-and-ai-music` directly).

## Prerequisites

1. **Browser toolset enabled** (`browser_exec`) attached to a persistent Chromium-family browser (Brave, Chrome, Chromium, or Edge) on port `9222` so Suno authentication survives across sessions.
2. **Cross-platform CDP browser launch** (Linux, macOS, and Windows) via the bundled helper script using `terminal`:
   ```python
   terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py cdp-launch --port 9222", timeout=30)
   ```
   Or launch manually per platform:
   - **Linux / macOS:**
     ```bash
     brave-browser --remote-debugging-port=9222 --user-data-dir="$HOME/.hermes/chrome-debug" --no-first-run --no-default-browser-check &
     ```
   - **Windows (PowerShell):**
     ```powershell
     Start-Process "chrome.exe" -ArgumentList "--remote-debugging-port=9222","--user-data-dir=$env:USERPROFILE\.hermes\chrome-debug","--no-first-run","--no-default-browser-check"
     ```
   Attach Hermes with `/browser connect http://127.0.0.1:9222` or `hermes config set browser.cdp_url http://127.0.0.1:9222`.
3. **One-time manual login**: Log into `https://suno.com` once in the `chrome-debug` browser window; session cookies persist in the profile directory.
4. **Songwriting reference**: Load `skill_view(name="songwriting-and-ai-music")` before drafting style arcs or `[metatags]`.

## How to Run

Validate prompts, generate React-safe batch-fill IIFEs, and parse harvested track links with `scripts/suno_helper.py` via `terminal`, and drive `suno.com/create` with `browser_exec`:

```python
# 1. Check or launch the persistent CDP browser (Linux / macOS / Windows)
terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py cdp-status --port 9222", timeout=15)

# 2. Validate Title, Style (<=1000 chars, no artist names), and Lyrics (<=3000 chars, [metatags])
terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py validate-prompt --title 'Midnight Neon Drift' --style-file style.txt --lyrics-file lyrics.txt", timeout=15)

# 3. Emit an escaped IIFE for browser_exec js(...) that sets Title, Lyrics, and Style via native React setters
terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py emit-fill-js --title 'Midnight Neon Drift' --style-file style.txt --lyrics-file lyrics.txt", timeout=15)

# 4. Parse harvested `Title | https://suno.com/song/<id>` output into structured JSON variations
terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py parse-harvest --input harvested.txt", timeout=15)
```

## Quick Reference

| Action | Command / Tool Call |
|---|---|
| Check CDP browser on port 9222 | `terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py cdp-status", timeout=15)` |
| Launch CDP browser (cross-platform) | `terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py cdp-launch", timeout=30)` |
| Validate prompt constraints | `terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py validate-prompt --title '...' --style '...' --lyrics '...'", timeout=15)` |
| Emit batch-fill IIFE JS | `terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py emit-fill-js --title '...' --style-file s.txt --lyrics-file l.txt", timeout=15)` |
| Emit Create-button click JS | `terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py emit-click-js", timeout=15)` |
| Emit `/song/` link harvester JS | `terminal(command="python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py emit-harvest-js", timeout=15)` |
| Open Suno & inspect state | `browser_exec`: `new_tab("https://suno.com/create")`, `wait_for_load()`, `page_info()` |
| Visually poll generation spinners | `browser_exec`: `capture_screenshot()` + `vision_analyze` |

### Prompt Engineering Cheatsheet

- **Style Formula (up to 1,000 chars):** `Genre + Mood + Era + Instruments + Vocal Persona + Production + Dynamic Arc`
  - *Dynamic Arc Starter:* `"Begins whisper-quiet with sparse analog piano, layers in tribal percussion through verse 1, explodes in the chorus with full brass and driving bass, strips to solo cello in the bridge, and fades on detuned tape hiss."`
  - *Vocal Persona:* `"Weathered baritone with gravel and warmth, speak-sings verses, opens to full chest voice in the chorus."` / `"Crystal soprano with breathy intimacy in verses and belted chorus vibrato."`
- **Genre Blends:** `"lo-fi hip hop jazz"`, `"Appalachian folk gothic"`, `"synthwave darkwave retro"`, `"bossa nova neo-soul"`, `"orchestral dubstep cinematic"`, `"chamber pop art rock"`.
- **Advanced Options:** Vocal Gender (`M`/`F`/`None`), Weirdness (`0–100`, default `30`), Style Influence (`0–100`, default `50`), Exclude Styles (e.g., `"autotune, mumble rap"`).

## Procedure

### 1. Parse intent and load songwriting rules

Extract the song concept, mood, genre blend, vocal persona, target duration, and mode (**Single Song** vs. **4–8 track DJ Mode**). If the request is vague, ask at most 1–2 clarifying questions via `clarify`. Call `skill_view(name="songwriting-and-ai-music")` for `[metatag]` syntax and phonetic delivery rules. For DJ Mode, plan an energy and BPM/key arc first (keep adjacent tracks within ±10 BPM and in relative keys such as `Am → C` or `Am → Em`):
- **Track 1 (Intro, Energy 2–3)** → **Track 2 (Warm-up, 4–5)** → **Track 3 (Build, 6–7)** → **Track 4 (Peak 1, 8–9)** → **Track 5 (Breather, 5–6)** → **Track 6 (Peak 2, 9–10)** → **Track 7 (Wind-down, 4–5)** → **Track 8 (Outro, 2–3)**.
Done when the track plan or single-song brief is defined and `songwriting-and-ai-music` is loaded.

### 2. Engineer and validate the Suno prompt

Draft three fields for each song:
- **Title:** Evocative, `<= 8` words.
- **Style of Music (`<= 1,000` chars):** Describe the sonic palette, BPM, key, vocal persona, and section-by-section dynamic journey. Never include artist names or trademarks.
- **Lyrics (`<= 3,000` chars):** Include structural `[metatags]` (`[Intro]`, `[Verse 1]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Final Chorus]`, `[Outro]`), keep to `5–8` tags per section with no contradictory tags in the same section, spell out all numbers (`"twenty four seven"`), and space acronyms (`"A I"`).
Run `suno_helper.py validate-prompt` via `terminal`. Done when `validate-prompt` exits `0` with `"valid": true` and zero errors.

### 3. Attach to Suno and switch to the Advanced tab

Verify the CDP browser is listening with `suno_helper.py cdp-status` (or start it with `cdp-launch`), then open `https://suno.com/create` in `browser_exec`. Check `page_info()` to confirm the session is authenticated (if a login gate or CAPTCHA appears, pause and ask the user to complete it in the persistent CDP window). Switch the create UI to the **Advanced** tab (labeled `Advanced` on Suno v5.5, replacing the older "Custom Mode" toggle). See `references/browser-automation-patterns.md` for the exact selector snippets. Done when the Advanced tab is active and the Title, Lyrics, and Style inputs are present in the DOM.

### 4. Batch-fill fields and trigger generation

Generate the batch-fill IIFE with `suno_helper.py emit-fill-js` and execute it via `js(...)` inside `browser_exec` (using native `HTMLInputElement.prototype` and `HTMLTextAreaElement.prototype` value setters plus bubbling `input`/`change` events so React state updates cleanly). Then click the Create button using `suno_helper.py emit-click-js`.
- **DJ Mode parallel-prep technique:** Do **not** wait for Track `N` to finish rendering before filling Track `N+1`. Immediately after clicking Create on Track `N`, batch-fill Track `N+1`'s Title/Lyrics/Style and click Create while Track `N` is still generating — this cuts multi-track set wall-clock time roughly in half.
Done when the batch-fill call returns `true` for `title`, `lyrics`, and `style` and the Create button click reports `"clicked": true, "disabled": false`.

### 5. Poll completion and harvest both variations

Suno v5.5 generates **2 variations per click** over **60–120+ seconds**, and the two takes often finish 30–90 seconds apart. Poll every 30–60 seconds using `capture_screenshot()` in `browser_exec` and `vision_analyze` to confirm cover art and duration timestamps have replaced the loading spinners. Once complete, run the harvester snippet (`suno_helper.py emit-harvest-js`) inside `browser_exec` to collect all `a[href*="/song/"]` links, parse them with `suno_helper.py parse-harvest`, and present **both** variations per track to the user alongside follow-up options (extend song, generate stems, remix, or queue a sequel set). Done when both variation URLs per requested song are verified and reported to the user.

## Pitfalls

1. **Looking for a "Custom Mode" toggle on v5.5.** Suno v5.5 uses three top tabs (`Simple | Advanced | Sounds`). Click the **Advanced** tab to expose separate Title, Lyrics, and Style fields.
2. **Setting React inputs with plain `el.value = ...` or sequential typing.** Direct `.value` assignment bypasses React's synthetic event system and leaves the Create button disabled or submits blank fields. Always use the native prototype value setter (`HTMLInputElement.prototype` for Title/sliders, `HTMLTextAreaElement.prototype` for Lyrics/Style) inside a single IIFE and dispatch `{ bubbles: true }` `input` and `change` events.
3. **Top-level `const`/`let` redeclaration across console calls.** Evaluating bare `const x = ...` twice in the same page context throws `SyntaxError: Identifier has already been declared`. Wrap every JS snippet in `(() => { ... })()`.
4. **Artist names or trademarks in the Style field.** Suno blocks prompts naming real artists or franchises (`"Nirvana style"`, `"James Bond theme"`). Describe the instrumentation, era, and vocal texture instead (`"90s Seattle grunge"`, `"1960s Cold War spy brass"`).
5. **Unexpanded numbers or acronyms in lyrics.** `"24/7"` or `"AI"` is mispronounced by AI vocalists; spell out `"twenty four seven"` and space `"A I"`.
6. **Contradictory or excessive `[metatags]`.** Combining `[Calm]` and `[Aggressive]` in the same section or exceeding 8 tags per section confuses the model.
7. **Assuming both variations finish together or share a predictable duration.** Two takes from the same prompt can finish 60 seconds apart and span wildly different lengths (e.g., `3:03` vs `7:59`). Present both variations so the user can pick the shorter take for standalone listening or the longer take for DJ mixing.
8. **Rate limits and credit exhaustion.** Free tier allows ~10 songs/day; Pro allows 50–200/day. If credits are exhausted, report remaining credits and offer `heartmula` for local GPU generation.

## Verification

- [ ] `songwriting-and-ai-music` skill loaded before drafting prompts and `[metatags]`.
- [ ] `python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py validate-prompt` exits `0` with no artist names, style `<= 1,000` chars, lyrics `<= 3,000` chars, and numbers spelled out.
- [ ] Persistent CDP browser confirmed listening on port `9222` (`cdp-status`) and switched to the **Advanced** tab on `suno.com/create`.
- [ ] Title, Lyrics, and Style batch-filled via a single native-setter IIFE (`emit-fill-js`), returning `true` for all required fields.
- [ ] Create clicked via JS (`"clicked": true, "disabled": false`) and generation polled via `capture_screenshot()` / `vision_analyze` until spinners clear.
- [ ] Both `/song/` variations per track harvested (`emit-harvest-js` + `parse-harvest`) and delivered to the user.
