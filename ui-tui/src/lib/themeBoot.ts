/**
 * Flash-free theme boot — the TUI port of the desktop app's
 * `hermes-boot-background` / `hermes-boot-color-scheme` localStorage keys.
 *
 * Theme resolution is asynchronous by nature (gateway skin arrives after
 * connect; the OSC-11 background probe answers after the first frame; the
 * config mode pin arrives with config sync), so without a cache every launch
 * repaints through default-dark → skin → detected-mode. This module persists
 * the LAST RESOLVED theme + background to disk and replays them as the very
 * first frame, so a stable setup renders correctly from paint one and the
 * async signals merely confirm it.
 *
 * The cache is a hint, never an authority: explicit env pins beat it, and
 * every later signal overwrites it (then persists the new answer).
 */

import { readFileSync, renameSync, writeFileSync } from 'fs'
import { homedir } from 'os'
import { join } from 'path'

import type { Theme } from '../theme.js'

interface BootThemeFile {
  /** The resolved background hex that detection settled on, if any. */
  background?: string
  /** The config mode pin (`display.tui_theme`) active when this cache was
   *  written. Without it a pinned theme caches incoherently — a light
   *  resolved theme next to the dark PHYSICAL background — and the next
   *  launch flashes light → dark (skin resolves against the seeded
   *  background) → light (config pin rehydrates). */
  mode?: 'dark' | 'light'
  /** The fully-resolved Theme (palette + brand) from the last session. */
  theme?: Theme
  version: 1
}

// Profile-aware: the Python launcher exports HERMES_HOME (set by
// _apply_profile_override) before spawning the TUI. Falling back to
// ~/.hermes matches get_hermes_home()'s default.
const bootFilePath = () => join(process.env.HERMES_HOME ?? join(homedir(), '.hermes'), 'tui-theme-boot.json')

// Never touch the user's real ~/.hermes from test runs (the TS suite has no
// HERMES_HOME isolation fixture).
const isTestRun = () => !!process.env.VITEST || process.env.NODE_ENV === 'test'

const looksLikeTheme = (value: unknown): value is Theme => {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const theme = value as Partial<Theme>

  return (
    typeof theme.color === 'object' &&
    theme.color !== null &&
    typeof theme.color.text === 'string' &&
    typeof theme.color.primary === 'string' &&
    typeof theme.brand === 'object' &&
    theme.brand !== null &&
    typeof theme.brand.name === 'string'
  )
}

export interface BootTheme {
  background?: string
  mode?: 'dark' | 'light'
  theme: Theme
}

/** Read the cached boot theme. Null on first launch / damage / test runs. */
export function readBootTheme(): BootTheme | null {
  if (isTestRun()) {
    return null
  }

  try {
    const raw = JSON.parse(readFileSync(bootFilePath(), 'utf8')) as BootThemeFile

    if (raw.version !== 1 || !looksLikeTheme(raw.theme)) {
      return null
    }

    return {
      background: typeof raw.background === 'string' ? raw.background : undefined,
      mode: raw.mode === 'light' || raw.mode === 'dark' ? raw.mode : undefined,
      theme: raw.theme
    }
  } catch {
    return null
  }
}

let writeTimer: NodeJS.Timeout | null = null

/** Persist the resolved theme (debounced, atomic, fire-and-forget). */
export function writeBootTheme(theme: Theme, background?: string, mode?: 'dark' | 'light'): void {
  if (isTestRun()) {
    return
  }

  if (writeTimer) {
    clearTimeout(writeTimer)
  }

  writeTimer = setTimeout(() => {
    writeTimer = null

    try {
      const payload: BootThemeFile = { background, mode, theme, version: 1 }
      const path = bootFilePath()
      const tmp = `${path}.tmp`

      writeFileSync(tmp, JSON.stringify(payload))
      renameSync(tmp, path)
    } catch {
      // Cache write failures are cosmetic — next launch just flashes once.
    }
  }, 400)

  writeTimer.unref?.()
}

// ── Boot-time seeding (with provenance) ──────────────────────────────
//
// The cache seeds env slots so the first skin resolution matches the last
// session — but a previous-session background must NOT occupy the same
// authoritative slot as a current OSC answer forever. Provenance is
// recorded so that once the CURRENT terminal answers with a distrusted
// value (pure-black OSC-11, unusable OSC-10), the stale hint can be
// invalidated and detection falls through to foreground / COLORFGBG /
// macOS appearance / the default — instead of a light cache pinning a now-
// dark terminal to light indefinitely (review on #20379, finding 2).

export interface BootSeedResult {
  /** The background hex this boot seeded into env, or null. */
  seededBackground: null | string
  /** True when the cached config pin was seeded into HERMES_TUI_THEME. */
  seededPin: boolean
}

// Provenance of the last seeding pass — read by invalidateBootBackground.
// Module-load seeds process.env; tests re-seed against a fake env.
let seeded: BootSeedResult = { seededBackground: null, seededPin: false }

/** Seeding step (exported for tests — module-load runs it on process.env).
 *  Explicit user signals always outrank the cache. Records provenance so
 *  invalidateBootBackground can later demote the hint. */
export function seedBootEnvironment(boot: BootTheme | null, env: NodeJS.ProcessEnv): BootSeedResult {
  const result: BootSeedResult = { seededBackground: null, seededPin: false }

  seeded = result

  if (!boot || env.HERMES_TUI_THEME || env.HERMES_TUI_LIGHT) {
    return result
  }

  // Replay the config mode pin first: a pinned session caches a resolved
  // theme whose polarity intentionally disagrees with the physical
  // background, and without the pin the first skin resolution flips to the
  // physical pole before config hydration flips it back (multi-stage flash).
  if (boot.mode) {
    env.HERMES_TUI_THEME = boot.mode
    result.seededPin = true
  }

  if (
    boot.background &&
    // Never seed the untrusted "unset default" fingerprint — a cache written
    // before the distrust rule existed must not poison this session's
    // detection (it would also suppress the macOS-appearance fallback).
    boot.background.toLowerCase() !== '#000000' &&
    !env.HERMES_TUI_BACKGROUND
  ) {
    env.HERMES_TUI_BACKGROUND = boot.background
    result.seededBackground = boot.background
  }

  return result
}

/**
 * Invalidate the cache-seeded background: called when the CURRENT terminal
 * answered the background probe with an untrusted value, meaning the seeded
 * hint is from another era and must not outrank the live fallback chain.
 * Only clears the slot while it still holds the seeded value — a trusted
 * OSC answer or explicit export that overwrote it is left alone.
 * Returns true when the slot was cleared.
 */
export function invalidateBootBackground(env: NodeJS.ProcessEnv = process.env): boolean {
  if (!seeded.seededBackground || env.HERMES_TUI_BACKGROUND !== seeded.seededBackground) {
    return false
  }

  delete env.HERMES_TUI_BACKGROUND
  seeded.seededBackground = null

  return true
}

const boot = readBootTheme()

/** True when this boot replayed a cached config pin into HERMES_TUI_THEME.
 *  applyConfiguredTuiTheme treats it as config-owned so a later 'auto' can
 *  clear it — otherwise a stale cached pin masquerades as a user shell
 *  export and becomes unclearable. */
export const bootSeededPin: boolean = seedBootEnvironment(boot, process.env).seededPin

/** The cached theme for the first frame, or null on first launch. */
export const bootTheme: Theme | null = boot?.theme ?? null
