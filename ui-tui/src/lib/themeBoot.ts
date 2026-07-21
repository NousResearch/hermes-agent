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

/** Read the cached boot theme. Null on first launch / damage / test runs. */
export function readBootTheme(): { background?: string; theme: Theme } | null {
  if (isTestRun()) {
    return null
  }

  try {
    const raw = JSON.parse(readFileSync(bootFilePath(), 'utf8')) as BootThemeFile

    if (raw.version !== 1 || !looksLikeTheme(raw.theme)) {
      return null
    }

    return { background: typeof raw.background === 'string' ? raw.background : undefined, theme: raw.theme }
  } catch {
    return null
  }
}

let writeTimer: NodeJS.Timeout | null = null

/** Persist the resolved theme (debounced, atomic, fire-and-forget). */
export function writeBootTheme(theme: Theme, background?: string): void {
  if (isTestRun()) {
    return
  }

  if (writeTimer) {
    clearTimeout(writeTimer)
  }

  writeTimer = setTimeout(() => {
    writeTimer = null

    try {
      const payload: BootThemeFile = { background, theme, version: 1 }
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

/**
 * Boot-time seeding, run once at module load (imported by uiStore before the
 * first render): make the cached background visible to `detectLightMode`
 * unless an explicit signal already outranks it.
 */
const boot = readBootTheme()

if (
  boot?.background &&
  // Never seed the untrusted "unset default" fingerprint — a cache written
  // before the distrust rule existed must not poison this session's
  // detection (it would also suppress the macOS-appearance fallback).
  boot.background.toLowerCase() !== '#000000' &&
  !process.env.HERMES_TUI_BACKGROUND &&
  !process.env.HERMES_TUI_THEME &&
  !process.env.HERMES_TUI_LIGHT
) {
  process.env.HERMES_TUI_BACKGROUND = boot.background
}

/** The cached theme for the first frame, or null on first launch. */
export const bootTheme: Theme | null = boot?.theme ?? null
