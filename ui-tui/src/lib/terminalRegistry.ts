import type { ClipboardReadPath, ClipboardWritePath, KeyboardEncoding } from './terminalCapabilities.js'

/**
 * Terminal / multiplexer profile registry.
 *
 * The registry replaces hardcoded switch/if chains for terminal detection,
 * keyboard encoding, clipboard routing, paste shortcuts, and mouse behavior.
 * Adding a new terminal = adding one entry here. No code changes needed.
 *
 * Design inspired by terminfo(5): a database of terminal capabilities
 * queried by detection signals rather than by TERM name alone.
 */

export interface TerminalProfile {
  id: string
  /** Priority: higher numbers checked first. Multiplexers lower (50-59),
   *  explicit terminals higher (100+). */
  priority: number
  /** How to detect this profile from raw signals.
   *  The first matching profile (by priority) wins. */
  detect: {
    /** Required env var must be present (value ignored, just checks existence) */
    envVar?: string
    /** TERM_PROGRAM exact match */
    termProgram?: string
    /** TERM value pattern (substring match, case-insensitive) */
    termPattern?: string
    /** Is this a multiplexer (tmux, screen, zellij, cy)? */
    multiplexer?: boolean
    /** Additional custom check */
    check?: (signals: { env: Record<string, string | undefined> }) => boolean
  }
  /** Capabilities this terminal provides */
  capabilities: {
    keyboard: KeyboardEncoding
    clipboard: {
      writePath: ClipboardWritePath
      readPath: ClipboardReadPath
    }
    pasteShortcuts: {
      darwin: string[]
      default: string[]
    }
    shiftDragHint: boolean
    copyOnSelect: boolean
  }
}

export const TERMINAL_REGISTRY: TerminalProfile[] = [
  // Multiplexers (priority 50-59) — detected by env var presence
  {
    id: 'tmux',
    priority: 55,
    detect: { envVar: 'TMUX', multiplexer: true },
    capabilities: {
      keyboard: 'legacy',
      clipboard: { writePath: 'tmux-buffer', readPath: 'none' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: true,
      copyOnSelect: false,
    },
  },
  {
    id: 'screen',
    priority: 54,
    detect: { envVar: 'STY', multiplexer: true },
    capabilities: {
      keyboard: 'legacy',
      clipboard: { writePath: 'screen-passthrough', readPath: 'none' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: true,
      copyOnSelect: false,
    },
  },
  {
    id: 'zellij',
    priority: 53,
    detect: { envVar: 'ZELLIJ', multiplexer: true },
    capabilities: {
      keyboard: 'legacy',
      clipboard: { writePath: 'zellij-passthrough', readPath: 'none' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: true,
      copyOnSelect: false,
    },
  },
  {
    id: 'cy',
    priority: 52,
    detect: { envVar: 'CY', multiplexer: true },
    capabilities: {
      keyboard: 'legacy',
      clipboard: { writePath: 'cy-passthrough', readPath: 'none' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: true,
      copyOnSelect: false,
    },
  },

  // Explicit terminal emulators (priority 100+) — detected by TERM_PROGRAM or env vars
  {
    id: 'kitty',
    priority: 110,
    detect: { envVar: 'KITTY_WINDOW_ID', termProgram: 'kitty', termPattern: 'kitty' },
    capabilities: {
      keyboard: 'kitty',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v', 'ctrl+v'], default: ['ctrl+shift+v', 'alt+v', 'ctrl+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'wezterm',
    priority: 109,
    detect: { envVar: 'WEZTERM_PANE', check: s => s.env['TERM'] === 'wezterm' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'ghostty',
    priority: 108,
    detect: { envVar: 'GHOSTTY_RESOURCES_DIR', termPattern: 'ghostty' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'warp',
    priority: 107,
    detect: { check: s => {
      const tp = s.env['TERM_PROGRAM']
      const term = s.env['TERM'] ?? ''
      return tp === 'WarpTerminal' || tp === 'warp' || term.toLowerCase().includes('warp')
    }},
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'iterm2',
    priority: 106,
    detect: { check: s => {
      const tp = s.env['TERM_PROGRAM']
      return s.env['ITERM_SESSION_ID'] !== undefined || tp === 'iTerm.app' || s.env['LC_TERMINAL'] === 'iTerm2'
    }},
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: true, // macOS only: copy-on-select makes sense on iTerm2
    },
  },
  {
    id: 'windows-terminal',
    priority: 105,
    detect: { envVar: 'WT_SESSION' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'vscode-xtermjs',
    priority: 104,
    detect: { termProgram: 'vscode' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'vte',
    priority: 103,
    detect: { envVar: 'VTE_VERSION' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'alacritty',
    priority: 102,
    detect: { termProgram: 'Alacritty', check: s => s.env['TERM'] === 'alacritty' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'foot',
    priority: 101,
    detect: { termProgram: 'foot', termPattern: 'foot' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
  {
    id: 'apple-terminal',
    priority: 100,
    detect: { termProgram: 'Apple_Terminal', envVar: 'TERM_SESSION_ID' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: true,
    },
  },
  {
    id: 'konsole',
    priority: 99,
    detect: { envVar: 'KONSOLE_VERSION' },
    capabilities: {
      keyboard: 'csi-u',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },

  // Legacy fallback (priority 0) — matched last
  {
    id: 'xterm',
    priority: 0,
    detect: { termPattern: 'xterm' },
    capabilities: {
      keyboard: 'legacy',
      clipboard: { writePath: 'native', readPath: 'native' },
      pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
      shiftDragHint: false,
      copyOnSelect: false,
    },
  },
]

/** Fallback profile for unknown terminals. */
export const UNKNOWN_PROFILE: TerminalProfile = {
  id: 'unknown',
  priority: -1,
  detect: {},
  capabilities: {
    keyboard: 'legacy',
    clipboard: { writePath: 'none', readPath: 'none' },
    pasteShortcuts: { darwin: ['cmd+v', 'ctrl+shift+v'], default: ['ctrl+shift+v', 'alt+v'] },
    shiftDragHint: false,
    copyOnSelect: false,
  },
}

/**
 * Find a terminal profile matching the given env signals.
 * Profiles are sorted by priority (highest first); the first match wins.
 *
 * Matching is OR-based per detection field: envVar must be present,
 * termProgram matches only if TERM_PROGRAM is set, termPattern matches
 * only if TERM is set. All specified checks must pass.
 */
export function findProfile(env: Record<string, string | undefined>): TerminalProfile {
  const sorted = [...TERMINAL_REGISTRY].sort((a, b) => b.priority - a.priority)

  for (const profile of sorted) {
    const d = profile.detect
    let matched = false

    // envVar: present = positive signal, missing = skip
    if (d.envVar) {
      if (env[d.envVar] !== undefined) matched = true
      else continue
    }

    // termProgram: match if TERM_PROGRAM is set and matches
    const tp = env['TERM_PROGRAM']
    if (d.termProgram) {
      if (tp === d.termProgram) matched = true
      else if (tp !== undefined) continue // TERM_PROGRAM set but doesn't match
      // If tp is undefined, don't skip — other signals may still match
    }

    // termPattern: match if TERM is set and contains the pattern
    const term = env['TERM']
    if (d.termPattern) {
      if (term !== undefined && term.toLowerCase().includes(d.termPattern.toLowerCase())) matched = true
      else if (term !== undefined) continue // TERM set but doesn't match
      // If term is undefined, don't skip — other signals may still match
    }

    // Custom check — positive signal only (can't reject if other signals matched)
    if (d.check) {
      if (d.check({ env })) matched = true
    }

    // Must have at least one positive signal
    if (matched) return profile
  }

  return UNKNOWN_PROFILE
}

/**
 * Find the multiplexer profile for a given env var key.
 * Returns undefined if the env var isn't set or no multiplexer matches.
 */
export function findMultiplexer(env: Record<string, string | undefined>): TerminalProfile | undefined {
  for (const profile of TERMINAL_REGISTRY) {
    if (profile.detect.multiplexer && profile.detect.envVar && env[profile.detect.envVar] !== undefined) {
      return profile
    }
  }
  return undefined
}
