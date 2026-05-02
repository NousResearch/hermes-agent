/** Platform-aware keybinding helpers.
 *
 * On macOS the "action" modifier is Cmd. Modern terminals that support kitty
 * keyboard protocol report Cmd as `key.super`; legacy terminals often surface it
 * as `key.meta`. Some macOS terminals also translate Cmd+Left/Right/Backspace
 * into readline-style Ctrl+A/Ctrl+E/Ctrl+U before the app sees them.
 * On other platforms the action modifier is Ctrl.
 * Ctrl+C stays the interrupt key on macOS. On non-mac terminals it can also
 * copy an active TUI selection, matching common terminal selection behavior.
 */

export const isMac = process.platform === 'darwin'

/** True when the platform action-modifier is pressed (Cmd on macOS, Ctrl elsewhere). */
export const isActionMod = (key: { ctrl: boolean; meta: boolean; super?: boolean }): boolean =>
  isMac ? key.meta || key.super === true : key.ctrl

/**
 * Accept raw Ctrl+<letter> as an action shortcut on macOS, where `isActionMod`
 * otherwise means Cmd. Two motivations:
 *   - Some macOS terminals rewrite Cmd navigation/deletion into readline control
 *     keys (Cmd+Left → Ctrl+A, Cmd+Right → Ctrl+E, Cmd+Backspace → Ctrl+U).
 *   - Ctrl+K (kill-to-end) and Ctrl+W (delete-word-back) are standard readline
 *     bindings that users expect to work regardless of platform, even though
 *     no terminal rewrites Cmd into them.
 */
export const isMacActionFallback = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  target: 'a' | 'e' | 'u' | 'k' | 'w'
): boolean => isMac && key.ctrl && !key.meta && key.super !== true && ch.toLowerCase() === target

/** Match action-modifier + a single character (case-insensitive). */
export const isAction = (key: { ctrl: boolean; meta: boolean; super?: boolean }, ch: string, target: string): boolean =>
  isActionMod(key) && ch.toLowerCase() === target

export const isRemoteShell = (env: NodeJS.ProcessEnv = process.env): boolean =>
  Boolean(env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY)

export const isCopyShortcut = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  env: NodeJS.ProcessEnv = process.env
): boolean =>
  ch.toLowerCase() === 'c' &&
  (isAction(key, ch, 'c') ||
    (isRemoteShell(env) && (key.meta || key.super === true)) ||
    // VS Code/Cursor/Windsurf terminal setup forwards Cmd+C as a CSI-u
    // sequence with the super bit plus a benign ctrl bit. Accept that shape
    // even though raw Ctrl+C should remain interrupt on local macOS.
    (isMac && key.ctrl && (key.meta || key.super === true)))

/**
 * Voice recording toggle key — configurable via ``voice.record_key`` in
 * ``config.yaml`` (default ``ctrl+b``).
 *
 * Documented in tips.py, the Python CLI prompt_toolkit handler, and the
 * config.yaml default. The TUI honours the same config knob (#18994);
 * when ``voice.record_key`` is e.g. ``ctrl+o`` the TUI binds Ctrl+O.
 *
 * On macOS we additionally accept the platform action modifier (Cmd) for
 * the configured letter so existing macOS muscle memory keeps working
 * alongside the documented Ctrl+<letter> shortcut.
 */
export type VoiceRecordKeyMod = 'alt' | 'ctrl' | 'meta' | 'super'

export interface ParsedVoiceRecordKey {
  ch: string
  mod: VoiceRecordKeyMod
  raw: string
}

export const DEFAULT_VOICE_RECORD_KEY: ParsedVoiceRecordKey = {
  ch: 'b',
  mod: 'ctrl',
  raw: 'ctrl+b'
}

const _MOD_ALIASES: Record<string, VoiceRecordKeyMod> = {
  alt: 'alt',
  cmd: 'meta',
  command: 'meta',
  control: 'ctrl',
  ctrl: 'ctrl',
  meta: 'meta',
  option: 'alt',
  opt: 'alt',
  super: 'super',
  win: 'super',
  windows: 'super'
}

/**
 * Parse a config-string voice record key like ``ctrl+b`` / ``alt+r`` /
 * ``cmd+space`` into ``{mod, ch}``. Falls back to the documented Ctrl+B
 * default for empty / malformed input so a typo never silently disables
 * the shortcut.
 */
export const parseVoiceRecordKey = (raw: string): ParsedVoiceRecordKey => {
  const lower = (raw ?? '').trim().toLowerCase()

  if (!lower) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  const parts = lower.split('+').map(p => p.trim()).filter(Boolean)

  if (!parts.length) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  const ch = parts[parts.length - 1]
  const modCandidates = parts.slice(0, -1)

  let mod: VoiceRecordKeyMod = 'ctrl'

  for (const cand of modCandidates) {
    const norm = _MOD_ALIASES[cand]

    if (norm) {
      mod = norm
      break
    }
  }

  // Reject multi-character chunks (e.g. "ctrl+space" → ch="space" — we
  // only support single-character bindings, matching the Python side's
  // prompt_toolkit binding shape).
  if (ch.length !== 1) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  return { ch, mod, raw: lower }
}

/** Render a parsed key back as ``Ctrl+B`` for status text. */
export const formatVoiceRecordKey = (parsed: ParsedVoiceRecordKey): string => {
  const modLabel = parsed.mod === 'meta' ? 'Cmd' : parsed.mod[0].toUpperCase() + parsed.mod.slice(1)

  return `${modLabel}+${parsed.ch.toUpperCase()}`
}

export const isVoiceToggleKey = (
  key: { alt?: boolean; ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  configured: ParsedVoiceRecordKey = DEFAULT_VOICE_RECORD_KEY
): boolean => {
  if (ch.toLowerCase() !== configured.ch) {
    return false
  }

  switch (configured.mod) {
    case 'alt':
      // Most terminals surface Alt as either ``alt`` or ``meta``; accept
      // both so the binding works across xterm-style and kitty-style
      // protocols.
      return key.alt === true || key.meta
    case 'ctrl':
      // Doc default — also accept the platform action modifier so macOS
      // Cmd+<letter> muscle memory keeps working alongside Ctrl+<letter>.
      return key.ctrl || isActionMod(key)
    case 'meta':
      return key.meta || key.super === true
    case 'super':
      return key.super === true
  }
}
