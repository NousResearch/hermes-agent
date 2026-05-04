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

export const DEFAULT_VOICE_RECORD_KEY = 'ctrl+b'

type VoiceRecordModifier = 'alt' | 'ctrl'

interface VoiceRecordKeySpec {
  key: string
  modifier: VoiceRecordModifier
}

const VOICE_RECORD_KEY_PATTERN = /^(ctrl|control|c|alt|option|meta|a)\s*\+\s*(.)$/

const parseVoiceRecordKeySpec = (raw: string): null | VoiceRecordKeySpec => {
  const match = raw.trim().toLowerCase().match(VOICE_RECORD_KEY_PATTERN)

  if (!match) {
    return null
  }

  const [, modifierToken = '', key = ''] = match
  const modifier: VoiceRecordModifier = ['alt', 'option', 'meta', 'a'].includes(modifierToken) ? 'alt' : 'ctrl'

  return { key, modifier }
}

const DEFAULT_VOICE_RECORD_KEY_SPEC = (() => {
  const spec = parseVoiceRecordKeySpec(DEFAULT_VOICE_RECORD_KEY)

  if (!spec) {
    throw new Error(`Invalid DEFAULT_VOICE_RECORD_KEY: ${DEFAULT_VOICE_RECORD_KEY}`)
  }

  return spec
})()

const parseVoiceRecordKey = (raw: unknown): VoiceRecordKeySpec => {
  if (typeof raw !== 'string') {
    return DEFAULT_VOICE_RECORD_KEY_SPEC
  }

  return parseVoiceRecordKeySpec(raw) ?? DEFAULT_VOICE_RECORD_KEY_SPEC
}

export const normalizeVoiceRecordKey = (raw: unknown): string => {
  const spec = parseVoiceRecordKey(raw)

  return `${spec.modifier}+${spec.key}`
}

export const formatVoiceRecordKey = (raw: unknown): string => {
  const spec = parseVoiceRecordKey(raw)
  const modifier = spec.modifier === 'ctrl' ? 'Ctrl' : 'Alt'

  return `${modifier}+${spec.key.toUpperCase()}`
}

/**
 * Voice recording toggle key from config.yaml's voice.record_key.
 *
 * Supported config spellings mirror the classic CLI's documented forms:
 * `ctrl+x` and `alt+x` (case-insensitive). Invalid values fall back to the
 * config default, Ctrl+B. For the default chord we keep the existing macOS
 * action-modifier+B compatibility; explicit non-default chords must match the
 * configured modifier/key so the TUI does not keep hardcoding Ctrl+B.
 */
export const isVoiceToggleKey = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  recordKey: string = DEFAULT_VOICE_RECORD_KEY
): boolean => {
  const spec = parseVoiceRecordKey(recordKey)
  const target = ch.toLowerCase() === spec.key

  if (!target) {
    return false
  }

  if (spec.modifier === 'alt') {
    return key.meta && !key.ctrl && key.super !== true
  }

  return (
    key.ctrl ||
    (normalizeVoiceRecordKey(recordKey) === DEFAULT_VOICE_RECORD_KEY && isActionMod(key))
  )
}
