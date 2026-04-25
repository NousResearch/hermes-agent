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

const isRemoteShell = (env: NodeJS.ProcessEnv = process.env): boolean =>
  Boolean(env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY)

export const isCopyShortcut = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  env: NodeJS.ProcessEnv = process.env
): boolean =>
  isAction(key, ch, 'c') || (isRemoteShell(env) && (key.meta || key.super === true) && ch.toLowerCase() === 'c')

/**
 * Voice recording toggle key (Ctrl+B).
 *
 * Documented as "Ctrl+B" everywhere: tips.py, config.yaml's voice.record_key
 * default, and the Python CLI prompt_toolkit handler. We accept raw Ctrl+B on
 * every platform so the TUI matches those docs. On macOS we additionally
 * accept Cmd+B (the platform action modifier) so existing macOS muscle memory
 * keeps working.
 */
export const isVoiceToggleKey = (key: { ctrl: boolean; meta: boolean; super?: boolean }, ch: string): boolean =>
  (key.ctrl || isActionMod(key)) && ch.toLowerCase() === 'b'
