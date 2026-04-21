/** Platform-aware keybinding helpers.
 *
 * On macOS the "action" modifier is Cmd. Modern terminals that support kitty
 * keyboard protocol report Cmd as `key.super`; legacy terminals often surface it
 * as `key.meta`. Some macOS terminals also translate Cmd+Left/Right/Backspace
 * into readline-style Ctrl+A/Ctrl+E/Ctrl+U before the app sees them.
 * On other platforms the action modifier is Ctrl.
 * Ctrl+C is ALWAYS the interrupt key regardless of platform — it must never be
 * remapped to copy.
 */

export const isMac = process.platform === 'darwin'

/** True when the platform action-modifier is pressed (Cmd on macOS, Ctrl elsewhere). */
export const isActionMod = (key: { ctrl: boolean; meta: boolean; super?: boolean }): boolean =>
  isMac ? key.meta || key.super === true : key.ctrl

/**
 * Some macOS terminals rewrite Cmd navigation/deletion into readline control keys.
 * Treat those as action shortcuts too, but only for the specific fallbacks we
 * have observed from terminals: Cmd+Left → Ctrl+A, Cmd+Right → Ctrl+E,
 * Cmd+Backspace → Ctrl+U.
 */
export const isMacActionFallback = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  target: 'a' | 'e' | 'u'
): boolean => isMac && key.ctrl && !key.meta && key.super !== true && ch.toLowerCase() === target

/** Match action-modifier + a single character (case-insensitive). */
export const isAction = (key: { ctrl: boolean; meta: boolean; super?: boolean }, ch: string, target: string): boolean =>
  isActionMod(key) && ch.toLowerCase() === target
