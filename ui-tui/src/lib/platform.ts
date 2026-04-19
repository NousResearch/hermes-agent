/** Platform-aware keybinding helpers.
 *
 * On macOS the "action" modifier is Cmd (key.meta in Ink), on other platforms
 * it is Ctrl.  Ctrl+C is ALWAYS the interrupt key regardless of platform —
 * it must never be remapped to copy.
 */

export const isMac = process.platform === 'darwin'

/** The display label for the action modifier key. */
export const modLabel = isMac ? '⌘' : 'Ctrl'

/** True when the platform action-modifier is pressed (Cmd on macOS, Ctrl elsewhere). */
export const isActionMod = (key: { ctrl: boolean; meta: boolean }): boolean =>
  isMac ? key.meta : key.ctrl

/** Match action-modifier + a single character (case-insensitive). */
export const isAction = (key: { ctrl: boolean; meta: boolean }, ch: string, target: string): boolean =>
  isActionMod(key) && ch.toLowerCase() === target
