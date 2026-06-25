import { FACES as DEFAULT_FACES } from '../content/faces.js'
import { VERBS as DEFAULT_VERBS } from '../content/verbs.js'

/**
 * Compute the padded verb string for the status ticker.
 * Uses the active verb list from skin (if available) or falls back to defaults.
 *
 * The padding length is computed dynamically based on the actual verb list
 * to prevent jitter when custom skins have longer/shorter verbs.
 */
export function computeVerbPadLen(verbs: string[]): number {
  if (verbs.length === 0) {
    return 0
  }
  return verbs.reduce((max, v) => Math.max(max, v.length), 0) + 1 // + ellipsis
}

/**
 * Pad a verb with trailing spaces so the status bar doesn't jitter
 * when the ticker rotates between short and long verbs.
 */
export function padVerb(verb: string, padLen: number): string {
  return `${verb}…`.padEnd(padLen, ' ')
}

/**
 * Get the active verb list: use skin-provided verbs if available,
 * otherwise fall back to hardcoded defaults.
 */
export function getActiveVerbs(skinVerbs: string[] | undefined): string[] {
  return skinVerbs && skinVerbs.length > 0 ? skinVerbs : DEFAULT_VERBS
}

/**
 * Get the active face list: use skin-provided faces if available,
 * otherwise fall back to hardcoded defaults.
 */
export function getActiveFaces(skinFaces: string[] | undefined): string[] {
  return skinFaces && skinFaces.length > 0 ? skinFaces : DEFAULT_FACES
}
