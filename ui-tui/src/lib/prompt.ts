import { stringWidth } from '@hermes/ink'

const TERMUX_SAFE_PROMPT = '>'

/**
 * Ensure the prompt string fills its display width so re-renders don't leave
 * ghost artifacts from ambiguous-width glyphs (e.g. ❯ U+276F which some
 * terminals render at 2 cells while stringWidth reports 1).  Pad with spaces
 * so Ink properly clears all previously-occupied cells on the next render.
 */
const widthSafePrompt = (text: string): string => {
  const w = stringWidth(text)

  // If the measured width is below the raw char length, we have an
  // ambiguous-width situation — pad by one space to give the terminal
  // room to clear the prior render's cells.
  return text.length > w ? text.padEnd(text.length + 1) : text
}

export function composerPromptText(
  prompt: string,
  profileName?: null | string,
  shellMode = false,
  termuxMode = false,
  totalCols?: number
): string {
  if (shellMode) {
    return '$'
  }

  if (termuxMode) {
    // Termux fonts/terminal backends can render decorative prompt glyphs with
    // ambiguous width; keep the live composer marker strictly single-cell ASCII
    // so we never leave stale arrow artifacts while typing.
    const basePrompt = TERMUX_SAFE_PROMPT

    // On very wide panes we can still include profile context. On narrow/mobile
    // panes this burns precious columns and increases wrap/clipping risk.
    const wideEnoughForProfile = typeof totalCols === 'number' ? totalCols >= 90 : false
    if (wideEnoughForProfile && profileName && !['default', 'custom'].includes(profileName)) {
      return widthSafePrompt(`${profileName} ${basePrompt}`)
    }

    return widthSafePrompt(basePrompt)
  }

  if (profileName && !['default', 'custom'].includes(profileName)) {
    return widthSafePrompt(`${profileName} ${prompt}`)
  }

  return widthSafePrompt(prompt)
}
