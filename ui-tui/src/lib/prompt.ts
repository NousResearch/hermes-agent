const TERMUX_SAFE_PROMPT = '>'

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
      return `${profileName} ${basePrompt}`
    }

    return basePrompt
  }

  if (profileName && !['default', 'custom'].includes(profileName)) {
    return `${profileName} ${prompt}`
  }

  return prompt
}
