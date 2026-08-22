const TERMUX_SAFE_PROMPT = '>'

// Profile names that never surface as identity chrome: `default` is the stock
// instance and `custom` marks an unrecognized HERMES_HOME — neither is a real
// named profile. Shared by the composer prefix and the status-bar segment so
// the two always agree on when a profile is displayable (#36081).
const HIDDEN_PROFILE_NAMES = ['default', 'custom']

export function showsNamedProfile(profileName?: null | string): profileName is string {
  return !!profileName && !HIDDEN_PROFILE_NAMES.includes(profileName)
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

    if (wideEnoughForProfile && showsNamedProfile(profileName)) {
      return `${profileName} ${basePrompt}`
    }

    return basePrompt
  }

  if (showsNamedProfile(profileName)) {
    return `${profileName} ${prompt}`
  }

  return prompt
}
