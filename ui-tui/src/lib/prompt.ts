const TERMUX_SAFE_PROMPT = '>'

export function composerPromptText(
  prompt: string,
  profileName?: null | string,
  shellMode = false,
  termuxMode = false,
  totalCols?: number,
  brandName?: null | string
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
    const prefix = promptPrefix(profileName, brandName)
    if (wideEnoughForProfile && prefix) {
      return `${prefix} ${basePrompt}`
    }

    return basePrompt
  }

  const prefix = promptPrefix(profileName, brandName)
  if (prefix) {
    return `${prefix} ${prompt}`
  }

  return prompt
}

function promptPrefix(profileName?: null | string, brandName?: null | string): string {
  const profile = (profileName ?? '').trim()
  if (profile && !['custom', 'default'].includes(profile)) {
    return profile
  }
  const brand = (brandName ?? '').trim()
  if (brand && profile !== 'custom') {
    return brand.replace(/\s+Agent$/i, '')
  }
  return ''
}
