export function composerPromptText(prompt: string, profileName?: null | string, shellMode = false): string {
  if (shellMode) {
    return '$'
  }

  if (profileName && !['default', 'custom'].includes(profileName)) {
    return `${profileName} ${prompt}`
  }

  return prompt
}
