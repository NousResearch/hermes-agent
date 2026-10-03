// Credential prompts: the sudo password, secret (env var) and password-manager
// unlock prompts hosted in MaskedPrompt. Owned namespace: `secure`.
//
// Only labels/hints live here — the typed values never touch the catalog.
// (maskedPrompt.tsx itself has no prose; its `> ` caret is a token.)

export const secureEn = {
  secure: {
    sudo: {
      title: 'sudo password required'
    },
    secret: {
      forEnvVar: (envVar: string) => `for ${envVar}`
    },
    vault: {
      unlockTitle: (displayName: string) => `Unlock ${displayName} for this session`,
      unlockHint: 'master password · hidden · goes to the manager CLI only · Esc keeps it locked'
    }
  }
}
