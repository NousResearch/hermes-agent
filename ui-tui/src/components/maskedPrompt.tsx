import { Box, Text } from '@hermes/ink'
import { useState } from 'react'

import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

export function MaskedPrompt({ cols = 80, icon, label, onSubmit, sub, t }: MaskedPromptProps) {
  const [value, setValue] = useState('')

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        {icon} {label}
      </Text>

      {sub && <Text color={t.color.muted}> {sub}</Text>}

      <Box>
        <Text color={t.color.label}>{'> '}</Text>
        <TextInput
          color={t.color.text}
          columns={Math.max(20, cols - 6)}
          mask="*"
          onChange={setValue}
          onSubmit={onSubmit}
          value={value}
        />
      </Box>
    </Box>
  )
}

interface MaskedPromptProps {
  cols?: number
  icon: string
  label: string
  onSubmit: (v: string) => void
  sub?: string
  t: Theme
}

interface SecurePromptProps {
  cols?: number
  onSubmit: (v: string) => void
  t: Theme
}

/** Masked prompt for the `sudo` password the backend asked for. */
export function SudoPrompt({ cols, onSubmit, t }: SecurePromptProps) {
  const T = useT()

  return <MaskedPrompt cols={cols} icon="🔐" label={T.secure.sudo.title} onSubmit={onSubmit} t={t} />
}

/** Masked prompt for a secret the backend wants stored under `envVar`; `prompt` is the backend's own wording. */
export function SecretPrompt({
  cols,
  envVar,
  onSubmit,
  prompt,
  t
}: SecurePromptProps & { envVar: string; prompt: string }) {
  const T = useT()

  return (
    <MaskedPrompt
      cols={cols}
      icon="🔑"
      label={prompt}
      onSubmit={onSubmit}
      sub={T.secure.secret.forEnvVar(envVar)}
      t={t}
    />
  )
}

/** Masked prompt for a password manager's master password; the value goes to the manager CLI only. */
export function VaultUnlockPrompt({ cols, displayName, onSubmit, t }: SecurePromptProps & { displayName: string }) {
  const T = useT()

  return (
    <MaskedPrompt
      cols={cols}
      icon="🔐"
      label={T.secure.vault.unlockTitle(displayName)}
      onSubmit={onSubmit}
      sub={T.secure.vault.unlockHint}
      t={t}
    />
  )
}
