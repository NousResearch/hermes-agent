import { Box, Text } from '@hermes/ink'
import { useState } from 'react'

import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

export function MaskedPrompt({ cols = 80, icon, label, onSubmit, reveal, sub, t }: MaskedPromptProps) {
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
          mask={reveal ? undefined : '*'}
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
  /** Show what's typed (a username, not a secret). */
  reveal?: boolean
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

/**
 * `vault.save_login`: the identifier as typed, then the masked password. Either
 * step left empty declines; the pair goes only to the encrypted vault.
 */
export function VaultSaveLoginPrompt({
  cols,
  onReady,
  site,
  t
}: Omit<SecurePromptProps, 'onSubmit'> & { onReady: (identifier: string, password: string) => void; site: string }) {
  const T = useT()
  const [identifier, setIdentifier] = useState('')

  if (identifier) {
    return (
      <MaskedPrompt
        cols={cols}
        icon="🔑"
        key="password"
        label={T.secure.vault.savePasswordTitle(identifier)}
        onSubmit={password => onReady(identifier, password)}
        sub={T.secure.vault.savePasswordHint(site)}
        t={t}
      />
    )
  }

  return (
    <MaskedPrompt
      cols={cols}
      icon="🔑"
      key="identifier"
      label={T.secure.vault.saveTitle(site)}
      onSubmit={value => (value ? setIdentifier(value) : onReady('', ''))}
      reveal
      sub={T.secure.vault.saveIdentifierHint}
      t={t}
    />
  )
}

/** `vault.code`: a one-time sign-in code; it is typed into the page, never shown to the model. */
export function VaultCodePrompt({ cols, hint, onSubmit, site, t }: SecurePromptProps & { hint: string; site: string }) {
  const T = useT()

  return (
    <MaskedPrompt
      cols={cols}
      icon="🔢"
      label={T.secure.vault.codeTitle(site)}
      onSubmit={onSubmit}
      reveal
      sub={hint ? `${hint} · ${T.secure.vault.codeHint}` : T.secure.vault.codeHint}
      t={t}
    />
  )
}
