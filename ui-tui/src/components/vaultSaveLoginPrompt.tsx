import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

interface VaultSaveLoginPromptProps {
  cols?: number
  onSubmit: (login: { identifier: string; password: string }) => void
  origin: string
  site: string
  t: Theme
}

/**
 * Collect a new login entirely inside the prompt surface. Values live only in
 * this mounted component and are handed directly to vault.save_login.respond;
 * neither field is copied into the composer or transcript.
 */
export function VaultSaveLoginPrompt({ cols = 80, onSubmit, origin, site, t }: VaultSaveLoginPromptProps) {
  const [field, setField] = useState<'identifier' | 'password'>('identifier')
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')

  useInput((_ch, key) => {
    if (key.tab || key.upArrow || key.downArrow) {
      setField(current => (current === 'identifier' ? 'password' : 'identifier'))
    }
  })

  const submit = () => {
    if (identifier && password) {
      onSubmit({ identifier, password })
    }
  }

  const inputColumns = Math.max(20, cols - 24)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        🔐 Save login for {site || origin}
      </Text>
      <Text color={t.color.muted}> bound to {origin}</Text>
      <Box>
        <Text color={t.color.label}> Email or username </Text>
        <TextInput
          color={t.color.text}
          columns={inputColumns}
          focus={field === 'identifier'}
          onChange={setIdentifier}
          onSubmit={() => setField('password')}
          value={identifier}
        />
      </Box>
      <Box>
        <Text color={t.color.label}> Password          </Text>
        <TextInput
          color={t.color.text}
          columns={inputColumns}
          focus={field === 'password'}
          mask="*"
          onChange={setPassword}
          onSubmit={submit}
          value={password}
        />
      </Box>
      <Text color={t.color.muted}> Tab/↑↓ switch · Enter saves and signs in · Esc does not save</Text>
    </Box>
  )
}
