import { Box, Text } from '@hermes/ink'
import { useState } from 'react'

import type { VaultSaveLoginReq } from '../types.js'
import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

export function VaultSaveLoginPrompt({ cols = 80, onSubmit, request, t }: VaultSaveLoginPromptProps) {
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [field, setField] = useState<'identifier' | 'password'>('identifier')
  const width = Math.max(20, cols - 6)

  const submitPassword = () => {
    if (identifier.trim() && password) {
      onSubmit(identifier, password)
    }
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        🔐 Save login for {request.site}
      </Text>
      <Text color={t.color.muted}> {request.origin}</Text>
      <Text color={t.color.muted}> encrypted vault only · password · hidden · Esc does not save</Text>
      <Box>
        <Text color={t.color.label}>identifier {'> '}</Text>
        <TextInput
          color={t.color.text}
          columns={width}
          onChange={setIdentifier}
          onSubmit={() => setField('password')}
          value={identifier}
        />
      </Box>
      {field === 'password' && (
        <Box>
          <Text color={t.color.label}>password {'> '}</Text>
          <TextInput
            color={t.color.text}
            columns={width}
            mask="*"
            onChange={setPassword}
            onSubmit={submitPassword}
            value={password}
          />
        </Box>
      )}
    </Box>
  )
}

interface VaultSaveLoginPromptProps {
  cols?: number
  onSubmit: (identifier: string, password: string) => void
  request: VaultSaveLoginReq
  t: Theme
}
