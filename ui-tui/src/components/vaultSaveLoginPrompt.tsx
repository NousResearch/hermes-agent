import { Box, Text } from '@hermes/ink'
import { useRef, useState } from 'react'

import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

interface VaultSaveLoginPromptProps {
  cols?: number
  onSubmit: (value: string) => void
  origin: string
  site: string
  t: Theme
}

export function VaultSaveLoginPrompt({ cols = 80, onSubmit, origin, site, t }: VaultSaveLoginPromptProps) {
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [step, setStep] = useState<'identifier' | 'password'>('identifier')
  const submitted = useRef(false)
  const inputColumns = Math.max(20, cols - 6)

  const submitIdentifier = (value: string) => {
    const next = value.trim()

    if (!next) {
      return
    }

    setIdentifier(next)
    setStep('password')
  }

  const submitPassword = (value: string) => {
    if (!value || submitted.current) {
      return
    }

    submitted.current = true
    onSubmit(JSON.stringify({ identifier, password: value }))
    setPassword('')
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        🔐 Save login for {site}
      </Text>
      <Text color={t.color.muted}> {origin} · sent directly to your credential vault · Esc cancels</Text>

      <Box>
        <Text color={t.color.label}>{step === 'identifier' ? 'Username > ' : 'Password > '}</Text>
        {step === 'identifier' ? (
          <TextInput columns={inputColumns} onChange={setIdentifier} onSubmit={submitIdentifier} value={identifier} />
        ) : (
          <TextInput
            columns={inputColumns}
            mask="*"
            onChange={setPassword}
            onSubmit={submitPassword}
            value={password}
          />
        )}
      </Box>
    </Box>
  )
}
