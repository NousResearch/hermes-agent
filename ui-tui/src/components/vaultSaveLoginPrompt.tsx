import { Box, Text } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'

import { MaskedPrompt } from './maskedPrompt.js'
import { TextInput } from './textInput.js'

/**
 * Two-step capture for `vault.save_login.request` (CLI parity: identifier
 * shown as typed, then a masked password). Either step submitted empty
 * declines the whole save. The values travel only inside
 * `vault.save_login.respond` to the encrypted vault — never to the
 * transcript, logs, or model context.
 */
export function VaultSaveLoginPrompt({ cols = 80, onReady, site, t }: VaultSaveLoginPromptProps) {
  const [identifier, setIdentifier] = useState('')
  const [masked, setMasked] = useState(false)

  if (masked) {
    return (
      <MaskedPrompt
        cols={cols}
        icon="🔑"
        label={`Password for ${identifier}`}
        onSubmit={password => onReady(identifier, password)}
        sub={`stored in the encrypted vault for ${site} · hidden · never shown to the model`}
        t={t}
      />
    )
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        🔑 Save a login for {site}
      </Text>

      <Text color={t.color.muted}> identifier (shown as typed) · password comes next · empty declines</Text>

      <Box>
        <Text color={t.color.label}>{'> '}</Text>
        <TextInput
          color={t.color.text}
          columns={Math.max(20, cols - 6)}
          onChange={setIdentifier}
          onSubmit={value => (value ? setMasked(true) : onReady('', ''))}
          value={identifier}
        />
      </Box>
    </Box>
  )
}

interface VaultSaveLoginPromptProps {
  cols?: number
  onReady: (identifier: string, password: string) => void
  site: string
  t: Theme
}
