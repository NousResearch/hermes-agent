import { Box, Text, TextInput } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'

export function MaskedPrompt({
  icon,
  label,
  onSubmit,
  sub,
  t
}: {
  icon: string
  label: string
  onSubmit: (v: string) => void
  sub?: string
  t: Theme
}) {
  const [value, setValue] = useState('')

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        {icon} {label}
      </Text>
      {sub && <Text color={t.color.dim}> {sub}</Text>}

      <Box>
        <Text color={t.color.label}>{'> '}</Text>
        <TextInput mask="*" onChange={setValue} onSubmit={onSubmit} value={value} />
      </Box>
    </Box>
  )
}
