import { Box, Text } from '@hermes/ink'
import { useState } from 'react'

import { TERMUX_TUI_MODE } from '../config/env.js'
import { terminalFloor } from '../lib/inputMetrics.js'
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
          columns={terminalFloor(cols - 6, 20, TERMUX_TUI_MODE)}
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
