import { Box, Text } from 'ink'

import { COMMANDS } from '../constants.js'
import type { Theme } from '../theme.js'

export function CommandPalette({ filter, t }: { filter: string; t: Theme }) {
  const matches = COMMANDS.filter(([cmd]) => cmd.startsWith(filter))

  if (!matches.length) {
    return null
  }

  return (
    <Box flexDirection="column">
      {matches.map(([cmd, desc]) => (
        <Text key={cmd}>
          <Text bold color={t.color.amber}>
            {cmd}
          </Text>
          <Text color={t.color.dim}> — {desc}</Text>
        </Text>
      ))}
    </Box>
  )
}
