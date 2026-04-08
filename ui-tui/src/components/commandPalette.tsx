import { Box, Text } from 'ink'

import type { Theme } from '../theme.js'

export function CommandPalette({ matches, t }: { matches: [string, string][]; t: Theme }) {
  if (!matches.length) {
    return null
  }

  return (
    <Box borderColor={t.color.bronze} borderStyle="single" flexDirection="column" paddingX={1}>
      {matches.map(([cmd, desc], i) => (
        <Text key={`${i}-${cmd}`}>
          <Text bold color={t.color.amber}>
            {cmd}
          </Text>
          {desc ? <Text color={t.color.dim}> — {desc}</Text> : null}
        </Text>
      ))}
    </Box>
  )
}
