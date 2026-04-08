import { Box, Text } from 'ink'

import type { Theme } from '../theme.js'
import type { ActivityItem } from '../types.js'

export function ActivityLane({ items, t }: { items: ActivityItem[]; t: Theme }) {
  if (!items.length) {
    return null
  }

  const visible = items.slice(-4)

  return (
    <Box flexDirection="column" marginTop={1}>
      {visible.map(item => {
        const color = item.tone === 'error' ? t.color.error : item.tone === 'warn' ? t.color.warn : t.color.dim

        return (
          <Text color={color} dimColor={item.tone === 'info'} key={item.id}>
            {t.brand.tool} {item.text}
          </Text>
        )
      })}
    </Box>
  )
}
