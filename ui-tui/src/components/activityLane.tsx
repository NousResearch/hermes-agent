import { Box, Text } from '@hermes/ink'

import type { Theme } from '../theme.js'
import type { ActivityItem } from '../types.js'

const toneColor = (item: ActivityItem, t: Theme) =>
  item.tone === 'error' ? t.color.error : item.tone === 'warn' ? t.color.warn : t.color.dim

export function ActivityLane({ items, t }: { items: ActivityItem[]; t: Theme }) {
  if (!items.length) {
    return null
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      {items.slice(-4).map(item => (
        <Text color={toneColor(item, t)} dimColor={item.tone === 'info'} key={item.id}>
          {t.brand.tool} {item.text}
        </Text>
      ))}
    </Box>
  )
}
