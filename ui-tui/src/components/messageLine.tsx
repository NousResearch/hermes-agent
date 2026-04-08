import { Box, Text } from 'ink'
import { memo } from 'react'

import { LONG_MSG, ROLE } from '../constants.js'
import { hasAnsi, userDisplay } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'
import { Md } from './markdown.js'

export const MessageLine = memo(function MessageLine({ cols, compact, msg, t }: { cols: number; compact?: boolean; msg: Msg; t: Theme }) {
  const { body, glyph, prefix } = ROLE[msg.role](t)
  const contentWidth = Math.max(20, cols - 5)

  if (msg.role === 'tool') {
    return (
      <Text color={t.color.dim} wrap="wrap">
        {'  '}{msg.text}
      </Text>
    )
  }

  const content = (() => {
    if (msg.role === 'assistant')
      return hasAnsi(msg.text) ? <Text wrap="wrap">{msg.text}</Text> : <Md compact={compact} t={t} text={msg.text} />

    if (msg.role === 'user' && msg.text.length > LONG_MSG) {
      const [head, ...rest] = userDisplay(msg.text).split('[long message]')

      return (
        <Text color={body}>
          {head}
          <Text color={t.color.dim} dimColor>[long message]</Text>
          {rest.join('')}
        </Text>
      )
    }

    return <Text {...(body ? { color: body } : {})}>{msg.text}</Text>
  })()

  return (
    <Box flexDirection="column">
      {(msg.role === 'user' || msg.role === 'assistant') && <Text>{' '}</Text>}

      <Box>
        <Box flexShrink={0} width={3}>
          <Text bold={msg.role === 'user'} color={prefix}>{glyph} </Text>
        </Box>

        <Box width={contentWidth}>
          {content}
        </Box>
      </Box>
    </Box>
  )
})
