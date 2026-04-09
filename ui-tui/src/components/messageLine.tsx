import { Box, Text } from 'ink'
import { memo } from 'react'

import { LONG_MSG, ROLE } from '../constants.js'
import { hasAnsi, userDisplay } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

import { Md } from './markdown.js'

export const MessageLine = memo(function MessageLine({
  cols,
  compact,
  msg,
  t
}: {
  cols: number
  compact?: boolean
  msg: Msg
  t: Theme
}) {
  const { body, glyph, prefix } = ROLE[msg.role](t)

  if (msg.role === 'tool') {
    return (
      <Box alignSelf="flex-start" borderColor={t.color.dim} borderStyle="round" marginLeft={3} paddingX={1}>
        <Text color={t.color.dim}>{msg.text}</Text>
      </Box>
    )
  }

  const content = (() => {
    if (msg.role === 'assistant') {
      return hasAnsi(msg.text) ? <Text wrap="wrap">{msg.text}</Text> : <Md compact={compact} t={t} text={msg.text} />
    }

    if (msg.role === 'user' && msg.text.length > LONG_MSG) {
      const [head, ...rest] = userDisplay(msg.text).split('[long message]')

      return (
        <Text color={body}>
          {head}
          <Text color={t.color.dim} dimColor>
            [long message]
          </Text>
          {rest.join('')}
        </Text>
      )
    }

    return <Text {...(body ? { color: body } : {})}>{msg.text}</Text>
  })()

  return (
    <Box flexDirection="column" marginTop={msg.role === 'user' ? 1 : 0}>
      {msg.thinking && (
        <Text color={t.color.dim} dimColor wrap="truncate-end">
          💭 {msg.thinking.replace(/\n/g, ' ').slice(0, 200)}
        </Text>
      )}

      <Box>
        <Box flexShrink={0} width={3}>
          <Text bold={msg.role === 'user'} color={prefix}>
            {glyph}{' '}
          </Text>
        </Box>

        <Box width={Math.max(20, cols - 5)}>{content}</Box>
      </Box>

      {!!msg.tools?.length && (
        <Box flexDirection="column" marginBottom={1} marginTop={1}>
          {msg.tools.map((tool, i) => (
            <Text
              color={tool.endsWith(' ✗') ? t.color.error : t.color.dim}
              dimColor={!tool.endsWith(' ✗')}
              key={`${tool}-${i}`}
            >
              {t.brand.tool} {tool}
            </Text>
          ))}
        </Box>
      )}
    </Box>
  )
})
