import { Box, Text } from 'ink'

import { LONG_MSG, ROLE } from '../constants.js'
import { hasAnsi, userDisplay } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

import { Md } from './markdown.js'

export function MessageLine({ compact, msg, t }: { compact?: boolean; msg: Msg; t: Theme }) {
  const { body, glyph, prefix } = ROLE[msg.role](t)

  const content = (() => {
    if (msg.role === 'assistant') {
      if (hasAnsi(msg.text)) {
        return <Text>{msg.text}</Text>
      }

      return <Md compact={compact} t={t} text={msg.text} />
    }

    if (msg.role === 'user' && msg.text.length > LONG_MSG) {
      const displayed = userDisplay(msg.text)
      const [head, ...rest] = displayed.split('[long message]')

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

    return <Text color={body}>{msg.text}</Text>
  })()

  return (
    <Box>
      <Box width={3}>
        <Text bold={msg.role === 'user'} color={prefix}>
          {glyph}{' '}
        </Text>
      </Box>
      {content}
    </Box>
  )
}
