import { Ansi, Box, NoSelect, Text } from '@hermes/ink'
import { memo } from 'react'

import { SECTION_NAMES, sectionMode } from '../domain/details.js'
import { LONG_MSG } from '../config/limits.js'
import { userDisplay } from '../domain/messages.js'
import { ROLE } from '../domain/roles.js'
import { compactPreview, hasAnsi, isPasteBackedText, stripAnsi } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { DetailsMode, Msg, SectionVisibility } from '../types.js'

import { Md } from './markdown.js'
import { ToolTrail } from './thinking.js'

export const MessageLine = memo(function MessageLine({
  cols,
  compact,
  detailsMode = 'collapsed',
  isStreaming = false,
  msg,
  sections,
  t
}: MessageLineProps) {
  if (msg.kind === 'trail' && msg.tools?.length) {
    // Per-section overrides win over the global mode, so don't pre-empt on
    // `detailsMode === 'hidden'` — only skip when EVERY section is hidden,
    // matching ToolTrail's own internal short-circuit.
    const anyVisible = SECTION_NAMES.some(s => sectionMode(s, detailsMode, sections) !== 'hidden')

    return anyVisible ? (
      <Box flexDirection="column" marginTop={1}>
        <ToolTrail detailsMode={detailsMode} sections={sections} t={t} trail={msg.tools} />
      </Box>
    ) : null
  }

  if (msg.role === 'tool') {
    const maxChars = Math.max(24, cols - 14)
    const stripped = hasAnsi(msg.text) ? stripAnsi(msg.text) : msg.text
    const preview = compactPreview(stripped, maxChars) || '(empty tool result)'

    return (
      <Box alignSelf="flex-start" borderColor={t.color.dim} borderStyle="round" marginLeft={3} paddingX={1}>
        {hasAnsi(msg.text) ? (
          <Text wrap="truncate-end">
            <Ansi>{msg.text}</Ansi>
          </Text>
        ) : (
          <Text color={t.color.dim} wrap="truncate-end">
            {preview}
          </Text>
        )}
      </Box>
    )
  }

  const { body, glyph, prefix } = ROLE[msg.role](t)
  const thinking = msg.thinking?.trim() ?? ''
  const showDetails = detailsMode !== 'hidden' && (Boolean(msg.tools?.length) || Boolean(thinking))

  const content = (() => {
    if (msg.kind === 'slash') {
      return <Text color={t.color.dim}>{msg.text}</Text>
    }

    if (msg.role !== 'user' && hasAnsi(msg.text)) {
      return <Ansi>{msg.text}</Ansi>
    }

    if (msg.role === 'assistant') {
      return isStreaming ? <Text color={body}>{msg.text}</Text> : <Md compact={compact} t={t} text={msg.text} />
    }

    if (msg.role === 'user' && msg.text.length > LONG_MSG && isPasteBackedText(msg.text)) {
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

  // Diff segments (emitted by pushInlineDiffSegment between narration
  // segments) need a blank line on both sides so the patch doesn't butt up
  // against the prose around it.
  const isDiffSegment = msg.kind === 'diff'

  return (
    <Box
      flexDirection="column"
      marginBottom={msg.role === 'user' || isDiffSegment ? 1 : 0}
      marginTop={msg.role === 'user' || msg.kind === 'slash' || isDiffSegment ? 1 : 0}
    >
      {showDetails && (
        <Box flexDirection="column" marginBottom={1}>
          <ToolTrail
            detailsMode={detailsMode}
            reasoning={thinking}
            reasoningTokens={msg.thinkingTokens}
            sections={sections}
            t={t}
            toolTokens={msg.toolTokens}
            trail={msg.tools}
          />
        </Box>
      )}

      <Box>
        <NoSelect flexShrink={0} fromLeftEdge width={3}>
          <Text bold={msg.role === 'user'} color={prefix}>
            {glyph}{' '}
          </Text>
        </NoSelect>

        <Box width={Math.max(20, cols - 5)}>{content}</Box>
      </Box>
    </Box>
  )
})

interface MessageLineProps {
  cols: number
  compact?: boolean
  detailsMode?: DetailsMode
  isStreaming?: boolean
  msg: Msg
  sections?: SectionVisibility
  t: Theme
}
