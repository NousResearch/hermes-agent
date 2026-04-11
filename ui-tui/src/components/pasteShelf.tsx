import { Box, Text } from '@hermes/ink'

import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { PendingPaste } from '../types.js'

const TOKEN_RE = /\[\[paste:(\d+)\]\]/g

const modeLabel = {
  attach: 'attach',
  excerpt: 'excerpt',
  inline: 'inline'
} as const

export function PasteShelf({ draft, pastes, t }: { draft: string; pastes: PendingPaste[]; t: Theme }) {
  if (!pastes.length) {
    return null
  }

  const inDraft = new Set<number>()

  for (const m of draft.matchAll(TOKEN_RE)) {
    inDraft.add(parseInt(m[1] ?? '-1', 10))
  }

  return (
    <Box borderColor={t.color.bronze} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
      <Text color={t.color.amber}>Paste shelf ({pastes.length})</Text>
      {pastes.slice(-4).map(paste => (
        <Text color={t.color.dim} key={paste.id}>
          #{paste.id} {modeLabel[paste.mode]} · {paste.lineCount}L · {paste.kind}
          {inDraft.has(paste.id) ? <Text color={t.color.label}> · in draft</Text> : ''}
          {' · '}
          <Text color={t.color.cornsilk}>{compactPreview(paste.text, 44) || '(empty)'}</Text>
        </Text>
      ))}
      {pastes.length > 4 && (
        <Text color={t.color.dim} dimColor>
          …and {pastes.length - 4} more
        </Text>
      )}
      <Text color={t.color.dim} dimColor>
        /paste mode {'<id>'} {'<attach|excerpt|inline>'} · /paste drop {'<id>'} · /paste clear
      </Text>
    </Box>
  )
}
