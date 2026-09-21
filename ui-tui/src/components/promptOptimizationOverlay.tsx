import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'
import type { PromptOptimizationReq } from '../types.js'

const OPTS = ['accept', 'reject', 'edit'] as const

const LABELS: Record<(typeof OPTS)[number], string> = {
  accept: 'Accept rewrite',
  edit: 'Edit before send',
  reject: 'Send original'
}

type OptKey = {
  downArrow?: boolean
  escape?: boolean
  return?: boolean
  upArrow?: boolean
}

type OptAction =
  | { kind: 'choose'; choice: (typeof OPTS)[number] }
  | { kind: 'move'; delta: -1 | 1 }
  | { kind: 'noop' }

export function promptOptimizationAction(ch: string, key: OptKey, sel: number): OptAction {
  if (key.escape) {
    return { kind: 'choose', choice: 'reject' }
  }

  const n = parseInt(ch, 10)

  if (n >= 1 && n <= OPTS.length) {
    return { kind: 'choose', choice: OPTS[n - 1]! }
  }

  if (key.return) {
    return { kind: 'choose', choice: OPTS[sel]! }
  }

  if (key.upArrow && sel > 0) {
    return { kind: 'move', delta: -1 }
  }

  if (key.downArrow && sel < OPTS.length - 1) {
    return { kind: 'move', delta: 1 }
  }

  return { kind: 'noop' }
}

export function PromptOptimizationOverlay({ onChoice, req, t }: PromptOptimizationOverlayProps) {
  const [sel, setSel] = useState(0)

  useInput((ch, key) => {
    const action = promptOptimizationAction(ch, key, sel)

    if (action.kind === 'choose') {
      onChoice(action.choice)
    } else if (action.kind === 'move') {
      setSel(s => s + action.delta)
    }
  })

  const p = req.preview
  const maxLines = Math.max(3, Math.min(10, p.rewritten.split('\n').length))

  const originalShort =
    p.original.length > 120 ? `${p.original.slice(0, 120)}…` : p.original

  const rewrittenShort =
    p.rewritten.length > 200 ? `${p.rewritten.slice(0, 200)}…` : p.rewritten

  return (
    <Box
      borderColor={t.color.accent}
      borderStyle="double"
      flexDirection="column"
      paddingX={1}
    >
      <Text bold color={t.color.accent}>
        ✨ prompt optimized · {p.model_profile}
        {p.template_name ? ` · ${p.template_name}` : ''}
      </Text>

      {req.reason ? (
        <Text color={t.color.muted}>reason: {req.reason}</Text>
      ) : null}

      <Box flexDirection="column" marginY={1}>
        <Text bold color={t.color.label}>
          Original:
        </Text>
        <Text color={t.color.text} wrap="truncate-end">
          {originalShort || ' '}
        </Text>

        <Text bold color={t.color.label} marginTop={1}>
          Rewritten (+{p.quality_after - p.quality_before > 0 ? '+' : ''}
          {p.quality_after - p.quality_before} quality · {p.token_delta_pct > 0 ? '+' : ''}
          {p.token_delta_pct}% tokens):
        </Text>
        <Text color={t.color.text} wrap="truncate-end">
          {rewrittenShort || ' '}
        </Text>
      </Box>

      <Text />

      {OPTS.map((o, i) => (
        <Text key={o}>
          <Text
            bold={sel === i}
            color={sel === i ? t.color.accent : t.color.muted}
            inverse={sel === i}
          >
            {sel === i ? '▸ ' : '  '}
            {i + 1}. {LABELS[o]}
          </Text>
        </Text>
      ))}

      <Text color={t.color.muted}>
        ↑/↓ select · Enter confirm · 1-3 quick pick · Esc/Ctrl+C send original
      </Text>
    </Box>
  )
}

interface PromptOptimizationOverlayProps {
  onChoice: (choice: string) => void
  req: PromptOptimizationReq
  t: Theme
}
