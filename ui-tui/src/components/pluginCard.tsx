import { Box, Text, useInput, useStdout } from '@hermes/ink'
import type { PluginCardWire } from '@hermes/shared/gateway-events'
import { useStore } from '@nanostores/react'
import { useMemo, useRef, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import {
  $pluginCards,
  activatePluginCard,
  closePluginCard,
  dismissPluginCard,
  getPluginCardState,
  openPluginCard
} from '../app/pluginCardStore.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { Dialog, Overlay } from './overlay.js'

interface ActionResult {
  card?: null | PluginCardWire
  kind: 'card' | 'text'
  text?: null | string
}

interface PluginCardSurfaceProps {
  dispatch: (pluginId: string, command: string, args: string) => Promise<ActionResult>
  onClose?: () => void
  sessionId: string
  theme: Theme
}

const wrapBody = (body: string, columns: number): string[] =>
  body.split('\n').flatMap(paragraph => {
    if (!paragraph) {
      return ['']
    }

    const lines: string[] = []
    let rest = paragraph

    while (rest.length > columns) {
      const space = rest.lastIndexOf(' ', columns)
      const end = space > 0 ? space : columns

      lines.push(rest.slice(0, end))
      rest = rest.slice(end + (space > 0 ? 1 : 0))
    }

    return [...lines, rest]
  })

export function PluginCardSurface({ dispatch, onClose = closePluginCard, sessionId, theme }: PluginCardSurfaceProps) {
  const state = useStore($pluginCards)
  const active = state.cards.find(entry => entry.key === state.activeKey)
  const [selected, setSelected] = useState(0)
  const [bodyOffset, setBodyOffset] = useState(0)
  const [pending, setPending] = useState(false)
  const pendingRef = useRef(false)
  const [feedback, setFeedback] = useState('')
  const [feedbackError, setFeedbackError] = useState(false)
  const { stdout } = useStdout()
  const width = Math.max(40, Math.min(88, (stdout?.columns ?? 80) - 6))
  const bodyHeight = Math.max(4, Math.min(12, (stdout?.rows ?? 24) - 10))
  const actions = active?.card.actions ?? []
  const bodyLines = active ? wrapBody(active.card.body, Math.max(20, width - 6)) : []
  const visibleBody = bodyLines.slice(bodyOffset, bodyOffset + bodyHeight).join('\n')

  const run = (index: number) => {
    const action = actions[index]

    if (!active || !action || pendingRef.current) {
      return
    }

    const originKey = active.key
    pendingRef.current = true
    setPending(true)
    setFeedback('')
    setFeedbackError(false)
    void dispatch(active.card.plugin_id, action.command, action.args)
      .then(result => {
        const current = getPluginCardState()

        if (current.sessionId !== sessionId || current.activeKey !== originKey) {
          return
        }

        if (result.kind === 'card' && result.card) {
          activatePluginCard(sessionId, result.card)
          setSelected(0)
          setBodyOffset(0)
        } else if (result.text?.trim()) {
          setFeedback(result.text.trim())
        } else {
          dismissPluginCard(originKey)
        }
      })
      .catch((reason: unknown) => {
        const current = getPluginCardState()

        if (current.sessionId === sessionId && current.activeKey === originKey) {
          setFeedback(rpcErrorMessage(reason))
          setFeedbackError(true)
        }
      })
      .finally(() => {
        pendingRef.current = false
        setPending(false)
      })
  }

  useInput((ch, key) => {
    if (!active) {
      return
    }

    if (key.escape) {
      onClose()
    } else if (key.upArrow || ch === 'k') {
      setBodyOffset(0)
    } else if (key.downArrow || ch === 'j') {
      setBodyOffset(Math.max(0, bodyLines.length - bodyHeight))
    } else if (key.leftArrow || (key.shift && key.tab)) {
      setSelected(index => Math.max(0, index - 1))
    } else if (key.rightArrow || key.tab) {
      setSelected(index => Math.min(Math.max(0, actions.length - 1), index + 1))
    } else if (key.return || ch === ' ') {
      run(selected)
    }
  })

  if (!active) {
    return null
  }

  return (
    <Overlay backdrop>
      <Dialog
        hint={pending ? 'Working…' : '↑↓ scroll · Tab choose · Enter run · Esc close'}
        title={`${active.card.plugin_name} · ${active.card.title}`}
        width={width}
      >
        <Box height={bodyHeight}>
          <Text color={theme.color.text}>{visibleBody}</Text>
        </Box>
        {actions.length > 0 ? (
          <Box flexDirection="row" gap={2} marginTop={1}>
            {actions.map((action, index) => (
              <Text
                bold={index === selected}
                color={index === selected ? theme.color.accent : theme.color.muted}
                key={action.id}
                onClick={() => run(index)}
              >
                {index === selected ? '› ' : '  '}
                {action.label}
              </Text>
            ))}
          </Box>
        ) : null}
        {feedback ? (
          <Box marginTop={1}>
            <Text color={feedbackError ? theme.color.error : theme.color.text}>{feedback}</Text>
          </Box>
        ) : null}
        <Box marginTop={1}>
          <Text color={theme.color.muted} onClick={() => dismissPluginCard(active.key)}>
            Dismiss
          </Text>
        </Box>
      </Dialog>
    </Overlay>
  )
}

export function PluginCardHost({ sessionId, theme }: { sessionId: string; theme: Theme }) {
  const { gw } = useGateway()
  const state = useStore($pluginCards)
  const count = state.cards.length
  const label = useMemo(() => `${count} plugin ${count === 1 ? 'card' : 'cards'} · Ctrl+O open`, [count])

  useInput((ch, key) => {
    if (!state.activeKey && count > 0 && key.ctrl && ch === 'o') {
      openPluginCard()
    }
  })

  if (!count) {
    return null
  }

  return (
    <>
      {!state.activeKey ? (
        <Box flexShrink={0} paddingX={1}>
          <Text color={theme.color.accent} onClick={() => openPluginCard()}>
            ◇ {label}
          </Text>
        </Box>
      ) : null}
      <PluginCardSurface
        dispatch={(pluginId, command, args) =>
          gw.request<ActionResult>('plugin.card.action', { args, command, plugin_id: pluginId, session_id: sessionId })
        }
        sessionId={sessionId}
        theme={theme}
      />
    </>
  )
}
