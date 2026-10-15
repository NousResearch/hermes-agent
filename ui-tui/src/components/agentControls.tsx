import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { messages } from '../i18n/runtime.js'
import { useT } from '../i18n/useT.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

export function rosterViewport(height: number, count: number, cursor: number) {
  const timelineRows = height >= 32 ? Math.min(4, count) : 0
  const rows = Math.max(1, height - 7 - (timelineRows ? timelineRows + 4 : 0))
  const start = Math.max(0, Math.min(Math.max(0, count - rows), cursor - Math.floor(rows / 2)))

  return { rows, start, timelineRows }
}

export async function sendAgentSteer(gw: GatewayClient, sid: string, id: string, text: string) {
  // The shared owner serves no subagent.steer: the sidecar would answer "session not found".
  if (gw.isCanonical) {
    return { accepted: false, message: messages().canonical.controls.notAvailable('agents steer') }
  }

  const result = asRpcResult<{ status: string }>(
    await gw.request('subagent.steer', { session_id: sid, subagent_id: id, text })
  )

  const accepted = result?.status === 'queued'
  const T = messages().hubs.agentControls

  return {
    accepted,
    message: accepted ? T.queued : T.notQueued
  }
}

interface ControlProps {
  gw: GatewayClient
  sid: string
  id: string
  t: Theme
}

export function AgentSteerForm({
  gw,
  sid,
  id,
  t,
  cols,
  onClose
}: ControlProps & { cols: number; onClose: () => void }) {
  const T = useT().hubs.agentControls
  const [text, setText] = useState('')
  const [feedback, setFeedback] = useState('')
  const [pending, setPending] = useState(false)
  useInput((_ch, key) => {
    if (key.escape && !pending) {
      onClose()
    }
  })

  const submit = async () => {
    if (!text.trim() || pending) {
      return
    }

    setPending(true)

    try {
      const result = await sendAgentSteer(gw, sid, id, text)
      setFeedback(result.message)

      if (result.accepted) {
        setText('')
      }
    } catch (error) {
      setFeedback(T.notQueuedError(error instanceof Error ? error.message : String(error)))
    } finally {
      setPending(false)
    }
  }

  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        {T.steerTitle(id)}
      </Text>
      <Text color={t.color.muted}>{T.steerIntro}</Text>
      <Box marginTop={1}>
        <Text color={t.color.accent}>❯ </Text>
        <TextInput
          color={t.color.text}
          columns={Math.max(1, cols - 6)}
          focus={!pending}
          onChange={setText}
          onSubmit={() => void submit()}
          value={text}
        />
      </Box>
      <Text color={t.color.muted}>{pending ? T.queueing : feedback}</Text>
      <Text color={t.color.muted}>{T.steerHint}</Text>
    </Box>
  )
}

export function AgentLiveTail({ gw, sid, id, t }: ControlProps) {
  const T = useT().hubs.agentControls
  const [tail, setTail] = useState(() => T.loadingTranscript)
  useEffect(() => {
    const M = messages().hubs.agentControls
    let active = true
    let pending = false

    const refresh = async () => {
      if (pending) {
        return
      }

      pending = true

      try {
        const result = asRpcResult<{ available: boolean; text: string; truncated: boolean }>(
          await gw.request('subagent.tail', { session_id: sid, subagent_id: id })
        )

        if (active) {
          setTail(
            result?.available
              ? `${result.truncated ? `${M.truncatedPrefix}\n` : ''}${result.text}`
              : M.transcriptUnavailable
          )
        }
      } catch {
        if (active) {
          setTail(M.transcriptRefreshFailed)
        }
      } finally {
        pending = false
      }
    }

    void refresh()
    const timer = setInterval(() => void refresh(), 1500)

    return () => {
      active = false
      clearInterval(timer)
    }
  }, [gw, sid, id])

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {T.transcriptTitle}
      </Text>
      <Text color={t.color.text} wrap="wrap">
        {tail}
      </Text>
    </Box>
  )
}
