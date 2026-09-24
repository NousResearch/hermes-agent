import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
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
  const result = asRpcResult<{ status: string }>(
    await gw.request('subagent.steer', { session_id: sid, subagent_id: id, text })
  )

  const accepted = result?.status === 'queued'

  return {
    accepted,
    message: accepted
      ? 'Queued for child — applied at the next tool boundary.'
      : 'Not queued: child has finished or is no longer accepting guidance.'
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
      setFeedback(`Not queued: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setPending(false)
    }
  }

  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        Steer {id}
      </Text>
      <Text color={t.color.muted}>Guidance queues at the next tool boundary; current work is not interrupted.</Text>
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
      <Text color={t.color.muted}>{pending ? 'Queueing…' : feedback}</Text>
      <Text color={t.color.muted}>Enter queue · Esc back · main composer draft is preserved</Text>
    </Box>
  )
}

export function AgentLiveTail({ gw, sid, id, t }: ControlProps) {
  const [tail, setTail] = useState('Loading live transcript…')
  useEffect(() => {
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
              ? `${result.truncated ? '[last 16 KiB]\n' : ''}${result.text}`
              : 'Live transcript unavailable; child may have finished. Progress and output remain below.'
          )
        }
      } catch {
        if (active) {
          setTail('Could not refresh live transcript.')
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
        Live transcript
      </Text>
      <Text color={t.color.text} wrap="wrap">
        {tail}
      </Text>
    </Box>
  )
}
