import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useState } from 'react'

import { patchOverlayState } from '../app/overlayStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

interface CrCounts {
  needs_you?: number
  agents_active?: number
  messages_unread?: number
}

interface CrSnapshotBadge {
  counts?: CrCounts
}

const POLL_MS = 5000

/**
 * KENSEI CUSTOM: compact, non-focus-stealing Control Room attention badge
 * (CR-305). Renders the plan's status strip — `● N need you · Ctrl+P Control
 * Room` — immediately around the input area. Bounded polling (5s) rides the
 * server's 2s snapshot cache, so this never becomes a high-frequency loop.
 * It never steals focus and never auto-opens the overlay.
 */
export function ControlRoomAttentionBadge({ gw, t }: { gw: GatewayClient; t: Theme }) {
  const [counts, setCounts] = useState<CrCounts | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null

    const fetchCounts = () => {
      gw.request<CrSnapshotBadge>('control.room.snapshot', { profile: 'default' })
        .then(raw => {
          const snap = asRpcResult<CrSnapshotBadge>(raw)

          if (snap && !cancelled) {setCounts(snap.counts ?? null)}
        })
        .catch(() => {
          // Gateway unavailable — stay silent (never render a fake zero).
        })
    }

    fetchCounts()
    timer = setInterval(fetchCounts, POLL_MS)

    return () => {
      cancelled = true

      if (timer) {clearInterval(timer)}
    }
  }, [gw])

  useInput((ch, key) => {
    // Ctrl+P from the badge area opens Control Room without stealing focus.
    if (key.ctrl && ch.toLowerCase() === 'p') {
      patchOverlayState({ controlRoom: true })
    }
  })

  if (!counts) {
    return null
  }

  const bits: string[] = []

  if (counts.needs_you) {bits.push(`● ${counts.needs_you} need you`)}

  if (counts.agents_active) {bits.push(`${counts.agents_active} agents active`)}

  if (counts.messages_unread) {bits.push(`${counts.messages_unread} unread`)}

  if (!bits.length) {return null}

  return (
    <Box>
      <Text color={t.color.label}>
        {bits.join(' · ')} · Ctrl+P Control Room
      </Text>
    </Box>
  )
}
