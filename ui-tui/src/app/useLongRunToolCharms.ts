import { useEffect, useRef } from 'react'

import { toolTrailLabel } from '../lib/text.js'
import type { ActiveTool, ActivityItem } from '../types.js'

const DELAY_MS = 8_000
const INTERVAL_MS = 10_000
const MAX = 2
const CHARMS = ['still cooking…', 'polishing edges…', 'asking the void nicely…']

export function useLongRunToolCharms(
  busy: boolean,
  tools: ActiveTool[],
  pushActivity: (text: string, tone?: ActivityItem['tone'], replaceLabel?: string) => void
) {
  const slotRef = useRef(new Map<string, { count: number; lastAt: number }>())

  useEffect(() => {
    if (!busy || !tools.length) {
      slotRef.current.clear()

      return
    }

    const tick = () => {
      const now = Date.now()
      const liveIds = new Set(tools.map(t => t.id))

      for (const key of [...slotRef.current.keys()]) {
        if (!liveIds.has(key)) {
          slotRef.current.delete(key)
        }
      }

      for (const tool of tools) {
        if (!tool.startedAt || now - tool.startedAt < DELAY_MS) {
          continue
        }

        const slot = slotRef.current.get(tool.id) ?? { count: 0, lastAt: 0 }

        if (slot.count >= MAX || now - slot.lastAt < INTERVAL_MS) {
          continue
        }

        slot.count += 1
        slot.lastAt = now
        slotRef.current.set(tool.id, slot)

        const charm = CHARMS[Math.floor(Math.random() * CHARMS.length)]!
        const sec = Math.round((now - tool.startedAt) / 1000)

        pushActivity(`${charm} (${toolTrailLabel(tool.name)} · ${sec}s)`)
      }
    }

    tick()
    const id = setInterval(tick, 1000)

    return () => clearInterval(id)
  }, [busy, pushActivity, tools])
}
