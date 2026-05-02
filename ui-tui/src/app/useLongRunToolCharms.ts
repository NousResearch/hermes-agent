import { useEffect, useRef } from 'react'

import { LONG_RUN_CHARMS } from '../content/charms.js'
import { pick, toolTrailLabel } from '../lib/text.js'

import { turnController } from './turnController.js'
import { useTurnSelector } from './turnStore.js'
import { getUiState } from './uiStore.js'

const DELAY_MS = 8_000
const INTERVAL_MS = 10_000
const MAX_CHARMS_PER_TOOL = 2

interface Slot {
  count: number
  lastAt: number
}

export function useLongRunToolCharms() {
  const tools = useTurnSelector(state => state.tools)
  const slots = useRef(new Map<string, Slot>())

  useEffect(() => {
    if (!getUiState().busy || !tools.length) {
      slots.current.clear()

      return
    }

    const tick = () => {
      if (!getUiState().busy) {
        slots.current.clear()

        return
      }

      const now = Date.now()
      const liveIds = new Set(tools.map(t => t.id))

      for (const key of Array.from(slots.current.keys())) {
        if (!liveIds.has(key)) {
          slots.current.delete(key)
        }
      }

      for (const tool of tools) {
        if (!tool.startedAt || now - tool.startedAt < DELAY_MS) {
          continue
        }

        const slot = slots.current.get(tool.id) ?? { count: 0, lastAt: 0 }

        if (slot.count >= MAX_CHARMS_PER_TOOL || now - slot.lastAt < INTERVAL_MS) {
          continue
        }

        slots.current.set(tool.id, { count: slot.count + 1, lastAt: now })
        turnController.pushActivity(
          `${pick(LONG_RUN_CHARMS)} (${toolTrailLabel(tool.name)} · ${Math.round((now - tool.startedAt) / 1000)}s)`
        )
      }
    }

    tick()
    const id = setInterval(tick, 1000)

    return () => clearInterval(id)
  }, [tools])
}
