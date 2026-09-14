import type { SystemBatteryResult } from '@hermes/shared/gateway-events'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'

import { $uiState, patchUiState } from './uiStore.js'

const BATTERY_POLL_MS = 30_000

/** Clamp a `system.battery` reading into the 0-100 the status bar renders. */
export const toBatteryInfo = (r: null | SystemBatteryResult): SystemBatteryResult | null => {
  if (!r) {
    return null
  }

  const percent =
    r.percent === null || !Number.isFinite(r.percent) ? null : Math.max(0, Math.min(100, Math.round(r.percent)))

  return { ...r, percent }
}

/**
 * Poll the host battery while the status-bar indicator is enabled.
 *
 * The reading is a system property (not per-session), so this runs whenever
 * `display.battery` is on — no `sid` gate. Python memoises the read, so a
 * 30s cadence is plenty to keep the read-out fresh without churn. When the
 * indicator is toggled off the cached reading is cleared.
 */
export function useBatteryPoll(gw: GatewayClient) {
  const enabled = useStore($uiState).battery

  useEffect(() => {
    if (!enabled) {
      patchUiState({ batteryStatus: null })

      return
    }

    let cancelled = false

    const poll = async () => {
      try {
        const r = await gw.request('system.battery', {})

        if (!cancelled) {
          patchUiState({ batteryStatus: toBatteryInfo(r) })
        }
      } catch {
        // Keep the last-good reading on a transient RPC failure.
      }
    }

    void poll()
    const id = setInterval(() => void poll(), BATTERY_POLL_MS)

    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [enabled, gw])
}
