import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { SystemBatteryResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'

import type { BatteryCategory, BatteryInfo } from './interfaces.js'
import { $uiState, patchUiState } from './uiStore.js'

const BATTERY_POLL_MS = 30_000

const CATEGORIES: ReadonlySet<BatteryCategory> = new Set(['bad', 'critical', 'dim', 'good', 'warn'])

const normalizeCategory = (raw: unknown): BatteryCategory =>
  typeof raw === 'string' && CATEGORIES.has(raw as BatteryCategory) ? (raw as BatteryCategory) : 'dim'

/** Coerce a `system.battery` RPC payload into the UI's BatteryInfo shape. */
export const toBatteryInfo = (r: null | SystemBatteryResponse): BatteryInfo | null => {
  if (!r) {
    return null
  }

  const percent =
    typeof r.percent === 'number' && Number.isFinite(r.percent)
      ? Math.max(0, Math.min(100, Math.round(r.percent)))
      : null

  return {
    available: !!r.available,
    category: normalizeCategory(r.category),
    percent,
    plugged: typeof r.plugged === 'boolean' ? r.plugged : null
  }
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
        const r = asRpcResult<SystemBatteryResponse>(await gw.request<SystemBatteryResponse>('system.battery', {}))

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
