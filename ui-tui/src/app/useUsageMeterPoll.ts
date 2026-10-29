import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { SystemUsageMeterProvider, SystemUsageMeterResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'

import type { AccountUsageMeterInfo, UsageMeterInfo } from './interfaces.js'
import { $uiState, patchUiState } from './uiStore.js'

const USAGE_METER_POLL_MS = 30_000

const normalizeProvider = (provider: SystemUsageMeterProvider | undefined): AccountUsageMeterInfo => {
  const windows = Array.isArray(provider?.windows)
    ? provider.windows.flatMap(window => {
        if (
          !window ||
          typeof window.label !== 'string' ||
          typeof window.used_percent !== 'number' ||
          !Number.isFinite(window.used_percent)
        ) {
          return []
        }

        return [
          {
            label: window.label,
            usedPercent: Math.max(0, Math.min(100, Math.round(window.used_percent)))
          }
        ]
      })
    : []

  return {
    available: provider?.available === true && windows.length > 0,
    ...(typeof provider?.unavailable_reason === 'string'
      ? { unavailableReason: provider.unavailable_reason }
      : {}),
    windows
  }
}

export const toUsageMeterInfo = (response: null | SystemUsageMeterResponse): null | UsageMeterInfo =>
  response
    ? {
        claude: normalizeProvider(response.claude),
        codex: normalizeProvider(response.codex)
      }
    : null

export function useUsageMeterPoll(gw: GatewayClient) {
  const enabled = useStore($uiState).usageMeter

  useEffect(() => {
    if (!enabled) {
      patchUiState({ usageMeterStatus: null })

      return
    }

    let cancelled = false

    const poll = async () => {
      try {
        const response = asRpcResult<SystemUsageMeterResponse>(
          await gw.request<SystemUsageMeterResponse>('system.usage_meter', {})
        )

        if (!cancelled) {
          patchUiState({ usageMeterStatus: toUsageMeterInfo(response) })
        }
      } catch {
        // Preserve the last good dual-provider reading during transient failures.
      }
    }

    void poll()
    const id = setInterval(() => void poll(), USAGE_METER_POLL_MS)

    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [enabled, gw])
}
