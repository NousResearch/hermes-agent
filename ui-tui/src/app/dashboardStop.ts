import { useStdin } from '@hermes/ink'
import { useEffect } from 'react'

import { type InterruptDeps, turnController } from './turnController.js'
import { $uiState, getUiState } from './uiStore.js'

const DASHBOARD_OSC = 6973
const validNonce = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{32}$/.test(value)

export type DashboardResponse = { type: string; code?: number; data?: string }
type DashboardDeps = Omit<InterruptDeps, 'sid'>

function dashboardMessage(response: DashboardResponse): Record<string, unknown> | null {
  if (response.type !== 'osc' || response.code !== DASHBOARD_OSC || typeof response.data !== 'string') {return null}

  try {
    const value: unknown = JSON.parse(response.data)

    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
  } catch {
    return null
  }
}

function currentBinding(gw: DashboardDeps['gw']) {
  const live = getUiState()
  const info = live.info

  return gw.isCanonical && live.busy && !turnController.interrupted && live.sid &&
    typeof info?.execution_epoch === 'string' && Number.isSafeInteger(info.execution_generation)
    ? { sid: live.sid, epoch: info.execution_epoch, generation: info.execution_generation } : null
}

export async function handleDashboardStop(response: DashboardResponse, deps: DashboardDeps): Promise<void> {
  const message = dashboardMessage(response)
  const bound = currentBinding(deps.gw)

  if (!bound || message?.action !== 'stop' || !validNonce(message.nonce) || message.sid !== bound.sid ||
      message.epoch !== bound.epoch || message.generation !== bound.generation) {return}

  await turnController.interruptTurn({ ...deps, sid: bound.sid })
}

export function useDashboardStop({ enabled, stdout, gw, appendMessage, sys }: DashboardDeps & {
  enabled: boolean
  stdout?: NodeJS.WriteStream
}) {
  const { inputEmitter } = useStdin()
  useEffect(() => {
    if (!enabled || !stdout) {return}
    let last = ''
    let active = true

    const publish = (nonce?: string) => {
      if (!active) {return}
      const bound = currentBinding(gw)
      const state = JSON.stringify({ action: 'state', bound })

      if (!nonce && state === last) {return}
      last = state
      stdout.write(`\x1b]${DASHBOARD_OSC};${JSON.stringify({ action: 'state', bound, nonce })}\x07`)
    }

    const onResponse = (response: DashboardResponse) => {
      const message = dashboardMessage(response)

      if (message?.action === 'query' && validNonce(message.nonce)) {
        publish(message.nonce)
      } else if (message?.action === 'stop' && validNonce(message.nonce)) {
        const nonce = message.nonce
        void handleDashboardStop(response, { gw, appendMessage, sys }).catch(() => {}).finally(() => publish(nonce))
      }
    }

    inputEmitter.on('terminalresponse', onResponse)
    const unsubscribe = $uiState.subscribe(() => publish())

    return () => { active = false; unsubscribe(); inputEmitter.off('terminalresponse', onResponse) }
  }, [appendMessage, enabled, gw, inputEmitter, stdout, sys])
}
