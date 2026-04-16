import type { MutableRefObject } from 'react'

import type { SlashExecResponse } from '../../gatewayTypes.js'
import { rpcErrorMessage } from '../../lib/rpc.js'
import { getUiState } from '../uiStore.js'

export interface SlashShared {
  showSlashOutput: (opts: { command: string; flight: number; sid: null | string; title: string }) => void
}

interface SlashSharedDeps {
  gw: { request: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T> }
  page: (text: string, title?: string) => void
  slashFlightRef: MutableRefObject<number>
  sys: (text: string) => void
}

export const createSlashShared = ({ gw, page, slashFlightRef, sys }: SlashSharedDeps): SlashShared => ({
  showSlashOutput: ({ command, flight, sid, title }) => {
    const stale = () => flight !== slashFlightRef.current || getUiState().sid !== sid

    gw.request<SlashExecResponse>('slash.exec', { command, session_id: sid })
      .then(r => {
        if (stale()) {
          return
        }

        const text = r?.warning ? `warning: ${r.warning}\n${r?.output || '(no output)'}` : r?.output || '(no output)'

        text.split('\n').filter(Boolean).length > 2 || text.length > 180 ? page(text, title) : sys(text)
      })
      .catch((e: unknown) => {
        if (!stale()) {
          sys(`error: ${rpcErrorMessage(e)}`)
        }
      })
  }
})
