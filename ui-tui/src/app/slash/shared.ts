import type { SlashExecResponse } from '../../gatewayTypes.js'
import { rpcErrorMessage } from '../../lib/rpc.js'

export const parseSlashCommand = (cmd: string): ParsedSlashCommand => {
  const [rawName = '', ...rest] = cmd.slice(1).split(/\s+/)

  return {
    arg: rest.join(' '),
    cmd,
    name: rawName.toLowerCase()
  }
}

export const createSlashShared = ({ gw, page, sys }: SlashSharedDeps): SlashShared => ({
  showSlashOutput: (title, command, sid) => {
    gw.request<SlashExecResponse>('slash.exec', { command, session_id: sid })
      .then(r => {
        const text = r?.warning ? `warning: ${r.warning}\n${r?.output || '(no output)'}` : r?.output || '(no output)'

        const lines = text.split('\n').filter(Boolean)

        if (lines.length > 2 || text.length > 180) {
          page(text, title)
        } else {
          sys(text)
        }
      })
      .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
  }
})

export interface ParsedSlashCommand {
  arg: string
  cmd: string
  name: string
}

export interface SlashShared {
  showSlashOutput: (title: string, command: string, sid: null | string) => void
}

interface SlashSharedDeps {
  gw: {
    request: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  }
  page: (text: string, title?: string) => void
  sys: (text: string) => void
}
