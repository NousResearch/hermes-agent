import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'

import type { SlashHandlerContext } from './interfaces.js'
import { createSlashCoreHandler } from './slash/createSlashCoreHandler.js'
import { createSlashOpsHandler } from './slash/createSlashOpsHandler.js'
import { createSlashSessionHandler } from './slash/createSlashSessionHandler.js'
import { createSlashShared, parseSlashCommand } from './slash/shared.js'
import { getUiState } from './uiStore.js'

export function createSlashHandler(ctx: SlashHandlerContext): (cmd: string) => boolean {
  const { gw } = ctx.gateway
  const { catalog } = ctx.local
  const { send, sys } = ctx.transcript
  const shared = createSlashShared({ ...ctx.transcript, gw })
  const handleCore = createSlashCoreHandler(ctx)
  const handleSession = createSlashSessionHandler(ctx, shared)
  const handleOps = createSlashOpsHandler(ctx)

  const handler = (cmd: string): boolean => {
    const ui = getUiState()
    const parsed = { ...parseSlashCommand(cmd), sid: ui.sid, ui }
    const argTail = parsed.arg ? ` ${parsed.arg}` : ''

    if (handleCore(parsed) || handleSession(parsed) || handleOps(parsed)) {
      return true
    }

    if (catalog?.canon) {
      const needle = `/${parsed.name}`.toLowerCase()

      const matches = [
        ...new Set(
          Object.entries(catalog.canon)
            .filter(([alias]) => alias.startsWith(needle))
            .map(([, canon]) => canon)
        )
      ]

      if (matches.length === 1 && matches[0]!.toLowerCase() !== needle) {
        return handler(`${matches[0]}${argTail}`)
      }

      if (matches.length > 1) {
        sys(`ambiguous command: ${matches.slice(0, 6).join(', ')}${matches.length > 6 ? ', …' : ''}`)

        return true
      }
    }

    gw.request('slash.exec', { command: cmd.slice(1), session_id: ui.sid })
      .then((r: any) => {
        sys(
          r?.warning
            ? `warning: ${r.warning}\n${r?.output || `/${parsed.name}: no output`}`
            : r?.output || `/${parsed.name}: no output`
        )
      })
      .catch(() => {
        gw.request('command.dispatch', { name: parsed.name, arg: parsed.arg, session_id: ui.sid })
          .then((raw: any) => {
            const d = asRpcResult(raw)

            if (!d?.type) {
              sys('error: invalid response: command.dispatch')

              return
            }

            if (d.type === 'exec' || d.type === 'plugin') {
              sys(d.output || '(no output)')
            } else if (d.type === 'alias') {
              handler(`/${d.target}${argTail}`)
            } else if (d.type === 'skill') {
              sys(`⚡ loading skill: ${d.name}`)

              if (typeof d.message === 'string' && d.message.trim()) {
                send(d.message)
              } else {
                sys(`/${parsed.name}: skill payload missing message`)
              }
            }
          })
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
      })

    return true
  }

  return handler
}
