import { parseSlashCommand } from '../domain/slash.js'
import type { SlashExecResponse } from '../gatewayTypes.js'
import { asCommandDispatch, rpcErrorMessage } from '../lib/rpc.js'

import type { SlashHandlerContext } from './interfaces.js'
import { findSlashCommand } from './slash/registry.js'
import { createSlashShared } from './slash/shared.js'
import type { SlashRunCtx } from './slash/types.js'
import { getUiState } from './uiStore.js'

export function createSlashHandler(ctx: SlashHandlerContext): (cmd: string) => boolean {
  const { gw } = ctx.gateway
  const { catalog } = ctx.local
  const { send, sys } = ctx.transcript
  const shared = createSlashShared({ ...ctx.transcript, gw, slashFlightRef: ctx.slashFlightRef })

  const handler = (cmd: string): boolean => {
    const flight = ++ctx.slashFlightRef.current
    const ui = getUiState()
    const sid = ui.sid
    const parsed = parseSlashCommand(cmd)
    const argTail = parsed.arg ? ` ${parsed.arg}` : ''

    const stale = () => flight !== ctx.slashFlightRef.current || getUiState().sid !== sid

    const guarded =
      <T>(fn: (r: T) => void) =>
      (r: null | T): void => {
        if (!stale() && r) {
          fn(r)
        }
      }

    const guardedErr = (e: unknown) => {
      if (!stale()) {
        sys(`error: ${rpcErrorMessage(e)}`)
      }
    }

    const runCtx: SlashRunCtx = { ...ctx, flight, guarded, guardedErr, shared, sid, stale, ui }

    const found = findSlashCommand(parsed.name)

    if (found) {
      found.run(parsed.arg, runCtx, cmd)

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

    gw.request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: sid })
      .then(r => {
        if (stale()) {
          return
        }

        sys(
          r?.warning
            ? `warning: ${r.warning}\n${r?.output || `/${parsed.name}: no output`}`
            : r?.output || `/${parsed.name}: no output`
        )
      })
      .catch(() => {
        gw.request('command.dispatch', { arg: parsed.arg, name: parsed.name, session_id: sid })
          .then((raw: unknown) => {
            if (stale()) {
              return
            }

            const d = asCommandDispatch(raw)

            if (!d) {
              return sys('error: invalid response: command.dispatch')
            }

            if (d.type === 'exec' || d.type === 'plugin') {
              return sys(d.output || '(no output)')
            }

            if (d.type === 'alias') {
              return handler(`/${d.target}${argTail}`)
            }

            if (d.type === 'skill') {
              sys(`⚡ loading skill: ${d.name}`)

              return d.message?.trim() ? send(d.message) : sys(`/${parsed.name}: skill payload missing message`)
            }
          })
          .catch(guardedErr)
      })

    return true
  }

  return handler
}
