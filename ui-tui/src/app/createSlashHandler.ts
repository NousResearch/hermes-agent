import { parseSlashCommand } from '../domain/slash.js'
import type { SlashExecResponse } from '../gatewayTypes.js'
import { asCommandDispatch, rpcErrorMessage } from '../lib/rpc.js'

import type { SlashHandlerContext } from './interfaces.js'
import { findSlashCommand } from './slash/registry.js'
import type { SlashRunCtx } from './slash/types.js'
import { getUiState } from './uiStore.js'

export const NATIVE_PRODUCT_ROUTE_COMMANDS = ['setup', 'skills', 'swarm', 'tools'] as const
export const LIVE_SLASH_EXEC_COMMANDS = ['handoff', 'init-deep', 'model', 'provider', 'ralph-loop', 'start-work', 'ulw-loop'] as const
export const LEGACY_SLASH_EXEC_COMMANDS = [
  'agents',
  'browser',
  'config',
  'cron',
  'debug',
  'fast',
  'gquota',
  'history',
  'insights',
  'platforms',
  'plugins',
  'profile',
  'reload',
  'reload-mcp',
  'rollback',
  'save',
  'snapshot',
  'status',
  'stop',
  'title',
  'toolsets'
] as const

const NATIVE_PRODUCT_ROUTE_COMMAND_SET = new Set<string>(NATIVE_PRODUCT_ROUTE_COMMANDS)
const SLASH_EXEC_COMMANDS = new Set<string>([...LIVE_SLASH_EXEC_COMMANDS, ...LEGACY_SLASH_EXEC_COMMANDS])

const resolveCatalogCanonical = (name: string, canon?: null | Record<string, string>): null | string => {
  if (!canon) {
    return null
  }

  return canon[`/${name}`.toLowerCase()]?.slice(1) ?? null
}

export function createSlashHandler(ctx: SlashHandlerContext): (cmd: string) => boolean {
  const { gw } = ctx.gateway
  const { catalog } = ctx.local
  const { page, send, sys } = ctx.transcript

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

    const runCtx: SlashRunCtx = { ...ctx, flight, guarded, guardedErr, sid, stale, ui }
    const found = findSlashCommand(parsed.name)
    const canonical = resolveCatalogCanonical(parsed.name, catalog?.canon)
    const routeName = canonical ?? parsed.name

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

    if (NATIVE_PRODUCT_ROUTE_COMMAND_SET.has(routeName)) {
      sys(`error: /${routeName} is handled by native product routing`)

      return true
    }

    if (SLASH_EXEC_COMMANDS.has(routeName)) {
      gw.request<SlashExecResponse>('slash.exec', { command: `${routeName}${argTail}`.trim(), session_id: sid })
        .then(r => {
          if (stale()) {
            return
          }

          const body = r?.output || `/${routeName}: no output`
          const text = r?.warning ? `warning: ${r.warning}\n${body}` : body
          const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

          long ? page(text, routeName[0]!.toUpperCase() + routeName.slice(1)) : sys(text)
        })
        .catch(guardedErr)

      return true
    }

    gw.request('command.dispatch', { arg: parsed.arg, name: canonical ?? parsed.name, session_id: sid })
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

    return true
  }

  return handler
}
