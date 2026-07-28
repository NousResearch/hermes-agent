import { parseSlashCommand } from '../domain/slash.js'
import type { SlashExecResponse } from '../gatewayTypes.js'
import { asCommandDispatch, rpcErrorMessage } from '../lib/rpc.js'
import { launchWidget } from '../sdk/host.js'
import { getWidgetApp } from '../sdk/registry.js'

import type { SlashHandlerContext } from './interfaces.js'
import { findSlashCommand } from './slash/registry.js'
import type { SlashRunCtx } from './slash/types.js'
import { getUiState } from './uiStore.js'

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

    if (found) {
      found.run(parsed.arg, runCtx, cmd)

      return true
    }

    // Registry-first fallback: widget apps registered AFTER the static
    // command table was built (user widgets from $HERMES_HOME/tui-widgets,
    // /widgets-reload) dispatch straight off the live registry.
    if (getWidgetApp(parsed.name)) {
      const err = launchWidget(parsed.name, parsed.arg)

      if (err) {
        sys(err)
      }

      return true
    }

    if (catalog?.canon) {
      const needle = `/${parsed.name}`.toLowerCase()
      const exact = Object.entries(catalog.canon).find(([alias]) => alias.toLowerCase() === needle)?.[1]

      if (exact) {
        if (exact.toLowerCase() !== needle) {
          return handler(`${exact}${argTail}`)
        }
      } else {
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
    }

    const handleDispatch = (raw: unknown): void => {
      const d = asCommandDispatch(raw)

      if (!d) {
        return sys('error: invalid response: command.dispatch')
      }

      if (d.type === 'exec' || d.type === 'plugin') {
        return sys(d.output || '(no output)')
      }

      if (d.type === 'alias') {
        return void handler(`/${d.target}${argTail}`)
      }

      // A skill/bundle dispatch's `message` is the expanded skill body —
      // model-facing scaffolding. `display` is the invocation the gateway
      // projected; the transcript shows that instead. An ordinary send has no
      // projection and goes through unchanged. No client-side fallback here:
      // the TUI spawns its gateway from this same checkout, so the two can't
      // version-skew (unlike the desktop, which can meet an older backend).
      const sendDispatch = (display: string | undefined, message: string) => {
        const shown = display?.trim()

        return shown ? send(message, true, shown) : send(message)
      }

      if (d.type === 'skill') {
        return d.message?.trim()
          ? sendDispatch(d.display, d.message)
          : sys(`/${parsed.name}: skill payload missing message`)
      }

      if (d.type === 'send') {
        if (d.notice?.trim()) {
          sys(d.notice)
        }

        return d.message?.trim() ? sendDispatch(d.display, d.message) : sys(`/${parsed.name}: empty message`)
      }

      if (d.type === 'prefill') {
        // /undo returns prefill: drop the backed-up message text into
        // the composer so the user can edit and resubmit, instead of
        // submitting it immediately like 'send'.
        if (d.notice?.trim()) {
          sys(d.notice)
        }

        if (d.message) {
          ctx.composer.setInput(d.message)
        }
      }
    }

    gw.request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: sid })
      .then(r => {
        if (stale()) {
          return
        }

        if (asCommandDispatch(r)) {
          return handleDispatch(r)
        }

        const body = r?.output || `/${parsed.name}: no output`
        const text = r?.warning ? `warning: ${r.warning}\n${body}` : body
        const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

        long ? page(text, parsed.name[0]!.toUpperCase() + parsed.name.slice(1)) : sys(text)
      })
      .catch(() => {
        gw.request('command.dispatch', { arg: parsed.arg, name: parsed.name, session_id: sid })
          .then((raw: unknown) => {
            if (stale()) {
              return
            }

            handleDispatch(raw)
          })
          .catch(guardedErr)
      })

    return true
  }

  return handler
}
