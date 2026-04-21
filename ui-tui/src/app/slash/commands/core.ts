import { NO_CONFIRM_DESTRUCTIVE } from '../../../config/env.js'
import { dailyFortune, randomFortune } from '../../../content/fortunes.js'
import { HOTKEYS } from '../../../content/hotkeys.js'
import { nextDetailsMode, parseDetailsMode } from '../../../domain/details.js'
import type {
  ConfigGetValueResponse,
  ConfigSetResponse,
  SessionSteerResponse,
  SessionUndoResponse
} from '../../../gatewayTypes.js'
import { writeOsc52Clipboard } from '../../../lib/osc52.js'
import { configureDetectedTerminalKeybindings, configureTerminalKeybindings } from '../../../lib/terminalSetup.js'
import type { DetailsMode, Msg, PanelSection } from '../../../types.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

const flagFromArg = (arg: string, current: boolean): boolean | null => {
  if (!arg) {
    return !current
  }

  const mode = arg.trim().toLowerCase()

  if (mode === 'on') {
    return true
  }

  if (mode === 'off') {
    return false
  }

  if (mode === 'toggle') {
    return !current
  }

  return null
}

const DETAIL_MODES = new Set(['collapsed', 'cycle', 'expanded', 'hidden', 'toggle'])

export const coreCommands: SlashCommand[] = [
  {
    help: 'list commands + hotkeys',
    name: 'help',
    run: (_arg, ctx) => {
      const sections: PanelSection[] = (ctx.local.catalog?.categories ?? []).map(cat => ({
        rows: cat.pairs,
        title: cat.name
      }))

      if (ctx.local.catalog?.skillCount) {
        sections.push({ text: `${ctx.local.catalog.skillCount} skill commands available — /skills to browse` })
      }

      sections.push(
        {
          rows: [
            ['/details [hidden|collapsed|expanded|cycle]', 'set agent detail visibility mode'],
            ['/fortune [random|daily]', 'show a random or daily local fortune']
          ],
          title: 'TUI'
        },
        { rows: HOTKEYS, title: 'Hotkeys' }
      )

      ctx.transcript.panel(ctx.ui.theme.brand.helpHeader, sections)
    }
  },

  {
    aliases: ['exit', 'q'],
    help: 'exit hermes',
    name: 'quit',
    run: (_arg, ctx) => ctx.session.die()
  },

  {
    aliases: ['new'],
    help: 'start a new session',
    name: 'clear',
    run: (_arg, ctx, cmd) => {
      if (ctx.session.guardBusySessionSwitch('switch sessions')) {
        return
      }

      const isNew = cmd.startsWith('/new')

      const commit = () => {
        patchUiState({ status: 'forging session…' })
        ctx.session.newSession(isNew ? 'new session started' : undefined)
      }

      if (NO_CONFIRM_DESTRUCTIVE) {
        return commit()
      }

      patchOverlayState({
        confirm: {
          cancelLabel: 'No, keep going',
          confirmLabel: isNew ? 'Yes, start a new session' : 'Yes, clear the session',
          danger: true,
          detail: 'This ends the current conversation and clears the transcript.',
          onConfirm: commit,
          title: isNew ? 'Start a new session?' : 'Clear the current session?'
        }
      })
    }
  },

  {
    help: 'resume a prior session',
    name: 'resume',
    run: (arg, ctx) => {
      if (ctx.session.guardBusySessionSwitch('switch sessions')) {
        return
      }

      arg ? ctx.session.resumeById(arg) : patchOverlayState({ picker: true })
    }
  },

  {
    help: 'toggle compact transcript',
    name: 'compact',
    run: (arg, ctx) => {
      const next = flagFromArg(arg, ctx.ui.compact)

      if (next === null) {
        return ctx.transcript.sys('usage: /compact [on|off|toggle]')
      }

      patchUiState({ compact: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'compact', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`compact ${next ? 'on' : 'off'}`))
    }
  },

  {
    aliases: ['detail'],
    help: 'control agent detail visibility',
    name: 'details',
    run: (arg, ctx) => {
      const { gateway, transcript, ui } = ctx

      if (!arg) {
        gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'details_mode' })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            const mode = parseDetailsMode(r?.value) ?? ui.detailsMode

            patchUiState({ detailsMode: mode })
            transcript.sys(`details: ${mode}`)
          })
          .catch(() => {
            if (!ctx.stale()) {
              transcript.sys(`details: ${ui.detailsMode}`)
            }
          })

        return
      }

      const mode = arg.trim().toLowerCase()

      if (!DETAIL_MODES.has(mode)) {
        return transcript.sys('usage: /details [hidden|collapsed|expanded|cycle]')
      }

      const next = mode === 'cycle' || mode === 'toggle' ? nextDetailsMode(ui.detailsMode) : (mode as DetailsMode)

      patchUiState({ detailsMode: next })
      gateway.rpc<ConfigSetResponse>('config.set', { key: 'details_mode', value: next }).catch(() => {})
      transcript.sys(`details: ${next}`)
    }
  },

  {
    help: 'local fortune',
    name: 'fortune',
    run: (arg, ctx) => {
      const key = arg.trim().toLowerCase()

      if (!arg || key === 'random') {
        return ctx.transcript.sys(randomFortune())
      }

      if (['daily', 'stable', 'today'].includes(key)) {
        return ctx.transcript.sys(dailyFortune(ctx.sid))
      }

      ctx.transcript.sys('usage: /fortune [random|daily]')
    }
  },

  {
    help: 'copy selection or assistant message',
    name: 'copy',
    run: (arg, ctx) => {
      const { sys } = ctx.transcript

      if (!arg && ctx.composer.hasSelection && ctx.composer.selection.copySelection()) {
        return sys('copied selection')
      }

      if (arg && Number.isNaN(parseInt(arg, 10))) {
        return sys('usage: /copy [number]')
      }

      const all = ctx.local.getHistoryItems().filter(m => m.role === 'assistant')
      const target = all[arg ? Math.min(parseInt(arg, 10), all.length) - 1 : all.length - 1]

      if (!target) {
        return sys('nothing to copy')
      }

      writeOsc52Clipboard(target.text)
      sys('sent OSC52 copy sequence (terminal support required)')
    }
  },

  {
    help: 'attach clipboard image',
    name: 'paste',
    run: (arg, ctx) => (arg ? ctx.transcript.sys('usage: /paste') : ctx.composer.paste())
  },

  {
    help: 'configure IDE terminal keybindings for multiline + undo/redo',
    name: 'terminal-setup',
    run: (arg, ctx) => {
      const target = arg.trim().toLowerCase()

      if (target && !['auto', 'cursor', 'vscode', 'windsurf'].includes(target)) {
        return ctx.transcript.sys('usage: /terminal-setup [auto|vscode|cursor|windsurf]')
      }

      const runner = !target || target === 'auto' ? configureDetectedTerminalKeybindings() : configureTerminalKeybindings(target as 'cursor' | 'vscode' | 'windsurf')

      void runner.then(result => {
        if (ctx.stale()) {
          return
        }

        ctx.transcript.sys(result.message)
        if (result.success && result.requiresRestart) {
          ctx.transcript.sys('restart the IDE terminal for the new keybindings to take effect')
        }
      }).catch(error => {
        if (!ctx.stale()) {
          ctx.transcript.sys(`terminal setup failed: ${String(error)}`)
        }
      })
    }
  },

  {
    help: 'view gateway logs',
    name: 'logs',
    run: (arg, ctx) => {
      const text = ctx.gateway.gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))

      text ? ctx.transcript.page(text, 'Logs') : ctx.transcript.sys('no gateway logs')
    }
  },

  {
    aliases: ['sb'],
    help: 'toggle status bar',
    name: 'statusbar',
    run: (arg, ctx) => {
      const next = flagFromArg(arg, ctx.ui.statusBar)

      if (next === null) {
        return ctx.transcript.sys('usage: /statusbar [on|off|toggle]')
      }

      patchUiState({ statusBar: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'statusbar', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`status bar ${next ? 'on' : 'off'}`))
    }
  },

  {
    help: 'inspect or enqueue a message',
    name: 'queue',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys(`${ctx.composer.queueRef.current.length} queued message(s)`)
      }

      ctx.composer.enqueue(arg)
      ctx.transcript.sys(`queued: "${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}"`)
    }
  },

  {
    help: 'inject a message after the next tool call (no interrupt)',
    name: 'steer',
    run: (arg, ctx) => {
      const payload = arg?.trim() ?? ''

      if (!payload) {
        return ctx.transcript.sys('usage: /steer <prompt>')
      }

      // If the agent isn't running, fall back to the queue so the user's
      // message isn't lost — identical semantics to the gateway handler.
      if (!ctx.ui.busy || !ctx.sid) {
        ctx.composer.enqueue(payload)
        ctx.transcript.sys(
          `no active turn — queued for next: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
        )

        return
      }

      ctx.gateway
        .rpc<SessionSteerResponse>('session.steer', { session_id: ctx.sid, text: payload })
        .then(
          ctx.guarded<SessionSteerResponse>(r => {
            if (r?.status === 'queued') {
              ctx.transcript.sys(
                `⏩ steer queued — arrives after next tool call: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
              )
            } else {
              ctx.transcript.sys('steer rejected')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'undo last exchange',
    name: 'undo',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('nothing to undo')
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) > 0) {
            ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
            ctx.transcript.sys(`undid ${r.removed} messages`)
          } else {
            ctx.transcript.sys('nothing to undo')
          }
        })
      )
    }
  },

  {
    help: 'retry last user message',
    name: 'retry',
    run: (_arg, ctx) => {
      const last = ctx.local.getLastUserMsg()

      if (!last) {
        return ctx.transcript.sys('nothing to retry')
      }

      if (!ctx.sid) {
        return ctx.transcript.send(last)
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) <= 0) {
            return ctx.transcript.sys('nothing to retry')
          }

          ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
          ctx.transcript.send(last)
        })
      )
    }
  }
]
