import { forceRedraw, type MouseTrackingMode } from '@hermes/ink'

import { DASHBOARD_TUI_MODE, NO_CONFIRM_DESTRUCTIVE } from '../../../config/env.js'
import { dailyFortune, randomFortune } from '../../../content/fortunes.js'
import { hotkeys } from '../../../content/hotkeys.js'
import { isSectionName, nextDetailsMode, parseDetailsMode, SECTION_NAMES } from '../../../domain/details.js'
import type {
  ConfigGetValueResponse,
  ConfigSetResponse,
  SessionSaveResponse,
  SessionStatusResponse,
  SessionSteerResponse,
  SessionTitleResponse,
  SessionUndoResponse,
  SystemBatteryResponse
} from '../../../gatewayTypes.js'
import { t } from '../../../i18n/runtime.js'
import { writeClipboardText } from '../../../lib/clipboard.js'
import { writeOsc52Clipboard } from '../../../lib/osc52.js'
import {
  configureDetectedTerminalKeybindings,
  configureTerminalKeybindings,
  isRemoteShellSession
} from '../../../lib/terminalSetup.js'
import type { Msg, PanelSection } from '../../../types.js'
import type { StatusBarMode } from '../../interfaces.js'
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

// `/mouse` toggles between full tracking and off when called bare so the
// old binary muscle-memory still works. Explicit presets (wheel / buttons /
// all) target the tmux-friendly hover-free subsets.
const MOUSE_MODE_ALIASES: Record<string, MouseTrackingMode> = {
  all: 'all',
  any: 'all',
  button: 'buttons',
  buttons: 'buttons',
  click: 'buttons',
  full: 'all',
  off: 'off',
  on: 'all',
  scroll: 'wheel',
  wheel: 'wheel'
}

const mouseModeFromArg = (arg: string, current: MouseTrackingMode): MouseTrackingMode | null => {
  if (!arg || arg.trim().toLowerCase() === 'toggle') {
    return current === 'off' ? 'all' : 'off'
  }

  return MOUSE_MODE_ALIASES[arg.trim().toLowerCase()] ?? null
}

const RESET_WORDS = new Set(['reset', 'clear', 'default'])
const CYCLE_WORDS = new Set(['cycle', 'toggle'])

const PREVIEW_CHARS = 50

/** Clip a queued/steered prompt for the confirmation line. */
const previewOf = (text: string): string => `${text.slice(0, PREVIEW_CHARS)}${text.length > PREVIEW_CHARS ? '…' : ''}`

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
        sections.push({ text: t('slashCmd.core.help.skillCommandsAvailable', String(ctx.local.catalog.skillCount)) })
      }

      sections.push(
        {
          rows: [
            ['/connectors', t('slashCmd.core.help.connectors')],
            ['/details [hidden|collapsed|expanded|cycle]', t('slashCmd.core.help.detailsGlobal')],
            ['/details <section> [hidden|collapsed|expanded|reset]', t('slashCmd.core.help.detailsSection')],
            ['/fortune [random|daily]', t('slashCmd.core.help.fortune')],
            ['/grid-test [cols]x[rows]', t('slashCmd.core.help.gridTest')],
            ['/dialog-test [zone]', t('slashCmd.core.help.dialogTest')]
          ],
          title: t('slashCmd.core.help.tuiSection')
        },
        { rows: hotkeys(), title: t('help.hotkeys') }
      )

      ctx.transcript.panel(ctx.ui.theme.brand.helpHeader, sections)
    }
  },

  {
    aliases: ['exit'],
    help: 'exit hermes',
    name: 'quit',
    run: (_arg, ctx) => {
      // In the hosted dashboard chat there is no in-page restart path after
      // the PTY child exits, so quitting bricks the tab until a refresh. The
      // keyboard idle-exit (Ctrl+C / Ctrl+D) and SIGINT handling already refuse
      // to die in this mode (see useInputHandlers + entry.tsx); gate /exit and
      // /quit on the same DASHBOARD_TUI_MODE flag. Unlike the keyboard path
      // (which auto-starts a fresh chat), the explicit quit command refuses and
      // instructs the user to run /new themselves.
      if (DASHBOARD_TUI_MODE) {
        ctx.transcript.sys(t('slashCmd.core.quit.dashboardDisabled'))

        return
      }

      ctx.session.die()
    }
  },

  {
    help: 'update Hermes Agent to the latest version (exits TUI)',
    name: 'update',
    run: (_arg, ctx) => {
      if (DASHBOARD_TUI_MODE) {
        ctx.transcript.sys(t('slashCmd.core.update.dashboardDisabled'))

        return
      }

      ctx.transcript.sys(t('slashCmd.core.update.exiting'))
      // Exit code 42 signals the Python wrapper to exec `hermes update`.
      // Use dieWithCode for proper cleanup (gateway kill + Ink unmount).
      setTimeout(() => ctx.session.dieWithCode(42), 100)
    }
  },

  {
    aliases: ['scroll'],
    help: 'set mouse tracking preset [on|off|toggle|wheel|buttons|all]',
    name: 'mouse',
    run: (arg, ctx) => {
      const current = ctx.ui.mouseTracking
      const next = mouseModeFromArg(arg, current)

      if (next === null) {
        return ctx.transcript.sys(t('slashCmd.core.mouse.usage'))
      }

      patchUiState({ mouseTracking: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'mouse', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(t('slashCmd.core.mouse.tracking', next)))
    }
  },

  {
    aliases: ['new'],
    help: 'start a new session',
    name: 'clear',
    run: (arg, ctx, cmd) => {
      if (ctx.session.guardBusySessionSwitch(t('slashCmd.core.clear.switchSessions'))) {
        return
      }

      const isNew = cmd.startsWith('/new')
      const requestedTitle = isNew ? arg.trim() : ''

      const commit = () => {
        patchUiState({ status: t('slashCmd.core.clear.forgingSession') })
        ctx.session.newSession(
          isNew ? t('slashCmd.core.clear.newSessionStarted') : undefined,
          requestedTitle || undefined
        )
      }

      if (NO_CONFIRM_DESTRUCTIVE || !ctx.ui.destructiveSlashConfirm) {
        return commit()
      }

      patchOverlayState({
        confirm: {
          cancelLabel: t('slashCmd.core.clear.cancelLabel'),
          confirmLabel: isNew ? t('slashCmd.core.clear.confirmNew') : t('slashCmd.core.clear.confirmClear'),
          danger: true,
          detail: t('slashCmd.core.clear.detail'),
          onConfirm: commit,
          title: isNew ? t('slashCmd.core.clear.titleNew') : t('slashCmd.core.clear.titleClear')
        }
      })
    }
  },

  {
    help: 'force a full UI repaint',
    name: 'redraw',
    run: (_arg, ctx) => {
      forceRedraw(process.stdout)
      ctx.transcript.sys(t('slashCmd.core.redraw.done'))
    }
  },

  {
    help: 'show live session info',
    name: 'status',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys(t('slashCmd.core.status.noActiveSession'))
      }

      ctx.gateway
        .rpc<SessionStatusResponse>('session.status', { session_id: ctx.sid })
        .then(
          ctx.guarded<SessionStatusResponse>(r =>
            ctx.transcript.page(r.output || t('slashCmd.core.status.empty'), t('slashCmd.core.status.pageTitle'))
          )
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'set or show current session title',
    name: 'title',
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys(t('slashCmd.core.title.noActiveSession'))
      }

      const title = arg.trim()

      if (!arg) {
        ctx.gateway
          .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid })
          .then(
            ctx.guarded<SessionTitleResponse>(r => {
              const current = (r?.title ?? '').trim()
              ctx.transcript.sys(current ? t('slashCmd.core.title.current', current) : t('slashCmd.core.title.none'))
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (!title) {
        return ctx.transcript.sys(t('slashCmd.core.title.usage'))
      }

      ctx.gateway
        .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid, title })
        .then(
          ctx.guarded<SessionTitleResponse>(r => {
            const next = (r?.title ?? title).trim()
            const suffix = r?.pending ? t('slashCmd.core.title.queuedSuffix') : ''
            patchUiState({ sessionTitle: next })
            ctx.transcript.sys(t('slashCmd.core.title.set', next, suffix))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'toggle compact display',
    name: 'density',
    run: (arg, ctx) => {
      const next = flagFromArg(arg, ctx.ui.compact)

      if (next === null) {
        return ctx.transcript.sys(t('slashCmd.core.density.usage'))
      }

      patchUiState({ compact: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'density', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(t('slashCmd.core.density.state', next ? 'on' : 'off')))
    }
  },

  {
    aliases: ['detail'],
    help: 'control agent detail visibility (global or per-section)',
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
            patchUiState({ detailsMode: mode, detailsModeCommandOverride: false })

            const overrides = SECTION_NAMES.filter(s => ui.sections[s])
              .map(s => `${s}=${ui.sections[s]}`)
              .join(' ')

            transcript.sys(t('slashCmd.core.details.current', mode, overrides ? `  (${overrides})` : ''))
          })
          .catch(() => !ctx.stale() && transcript.sys(t('slashCmd.core.details.current', ui.detailsMode, '')))

        return
      }

      const [first, second] = arg.trim().toLowerCase().split(/\s+/)

      if (second && isSectionName(first)) {
        const reset = RESET_WORDS.has(second)
        const mode = reset ? null : parseDetailsMode(second)

        if (!reset && !mode) {
          return transcript.sys(t('slashCmd.core.details.sectionUsage'))
        }

        const { [first]: _drop, ...rest } = ui.sections

        patchUiState({ sections: mode ? { ...rest, [first]: mode } : rest })
        gateway
          .rpc<ConfigSetResponse>('config.set', { key: `details_mode.${first}`, value: mode ?? '' })
          .catch(() => {})
        transcript.sys(t('slashCmd.core.details.section', first, mode ?? t('slashCmd.core.details.reset')))

        return
      }

      const next = CYCLE_WORDS.has(first ?? '') ? nextDetailsMode(ui.detailsMode) : parseDetailsMode(first)

      if (!next) {
        return transcript.sys(t('slashCmd.core.details.usage'))
      }

      const sections = Object.fromEntries(SECTION_NAMES.map(section => [section, next]))

      patchUiState({ detailsMode: next, detailsModeCommandOverride: true, sections })
      gateway.rpc<ConfigSetResponse>('config.set', { key: 'details_mode', value: next }).catch(() => {})
      transcript.sys(t('slashCmd.core.details.current', next, ''))
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

      ctx.transcript.sys(t('slashCmd.core.fortune.usage'))
    }
  },

  {
    help: 'copy selection or assistant message',
    name: 'copy',
    run: async (arg, ctx) => {
      const { sys } = ctx.transcript

      if (!arg && ctx.composer.hasSelection) {
        const text = await ctx.composer.selection.copySelection()

        if (text) {
          return sys(
            t(
              text.length === 1 ? 'slashCmd.core.copy.copiedCharsOne' : 'slashCmd.core.copy.copiedCharsOther',
              String(text.length)
            )
          )
        } else {
          return sys(t('slashCmd.core.copy.clipboardFailed'))
        }
      }

      if (arg && Number.isNaN(parseInt(arg, 10))) {
        return sys(t('slashCmd.core.copy.usage'))
      }

      const all = ctx.local.getHistoryItems().filter(m => m.role === 'assistant')
      const target = all[arg ? Math.min(parseInt(arg, 10), all.length) - 1 : all.length - 1]

      if (!target) {
        return sys(t('slashCmd.core.copy.nothingToCopy'))
      }

      const shouldUseTerminalClipboard = isRemoteShellSession(process.env)

      if (shouldUseTerminalClipboard) {
        writeOsc52Clipboard(target.text)

        return sys(t('slashCmd.core.copy.sentOsc52'))
      }

      void writeClipboardText(target.text)
        .then(nativeOk => {
          if (ctx.stale()) {
            return
          }

          if (nativeOk) {
            sys(t('slashCmd.core.copy.copied'))
          } else {
            writeOsc52Clipboard(target.text)
            sys(t('slashCmd.core.copy.sentOsc52'))
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            sys(t('slashCmd.core.copy.failed', String(error)))
          }
        })
    }
  },

  {
    help: 'attach clipboard image',
    name: 'paste',
    run: (arg, ctx) => (arg ? ctx.transcript.sys(t('slashCmd.core.paste.usage')) : ctx.composer.attachClipboardImage())
  },

  {
    aliases: ['compose'],
    help: 'compose your next prompt in $EDITOR (same as Ctrl+G)',
    name: 'prompt',
    run: (arg, ctx) => {
      if (arg) {
        // The TUI editor opens with the current composer draft; there is no
        // separate seed arg. Drop any inline text into the composer first so
        // it carries into the editor, matching the CLI's /prompt <text>.
        ctx.composer.setInput(arg)
      }

      void ctx.composer.openEditor().catch((err: unknown) => {
        ctx.transcript.sys(t('slashCmd.core.prompt.editorFailed', String(err)))
      })
    }
  },

  {
    help: 'configure IDE terminal keybindings for multiline + undo/redo',
    name: 'terminal-setup',
    run: (arg, ctx) => {
      const target = arg.trim().toLowerCase()

      if (target && !['auto', 'cursor', 'vscode', 'windsurf'].includes(target)) {
        return ctx.transcript.sys(t('slashCmd.core.terminalSetup.usage'))
      }

      const runner =
        !target || target === 'auto'
          ? configureDetectedTerminalKeybindings()
          : configureTerminalKeybindings(target as 'cursor' | 'vscode' | 'windsurf')

      void runner
        .then(result => {
          if (ctx.stale()) {
            return
          }

          ctx.transcript.sys(result.message)

          if (result.success && result.requiresRestart) {
            ctx.transcript.sys(t('slashCmd.core.terminalSetup.restartIde'))
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            ctx.transcript.sys(t('slashCmd.core.terminalSetup.failed', String(error)))
          }
        })
    }
  },

  {
    help: 'view gateway logs',
    name: 'logs',
    run: (arg, ctx) => {
      const text = ctx.gateway.gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))

      text
        ? ctx.transcript.page(text, t('slashCmd.core.logs.pageTitle'))
        : ctx.transcript.sys(t('slashCmd.core.logs.none'))
    }
  },

  {
    help: 'view current transcript (user + assistant messages)',
    name: 'history',
    run: (arg, ctx) => {
      // The CLI-side `/history` runs in a detached slash-worker subprocess
      // that never sees the TUI's turns — it only surfaces whatever was
      // persisted before this process started.  Render the TUI's own
      // transcript so `/history` actually reflects what the user just did.
      const items = ctx.local.getHistoryItems().filter(m => m.role === 'user' || m.role === 'assistant')

      if (!items.length) {
        return ctx.transcript.sys(t('slashCmd.core.history.noConversation'))
      }

      const preview = Math.max(80, parseInt(arg, 10) || 400)

      const lines = items.map((m, i) => {
        const index = String(i + 1)

        const tag =
          m.role === 'user' ? t('slashCmd.core.history.youTag', index) : t('slashCmd.core.history.hermesTag', index)

        const toolCount = m.tools?.length ?? 0

        const body =
          m.text.trim() ||
          (toolCount
            ? t(
                toolCount === 1 ? 'slashCmd.core.history.toolCallsOne' : 'slashCmd.core.history.toolCallsOther',
                String(toolCount)
              )
            : t('slashCmd.core.history.empty'))

        const clipped = body.length > preview ? `${body.slice(0, preview).trimEnd()}…` : body

        return `[${tag}]\n${clipped}`
      })

      ctx.transcript.page(lines.join('\n\n'), t('slashCmd.core.history.pageTitle'))
    }
  },

  {
    help: 'save the current transcript to JSON',
    name: 'save',
    run: (_arg, ctx) => {
      const hasConversation = ctx.local
        .getHistoryItems()
        .some(m => m.role === 'user' || m.role === 'assistant' || m.role === 'tool')

      if (!hasConversation) {
        return ctx.transcript.sys(t('slashCmd.core.save.noConversation'))
      }

      if (!ctx.sid) {
        return ctx.transcript.sys(t('slashCmd.core.save.noActiveSession'))
      }

      ctx.gateway
        .rpc<SessionSaveResponse>('session.save', { session_id: ctx.sid })
        .then(
          ctx.guarded<SessionSaveResponse>(r => {
            const file = r?.file

            if (file) {
              ctx.transcript.sys(t('slashCmd.core.save.saved', file))
            } else {
              ctx.transcript.sys(t('slashCmd.core.save.failed'))
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'toggle focus view — show only your prompt and the final response [on|off|status]',
    name: 'focus',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const current = ctx.ui.focusView

      // `/focus status` reports without writing, matching the CLI surface.
      if (mode === 'status' || mode === 'show' || mode === '?') {
        return ctx.transcript.sys(current ? t('slashCmd.core.focus.statusOn') : t('slashCmd.core.focus.statusOff'))
      }

      const next = flagFromArg(mode, current)

      if (next === null) {
        return ctx.transcript.sys(t('slashCmd.core.focus.usage'))
      }

      // Display-only: Python owns the tool_progress stash/restore so /focus off
      // returns to whatever /verbose mode the user had. Optimistically patch the
      // badge so the status bar flips on the same frame.
      patchUiState({ focusView: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'focus', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() =>
        ctx.transcript.sys(next ? t('slashCmd.core.focus.enabled') : t('slashCmd.core.focus.disabled'))
      )
    }
  },

  {
    aliases: ['sb'],
    help: 'status bar position (on|off|top|bottom)',
    name: 'statusbar',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const toggle: StatusBarMode = ctx.ui.statusBar === 'off' ? 'top' : 'off'

      const next: null | StatusBarMode =
        !mode || mode === 'toggle'
          ? toggle
          : mode === 'on' || mode === 'top'
            ? 'top'
            : mode === 'off' || mode === 'bottom'
              ? mode
              : null

      if (!next) {
        return ctx.transcript.sys(t('slashCmd.core.statusbar.usage'))
      }

      patchUiState({ statusBar: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'statusbar', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(t('slashCmd.core.statusbar.state', next)))
    }
  },

  {
    help: 'toggle a color-coded battery indicator in the status bar [on|off|status]',
    name: 'battery',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()

      // `/battery status` reports the current setting plus a live reading,
      // matching the CLI surface. Fetch on demand so it works even while the
      // indicator (and its poller) is off.
      if (mode === 'status' || mode === 'show') {
        const state = ctx.ui.battery ? 'on' : 'off'

        ctx.gateway
          .rpc<SystemBatteryResponse>('system.battery', {})
          .then(r => {
            if (r?.available && typeof r.percent === 'number') {
              ctx.transcript.sys(
                t('slashCmd.core.battery.statusLive', state, r.plugged ? '⚡' : '🔋', String(r.percent))
              )
            } else {
              ctx.transcript.sys(t('slashCmd.core.battery.statusNoBattery', state))
            }
          })
          .catch(() => ctx.transcript.sys(t('slashCmd.core.battery.state', state)))

        return
      }

      const next = flagFromArg(arg, ctx.ui.battery)

      if (next === null) {
        return ctx.transcript.sys(t('slashCmd.core.battery.usage'))
      }

      patchUiState({ battery: next, ...(next ? {} : { batteryStatus: null }) })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'battery', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(t('slashCmd.core.battery.state', next ? 'on' : 'off')))
    }
  },

  {
    aliases: ['q'],
    help: 'inspect or enqueue a message',
    name: 'queue',
    run: (arg, ctx) => {
      if (!arg) {
        const count = ctx.composer.queueRef.current.length

        return ctx.transcript.sys(
          t(count === 1 ? 'slashCmd.core.queue.countOne' : 'slashCmd.core.queue.countOther', String(count))
        )
      }

      ctx.composer.enqueue(arg)
      ctx.transcript.sys(t('slashCmd.core.queue.queued', previewOf(arg)))
    }
  },

  {
    aliases: ['s'],
    help: 'inject a message after the next tool call (no interrupt)',
    name: 'steer',
    run: (arg, ctx) => {
      const payload = arg?.trim() ?? ''

      if (!payload) {
        return ctx.transcript.sys(t('slashCmd.core.steer.usage'))
      }

      // If the agent isn't running, fall back to the queue so the user's
      // message isn't lost — identical semantics to the gateway handler.
      if (!ctx.ui.busy || !ctx.sid) {
        ctx.composer.enqueue(payload)
        ctx.transcript.sys(t('slashCmd.core.steer.noActiveTurnQueued', previewOf(payload)))

        return
      }

      ctx.gateway
        .rpc<SessionSteerResponse>('session.steer', { session_id: ctx.sid, text: payload })
        .then(
          ctx.guarded<SessionSteerResponse>(r => {
            if (r?.status === 'queued') {
              ctx.transcript.sys(t('slashCmd.core.steer.queued', previewOf(payload)))
            } else {
              // The turn ended before the steer landed (#64578): keep the words as the next turn.
              ctx.composer.enqueue(payload)
              ctx.transcript.sys(t('slashCmd.core.steer.rejected'))
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
        return ctx.transcript.sys(t('slashCmd.core.undo.nothing'))
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) > 0) {
            ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
            ctx.transcript.sys(
              t(r.removed === 1 ? 'slashCmd.core.undo.undidOne' : 'slashCmd.core.undo.undidOther', String(r.removed))
            )
          } else {
            ctx.transcript.sys(t('slashCmd.core.undo.nothing'))
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
        return ctx.transcript.sys(t('slashCmd.core.retry.nothing'))
      }

      if (!ctx.sid) {
        return ctx.transcript.send(last)
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { intent: 'retry', session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) <= 0) {
            return ctx.transcript.sys(t('slashCmd.core.retry.nothing'))
          }

          ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
          ctx.transcript.send(last)
        })
      )
    }
  }
]
