import { forceRedraw, type MouseTrackingMode } from '@hermes/ink'

import { DASHBOARD_TUI_MODE, NO_CONFIRM_DESTRUCTIVE } from '../../../config/env.js'
import { dailyFortune, randomFortune } from '../../../content/fortunes.js'
import { HOTKEYS } from '../../../content/hotkeys.js'
import { isSectionName, nextDetailsMode, parseDetailsMode, SECTION_NAMES } from '../../../domain/details.js'
import type {
  ConfigGetValueResponse,
  ConfigSetResponse,
  SessionSaveResponse,
  SessionStatusResponse,
  SessionSteerResponse,
  SessionTitleResponse,
  SessionUndoResponse
} from '../../../gatewayTypes.js'
import { type Locale, translate, type TranslationKey } from '../../../i18n/index.js'
import { writeClipboardText } from '../../../lib/clipboard.js'
import { writeOsc52Clipboard } from '../../../lib/osc52.js'
import { configureDetectedTerminalKeybindings, configureTerminalKeybindings } from '../../../lib/terminalSetup.js'
import type { Msg, PanelSection } from '../../../types.js'
import type { StatusBarMode } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

const makeT = (locale: Locale) => (key: TranslationKey, vars?: Record<string, string | number>) => translate(locale, key, vars)

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

const DETAILS_USAGE =
  'usage: /details [hidden|collapsed|expanded|cycle]  or  /details <section> [hidden|collapsed|expanded|reset]'

const DETAILS_SECTION_USAGE = 'usage: /details <section> [hidden|collapsed|expanded|reset]'

// Shown when /exit or /quit is refused in the hosted dashboard chat. Kept as a
// constant so the test asserts against the same source of truth as production.
export const DASHBOARD_EXIT_DISABLED_MESSAGE =
  'exit is disabled in hosted dashboard chat — use /new to start a fresh session'

export const DASHBOARD_UPDATE_DISABLED_MESSAGE =
  'update is disabled in hosted dashboard chat — the hosted environment is managed separately'

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
        sections.push({ text: translate(ctx.ui.locale,'sys.helpSkillCount', { count: String(ctx.local.catalog?.skillCount ?? 0) }) })
      }

      sections.push(
        {
          rows: [
            ['/details [hidden|collapsed|expanded|cycle]', translate(ctx.ui.locale,'sys.helpDetailGlobal')],
            [
              '/details <section> [hidden|collapsed|expanded|reset]',
              translate(ctx.ui.locale,'sys.helpDetailSection')
            ],
            ['/fortune [random|daily]', translate(ctx.ui.locale,'sys.helpFortune')]
          ],
          title: translate(ctx.ui.locale, 'section.tuiCommands')
        },
        { rows: HOTKEYS.map(([k, key]) => [k, translate(ctx.ui.locale,key)]), title: translate(ctx.ui.locale, 'section.hotkeys') }
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
        ctx.transcript.sys(DASHBOARD_EXIT_DISABLED_MESSAGE)

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
        ctx.transcript.sys(DASHBOARD_UPDATE_DISABLED_MESSAGE)

        return
      }

      ctx.transcript.sys(translate(ctx.ui.locale, 'sys.updateExiting'))
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
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.usageMouse'))
      }

      patchUiState({ mouseTracking: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'mouse', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(translate(ctx.ui.locale,'sys.mouseTracking', { state: next ? translate(ctx.ui.locale,'sys.on') : translate(ctx.ui.locale,'sys.off') })))
    }
  },

  {
    aliases: ['new'],
    help: 'start a new session',
    name: 'clear',
    run: (arg, ctx, cmd) => {
      if (ctx.session.guardBusySessionSwitch(translate(ctx.ui.locale, 'action.switchSessions'))) {
        return
      }

      const isNew = cmd.startsWith('/new')
      const requestedTitle = isNew ? arg.trim() : ''

      const commit = () => {
        patchUiState({ status: 'forging session…' })
        ctx.session.newSession(isNew ? translate(ctx.ui.locale, 'sys.newSessionStarted') : undefined, requestedTitle || undefined)
      }

      if (NO_CONFIRM_DESTRUCTIVE) {
        return commit()
      }

      patchOverlayState({
        confirm: {
          cancelLabel: translate(ctx.ui.locale,'sys.confirm.clearCancel'),
          confirmLabel: isNew ? translate(ctx.ui.locale,'sys.confirm.clearNew') : translate(ctx.ui.locale,'sys.confirm.clearCurrent'),
          danger: true,
          detail: translate(ctx.ui.locale,'sys.confirm.clearDetail'),
          onConfirm: commit,
          title: isNew ? translate(ctx.ui.locale,'sys.confirm.clearNewTitle') : translate(ctx.ui.locale,'sys.confirm.clearCurrentTitle')
        }
      })
    }
  },

  {
    help: 'force a full UI repaint',
    name: 'redraw',
    run: (_arg, ctx) => {
      forceRedraw(process.stdout)
      ctx.transcript.sys(translate(ctx.ui.locale,'sys.uiRedrawn'))
    }
  },

  {
    help: 'show live session info',
    name: 'status',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.noActiveSession'))
      }

      ctx.gateway
        .rpc<SessionStatusResponse>('session.status', { session_id: ctx.sid })
        .then(ctx.guarded<SessionStatusResponse>(r => ctx.transcript.page(r.output || translate(ctx.ui.locale, 'sys.noStatus'), translate(ctx.ui.locale, 'section.status'))))
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'set or show current session title',
    name: 'title',
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.noActiveSession'))
      }

      const title = arg.trim()

      if (!arg) {
        ctx.gateway
          .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid })
          .then(
            ctx.guarded<SessionTitleResponse>(r => {
              const current = (r?.title ?? '').trim()
              ctx.transcript.sys(current ? translate(ctx.ui.locale,'sys.titleCurrent', { title: current }) : translate(ctx.ui.locale,'sys.noTitleSet'))
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (!title) {
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.usageTitle'))
      }

      ctx.gateway
        .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid, title })
        .then(
          ctx.guarded<SessionTitleResponse>(r => {
            const next = (r?.title ?? title).trim()
            const suffix = r?.pending ? translate(ctx.ui.locale, 'session.titleQueuedSuffix') : ''
            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.titleSet', { title: next, suffix }))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'toggle compact transcript',
    name: 'compact',
    run: (arg, ctx) => {
      const next = flagFromArg(arg, ctx.ui.compact)

      if (next === null) {
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.usageCompact'))
      }

      patchUiState({ compact: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'compact', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(translate(ctx.ui.locale,'sys.compactMode', { state: next ? translate(ctx.ui.locale,'sys.on') : translate(ctx.ui.locale,'sys.off') })))
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

            transcript.sys(translate(ctx.ui.locale, 'sys.detailsGlobal', { mode, overrides: overrides ? `  (${overrides})` : '' }))
          })
          .catch(() => !ctx.stale() && transcript.sys(translate(ctx.ui.locale, 'sys.detailsGlobal', { mode: ui.detailsMode, overrides: '' })))

        return
      }

      const [first, second] = arg.trim().toLowerCase().split(/\s+/)

      if (second && isSectionName(first)) {
        const reset = RESET_WORDS.has(second)
        const mode = reset ? null : parseDetailsMode(second)

        if (!reset && !mode) {
          return transcript.sys(translate(ctx.ui.locale, 'sys.usageDetailsSection'))
        }

        const { [first]: _drop, ...rest } = ui.sections

        patchUiState({ sections: mode ? { ...rest, [first]: mode } : rest })
        gateway
          .rpc<ConfigSetResponse>('config.set', { key: `details_mode.${first}`, value: mode ?? '' })
          .catch(() => {})
        transcript.sys(translate(ctx.ui.locale, 'sys.detailsSection', { section: first, mode: mode ?? translate(ctx.ui.locale, 'common.reset') }))

        return
      }

      const next = CYCLE_WORDS.has(first ?? '') ? nextDetailsMode(ui.detailsMode) : parseDetailsMode(first)

      if (!next) {
        return transcript.sys(translate(ctx.ui.locale, 'sys.helpDetailGlobal'))
      }

      const sections = Object.fromEntries(SECTION_NAMES.map(section => [section, next]))

      patchUiState({ detailsMode: next, detailsModeCommandOverride: true, sections })
      gateway.rpc<ConfigSetResponse>('config.set', { key: 'details_mode', value: next }).catch(() => {})
      transcript.sys(translate(ctx.ui.locale, 'sys.detailsGlobal', { mode: next, overrides: '' }))
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

      ctx.transcript.sys(translate(ctx.ui.locale,'sys.usageFortune'))
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
          return sys(translate(ctx.ui.locale,'sys.copiedChars', { count: String(text.length) }))
        } else {
          return sys(translate(ctx.ui.locale,'sys.clipboardCopyFailed'))
        }
      }

      if (arg && Number.isNaN(parseInt(arg, 10))) {
        return sys(translate(ctx.ui.locale,'sys.usageCopy'))
      }

      const all = ctx.local.getHistoryItems().filter(m => m.role === 'assistant')
      const target = all[arg ? Math.min(parseInt(arg, 10), all.length) - 1 : all.length - 1]

      if (!target) {
        return sys(translate(ctx.ui.locale,'sys.nothingToCopy'))
      }

      void writeClipboardText(target.text)
        .then(nativeOk => {
          if (ctx.stale()) {
            return
          }

          if (nativeOk) {
            sys(translate(ctx.ui.locale,'sys.copiedToClipboard'))
          } else {
            writeOsc52Clipboard(target.text)
            sys(translate(ctx.ui.locale,'sys.sentOsc52'))
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            sys(translate(ctx.ui.locale,'sys.copyFailed', { error: String(error) }))
          }
        })
    }
  },

  {
    help: 'attach clipboard image',
    name: 'paste',
    run: (arg, ctx) => (arg ? ctx.transcript.sys(translate(ctx.ui.locale,'sys.usagePaste')) : ctx.composer.paste())
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
        ctx.transcript.sys(`editor failed: ${String(err)}`)
      })
    }
  },

  {
    help: 'configure IDE terminal keybindings for multiline + undo/redo',
    name: 'terminal-setup',
    run: (arg, ctx) => {
      const target = arg.trim().toLowerCase()

      if (target && !['auto', 'cursor', 'vscode', 'windsurf'].includes(target)) {
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.usageTerminalSetup'))
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
            ctx.transcript.sys(translate(ctx.ui.locale,'sys.restartIdeTerminal'))
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            ctx.transcript.sys(translate(ctx.ui.locale,'sys.terminalSetupFailed', { error: String(error) }))
          }
        })
    }
  },

  {
    help: 'view gateway logs',
    name: 'logs',
    run: (arg, ctx) => {
      const text = ctx.gateway.gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))

      text ? ctx.transcript.page(text, translate(ctx.ui.locale, 'section.log')) : ctx.transcript.sys(translate(ctx.ui.locale,'sys.noGatewayLogs'))
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
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.noConversation'))
      }

      const preview = Math.max(80, parseInt(arg, 10) || 400)

      const lines = items.map((m, i) => {
        const tag = m.role === 'user' ? translate(ctx.ui.locale,'sys.historyYou', { n: String(i + 1) }) : translate(ctx.ui.locale,'sys.historyHermes', { n: String(i + 1) })
        const body = m.text.trim() || (m.tools?.length ? translate(ctx.ui.locale,'sys.historyToolCalls', { n: String(m.tools.length) }) : translate(ctx.ui.locale,'sys.historyEmpty'))
        const clipped = body.length > preview ? `${body.slice(0, preview).trimEnd()}…` : body

        return `[${tag}]\n${clipped}`
      })

ctx.transcript.page(lines.join('\n\n'), translate(ctx.ui.locale, 'section.history'))
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
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.noConversation'))
      }

      if (!ctx.sid) {
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.noActiveSessionSave'))
      }

      ctx.gateway
        .rpc<SessionSaveResponse>('session.save', { session_id: ctx.sid })
        .then(
          ctx.guarded<SessionSaveResponse>(r => {
            const file = r?.file

            if (file) {
              ctx.transcript.sys(translate(ctx.ui.locale,'sys.savedToFile', { file }))
            } else {
              ctx.transcript.sys(translate(ctx.ui.locale,'sys.failedToSave'))
            }
          })
        )
        .catch(ctx.guardedErr)
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
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.usageStatusbar'))
      }

      patchUiState({ statusBar: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'statusbar', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(translate(ctx.ui.locale,'sys.statusBarMode', { mode: next })))
    }
  },

  {
    aliases: ['q'],
    help: 'inspect or enqueue a message',
    name: 'queue',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.queuedCount', { count: String(ctx.composer.queueRef.current.length), s: ctx.composer.queueRef.current.length === 1 ? '' : 's' }))
      }

      ctx.composer.enqueue(arg)
      ctx.transcript.sys(translate(ctx.ui.locale,'sys.queuedText', { text: `${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}` }))
    }
  },

  {
    help: 'inject a message after the next tool call (no interrupt)',
    name: 'steer',
    run: (arg, ctx) => {
      const payload = arg?.trim() ?? ''

      if (!payload) {
        return ctx.transcript.sys(translate(ctx.ui.locale,'sys.usageSteer'))
      }

      // If the agent isn't running, fall back to the queue so the user's
      // message isn't lost — identical semantics to the gateway handler.
      if (!ctx.ui.busy || !ctx.sid) {
        ctx.composer.enqueue(payload)
        ctx.transcript.sys(
          translate(ctx.ui.locale, 'sys.steerQueuedNoTurn', { text: `${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}` })
        )

        return
      }

      ctx.gateway
        .rpc<SessionSteerResponse>('session.steer', { session_id: ctx.sid, text: payload })
        .then(
          ctx.guarded<SessionSteerResponse>(r => {
            if (r?.status === 'queued') {
              ctx.transcript.sys(
                translate(ctx.ui.locale, 'sys.steerQueued', { text: `${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}` })
              )
            } else {
              ctx.transcript.sys(translate(ctx.ui.locale, 'sys.steerRejected'))
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
