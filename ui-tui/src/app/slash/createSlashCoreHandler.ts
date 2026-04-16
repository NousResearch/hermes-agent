import { HOTKEYS } from '../../constants.js'
import { writeOsc52Clipboard } from '../../lib/osc52.js'
import type { DetailsMode, PanelSection } from '../../types.js'
import { nextDetailsMode, parseDetailsMode } from '../helpers.js'
import type { SlashHandlerContext } from '../interfaces.js'
import { patchOverlayState } from '../overlayStore.js'
import { patchUiState } from '../uiStore.js'

import { isStaleSlash } from './isStaleSlash.js'

const FORTUNES = [
  'you are one clean refactor away from clarity',
  'a tiny rename today prevents a huge bug tomorrow',
  'your next commit message will be immaculate',
  'the edge case you are ignoring is already solved in your head',
  'minimal diff, maximal calm',
  'today favors bold deletions over new abstractions',
  'the right helper is already in your codebase',
  'you will ship before overthinking catches up',
  'tests are about to save your future self',
  'your instincts are correctly suspicious of that one branch'
]

const LEGENDARY_FORTUNES = [
  'legendary drop: one-line fix, first try',
  'legendary drop: every flaky test passes cleanly',
  'legendary drop: your diff teaches by itself'
]

const hash = (input: string) => {
  let out = 2166136261

  for (let i = 0; i < input.length; i++) {
    out ^= input.charCodeAt(i)
    out = Math.imul(out, 16777619)
  }

  return out >>> 0
}

const fortuneFromScore = (score: number) => {
  const rare = score % 20 === 0
  const bag = rare ? LEGENDARY_FORTUNES : FORTUNES

  return `${rare ? '🌟' : '🔮'} ${bag[score % bag.length]}`
}

const randomFortune = () => fortuneFromScore(Math.floor(Math.random() * 0x7fffffff))

const dailyFortune = (sid: null | string) => fortuneFromScore(hash(`${sid || 'anon'}|${new Date().toDateString()}`))

export function createSlashCoreHandler(ctx: SlashHandlerContext) {
  const { enqueue, hasSelection, paste, queueRef, selection } = ctx.composer
  const { catalog, getHistoryItems, getLastUserMsg } = ctx.local
  const { guardBusySessionSwitch, newSession, resumeById } = ctx.session
  const { panel, send, setHistoryItems, sys, trimLastExchange } = ctx.transcript

  return ({ arg, flight, name, sid, ui }: SlashCommand) => {
    switch (name) {
      case 'help': {
        const sections: PanelSection[] = (catalog?.categories ?? []).map(({ name: catName, pairs }: any) => ({
          title: catName,
          rows: pairs
        }))

        if (catalog?.skillCount) {
          sections.push({ text: `${catalog.skillCount} skill commands available — /skills to browse` })
        }

        sections.push({
          title: 'TUI',
          rows: [
            ['/details [hidden|collapsed|expanded|cycle]', 'set agent detail visibility mode'],
            ['/fortune [random|daily]', 'show a random or daily local fortune']
          ]
        })
        sections.push({ title: 'Hotkeys', rows: HOTKEYS })
        panel('Commands', sections)

        return true
      }

      case 'quit':

      case 'exit':

      case 'q':
        ctx.session.die()

        return true

      case 'clear':

      case 'new':
        if (guardBusySessionSwitch('switch sessions')) {
          return true
        }

        patchUiState({ status: 'forging session…' })
        newSession(name === 'new' ? 'new session started' : undefined)

        return true

      case 'resume':
        if (guardBusySessionSwitch('switch sessions')) {
          return true
        }

        arg ? resumeById(arg) : patchOverlayState({ picker: true })

        return true
      case 'compact': {
        const mode = arg.trim().toLowerCase()

        if (arg && !['on', 'off', 'toggle'].includes(mode)) {
          sys('usage: /compact [on|off|toggle]')

          return true
        }

        const next = mode === 'on' ? true : mode === 'off' ? false : !ui.compact

        patchUiState({ compact: next })
        ctx.gateway.rpc('config.set', { key: 'compact', value: next ? 'on' : 'off' }).catch(() => {})
        queueMicrotask(() => sys(`compact ${next ? 'on' : 'off'}`))

        return true
      }

      case 'details':

      case 'detail':
        if (!arg) {
          ctx.gateway
            .rpc('config.get', { key: 'details_mode' })
            .then((r: any) => {
              if (isStaleSlash(ctx, flight, sid)) {
                return
              }

              const mode = parseDetailsMode(r?.value) ?? ui.detailsMode

              patchUiState({ detailsMode: mode })
              sys(`details: ${mode}`)
            })
            .catch(() => {
              if (isStaleSlash(ctx, flight, sid)) {
                return
              }

              sys(`details: ${ui.detailsMode}`)
            })

          return true
        }

        {
          const mode = arg.trim().toLowerCase()

          if (!['hidden', 'collapsed', 'expanded', 'cycle', 'toggle'].includes(mode)) {
            sys('usage: /details [hidden|collapsed|expanded|cycle]')

            return true
          }

          const next = mode === 'cycle' || mode === 'toggle' ? nextDetailsMode(ui.detailsMode) : (mode as DetailsMode)

          patchUiState({ detailsMode: next })
          ctx.gateway.rpc('config.set', { key: 'details_mode', value: next }).catch(() => {})
          sys(`details: ${next}`)
        }

        return true

      case 'fortune':
        if (!arg || arg.trim().toLowerCase() === 'random') {
          sys(randomFortune())

          return true
        }

        if (['daily', 'today', 'stable'].includes(arg.trim().toLowerCase())) {
          sys(dailyFortune(sid))

          return true
        }

        sys('usage: /fortune [random|daily]')

        return true
      case 'copy': {
        if (!arg && hasSelection) {
          const copied = selection.copySelection()

          if (copied) {
            sys('copied selection')

            return true
          }
        }

        if (arg && Number.isNaN(parseInt(arg, 10))) {
          sys('usage: /copy [number]')

          return true
        }

        const all = getHistoryItems().filter((m: any) => m.role === 'assistant')
        const target = all[arg ? Math.min(parseInt(arg, 10), all.length) - 1 : all.length - 1]

        if (!target) {
          sys('nothing to copy')

          return true
        }

        writeOsc52Clipboard(target.text)
        sys('sent OSC52 copy sequence (terminal support required)')

        return true
      }

      case 'paste':
        if (!arg) {
          paste()

          return true
        }

        sys('usage: /paste')

        return true
      case 'logs': {
        const logText = ctx.gateway.gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))

        logText ? ctx.transcript.page(logText, 'Logs') : sys('no gateway logs')

        return true
      }

      case 'statusbar':
      case 'sb': {
        const mode = arg.trim().toLowerCase()

        if (arg && !['on', 'off', 'toggle'].includes(mode)) {
          sys('usage: /statusbar [on|off|toggle]')

          return true
        }

        const next = mode === 'on' ? true : mode === 'off' ? false : !ui.statusBar

        patchUiState({ statusBar: next })
        ctx.gateway.rpc('config.set', { key: 'statusbar', value: next ? 'on' : 'off' }).catch(() => {})
        queueMicrotask(() => sys(`status bar ${next ? 'on' : 'off'}`))

        return true
      }

      case 'queue':
        if (!arg) {
          sys(`${queueRef.current.length} queued message(s)`)

          return true
        }

        enqueue(arg)
        sys(`queued: "${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}"`)

        return true

      case 'undo':
        if (!sid) {
          sys('nothing to undo')

          return true
        }

        ctx.gateway.rpc('session.undo', { session_id: sid }).then((r: any) => {
          if (isStaleSlash(ctx, flight, sid) || !r) {
            return
          }

          if (r.removed > 0) {
            setHistoryItems((prev: any[]) => trimLastExchange(prev))
            sys(`undid ${r.removed} messages`)
          } else {
            sys('nothing to undo')
          }
        })

        return true
      case 'retry': {
        const lastUserMsg = getLastUserMsg()

        if (!lastUserMsg) {
          sys('nothing to retry')

          return true
        }

        if (!sid) {
          send(lastUserMsg)

          return true
        }

        ctx.gateway.rpc('session.undo', { session_id: sid }).then((r: any) => {
          if (isStaleSlash(ctx, flight, sid) || !r) {
            return
          }

          if (r.removed <= 0) {
            sys('nothing to retry')

            return
          }

          setHistoryItems((prev: any[]) => trimLastExchange(prev))
          send(lastUserMsg)
        })

        return true
      }
    }

    return false
  }
}

interface SlashCommand {
  arg: string
  flight: number
  name: string
  sid: null | string
  ui: {
    compact: boolean
    detailsMode: DetailsMode
    statusBar: boolean
  }
}
