import type { Dispatch, MutableRefObject, SetStateAction } from 'react'

import { HOTKEYS } from '../constants.js'
import { writeOsc52Clipboard } from '../lib/osc52.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { fmtK } from '../lib/text.js'
import type { DetailsMode, Msg, PanelSection, SessionInfo, SlashCatalog } from '../types.js'

import { imageTokenMeta, introMsg, nextDetailsMode, parseDetailsMode, toTranscriptMessages } from './helpers.js'
import type { GatewayServices } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { getUiState, patchUiState } from './uiStore.js'

export interface SlashHandlerContext {
  composer: {
    enqueue: (text: string) => void
    hasSelection: boolean
    paste: (quiet?: boolean) => void
    queueRef: MutableRefObject<string[]>
    selection: {
      copySelection: () => string
    }
    setInput: Dispatch<SetStateAction<string>>
  }
  gateway: GatewayServices
  local: {
    catalog: SlashCatalog | null
    lastUserMsg: string
    maybeWarn: (value: any) => void
    messages: Msg[]
  }
  session: {
    closeSession: (targetSid?: string | null) => Promise<unknown>
    die: () => void
    guardBusySessionSwitch: (what?: string) => boolean
    newSession: (msg?: string) => void
    resetVisibleHistory: (info?: SessionInfo | null) => void
    resumeById: (id: string) => void
    setSessionStartedAt: Dispatch<SetStateAction<number>>
  }
  transcript: {
    page: (text: string, title?: string) => void
    panel: (title: string, sections: PanelSection[]) => void
    send: (text: string) => void
    setHistoryItems: Dispatch<SetStateAction<Msg[]>>
    setMessages: Dispatch<SetStateAction<Msg[]>>
    sys: (text: string) => void
    trimLastExchange: (items: Msg[]) => Msg[]
  }
  voice: {
    setVoiceEnabled: Dispatch<SetStateAction<boolean>>
  }
}

export function createSlashHandler(ctx: SlashHandlerContext): (cmd: string) => boolean {
  const { enqueue, hasSelection, paste, queueRef, selection, setInput } = ctx.composer
  const { gw, rpc } = ctx.gateway
  const { catalog, lastUserMsg, maybeWarn, messages } = ctx.local

  const {
    closeSession,
    die,
    guardBusySessionSwitch,
    newSession,
    resetVisibleHistory,
    resumeById,
    setSessionStartedAt
  } = ctx.session

  const { page, panel, send, setHistoryItems, setMessages, sys, trimLastExchange } = ctx.transcript
  const { setVoiceEnabled } = ctx.voice

  const handler = (cmd: string): boolean => {
    const ui = getUiState()
    const detailsMode = ui.detailsMode
    const sid = ui.sid
    const [rawName, ...rest] = cmd.slice(1).split(/\s+/)
    const name = rawName.toLowerCase()
    const arg = rest.join(' ')

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
          rows: [['/details [hidden|collapsed|expanded|cycle]', 'set agent detail visibility mode']]
        })

        sections.push({ title: 'Hotkeys', rows: HOTKEYS })

        panel('Commands', sections)

        return true
      }

      case 'quit':

      case 'exit':

      case 'q':
        die()

        return true

      case 'clear':
        if (guardBusySessionSwitch('switch sessions')) {
          return true
        }

        patchUiState({ status: 'forging session…' })
        newSession()

        return true

      case 'new':
        if (guardBusySessionSwitch('switch sessions')) {
          return true
        }

        patchUiState({ status: 'forging session…' })
        newSession('new session started')

        return true

      case 'resume':
        if (guardBusySessionSwitch('switch sessions')) {
          return true
        }

        if (arg) {
          resumeById(arg)
        } else {
          patchOverlayState({ picker: true })
        }

        return true

      case 'compact':
        if (arg && !['on', 'off', 'toggle'].includes(arg.trim().toLowerCase())) {
          sys('usage: /compact [on|off|toggle]')

          return true
        }

        {
          const mode = arg.trim().toLowerCase()
          const next = mode === 'on' ? true : mode === 'off' ? false : !ui.compact

          patchUiState({ compact: next })
          rpc('config.set', { key: 'compact', value: next ? 'on' : 'off' }).catch(() => {})
          queueMicrotask(() => sys(`compact ${next ? 'on' : 'off'}`))
        }

        return true

      case 'details':

      case 'detail':
        if (!arg) {
          rpc('config.get', { key: 'details_mode' })
            .then((r: any) => {
              const mode = parseDetailsMode(r?.value) ?? detailsMode
              patchUiState({ detailsMode: mode })
              sys(`details: ${mode}`)
            })
            .catch(() => sys(`details: ${detailsMode}`))

          return true
        }

        {
          const mode = arg.trim().toLowerCase()

          if (!['hidden', 'collapsed', 'expanded', 'cycle', 'toggle'].includes(mode)) {
            sys('usage: /details [hidden|collapsed|expanded|cycle]')

            return true
          }

          const next = mode === 'cycle' || mode === 'toggle' ? nextDetailsMode(detailsMode) : (mode as DetailsMode)
          patchUiState({ detailsMode: next })
          rpc('config.set', { key: 'details_mode', value: next }).catch(() => {})
          sys(`details: ${next}`)
        }

        return true
      case 'copy': {
        if (!arg && hasSelection) {
          const copied = selection.copySelection()

          if (copied) {
            sys('copied selection')

            return true
          }
        }

        const all = messages.filter((m: any) => m.role === 'assistant')

        if (arg && Number.isNaN(parseInt(arg, 10))) {
          sys('usage: /copy [number]')

          return true
        }

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
        const logText = gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))
        logText ? page(logText, 'Logs') : sys('no gateway logs')

        return true
      }

      case 'statusbar':

      case 'sb':
        if (arg && !['on', 'off', 'toggle'].includes(arg.trim().toLowerCase())) {
          sys('usage: /statusbar [on|off|toggle]')

          return true
        }

        {
          const mode = arg.trim().toLowerCase()
          const next = mode === 'on' ? true : mode === 'off' ? false : !ui.statusBar

          patchUiState({ statusBar: next })
          rpc('config.set', { key: 'statusbar', value: next ? 'on' : 'off' }).catch(() => {})
          queueMicrotask(() => sys(`status bar ${next ? 'on' : 'off'}`))
        }

        return true

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

        rpc('session.undo', { session_id: sid }).then((r: any) => {
          if (!r) {
            return
          }

          if (r.removed > 0) {
            setMessages((prev: any[]) => trimLastExchange(prev))
            setHistoryItems((prev: any[]) => trimLastExchange(prev))
            sys(`undid ${r.removed} messages`)
          } else {
            sys('nothing to undo')
          }
        })

        return true

      case 'retry':
        if (!lastUserMsg) {
          sys('nothing to retry')

          return true
        }

        if (sid) {
          rpc('session.undo', { session_id: sid }).then((r: any) => {
            if (!r) {
              return
            }

            if (r.removed <= 0) {
              sys('nothing to retry')

              return
            }

            setMessages((prev: any[]) => trimLastExchange(prev))
            setHistoryItems((prev: any[]) => trimLastExchange(prev))
            send(lastUserMsg)
          })

          return true
        }

        send(lastUserMsg)

        return true

      case 'background':

      case 'bg':
        if (!arg) {
          sys('/background <prompt>')

          return true
        }

        rpc('prompt.background', { session_id: sid, text: arg }).then((r: any) => {
          if (!r?.task_id) {
            return
          }

          patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add(r.task_id) }))
          sys(`bg ${r.task_id} started`)
        })

        return true

      case 'btw':
        if (!arg) {
          sys('/btw <question>')

          return true
        }

        rpc('prompt.btw', { session_id: sid, text: arg }).then((r: any) => {
          if (!r) {
            return
          }

          patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add('btw:x') }))
          sys('btw running…')
        })

        return true

      case 'model':
        if (guardBusySessionSwitch('change models')) {
          return true
        }

        if (!arg) {
          patchOverlayState({ modelPicker: true })
        } else {
          rpc('config.set', { session_id: sid, key: 'model', value: arg.trim() }).then((r: any) => {
            if (!r) {
              return
            }

            if (!r.value) {
              sys('error: invalid response: model switch')

              return
            }

            sys(`model → ${r.value}`)
            maybeWarn(r)
            patchUiState(state => ({
              ...state,
              info: state.info ? { ...state.info, model: r.value } : { model: r.value, skills: {}, tools: {} }
            }))
          })
        }

        return true

      case 'image':
        rpc('image.attach', { session_id: sid, path: arg }).then((r: any) => {
          if (!r) {
            return
          }

          const meta = imageTokenMeta(r)
          sys(`attached image: ${r.name}${meta ? ` · ${meta}` : ''}`)

          if (r?.remainder) {
            setInput(r.remainder)
          }
        })

        return true

      case 'provider':
        gw.request('slash.exec', { command: 'provider', session_id: sid })
          .then((r: any) => {
            page(
              r?.warning ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}` : r?.output || '(no output)',
              'Provider'
            )
          })
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true

      case 'skin':
        if (arg) {
          rpc('config.set', { key: 'skin', value: arg }).then((r: any) => {
            if (!r?.value) {
              return
            }

            sys(`skin → ${r.value}`)
          })
        } else {
          rpc('config.get', { key: 'skin' }).then((r: any) => {
            if (!r) {
              return
            }

            sys(`skin: ${r.value || 'default'}`)
          })
        }

        return true

      case 'yolo':
        rpc('config.set', { session_id: sid, key: 'yolo' }).then((r: any) => {
          if (!r) {
            return
          }

          sys(`yolo ${r.value === '1' ? 'on' : 'off'}`)
        })

        return true

      case 'reasoning':
        if (!arg) {
          rpc('config.get', { key: 'reasoning' }).then((r: any) => {
            if (!r?.value) {
              return
            }

            sys(`reasoning: ${r.value} · display ${r.display || 'hide'}`)
          })
        } else {
          rpc('config.set', { session_id: sid, key: 'reasoning', value: arg }).then((r: any) => {
            if (!r?.value) {
              return
            }

            sys(`reasoning: ${r.value}`)
          })
        }

        return true

      case 'verbose':
        rpc('config.set', { session_id: sid, key: 'verbose', value: arg || 'cycle' }).then((r: any) => {
          if (!r?.value) {
            return
          }

          sys(`verbose: ${r.value}`)
        })

        return true

      case 'personality':
        if (arg) {
          rpc('config.set', { session_id: sid, key: 'personality', value: arg }).then((r: any) => {
            if (!r) {
              return
            }

            if (r.history_reset) {
              resetVisibleHistory(r.info ?? null)
            }

            sys(`personality: ${r.value || 'default'}${r.history_reset ? ' · transcript cleared' : ''}`)
            maybeWarn(r)
          })
        } else {
          gw.request('slash.exec', { command: 'personality', session_id: sid })
            .then((r: any) => {
              panel('Personality', [
                {
                  text: r?.warning
                    ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}`
                    : r?.output || '(no output)'
                }
              ])
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
        }

        return true

      case 'compress':
        rpc('session.compress', { session_id: sid, ...(arg ? { focus_topic: arg } : {}) }).then((r: any) => {
          if (!r) {
            return
          }

          if (Array.isArray(r.messages)) {
            const resumed = toTranscriptMessages(r.messages)
            setMessages(resumed)
            setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
          }

          if (r.info) {
            patchUiState({ info: r.info })
          }

          if (r.usage) {
            patchUiState(state => ({ ...state, usage: { ...state.usage, ...r.usage } }))
          }

          if ((r.removed ?? 0) <= 0) {
            sys('nothing to compress')

            return
          }

          sys(`compressed ${r.removed} messages${r.usage?.total ? ' · ' + fmtK(r.usage.total) + ' tok' : ''}`)
        })

        return true

      case 'stop':
        rpc('process.stop', {}).then((r: any) => {
          if (!r) {
            return
          }

          sys(`killed ${r.killed ?? 0} registered process(es)`)
        })

        return true

      case 'branch':
      case 'fork': {
        const prevSid = sid
        rpc('session.branch', { session_id: sid, name: arg }).then((r: any) => {
          if (r?.session_id) {
            void closeSession(prevSid)
            patchUiState({ sid: r.session_id })
            setSessionStartedAt(Date.now())
            setHistoryItems([])
            setMessages([])
            sys(`branched → ${r.title}`)
          }
        })

        return true
      }

      case 'reload-mcp':

      case 'reload_mcp':
        rpc('reload.mcp', { session_id: sid }).then((r: any) => {
          if (!r) {
            return
          }

          sys('MCP reloaded')
        })

        return true

      case 'title':
        rpc('session.title', { session_id: sid, ...(arg ? { title: arg } : {}) }).then((r: any) => {
          if (!r) {
            return
          }

          sys(`title: ${r.title || '(none)'}`)
        })

        return true

      case 'usage':
        rpc('session.usage', { session_id: sid }).then((r: any) => {
          if (r) {
            patchUiState({
              usage: { input: r.input ?? 0, output: r.output ?? 0, total: r.total ?? 0, calls: r.calls ?? 0 }
            })
          }

          if (!r?.calls) {
            sys('no API calls yet')

            return
          }

          const f = (v: number) => (v ?? 0).toLocaleString()

          const cost =
            r.cost_usd != null ? `${r.cost_status === 'estimated' ? '~' : ''}$${r.cost_usd.toFixed(4)}` : null

          const rows: [string, string][] = [
            ['Model', r.model ?? ''],
            ['Input tokens', f(r.input)],
            ['Cache read tokens', f(r.cache_read)],
            ['Cache write tokens', f(r.cache_write)],
            ['Output tokens', f(r.output)],
            ['Total tokens', f(r.total)],
            ['API calls', f(r.calls)]
          ]

          if (cost) {
            rows.push(['Cost', cost])
          }

          const sections: PanelSection[] = [{ rows }]

          if (r.context_max) {
            sections.push({ text: `Context: ${f(r.context_used)} / ${f(r.context_max)} (${r.context_percent}%)` })
          }

          if (r.compressions) {
            sections.push({ text: `Compressions: ${r.compressions}` })
          }

          panel('Usage', sections)
        })

        return true

      case 'save':
        rpc('session.save', { session_id: sid }).then((r: any) => {
          if (!r?.file) {
            return
          }

          sys(`saved: ${r.file}`)
        })

        return true

      case 'history':
        rpc('session.history', { session_id: sid }).then((r: any) => {
          if (typeof r?.count !== 'number') {
            return
          }

          sys(`${r.count} messages`)
        })

        return true

      case 'profile':
        rpc('config.get', { key: 'profile' }).then((r: any) => {
          if (!r) {
            return
          }

          const text = r.display || r.home || '(unknown profile)'
          const lines = text.split('\n').filter(Boolean)

          if (lines.length <= 2) {
            panel('Profile', [{ text }])
          } else {
            page(text, 'Profile')
          }
        })

        return true

      case 'voice':
        rpc('voice.toggle', { action: arg === 'on' || arg === 'off' ? arg : 'status' }).then((r: any) => {
          if (!r) {
            return
          }

          setVoiceEnabled(!!r?.enabled)
          sys(`voice: ${r.enabled ? 'on' : 'off'}`)
        })

        return true

      case 'insights':
        rpc('insights.get', { days: parseInt(arg) || 30 }).then((r: any) => {
          if (!r) {
            return
          }

          panel('Insights', [
            {
              rows: [
                ['Period', `${r.days} days`],
                ['Sessions', `${r.sessions}`],
                ['Messages', `${r.messages}`]
              ]
            }
          ])
        })

        return true
      case 'rollback': {
        const [sub, ...rArgs] = (arg || 'list').split(/\s+/)

        if (!sub || sub === 'list') {
          rpc('rollback.list', { session_id: sid }).then((r: any) => {
            if (!r) {
              return
            }

            if (!r.checkpoints?.length) {
              return sys('no checkpoints')
            }

            panel('Checkpoints', [
              {
                rows: r.checkpoints.map(
                  (c: any, i: number) => [`${i + 1} ${c.hash?.slice(0, 8)}`, c.message] as [string, string]
                )
              }
            ])
          })
        } else {
          const hash = sub === 'restore' || sub === 'diff' ? rArgs[0] : sub

          const filePath =
            sub === 'restore' || sub === 'diff' ? rArgs.slice(1).join(' ').trim() : rArgs.join(' ').trim()

          rpc(sub === 'diff' ? 'rollback.diff' : 'rollback.restore', {
            session_id: sid,
            hash,
            ...(sub === 'diff' || !filePath ? {} : { file_path: filePath })
          }).then((r: any) => {
            if (!r) {
              return
            }

            sys(r.rendered || r.diff || r.message || 'done')
          })
        }

        return true
      }

      case 'browser': {
        const [act, ...bArgs] = (arg || 'status').split(/\s+/)
        rpc('browser.manage', { action: act, ...(bArgs[0] ? { url: bArgs[0] } : {}) }).then((r: any) => {
          if (!r) {
            return
          }

          sys(r.connected ? `browser: ${r.url}` : 'browser: disconnected')
        })

        return true
      }

      case 'plugins':
        rpc('plugins.list', {}).then((r: any) => {
          if (!r) {
            return
          }

          if (!r.plugins?.length) {
            return sys('no plugins')
          }

          panel('Plugins', [
            {
              items: r.plugins.map((p: any) => `${p.name} v${p.version}${p.enabled ? '' : ' (disabled)'}`)
            }
          ])
        })

        return true
      case 'skills': {
        const [sub, ...sArgs] = (arg || '').split(/\s+/).filter(Boolean)

        if (!sub || sub === 'list') {
          rpc('skills.manage', { action: 'list' }).then((r: any) => {
            if (!r) {
              return
            }

            const sk = r.skills as Record<string, string[]> | undefined

            if (!sk || !Object.keys(sk).length) {
              return sys('no skills installed')
            }

            panel(
              'Installed Skills',
              Object.entries(sk).map(([cat, names]) => ({
                title: cat,
                items: names as string[]
              }))
            )
          })

          return true
        }

        if (sub === 'browse') {
          const pg = parseInt(sArgs[0] ?? '1', 10) || 1
          rpc('skills.manage', { action: 'browse', page: pg }).then((r: any) => {
            if (!r) {
              return
            }

            if (!r.items?.length) {
              return sys('no skills found in the hub')
            }

            const sections: PanelSection[] = [
              {
                rows: r.items.map(
                  (s: any) =>
                    [s.name ?? '', (s.description ?? '').slice(0, 60) + (s.description?.length > 60 ? '…' : '')] as [
                      string,
                      string
                    ]
                )
              }
            ]

            if (r.page < r.total_pages) {
              sections.push({ text: `/skills browse ${r.page + 1} → next page` })
            }

            if (r.page > 1) {
              sections.push({ text: `/skills browse ${r.page - 1} → prev page` })
            }

            panel(`Skills Hub (page ${r.page}/${r.total_pages}, ${r.total} total)`, sections)
          })

          return true
        }

        gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
          .then((r: any) => {
            sys(
              r?.warning
                ? `warning: ${r.warning}\n${r?.output || '/skills: no output'}`
                : r?.output || '/skills: no output'
            )
          })
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true
      }

      case 'agents':

      case 'tasks':
        rpc('agents.list', {})
          .then((r: any) => {
            if (!r) {
              return
            }

            const procs = r.processes ?? []
            const running = procs.filter((p: any) => p.status === 'running')
            const finished = procs.filter((p: any) => p.status !== 'running')
            const sections: PanelSection[] = []

            if (running.length) {
              sections.push({
                title: `Running (${running.length})`,
                rows: running.map((p: any) => [p.session_id.slice(0, 8), p.command])
              })
            }

            if (finished.length) {
              sections.push({
                title: `Finished (${finished.length})`,
                rows: finished.map((p: any) => [p.session_id.slice(0, 8), p.command])
              })
            }

            if (!sections.length) {
              sections.push({ text: 'No active processes' })
            }

            panel('Agents', sections)
          })
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true

      case 'cron':
        if (!arg || arg === 'list') {
          rpc('cron.manage', { action: 'list' })
            .then((r: any) => {
              if (!r) {
                return
              }

              const jobs = r.jobs ?? []

              if (!jobs.length) {
                return sys('no scheduled jobs')
              }

              panel('Cron', [
                {
                  rows: jobs.map(
                    (j: any) =>
                      [j.name || j.job_id?.slice(0, 12), `${j.schedule} · ${j.state ?? 'active'}`] as [string, string]
                  )
                }
              ])
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
        } else {
          gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
            .then((r: any) => {
              sys(r?.warning ? `warning: ${r.warning}\n${r?.output || '(no output)'}` : r?.output || '(no output)')
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
        }

        return true

      case 'config':
        rpc('config.show', {})
          .then((r: any) => {
            if (!r) {
              return
            }

            panel(
              'Config',
              (r.sections ?? []).map((s: any) => ({
                title: s.title,
                rows: s.rows
              }))
            )
          })
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true

      case 'tools':
        rpc('tools.list', { session_id: sid })
          .then((r: any) => {
            if (!r) {
              return
            }

            if (!r.toolsets?.length) {
              return sys('no tools')
            }

            panel(
              'Tools',
              r.toolsets.map((ts: any) => ({
                title: `${ts.enabled ? '*' : ' '} ${ts.name} [${ts.tool_count} tools]`,
                items: ts.tools
              }))
            )
          })
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true

      case 'toolsets':
        rpc('toolsets.list', { session_id: sid })
          .then((r: any) => {
            if (!r) {
              return
            }

            if (!r.toolsets?.length) {
              return sys('no toolsets')
            }

            panel('Toolsets', [
              {
                rows: r.toolsets.map(
                  (ts: any) =>
                    [`${ts.enabled ? '(*)' : '   '} ${ts.name}`, `[${ts.tool_count}] ${ts.description}`] as [
                      string,
                      string
                    ]
                )
              }
            ])
          })
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true

      default:
        gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
          .then((r: any) => {
            sys(
              r?.warning
                ? `warning: ${r.warning}\n${r?.output || `/${name}: no output`}`
                : r?.output || `/${name}: no output`
            )
          })
          .catch(() => {
            gw.request('command.dispatch', { name: name ?? '', arg, session_id: sid })
              .then((raw: any) => {
                const d = asRpcResult(raw)

                if (!d?.type) {
                  sys('error: invalid response: command.dispatch')

                  return
                }

                if (d.type === 'exec') {
                  sys(d.output || '(no output)')
                } else if (d.type === 'alias') {
                  handler(`/${d.target}${arg ? ' ' + arg : ''}`)
                } else if (d.type === 'plugin') {
                  sys(d.output || '(no output)')
                } else if (d.type === 'skill') {
                  sys(`⚡ loading skill: ${d.name}`)

                  if (typeof d.message === 'string' && d.message.trim()) {
                    send(d.message)
                  } else {
                    sys(`/${name}: skill payload missing message`)
                  }
                }
              })
              .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
          })

        return true
    }
  }

  return handler
}
