import type { BackgroundStartResponse, SessionHistoryResponse } from '../../gatewayTypes.js'
import { rpcErrorMessage } from '../../lib/rpc.js'
import { fmtK } from '../../lib/text.js'
import type { PanelSection } from '../../types.js'
import { imageTokenMeta, introMsg, toTranscriptMessages } from '../helpers.js'
import type { SlashHandlerContext } from '../interfaces.js'
import { patchOverlayState } from '../overlayStore.js'
import { patchUiState } from '../uiStore.js'

import type { ParsedSlashCommand, SlashShared } from './shared.js'

const SLASH_OUTPUT_PAGE: Record<string, string> = {
  debug: 'Debug',
  fast: 'Fast',
  platforms: 'Platforms',
  snapshot: 'Snapshot'
}

export function createSlashSessionHandler(ctx: SlashHandlerContext, shared: SlashShared) {
  const { setInput } = ctx.composer
  const { gw, rpc } = ctx.gateway
  const { maybeWarn } = ctx.local
  const { closeSession, guardBusySessionSwitch, resetVisibleHistory, setSessionStartedAt } = ctx.session
  const { page, panel, setHistoryItems, sys } = ctx.transcript
  const { setVoiceEnabled } = ctx.voice

  return ({ arg, cmd, name, sid }: SessionSlashCommand) => {
    const pageTitle = SLASH_OUTPUT_PAGE[name]

    if (pageTitle) {
      shared.showSlashOutput(pageTitle, cmd.slice(1), sid)

      return true
    }

    switch (name) {
      case 'background':

      case 'bg':
        if (!arg) {
          sys('/background <prompt>')

          return true
        }

        rpc<BackgroundStartResponse>('prompt.background', { session_id: sid, text: arg }).then(r => {
          const taskId = r?.task_id

          if (!taskId) {
            return
          }

          patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add(taskId) }))
          sys(`bg ${taskId} started`)
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

          return true
        }

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

        return true

      case 'image':
        rpc('image.attach', { session_id: sid, path: arg }).then((r: any) => {
          if (!r) {
            return
          }

          const meta = imageTokenMeta(r)

          sys(`attached image: ${r.name}${meta ? ` · ${meta}` : ''}`)
          r?.remainder && setInput(r.remainder)
        })

        return true

      case 'provider':
        gw.request('slash.exec', { command: 'provider', session_id: sid })
          .then((r: any) =>
            page(
              r?.warning ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}` : r?.output || '(no output)',
              'Provider'
            )
          )
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true

      case 'skin':
        if (arg) {
          rpc('config.set', { key: 'skin', value: arg }).then((r: any) => r?.value && sys(`skin → ${r.value}`))
        } else {
          rpc('config.get', { key: 'skin' }).then((r: any) => r && sys(`skin: ${r.value || 'default'}`))
        }

        return true

      case 'yolo':
        rpc('config.set', { session_id: sid, key: 'yolo' }).then(
          (r: any) => r && sys(`yolo ${r.value === '1' ? 'on' : 'off'}`)
        )

        return true

      case 'reasoning':
        if (!arg) {
          rpc('config.get', { key: 'reasoning' }).then(
            (r: any) => r?.value && sys(`reasoning: ${r.value} · display ${r.display || 'hide'}`)
          )
        } else {
          rpc('config.set', { session_id: sid, key: 'reasoning', value: arg }).then(
            (r: any) => r?.value && sys(`reasoning: ${r.value}`)
          )
        }

        return true

      case 'verbose':
        rpc('config.set', { session_id: sid, key: 'verbose', value: arg || 'cycle' }).then(
          (r: any) => r?.value && sys(`verbose: ${r.value}`)
        )

        return true

      case 'personality':
        if (arg) {
          rpc('config.set', { session_id: sid, key: 'personality', value: arg }).then((r: any) => {
            if (!r) {
              return
            }

            r.history_reset && resetVisibleHistory(r.info ?? null)
            sys(`personality: ${r.value || 'default'}${r.history_reset ? ' · transcript cleared' : ''}`)
            maybeWarn(r)
          })

          return true
        }

        gw.request('slash.exec', { command: 'personality', session_id: sid })
          .then((r: any) =>
            panel('Personality', [
              {
                text: r?.warning ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}` : r?.output || '(no output)'
              }
            ])
          )
          .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

        return true

      case 'compress':
        rpc('session.compress', { session_id: sid, ...(arg ? { focus_topic: arg } : {}) }).then((r: any) => {
          if (!r) {
            return
          }

          Array.isArray(r.messages) &&
            setHistoryItems(
              r.info ? [introMsg(r.info), ...toTranscriptMessages(r.messages)] : toTranscriptMessages(r.messages)
            )
          r.info && patchUiState({ info: r.info })
          r.usage && patchUiState(state => ({ ...state, usage: { ...state.usage, ...r.usage } }))

          if ((r.removed ?? 0) <= 0) {
            sys('nothing to compress')

            return
          }

          sys(`compressed ${r.removed} messages${r.usage?.total ? ` · ${fmtK(r.usage.total)} tok` : ''}`)
        })

        return true

      case 'stop':
        rpc('process.stop', {}).then((r: any) => r && sys(`killed ${r.killed ?? 0} registered process(es)`))

        return true

      case 'branch':
      case 'fork': {
        const prevSid = sid

        rpc('session.branch', { session_id: sid, name: arg }).then((r: any) => {
          if (!r?.session_id) {
            return
          }

          void closeSession(prevSid)
          patchUiState({ sid: r.session_id })
          setSessionStartedAt(Date.now())
          setHistoryItems([])
          sys(`branched → ${r.title}`)
        })

        return true
      }

      case 'reload-mcp':

      case 'reload_mcp':
        rpc('reload.mcp', { session_id: sid }).then((r: any) => r && sys('MCP reloaded'))

        return true

      case 'title':
        rpc('session.title', { session_id: sid, ...(arg ? { title: arg } : {}) }).then(
          (r: any) => r && sys(`title: ${r.title || '(none)'}`)
        )

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

          const sections: PanelSection[] = [{ rows }]

          cost && rows.push(['Cost', cost])
          r.context_max &&
            sections.push({ text: `Context: ${f(r.context_used)} / ${f(r.context_max)} (${r.context_percent}%)` })
          r.compressions && sections.push({ text: `Compressions: ${r.compressions}` })
          panel('Usage', sections)
        })

        return true

      case 'save':
        rpc('session.save', { session_id: sid }).then((r: any) => r?.file && sys(`saved: ${r.file}`))

        return true

      case 'history':
        rpc<SessionHistoryResponse>('session.history', { session_id: sid }).then(r => {
          if (typeof r?.count !== 'number') {
            return
          }

          if (!r.messages?.length) {
            sys(`${r.count} messages`)

            return
          }

          page(
            r.messages
              .map((msg, index) =>
                msg.role === 'tool'
                  ? `[Tool #${index + 1}] ${msg.name || 'tool'} ${msg.context || ''}`.trim()
                  : `[${msg.role === 'assistant' ? 'Hermes' : msg.role === 'user' ? 'You' : 'System'} #${index + 1}] ${msg.text || ''}`.trim()
              )
              .join('\n\n'),
            `History (${r.count})`
          )
        })

        return true

      case 'profile':
        rpc('config.get', { key: 'profile' }).then((r: any) => {
          if (!r) {
            return
          }

          const text = r.display || r.home || '(unknown profile)'
          const lines = text.split('\n').filter(Boolean)

          lines.length <= 2 ? panel('Profile', [{ text }]) : page(text, 'Profile')
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
    }

    return false
  }
}

interface SessionSlashCommand extends ParsedSlashCommand {
  sid: null | string
}
