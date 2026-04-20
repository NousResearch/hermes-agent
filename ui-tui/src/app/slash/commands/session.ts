import { imageTokenMeta, introMsg, toTranscriptMessages } from '../../../domain/messages.js'
import type {
  BackgroundStartResponse,
  BtwStartResponse,
  ConfigGetValueResponse,
  ConfigSetResponse,
  ImageAttachResponse,
  SessionBranchResponse,
  SessionCompressResponse,
  SessionUsageResponse,
  VoiceToggleResponse
} from '../../../gatewayTypes.js'
import { fmtK } from '../../../lib/text.js'
import type { PanelSection } from '../../../types.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

export const sessionCommands: SlashCommand[] = [
  {
    aliases: ['bg'],
    help: 'launch a background prompt',
    name: 'background',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys('/background <prompt>')
      }

      ctx.gateway.rpc<BackgroundStartResponse>('prompt.background', { session_id: ctx.sid, text: arg }).then(
        ctx.guarded<BackgroundStartResponse>(r => {
          if (!r.task_id) {
            return
          }

          patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add(r.task_id!) }))
          ctx.transcript.sys(`bg ${r.task_id} started`)
        })
      )
    }
  },

  {
    help: 'by-the-way follow-up',
    name: 'btw',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys('/btw <question>')
      }

      ctx.gateway.rpc<BtwStartResponse>('prompt.btw', { session_id: ctx.sid, text: arg }).then(
        ctx.guarded(() => {
          patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add('btw:x') }))
          ctx.transcript.sys('btw running…')
        })
      )
    }
  },

  {
    help: 'change or show model',
    name: 'model',
    run: (arg, ctx) => {
      if (ctx.session.guardBusySessionSwitch('change models')) {
        return
      }

      if (!arg) {
        return patchOverlayState({ modelPicker: true })
      }

      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'model', session_id: ctx.sid, value: arg.trim() }).then(
        ctx.guarded<ConfigSetResponse>(r => {
          if (!r.value) {
            return ctx.transcript.sys('error: invalid response: model switch')
          }

          ctx.transcript.sys(`model → ${r.value}`)
          ctx.local.maybeWarn(r)

          patchUiState(state => ({
            ...state,
            info: state.info ? { ...state.info, model: r.value! } : { model: r.value!, skills: {}, tools: {} }
          }))
        })
      )
    }
  },

  {
    help: 'attach an image',
    name: 'image',
    run: (arg, ctx) => {
      ctx.gateway.rpc<ImageAttachResponse>('image.attach', { path: arg, session_id: ctx.sid }).then(
        ctx.guarded<ImageAttachResponse>(r => {
          const meta = imageTokenMeta(r)

          ctx.transcript.sys(`attached image: ${r.name ?? ''}${meta ? ` · ${meta}` : ''}`)

          if (r.remainder) {
            ctx.composer.setInput(r.remainder)
          }
        })
      )
    }
  },

  {
    help: 'switch or reset personality (history reset on set)',
    name: 'personality',
    run: (arg, ctx) => {
      if (!arg) {
        return
      }

      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'personality', session_id: ctx.sid, value: arg }).then(
        ctx.guarded<ConfigSetResponse>(r => {
          if (r.history_reset) {
            ctx.session.resetVisibleHistory(r.info ?? null)
          }

          ctx.transcript.sys(`personality: ${r.value || 'default'}${r.history_reset ? ' · transcript cleared' : ''}`)
          ctx.local.maybeWarn(r)
        })
      )
    }
  },

  {
    help: 'compress transcript',
    name: 'compress',
    run: (arg, ctx) => {
      ctx.gateway
        .rpc<SessionCompressResponse>('session.compress', {
          session_id: ctx.sid,
          ...(arg ? { focus_topic: arg } : {})
        })
        .then(
          ctx.guarded<SessionCompressResponse>(r => {
            if (Array.isArray(r.messages)) {
              const rows = toTranscriptMessages(r.messages)

              ctx.transcript.setHistoryItems(r.info ? [introMsg(r.info), ...rows] : rows)
            }

            if (r.info) {
              patchUiState({ info: r.info })
            }

            if (r.usage) {
              patchUiState(state => ({ ...state, usage: { ...state.usage, ...r.usage } }))
            }

            if ((r.removed ?? 0) <= 0) {
              return ctx.transcript.sys('nothing to compress')
            }

            ctx.transcript.sys(
              `compressed ${r.removed} messages${r.usage?.total ? ` · ${fmtK(r.usage.total)} tok` : ''}`
            )
          })
        )
    }
  },

  {
    aliases: ['fork'],
    help: 'branch the session',
    name: 'branch',
    run: (arg, ctx) => {
      const prevSid = ctx.sid

      ctx.gateway.rpc<SessionBranchResponse>('session.branch', { name: arg, session_id: ctx.sid }).then(
        ctx.guarded<SessionBranchResponse>(r => {
          if (!r.session_id) {
            return
          }

          void ctx.session.closeSession(prevSid)
          patchUiState({ sid: r.session_id })
          ctx.session.setSessionStartedAt(Date.now())
          ctx.transcript.setHistoryItems([])
          ctx.transcript.sys(`branched → ${r.title ?? ''}`)
        })
      )
    }
  },

  {
    help: 'toggle voice input',
    name: 'voice',
    run: (arg, ctx) => {
      const action = arg === 'on' || arg === 'off' ? arg : 'status'

      ctx.gateway.rpc<VoiceToggleResponse>('voice.toggle', { action }).then(
        ctx.guarded<VoiceToggleResponse>(r => {
          ctx.voice.setVoiceEnabled(!!r.enabled)
          ctx.transcript.sys(`voice: ${r.enabled ? 'on' : 'off'}`)
        })
      )
    }
  },

  {
    help: 'switch theme skin (fires skin.changed)',
    name: 'skin',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'skin' })
          .then(ctx.guarded<ConfigGetValueResponse>(r => ctx.transcript.sys(`skin: ${r.value || 'default'}`)))
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'skin', value: arg })
        .then(ctx.guarded<ConfigSetResponse>(r => r.value && ctx.transcript.sys(`skin → ${r.value}`)))
    }
  },

  {
    help: 'toggle yolo mode (per-session approvals)',
    name: 'yolo',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'yolo', session_id: ctx.sid })
        .then(ctx.guarded<ConfigSetResponse>(r => ctx.transcript.sys(`yolo ${r.value === '1' ? 'on' : 'off'}`)))
    }
  },

  {
    help: 'inspect or set reasoning effort (updates live agent)',
    name: 'reasoning',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'reasoning' })
          .then(
            ctx.guarded<ConfigGetValueResponse>(
              r => r.value && ctx.transcript.sys(`reasoning: ${r.value} · display ${r.display || 'hide'}`)
            )
          )
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'reasoning', session_id: ctx.sid, value: arg })
        .then(ctx.guarded<ConfigSetResponse>(r => r.value && ctx.transcript.sys(`reasoning: ${r.value}`)))
    }
  },

  {
    help: 'cycle verbose tool-output mode (updates live agent)',
    name: 'verbose',
    run: (arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'verbose', session_id: ctx.sid, value: arg || 'cycle' })
        .then(ctx.guarded<ConfigSetResponse>(r => r.value && ctx.transcript.sys(`verbose: ${r.value}`)))
    }
  },

  {
    help: 'session usage (live counts — worker sees zeros)',
    name: 'usage',
    run: (_arg, ctx) => {
      ctx.gateway.rpc<SessionUsageResponse>('session.usage', { session_id: ctx.sid }).then(r => {
        if (ctx.stale()) {
          return
        }

        if (r) {
          patchUiState({
            usage: { calls: r.calls ?? 0, input: r.input ?? 0, output: r.output ?? 0, total: r.total ?? 0 }
          })
        }

        if (!r?.calls) {
          return ctx.transcript.sys('no API calls yet')
        }

        const f = (v: number | undefined) => (v ?? 0).toLocaleString()
        const cost = r.cost_usd != null ? `${r.cost_status === 'estimated' ? '~' : ''}$${r.cost_usd.toFixed(4)}` : null

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

        ctx.transcript.panel('Usage', sections)
      })
    }
  }
]
