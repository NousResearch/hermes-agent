import { imageTokenMeta, introMsg, toTranscriptMessages } from '../../../domain/messages.js'
import type {
  BackgroundStartResponse,
  BtwStartResponse,
  ConfigGetValueResponse,
  ConfigSetResponse,
  ImageAttachResponse,
  InsightsResponse,
  ReloadMcpResponse,
  SessionBranchResponse,
  SessionCompressResponse,
  SessionHistoryResponse,
  SessionSaveResponse,
  SessionTitleResponse,
  SessionUsageResponse,
  SlashExecResponse,
  VoiceToggleResponse
} from '../../../gatewayTypes.js'
import { fmtK } from '../../../lib/text.js'
import type { PanelSection } from '../../../types.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

const PAGE_TITLES: Record<string, string> = {
  debug: 'Debug',
  fast: 'Fast',
  platforms: 'Platforms',
  snapshot: 'Snapshot'
}

const passthrough = (name: string): SlashCommand => ({
  name,
  run: (_arg, ctx, cmd) =>
    ctx.shared.showSlashOutput({
      command: cmd.slice(1),
      flight: ctx.flight,
      sid: ctx.sid,
      title: PAGE_TITLES[name] ?? name
    })
})

const historyLabel = (role: string) => (role === 'assistant' ? 'Hermes' : role === 'user' ? 'You' : 'System')

export const sessionCommands: SlashCommand[] = [
  passthrough('debug'),
  passthrough('fast'),
  passthrough('platforms'),
  passthrough('snapshot'),

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
          r.remainder && ctx.composer.setInput(r.remainder)
        })
      )
    }
  },

  {
    help: 'show provider details',
    name: 'provider',
    run: (_arg, ctx) => {
      ctx.gateway.gw
        .request<SlashExecResponse>('slash.exec', { command: 'provider', session_id: ctx.sid })
        .then(r => {
          if (ctx.stale()) {
            return
          }

          ctx.transcript.page(
            r?.warning ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}` : r?.output || '(no output)',
            'Provider'
          )
        })
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'switch theme skin',
    name: 'skin',
    run: (arg, ctx) => {
      if (arg) {
        return ctx.gateway
          .rpc<ConfigSetResponse>('config.set', { key: 'skin', value: arg })
          .then(ctx.guarded<ConfigSetResponse>(r => r.value && ctx.transcript.sys(`skin → ${r.value}`)))
      }

      ctx.gateway
        .rpc<ConfigGetValueResponse>('config.get', { key: 'skin' })
        .then(ctx.guarded<ConfigGetValueResponse>(r => ctx.transcript.sys(`skin: ${r.value || 'default'}`)))
    }
  },

  {
    help: 'toggle yolo mode',
    name: 'yolo',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'yolo', session_id: ctx.sid })
        .then(ctx.guarded<ConfigSetResponse>(r => ctx.transcript.sys(`yolo ${r.value === '1' ? 'on' : 'off'}`)))
    }
  },

  {
    help: 'inspect or set reasoning mode',
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
    help: 'cycle verbose output',
    name: 'verbose',
    run: (arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'verbose', session_id: ctx.sid, value: arg || 'cycle' })
        .then(ctx.guarded<ConfigSetResponse>(r => r.value && ctx.transcript.sys(`verbose: ${r.value}`)))
    }
  },

  {
    help: 'personality panel or switch',
    name: 'personality',
    run: (arg, ctx) => {
      if (arg) {
        return ctx.gateway
          .rpc<ConfigSetResponse>('config.set', { key: 'personality', session_id: ctx.sid, value: arg })
          .then(
            ctx.guarded<ConfigSetResponse>(r => {
              r.history_reset && ctx.session.resetVisibleHistory(r.info ?? null)
              ctx.transcript.sys(
                `personality: ${r.value || 'default'}${r.history_reset ? ' · transcript cleared' : ''}`
              )
              ctx.local.maybeWarn(r)
            })
          )
      }

      ctx.gateway.gw
        .request<SlashExecResponse>('slash.exec', { command: 'personality', session_id: ctx.sid })
        .then(r => {
          if (ctx.stale()) {
            return
          }

          ctx.transcript.panel('Personality', [
            {
              text: r?.warning ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}` : r?.output || '(no output)'
            }
          ])
        })
        .catch(ctx.guardedErr)
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

            r.info && patchUiState({ info: r.info })
            r.usage && patchUiState(state => ({ ...state, usage: { ...state.usage, ...r.usage } }))

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
    help: 'stop background processes',
    name: 'stop',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<{ killed?: number }>('process.stop', {})
        .then(
          ctx.guarded<{ killed?: number }>(r => ctx.transcript.sys(`killed ${r.killed ?? 0} registered process(es)`))
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
    aliases: ['reload_mcp'],
    help: 'reload MCP servers',
    name: 'reload-mcp',
    run: (_arg, ctx) =>
      ctx.gateway
        .rpc<ReloadMcpResponse>('reload.mcp', { session_id: ctx.sid })
        .then(ctx.guarded(() => ctx.transcript.sys('MCP reloaded')))
  },

  {
    help: 'inspect or set session title',
    name: 'title',
    run: (arg, ctx) => {
      ctx.gateway
        .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid, ...(arg ? { title: arg } : {}) })
        .then(ctx.guarded<SessionTitleResponse>(r => ctx.transcript.sys(`title: ${r.title || '(none)'}`)))
    }
  },

  {
    help: 'session usage',
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

        const sections: PanelSection[] = [{ rows }]

        cost && rows.push(['Cost', cost])
        r.context_max &&
          sections.push({ text: `Context: ${f(r.context_used)} / ${f(r.context_max)} (${r.context_percent}%)` })
        r.compressions && sections.push({ text: `Compressions: ${r.compressions}` })

        ctx.transcript.panel('Usage', sections)
      })
    }
  },

  {
    help: 'save transcript to disk',
    name: 'save',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<SessionSaveResponse>('session.save', { session_id: ctx.sid })
        .then(ctx.guarded<SessionSaveResponse>(r => r.file && ctx.transcript.sys(`saved: ${r.file}`)))
    }
  },

  {
    help: 'view message history',
    name: 'history',
    run: (_arg, ctx) => {
      ctx.gateway.rpc<SessionHistoryResponse>('session.history', { session_id: ctx.sid }).then(r => {
        if (ctx.stale() || typeof r?.count !== 'number') {
          return
        }

        if (!r.messages?.length) {
          return ctx.transcript.sys(`${r.count} messages`)
        }

        const body = r.messages
          .map((m, i) =>
            m.role === 'tool'
              ? `[Tool #${i + 1}] ${m.name || 'tool'} ${m.context || ''}`.trim()
              : `[${historyLabel(m.role)} #${i + 1}] ${m.text || ''}`.trim()
          )
          .join('\n\n')

        ctx.transcript.page(body, `History (${r.count})`)
      })
    }
  },

  {
    help: 'show current profile',
    name: 'profile',
    run: (_arg, ctx) => {
      ctx.gateway.rpc<ConfigGetValueResponse>('config.get', { key: 'profile' }).then(
        ctx.guarded<ConfigGetValueResponse>(r => {
          const text = r.display || r.home || '(unknown profile)'
          const lines = text.split('\n').filter(Boolean)

          lines.length <= 2 ? ctx.transcript.panel('Profile', [{ text }]) : ctx.transcript.page(text, 'Profile')
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
    help: 'view usage insights',
    name: 'insights',
    run: (arg, ctx) => {
      ctx.gateway.rpc<InsightsResponse>('insights.get', { days: parseInt(arg) || 30 }).then(
        ctx.guarded<InsightsResponse>(r =>
          ctx.transcript.panel('Insights', [
            {
              rows: [
                ['Period', `${r.days ?? 0} days`],
                ['Sessions', `${r.sessions ?? 0}`],
                ['Messages', `${r.messages ?? 0}`]
              ]
            }
          ])
        )
      )
    }
  }
]
