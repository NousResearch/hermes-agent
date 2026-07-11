import { attachedImageNotice, introMsg, toTranscriptMessages } from '../../../domain/messages.js'
import { TUI_SESSION_MODEL_FLAG } from '../../../domain/slash.js'
import type {
  BackgroundStartResponse,
  ConfigGetValueResponse,
  ConfigSetResponse,
  ImageAttachResponse,
  SessionBranchResponse,
  SessionCompressResponse,
  SessionUsageResponse,
  SlashExecResponse,
  VoiceToggleResponse
} from '../../../gatewayTypes.js'
import { translate, type TranslationKey } from '../../../i18n/index.js'
import { formatVoiceRecordKey, parseVoiceRecordKey } from '../../../lib/platform.js'
import { fmtK } from '../../../lib/text.js'
import type { PanelSection } from '../../../types.js'
import { DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES, type IndicatorStyle } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

const TUI_SESSION_MODEL_RE = new RegExp(`(?:^|\\s)${TUI_SESSION_MODEL_FLAG}(?:\\s|$)`)
const TUI_SESSION_STRIP_RE = new RegExp(`\\s*${TUI_SESSION_MODEL_FLAG}\\b\\s*`, 'g')

const stripTuiSessionFlag = (trimmed: string) => trimmed.replace(TUI_SESSION_STRIP_RE, ' ').replace(/\s+/g, ' ').trim()

const formatUsageCost = (r: SessionUsageResponse) =>
  r.cost_usd != null ? `${r.cost_status === 'estimated' ? '~' : ''}$${r.cost_usd.toFixed(4)}` : null

const modelValueForConfigSet = (arg: string) => {
  const trimmed = arg.trim()

  if (!trimmed) {
    return trimmed
  }

  if (TUI_SESSION_MODEL_RE.test(trimmed)) {
    return stripTuiSessionFlag(trimmed)
  }

  return trimmed
}

export const sessionCommands: SlashCommand[] = [
  {
    aliases: ['bg', 'btw'],
    help: 'launch a background prompt',
    name: 'background',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.backgroundUsage'))
      }

      ctx.gateway.rpc<BackgroundStartResponse>('prompt.background', { session_id: ctx.sid, text: arg }).then(
        ctx.guarded<BackgroundStartResponse>(r => {
          if (!r.task_id) {
            return
          }

          patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add(r.task_id!) }))
          ctx.transcript.sys(translate(ctx.ui.locale, 'sys.backgroundStarted', { taskId: r.task_id }))
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

      if (!arg.trim()) {
        return patchOverlayState({ modelPicker: true })
      }

      if (arg.trim() === '--refresh') {
        return patchOverlayState({ modelPicker: { refresh: true } })
      }

      const switchModel = (confirmExpensiveModel = false) =>
        ctx.gateway
          .rpc<ConfigSetResponse>('config.set', {
            confirm_expensive_model: confirmExpensiveModel,
            key: 'model',
            session_id: ctx.sid,
            value: modelValueForConfigSet(arg)
          })
          .then(
            ctx.guarded<ConfigSetResponse>(r => {
              if (r.confirm_required) {
                patchOverlayState({
                  confirm: {
                    cancelLabel: translate(ctx.ui.locale, 'common.cancel'),
                    confirmLabel: translate(ctx.ui.locale, 'sys.switchAnyway'),
                    danger: true,
                    detail: r.confirm_message || r.warning || translate(ctx.ui.locale, 'sys.expensiveModelDetail'),
                    onConfirm: () => switchModel(true),
                    title: translate(ctx.ui.locale, 'sys.expensiveModelTitle')
                  }
                })

                return
              }

              if (!r.value) {
                return ctx.transcript.sys(
                  translate(ctx.ui.locale, 'errors.invalidResponse', { method: 'model switch' })
                )
              }

              ctx.transcript.sys(translate(ctx.ui.locale, 'sys.modelSet', { model: r.value }))
              ctx.local.maybeWarn(r)

              patchUiState(state => ({
                ...state,
                info: state.info ? { ...state.info, model: r.value! } : { model: r.value!, skills: {}, tools: {} }
              }))
            })
          )

      switchModel()
    }
  },

  {
    aliases: ['switch', 'session', 'resume'],
    help: 'browse, switch, or resume sessions',
    name: 'sessions',
    run: (arg, ctx) => {
      const trimmed = arg.trim()

      // A new *live* session keeps the current one running in the background
      // (it doesn't close it), so fanning out while busy is allowed — that's
      // the whole point of multiple live sessions.
      if (trimmed.toLowerCase() === 'new') {
        return ctx.session.newLiveSession()
      }

      // `/resume <id|title>` (and `/sessions <id>`) load a cold session and
      // CLOSE the current one, so guard it while a turn is in-flight to avoid
      // corrupting streaming/busy state. Bare opens the overlay to browse.
      if (trimmed) {
        if (ctx.session.guardBusySessionSwitch('switch sessions')) {
          return
        }

        return ctx.session.resumeById(trimmed)
      }

      patchOverlayState({ sessions: true })
    }
  },

  {
    help: 'attach an image',
    name: 'image',
    run: (arg, ctx) => {
      ctx.gateway.rpc<ImageAttachResponse>('image.attach', { path: arg, session_id: ctx.sid }).then(
        ctx.guarded<ImageAttachResponse>(r => {
          ctx.transcript.sys(attachedImageNotice(r, ctx.ui.locale))

          if (r.remainder) {
            ctx.composer.setInput(r.remainder)
          }
        })
      )
    }
  },

  {
    help: 'switch personality for this session',
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

          ctx.transcript.sys(
            translate(ctx.ui.locale, 'sys.personality', {
              value: r.value || translate(ctx.ui.locale, 'common.default'),
              suffix: r.history_reset ? ` · ${translate(ctx.ui.locale, 'common.transcriptCleared')}` : ''
            })
          )
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

            if (r.summary?.headline) {
              const prefix = r.summary.noop ? '' : '✓ '

              ctx.transcript.sys(`${prefix}${r.summary.headline}`)

              if (r.summary.token_line) {
                ctx.transcript.sys(`  ${r.summary.token_line}`)
              }

              if (r.summary.note) {
                ctx.transcript.sys(`  ${r.summary.note}`)
              }

              return
            }

            if ((r.removed ?? 0) <= 0) {
              return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.nothingToCompress'))
            }

            ctx.transcript.sys(
              translate(ctx.ui.locale, 'sys.compressedMessages', {
                count: r.removed ?? 0,
                tokens: r.usage?.total ? ` · ${fmtK(r.usage.total)} tok` : ''
              })
            )
          })
        )
        .catch(ctx.guardedErr)
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
          ctx.transcript.sys(translate(ctx.ui.locale, 'sys.branched', { title: r.title ?? '' }))
        })
      )
    }
  },

  {
    help: 'voice mode: [on|off|tts|status]',
    name: 'voice',
    run: (arg, ctx) => {
      const normalized = (arg ?? '').trim().toLowerCase()

      const action =
        normalized === 'on' || normalized === 'off' || normalized === 'tts' || normalized === 'status'
          ? normalized
          : 'status'

      const ti = (key: TranslationKey, vars?: Record<string, string | number>) => translate(ctx.ui.locale, key, vars)
      ctx.gateway.rpc<VoiceToggleResponse>('voice.toggle', { action }).then(
        ctx.guarded<VoiceToggleResponse>(r => {
          ctx.voice.setVoiceEnabled(!!r.enabled)
          ctx.voice.setVoiceTts(!!r.tts)

          // Render the configured record key (config.yaml ``voice.record_key``)
          // instead of hardcoded "Ctrl+B" — the gateway response carries the
          // current value so /voice status and /voice on stay in sync with
          // both the CLI and the TUI's actual binding (#18994).
          //
          // Copilot review on #19835 caught that rendering from the fresh
          // backend response WITHOUT updating the frontend ``voice.recordKey``
          // state would skew display and binding between config-edit and
          // the next ``mtime`` poll (~5s). Parse once, push into state so
          // ``useInputHandlers()`` picks up the new binding immediately.
          //
          // Round-2 follow-up: only push state when the response actually
          // carries ``record_key`` — otherwise an older gateway (or a future
          // branch that forgets to include it) would clobber a custom user
          // binding back to the default on every /voice invocation. The
          // label still falls back to the documented default for display.
          const parsed = r.record_key ? parseVoiceRecordKey(r.record_key) : undefined

          if (parsed) {
            ctx.voice.setVoiceRecordKey(parsed)
          }

          const recordKeyLabel = formatVoiceRecordKey(parsed ?? parseVoiceRecordKey('ctrl+b'))

          // Match CLI's _show_voice_status / _enable_voice_mode /
          // _toggle_voice_tts output shape so users don't have to learn
          // two vocabularies.
          if (action === 'status') {
            const mode = r.enabled ? 'ON' : 'OFF'
            const tts = r.tts ? 'ON' : 'OFF'
            ctx.transcript.sys(ti('voice.statusTitle'))
            ctx.transcript.sys(`  ${ti('voice.modeLabel')}:       ${mode}`)
            ctx.transcript.sys(`  ${ti('voice.ttsLabel')}:        ${tts}`)
            ctx.transcript.sys(`  ${ti('voice.recordKeyLabel')}: ${recordKeyLabel}`)

            // CLI's "Requirements:" block — surfaces STT/audio setup issues
            // so the user sees "STT provider: MISSING ..." instead of
            // silently failing on every record-key press.
            if (r.details) {
              ctx.transcript.sys('')
              ctx.transcript.sys(`  ${ti('voice.requirements')}:`)

              for (const line of r.details.split('\n')) {
                if (line.trim()) {
                  ctx.transcript.sys(`    ${line}`)
                }
              }
            }

            return
          }

          if (action === 'tts') {
            ctx.transcript.sys(`${r.tts ? ti('voice.ttsEnabled') : ti('voice.ttsDisabled')}`)

            return
          }

          // on/off — mirror cli.py:_enable_voice_mode's 3-line output
          if (r.enabled) {
            const tts = r.tts ? ti('voice.ttsEnabledSuffix') : ''
            ctx.transcript.sys(ti('voice.modeEnabled', { tts }))
            ctx.transcript.sys(ti('voice.recordHint', { key: recordKeyLabel }))
            ctx.transcript.sys(ti('voice.ttsToggleHint'))
            ctx.transcript.sys(ti('voice.disableHint'))
          } else {
            ctx.transcript.sys(ti('voice.modeDisabled'))
          }
        })
      )
    }
  },

  {
    help: 'toggle / adopt / resize an animated pet',
    name: 'pet',
    usage: '/pet [toggle | list | scale <n> | <slug>]',
    run: (arg, ctx, cmd) => {
      const sub = arg.trim().toLowerCase()

      // Gallery picker — the interactive browse surface.
      if (sub === 'list') {
        return patchOverlayState({ petPicker: true })
      }

      // Bare /pet and /pet toggle flip display.pet.enabled via the slash worker.
      ctx.gateway.gw
        .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
        .then(
          ctx.guarded<SlashExecResponse>(r => {
            const body = r.output || '/pet: no output'
            ctx.transcript.sys(r.warning ? `warning: ${r.warning}\n${body}` : body)
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'switch theme skin (fires skin.changed)',
    name: 'skin',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.gateway.rpc<ConfigGetValueResponse>('config.get', { key: 'skin' }).then(
          ctx.guarded<ConfigGetValueResponse>(r =>
            ctx.transcript.sys(
              translate(ctx.ui.locale, 'sys.skinCurrent', {
                value: r.value || translate(ctx.ui.locale, 'common.default')
              })
            )
          )
        )
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'skin', value: arg })
        .then(
          ctx.guarded<ConfigSetResponse>(
            r => r.value && ctx.transcript.sys(translate(ctx.ui.locale, 'sys.skinSet', { value: r.value }))
          )
        )
    }
  },

  {
    help: 'pick the busy indicator: kaomoji (default), emoji, unicode (braille), or ascii',
    name: 'indicator',
    usage: `/indicator [${INDICATOR_STYLES.join('|')}]`,
    run: (arg, ctx) => {
      const value = arg.trim().toLowerCase()

      if (!value) {
        return ctx.gateway.rpc<ConfigGetValueResponse>('config.get', { key: 'indicator' }).then(
          ctx.guarded<ConfigGetValueResponse>(r =>
            ctx.transcript.sys(
              translate(ctx.ui.locale, 'sys.indicatorStyle', {
                value: r.value || DEFAULT_INDICATOR_STYLE
              })
            )
          )
        )
      }

      if (!(INDICATOR_STYLES as readonly string[]).includes(value)) {
        return ctx.transcript.sys(
          translate(ctx.ui.locale, 'sys.usageIndicator', { styles: INDICATOR_STYLES.join('|') })
        )
      }

      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'indicator', value }).then(
        ctx.guarded<ConfigSetResponse>(r => {
          if (!r.value) {
            return
          }

          // Hot-swap the running TUI immediately so the next render
          // uses the new style without waiting for the 5s mtime poll
          // to re-apply config.full.
          patchUiState({ indicatorStyle: value as IndicatorStyle })
          ctx.transcript.sys(translate(ctx.ui.locale, 'sys.indicatorStyle', { value: r.value }))
        })
      )
    }
  },

  {
    help: 'toggle yolo mode (per-session approvals)',
    name: 'yolo',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'yolo', session_id: ctx.sid })
        .then(
          ctx.guarded<ConfigSetResponse>(r =>
            ctx.transcript.sys(translate(ctx.ui.locale, r.value === '1' ? 'sys.yoloOn' : 'sys.yoloOff'))
          )
        )
    }
  },

  {
    help: 'inspect or set reasoning effort (updates live agent)',
    name: 'reasoning',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.gateway.rpc<ConfigGetValueResponse>('config.get', { key: 'reasoning' }).then(
          ctx.guarded<ConfigGetValueResponse>(
            r =>
              r.value &&
              ctx.transcript.sys(
                translate(ctx.ui.locale, 'sys.reasoningCurrent', {
                  value: r.value,
                  display: r.display || 'hide'
                })
              )
          )
        )
      }

      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'reasoning', session_id: ctx.sid, value: arg }).then(
        ctx.guarded<ConfigSetResponse>(r => {
          if (!r.value) {
            return
          }

          if (r.value === 'hide') {
            patchUiState(state => ({
              ...state,
              sections: { ...state.sections, thinking: 'hidden' },
              showReasoning: false
            }))
          } else if (r.value === 'show') {
            patchUiState(state => ({
              ...state,
              sections: { ...state.sections, thinking: 'expanded' },
              showReasoning: true
            }))
          }

          ctx.transcript.sys(translate(ctx.ui.locale, 'sys.reasoningSet', { value: r.value }))
        })
      )
    }
  },

  {
    help: 'toggle fast mode [normal|fast|status|on|off|toggle]',
    name: 'fast',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const valid = new Set(['', 'status', 'normal', 'fast', 'on', 'off', 'toggle'])

      if (!valid.has(mode)) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageFast'))
      }

      if (!mode || mode === 'status') {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'fast', session_id: ctx.sid })
          .then(
            ctx.guarded<ConfigGetValueResponse>(r =>
              ctx.transcript.sys(
                translate(ctx.ui.locale, 'sys.fastMode', { value: r.value === 'fast' ? 'fast' : 'normal' })
              )
            )
          )
          .catch(ctx.guardedErr)
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'fast', session_id: ctx.sid, value: mode })
        .then(
          ctx.guarded<ConfigSetResponse>(r => {
            const next = r.value === 'fast' ? 'fast' : 'normal'
            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.fastMode', { value: next }))
            patchUiState(state => ({
              ...state,
              info: state.info
                ? {
                    ...state.info,
                    fast: next === 'fast',
                    service_tier: next === 'fast' ? 'priority' : ''
                  }
                : state.info
            }))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'control busy enter mode [queue|steer|interrupt|status]',
    name: 'busy',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const valid = new Set(['', 'status', 'queue', 'steer', 'interrupt'])

      if (!valid.has(mode)) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageBusy'))
      }

      if (!mode || mode === 'status') {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'busy' })
          .then(
            ctx.guarded<ConfigGetValueResponse>(r => {
              const current = r.value || 'interrupt'
              ctx.transcript.sys(translate(ctx.ui.locale, 'sys.busyInputMode', { value: current }))
            })
          )
          .catch(ctx.guardedErr)
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'busy', value: mode })
        .then(
          ctx.guarded<ConfigSetResponse>(r => {
            const next = r.value || mode
            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.busyInputMode', { value: next }))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'cycle verbose tool-output mode (updates live agent)',
    name: 'verbose',
    run: (arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'verbose', session_id: ctx.sid, value: arg || 'cycle' })
        .then(
          ctx.guarded<ConfigSetResponse>(
            r => r.value && ctx.transcript.sys(translate(ctx.ui.locale, 'sys.verboseMode', { value: r.value }))
          )
        )
    }
  },

  {
    help: 'session usage + Nous credits',
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

        // Nous credits block is agent-independent (a portal fetch), so it shows
        // even with zero API calls or on a resumed session. Render it whenever
        // present, before the token panel.
        const creditsLines = r?.credits_lines ?? []

        if (creditsLines.length) {
          ctx.transcript.panel('Nous credits', [{ text: creditsLines.join('\n') }])
        }

        if (!r?.calls) {
          if (!creditsLines.length) {
            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.noApiCalls'))
          }

          return
        }

        const f = (v: number | undefined) => (v ?? 0).toLocaleString()
        const cost = formatUsageCost(r)

        const t = (key: TranslationKey, vars?: Record<string, string | number>) => translate(ctx.ui.locale, key, vars)

        const rows: [string, string][] = [
          [t('usage.model'), r.model ?? ''],
          [t('usage.inputTokens'), f(r.input)],
          [t('usage.cacheReadTokens'), f(r.cache_read)],
          [t('usage.cacheWriteTokens'), f(r.cache_write)],
          [t('usage.outputTokens'), f(r.output)],
          [t('usage.totalTokens'), f(r.total)],
          [t('usage.apiCalls'), f(r.calls)]
        ]

        if (cost) {
          rows.push([t('usage.cost'), cost])
        }

        const sections: PanelSection[] = [{ rows }]

        if (r.context_max) {
          sections.push({
            text: t('usage.context', {
              max: f(r.context_max),
              percent: String(r.context_percent),
              used: f(r.context_used)
            })
          })
        }

        if (r.compressions) {
          sections.push({ text: t('usage.compressions', { count: String(r.compressions) }) })
        }

        ctx.transcript.panel(t('usage.panelTitle'), sections)
      })
    }
  }
]
