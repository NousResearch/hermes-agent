import type { GatewayEvent } from '../gatewayClient.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import {
  buildToolTrailLine,
  estimateTokensRough,
  isToolTrailResultLine,
  sameToolTrailGroup,
  toolTrailLabel
} from '../lib/text.js'
import { fromSkin } from '../theme.js'

import { introMsg, toTranscriptMessages } from './helpers.js'
import type { GatewayEventHandlerContext } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { getUiState, patchUiState } from './uiStore.js'

export function createGatewayEventHandler(ctx: GatewayEventHandlerContext): (ev: GatewayEvent) => void {
  const { dequeue, queueEditRef, sendQueued } = ctx.composer
  const { gw, rpc } = ctx.gateway
  const { STARTUP_RESUME_ID, colsRef, newSession, resetSession, setCatalog } = ctx.session
  const { bellOnComplete, stdout, sys } = ctx.system
  const { appendMessage, setHistoryItems, setMessages } = ctx.transcript

  const {
    clearReasoning,
    endReasoningPhase,
    idle,
    pruneTransient,
    pulseReasoningStreaming,
    pushActivity,
    pushTrail,
    scheduleReasoning,
    scheduleStreaming,
    setActivity,
    setReasoningTokens,
    setStreaming,
    setToolTokens,
    setTools,
    setTurnTrail
  } = ctx.turn.actions

  const {
    activeToolsRef,
    bufRef,
    interruptedRef,
    lastStatusNoteRef,
    persistedToolLabelsRef,
    protocolWarnedRef,
    reasoningRef,
    statusTimerRef,
    toolTokenAccRef,
    toolCompleteRibbonRef,
    turnToolsRef
  } = ctx.turn.refs

  return (ev: GatewayEvent) => {
    const sid = getUiState().sid

    if (ev.session_id && sid && ev.session_id !== sid && !ev.type.startsWith('gateway.')) {
      return
    }

    const p = ev.payload as any

    switch (ev.type) {
      case 'gateway.ready':
        if (p?.skin) {
          patchUiState({
            theme: fromSkin(
              p.skin.colors ?? {},
              p.skin.branding ?? {},
              p.skin.banner_logo ?? '',
              p.skin.banner_hero ?? ''
            )
          })
        }

        rpc('commands.catalog', {})
          .then((r: any) => {
            if (!r?.pairs) {
              return
            }

            setCatalog({
              canon: (r.canon ?? {}) as Record<string, string>,
              categories: (r.categories ?? []) as any,
              pairs: r.pairs as [string, string][],
              skillCount: (r.skill_count ?? 0) as number,
              sub: (r.sub ?? {}) as Record<string, string[]>
            })

            if (r.warning) {
              pushActivity(String(r.warning), 'warn')
            }
          })
          .catch((e: unknown) => pushActivity(`command catalog unavailable: ${rpcErrorMessage(e)}`, 'warn'))

        if (STARTUP_RESUME_ID) {
          patchUiState({ status: 'resuming…' })
          gw.request('session.resume', { cols: colsRef.current, session_id: STARTUP_RESUME_ID })
            .then((raw: any) => {
              const r = asRpcResult(raw)

              if (!r) {
                throw new Error('invalid response: session.resume')
              }

              resetSession()
              const resumed = toTranscriptMessages(r.messages)

              patchUiState({
                info: r.info ?? null,
                sid: r.session_id,
                status: 'ready',
                usage: r.info?.usage ?? getUiState().usage
              })
              setMessages(resumed)
              setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
            })
            .catch((e: unknown) => {
              sys(`resume failed: ${rpcErrorMessage(e)}`)
              patchUiState({ status: 'forging session…' })
              newSession('started a new session')
            })
        } else {
          patchUiState({ status: 'forging session…' })
          newSession()
        }

        break

      case 'skin.changed':
        if (p) {
          patchUiState({
            theme: fromSkin(p.colors ?? {}, p.branding ?? {}, p.banner_logo ?? '', p.banner_hero ?? '')
          })
        }

        break

      case 'session.info':
        patchUiState(state => ({
          ...state,
          info: p as any,
          usage: p?.usage ? { ...state.usage, ...p.usage } : state.usage
        }))

        break

      case 'thinking.delta':
        if (p && Object.prototype.hasOwnProperty.call(p, 'text')) {
          patchUiState({ status: p.text ? String(p.text) : getUiState().busy ? 'running…' : 'ready' })
        }

        break

      case 'message.start':
        patchUiState({ busy: true })
        endReasoningPhase()
        clearReasoning()
        setActivity([])
        setTurnTrail([])
        activeToolsRef.current = []
        setTools([])
        turnToolsRef.current = []
        persistedToolLabelsRef.current.clear()
        toolTokenAccRef.current = 0
        setToolTokens(0)

        break

      case 'status.update':
        if (p?.text) {
          patchUiState({ status: p.text })

          if (p.kind && p.kind !== 'status') {
            if (lastStatusNoteRef.current !== p.text) {
              lastStatusNoteRef.current = p.text
              pushActivity(
                p.text,
                p.kind === 'error' ? 'error' : p.kind === 'warn' || p.kind === 'approval' ? 'warn' : 'info'
              )
            }

            if (statusTimerRef.current) {
              clearTimeout(statusTimerRef.current)
            }

            statusTimerRef.current = setTimeout(() => {
              statusTimerRef.current = null
              patchUiState({ status: getUiState().busy ? 'running…' : 'ready' })
            }, 4000)
          }
        }

        break

      case 'gateway.stderr':
        if (p?.line) {
          const line = String(p.line).slice(0, 120)
          const tone = /\b(error|traceback|exception|failed|spawn)\b/i.test(line) ? 'error' : 'warn'

          pushActivity(line, tone)
        }

        break

      case 'gateway.start_timeout':
        patchUiState({ status: 'gateway startup timeout' })
        pushActivity(
          `gateway startup timed out${p?.python || p?.cwd ? ` · ${String(p?.python || '')} ${String(p?.cwd || '')}`.trim() : ''} · /logs to inspect`,
          'error'
        )

        break

      case 'gateway.protocol_error':
        patchUiState({ status: 'protocol warning' })

        if (statusTimerRef.current) {
          clearTimeout(statusTimerRef.current)
        }

        statusTimerRef.current = setTimeout(() => {
          statusTimerRef.current = null
          patchUiState({ status: getUiState().busy ? 'running…' : 'ready' })
        }, 4000)

        if (!protocolWarnedRef.current) {
          protocolWarnedRef.current = true
          pushActivity('protocol noise detected · /logs to inspect', 'warn')
        }

        if (p?.preview) {
          pushActivity(`protocol noise: ${String(p.preview).slice(0, 120)}`, 'warn')
        }

        break

      case 'reasoning.delta':
        if (p?.text) {
          reasoningRef.current += p.text
          setReasoningTokens(estimateTokensRough(reasoningRef.current))
          scheduleReasoning()
          pulseReasoningStreaming()
        }

        break
      case 'reasoning.available': {
        const incoming = String(p?.text ?? '').trim()

        if (!incoming) {
          break
        }

        const current = reasoningRef.current.trim()

        // `reasoning.available` is a backend fallback preview that can arrive after
        // streamed reasoning. Preserve the live-visible reasoning/counts if we
        // already saw deltas; only hydrate from this event when streaming gave us
        // nothing.
        if (!current) {
          reasoningRef.current = incoming
          setReasoningTokens(estimateTokensRough(reasoningRef.current))
          scheduleReasoning()
          pulseReasoningStreaming()
        }

        break
      }

      case 'tool.progress':
        if (p?.preview) {
          const index = activeToolsRef.current.findIndex(tool => tool.name === p.name)

          if (index >= 0) {
            const next = [...activeToolsRef.current]

            next[index] = { ...next[index]!, context: p.preview as string }
            activeToolsRef.current = next
            setTools(next)
          }
        }

        break

      case 'tool.generating':
        if (p?.name) {
          pushTrail(`drafting ${p.name}…`)
        }

        break
      case 'tool.start': {
        pruneTransient()
        endReasoningPhase()
        const ctx = (p.context as string) || ''
        const sample = `${String(p.name ?? '')} ${ctx}`.trim()
        toolTokenAccRef.current += sample ? estimateTokensRough(sample) : 0
        setToolTokens(toolTokenAccRef.current)
        activeToolsRef.current = [
          ...activeToolsRef.current,
          { id: p.tool_id, name: p.name, context: ctx, startedAt: Date.now() }
        ]
        setTools(activeToolsRef.current)

        break
      }

      case 'tool.complete': {
        toolCompleteRibbonRef.current = null
        const done = activeToolsRef.current.find(tool => tool.id === p.tool_id)
        const name = done?.name ?? p.name
        const label = toolTrailLabel(name)

        const line = buildToolTrailLine(
          name,
          done?.context || '',
          !!p.error,
          (p.error as string) || (p.summary as string) || ''
        )

        const next = [...turnToolsRef.current.filter(item => !sameToolTrailGroup(label, item)), line]

        activeToolsRef.current = activeToolsRef.current.filter(tool => tool.id !== p.tool_id)
        setTools(activeToolsRef.current)
        toolCompleteRibbonRef.current = { label, line }

        if (!activeToolsRef.current.length) {
          next.push('analyzing tool output…')
        }

        turnToolsRef.current = next.slice(-8)
        setTurnTrail(turnToolsRef.current)

        if (p?.inline_diff) {
          sys(p.inline_diff as string)
        }

        break
      }

      case 'clarify.request':
        patchOverlayState({ clarify: { choices: p.choices, question: p.question, requestId: p.request_id } })
        patchUiState({ status: 'waiting for input…' })

        break

      case 'approval.request':
        patchOverlayState({ approval: { command: p.command, description: p.description } })
        patchUiState({ status: 'approval needed' })

        break

      case 'sudo.request':
        patchOverlayState({ sudo: { requestId: p.request_id } })
        patchUiState({ status: 'sudo password needed' })

        break

      case 'secret.request':
        patchOverlayState({ secret: { envVar: p.env_var, prompt: p.prompt, requestId: p.request_id } })
        patchUiState({ status: 'secret input needed' })

        break

      case 'background.complete':
        patchUiState(state => {
          const next = new Set(state.bgTasks)

          next.delete(p.task_id)

          return { ...state, bgTasks: next }
        })
        sys(`[bg ${p.task_id}] ${p.text}`)

        break

      case 'btw.complete':
        patchUiState(state => {
          const next = new Set(state.bgTasks)

          next.delete('btw:x')

          return { ...state, bgTasks: next }
        })
        sys(`[btw] ${p.text}`)

        break

      case 'message.delta':
        pruneTransient()
        endReasoningPhase()

        if (p?.text && !interruptedRef.current) {
          bufRef.current = p.rendered ?? bufRef.current + p.text
          scheduleStreaming()
        }

        break
      case 'message.complete': {
        const finalText = (p?.rendered ?? p?.text ?? bufRef.current).trimStart()
        const persisted = persistedToolLabelsRef.current
        const savedReasoning = reasoningRef.current.trim()
        const savedReasoningTokens = savedReasoning ? estimateTokensRough(savedReasoning) : 0
        const savedToolTokens = toolTokenAccRef.current

        const savedTools = turnToolsRef.current.filter(
          line => isToolTrailResultLine(line) && ![...persisted].some(item => sameToolTrailGroup(item, line))
        )

        const wasInterrupted = interruptedRef.current

        if (!wasInterrupted) {
          appendMessage({
            role: 'assistant',
            text: finalText,
            thinking: savedReasoning || undefined,
            thinkingTokens: savedReasoning ? savedReasoningTokens : undefined,
            toolTokens: savedTools.length ? savedToolTokens : undefined,
            tools: savedTools.length ? savedTools : undefined
          })

          if (bellOnComplete && stdout?.isTTY) {
            stdout.write('\x07')
          }
        }

        idle()
        clearReasoning()

        turnToolsRef.current = []
        persistedToolLabelsRef.current.clear()
        setActivity([])
        bufRef.current = ''
        patchUiState({ status: 'ready' })

        if (p?.usage) {
          patchUiState({ usage: p.usage })
        }

        if (queueEditRef.current !== null) {
          break
        }

        const next = dequeue()

        if (next) {
          sendQueued(next)
        }

        break
      }

      case 'error':
        idle()
        clearReasoning()
        turnToolsRef.current = []
        persistedToolLabelsRef.current.clear()

        if (statusTimerRef.current) {
          clearTimeout(statusTimerRef.current)
          statusTimerRef.current = null
        }

        pushActivity(String(p?.message || 'unknown error'), 'error')
        sys(`error: ${p?.message}`)
        patchUiState({ status: 'ready' })

        break
    }
  }
}
