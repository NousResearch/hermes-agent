import type { CommandsCatalogResponse, GatewayEvent, SessionResumeResponse } from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import {
  buildToolTrailLine,
  estimateTokensRough,
  formatToolCall,
  isToolTrailResultLine,
  sameToolTrailGroup,
  toolTrailLabel
} from '../lib/text.js'
import { fromSkin } from '../theme.js'

import { STREAM_BATCH_MS } from './constants.js'
import { introMsg, toTranscriptMessages } from './helpers.js'
import type { GatewayEventHandlerContext } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { getUiState, patchUiState } from './uiStore.js'

export function createGatewayEventHandler(ctx: GatewayEventHandlerContext): (ev: GatewayEvent) => void {
  const { dequeue, queueEditRef, sendQueued } = ctx.composer
  const { gw, rpc } = ctx.gateway
  const { STARTUP_RESUME_ID, colsRef, newSession, resetSession, setCatalog } = ctx.session
  const { bellOnComplete, stdout, sys } = ctx.system
  const { appendMessage, setHistoryItems } = ctx.transcript

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
    setStreaming,
    setSubagents,
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

  let pendingThinkingStatus = ''
  let thinkingStatusTimer: ReturnType<typeof setTimeout> | null = null
  let toolProgressTimer: ReturnType<typeof setTimeout> | null = null

  const cancelThinkingStatus = () => {
    pendingThinkingStatus = ''

    if (thinkingStatusTimer) {
      clearTimeout(thinkingStatusTimer)
      thinkingStatusTimer = null
    }
  }

  const setStatus = (status: string) => {
    cancelThinkingStatus()
    patchUiState({ status })
  }

  const scheduleThinkingStatus = (status: string) => {
    pendingThinkingStatus = status

    if (thinkingStatusTimer) {
      return
    }

    thinkingStatusTimer = setTimeout(() => {
      thinkingStatusTimer = null
      patchUiState({ status: pendingThinkingStatus || (getUiState().busy ? 'running…' : 'ready') })
    }, STREAM_BATCH_MS)
  }

  const scheduleToolProgress = () => {
    if (toolProgressTimer) {
      return
    }

    toolProgressTimer = setTimeout(() => {
      toolProgressTimer = null
      setTools([...activeToolsRef.current])
    }, STREAM_BATCH_MS)
  }

  const upsertSubagent = (
    taskIndex: number,
    taskCount: number,
    goal: string,
    update: (current: {
      durationSeconds?: number
      goal: string
      id: string
      index: number
      notes: string[]
      status: 'completed' | 'failed' | 'interrupted' | 'running'
      summary?: string
      taskCount: number
      thinking: string[]
      tools: string[]
    }) => {
      durationSeconds?: number
      goal: string
      id: string
      index: number
      notes: string[]
      status: 'completed' | 'failed' | 'interrupted' | 'running'
      summary?: string
      taskCount: number
      thinking: string[]
      tools: string[]
    }
  ) => {
    const id = `sa:${taskIndex}:${goal || 'subagent'}`

    setSubagents(prev => {
      const index = prev.findIndex(item => item.id === id)

      const base =
        index >= 0
          ? prev[index]!
          : {
              id,
              index: taskIndex,
              taskCount,
              goal,
              notes: [],
              status: 'running' as const,
              thinking: [],
              tools: []
            }

      const nextItem = update(base)

      if (index < 0) {
        return [...prev, nextItem].sort((a, b) => a.index - b.index)
      }

      const next = [...prev]
      next[index] = nextItem

      return next
    })
  }

  return (ev: GatewayEvent) => {
    const sid = getUiState().sid

    if (ev.session_id && sid && ev.session_id !== sid && !ev.type.startsWith('gateway.')) {
      return
    }

    switch (ev.type) {
      case 'gateway.ready': {
        const p = ev.payload

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

        rpc<CommandsCatalogResponse>('commands.catalog', {})
          .then(r => {
            if (!r?.pairs) {
              return
            }

            setCatalog({
              canon: (r.canon ?? {}) as Record<string, string>,
              categories: r.categories ?? [],
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
          gw.request<SessionResumeResponse>('session.resume', { cols: colsRef.current, session_id: STARTUP_RESUME_ID })
            .then(raw => {
              const r = asRpcResult<SessionResumeResponse>(raw)

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
      }

      case 'skin.changed': {
        const p = ev.payload

        if (p) {
          patchUiState({
            theme: fromSkin(p.colors ?? {}, p.branding ?? {}, p.banner_logo ?? '', p.banner_hero ?? '')
          })
        }

        break
      }

      case 'session.info': {
        const p = ev.payload

        patchUiState(state => ({
          ...state,
          info: p,
          usage: p.usage ? { ...state.usage, ...p.usage } : state.usage
        }))

        break
      }

      case 'thinking.delta': {
        const p = ev.payload

        if (p && Object.prototype.hasOwnProperty.call(p, 'text')) {
          scheduleThinkingStatus(p.text ? String(p.text) : getUiState().busy ? 'running…' : 'ready')
        }

        break
      }

      case 'message.start':
        patchUiState({ busy: true })
        endReasoningPhase()
        clearReasoning()
        setActivity([])
        setSubagents([])
        setTurnTrail([])
        activeToolsRef.current = []
        setTools([])
        turnToolsRef.current = []
        persistedToolLabelsRef.current.clear()
        toolTokenAccRef.current = 0
        setToolTokens(0)

        break
      case 'status.update': {
        const p = ev.payload

        if (p?.text) {
          setStatus(p.text)

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
      }

      case 'gateway.stderr': {
        const p = ev.payload

        if (p?.line) {
          const line = String(p.line).slice(0, 120)
          const tone = /\b(error|traceback|exception|failed|spawn)\b/i.test(line) ? 'error' : 'warn'

          pushActivity(line, tone)
        }

        break
      }

      case 'gateway.start_timeout': {
        const p = ev.payload

        setStatus('gateway startup timeout')
        pushActivity(
          `gateway startup timed out${p?.python || p?.cwd ? ` · ${String(p?.python || '')} ${String(p?.cwd || '')}`.trim() : ''} · /logs to inspect`,
          'error'
        )

        break
      }

      case 'gateway.protocol_error': {
        const p = ev.payload

        setStatus('protocol warning')

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
      }

      case 'reasoning.delta': {
        const p = ev.payload

        if (p?.text) {
          reasoningRef.current += p.text
          scheduleReasoning()
          pulseReasoningStreaming()
        }

        break
      }

      case 'reasoning.available': {
        const p = ev.payload
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
          scheduleReasoning()
          pulseReasoningStreaming()
        }

        break
      }

      case 'tool.progress': {
        const p = ev.payload

        if (p?.preview) {
          const index = activeToolsRef.current.findIndex(tool => tool.name === p.name)

          if (index >= 0) {
            const next = [...activeToolsRef.current]

            next[index] = { ...next[index]!, context: p.preview as string }
            activeToolsRef.current = next
            scheduleToolProgress()
          }
        }

        break
      }

      case 'tool.generating': {
        const p = ev.payload

        if (p?.name) {
          pushTrail(`drafting ${p.name}…`)
        }

        break
      }

      case 'tool.start': {
        const p = ev.payload
        pruneTransient()
        endReasoningPhase()
        const name = p.name ?? 'tool'
        const ctx = p.context ?? ''
        const sample = `${String(p.name ?? '')} ${ctx}`.trim()
        toolTokenAccRef.current += sample ? estimateTokensRough(sample) : 0
        setToolTokens(toolTokenAccRef.current)
        activeToolsRef.current = [
          ...activeToolsRef.current,
          { id: p.tool_id, name, context: ctx, startedAt: Date.now() }
        ]
        setTools(activeToolsRef.current)

        break
      }

      case 'tool.complete': {
        const p = ev.payload
        toolCompleteRibbonRef.current = null
        const done = activeToolsRef.current.find(tool => tool.id === p.tool_id)
        const name = done?.name ?? p.name ?? 'tool'
        const label = toolTrailLabel(name)

        const line = buildToolTrailLine(name, done?.context || '', !!p.error, p.error || p.summary || '')

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
          sys(p.inline_diff)
        }

        break
      }

      case 'clarify.request': {
        const p = ev.payload
        patchOverlayState({ clarify: { choices: p.choices, question: p.question, requestId: p.request_id } })
        setStatus('waiting for input…')

        break
      }

      case 'approval.request': {
        const p = ev.payload
        patchOverlayState({ approval: { command: p.command, description: p.description } })
        setStatus('approval needed')

        break
      }

      case 'sudo.request': {
        const p = ev.payload
        patchOverlayState({ sudo: { requestId: p.request_id } })
        setStatus('sudo password needed')

        break
      }

      case 'secret.request': {
        const p = ev.payload
        patchOverlayState({ secret: { envVar: p.env_var, prompt: p.prompt, requestId: p.request_id } })
        setStatus('secret input needed')

        break
      }

      case 'background.complete': {
        const p = ev.payload
        patchUiState(state => {
          const next = new Set(state.bgTasks)

          next.delete(p.task_id)

          return { ...state, bgTasks: next }
        })
        sys(`[bg ${p.task_id}] ${p.text}`)

        break
      }

      case 'btw.complete': {
        const p = ev.payload
        patchUiState(state => {
          const next = new Set(state.bgTasks)

          next.delete('btw:x')

          return { ...state, bgTasks: next }
        })
        sys(`[btw] ${p.text}`)

        break
      }

      case 'subagent.start': {
        const p = ev.payload

        upsertSubagent(p.task_index, p.task_count ?? 1, p.goal, current => ({
          ...current,
          goal: p.goal || current.goal,
          status: 'running',
          taskCount: p.task_count ?? current.taskCount
        }))

        break
      }

      case 'subagent.thinking': {
        const p = ev.payload
        const text = String(p.text ?? '').trim()

        if (!text) {
          break
        }

        upsertSubagent(p.task_index, p.task_count ?? 1, p.goal, current => ({
          ...current,
          goal: p.goal || current.goal,
          status: current.status === 'completed' ? current.status : 'running',
          taskCount: p.task_count ?? current.taskCount,
          thinking: current.thinking.at(-1) === text ? current.thinking : [...current.thinking, text].slice(-6)
        }))

        break
      }

      case 'subagent.tool': {
        const p = ev.payload
        const line = formatToolCall(p.tool_name ?? 'delegate_task', p.tool_preview ?? p.text ?? '')

        upsertSubagent(p.task_index, p.task_count ?? 1, p.goal, current => ({
          ...current,
          goal: p.goal || current.goal,
          status: current.status === 'completed' ? current.status : 'running',
          taskCount: p.task_count ?? current.taskCount,
          tools: current.tools.at(-1) === line ? current.tools : [...current.tools, line].slice(-8)
        }))

        break
      }

      case 'subagent.progress': {
        const p = ev.payload
        const text = String(p.text ?? '').trim()

        if (!text) {
          break
        }

        upsertSubagent(p.task_index, p.task_count ?? 1, p.goal, current => ({
          ...current,
          goal: p.goal || current.goal,
          status: current.status === 'completed' ? current.status : 'running',
          taskCount: p.task_count ?? current.taskCount,
          notes: current.notes.at(-1) === text ? current.notes : [...current.notes, text].slice(-6)
        }))

        break
      }

      case 'subagent.complete': {
        const p = ev.payload
        const status = p.status ?? 'completed'

        upsertSubagent(p.task_index, p.task_count ?? 1, p.goal, current => ({
          ...current,
          durationSeconds: p.duration_seconds ?? current.durationSeconds,
          goal: p.goal || current.goal,
          status,
          summary: p.summary || p.text || current.summary,
          taskCount: p.task_count ?? current.taskCount
        }))

        break
      }

      case 'message.delta': {
        const p = ev.payload
        pruneTransient()
        endReasoningPhase()

        if (p?.text && !interruptedRef.current) {
          bufRef.current = p.rendered ?? bufRef.current + p.text
          scheduleStreaming()
        }

        break
      }

      case 'message.complete': {
        const p = ev.payload
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
        setStatus('ready')

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

      case 'error': {
        const p = ev.payload
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
        setStatus('ready')

        break
      }
    }
  }
}
