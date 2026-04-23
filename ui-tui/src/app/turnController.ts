import { REASONING_PULSE_MS, STREAM_BATCH_MS } from '../config/timing.js'
import type { SessionInterruptResponse, SubagentEventPayload } from '../gatewayTypes.js'
import { hasReasoningTag, splitReasoning } from '../lib/reasoning.js'
import {
  buildToolTrailLine,
  estimateTokensRough,
  isTransientTrailLine,
  sameToolTrailGroup,
  toolTrailLabel
} from '../lib/text.js'
import type { ActiveTool, ActivityItem, Msg, SubagentProgress } from '../types.js'

import { resetFlowOverlays } from './overlayStore.js'
import { pushSnapshot } from './spawnHistoryStore.js'
import { getTurnState, patchTurnState, resetTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const INTERRUPT_COOLDOWN_MS = 1500
const ACTIVITY_LIMIT = 8
const TRAIL_LIMIT = 8

export interface InterruptDeps {
  appendMessage: (msg: Msg) => void
  gw: { request: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T> }
  sid: string
  sys: (text: string) => void
}

type Timer = null | ReturnType<typeof setTimeout>

const clear = (t: Timer): null => {
  if (t) {
    clearTimeout(t)
  }

  return null
}

class TurnController {
  bufRef = ''
  interrupted = false
  lastStatusNote = ''
  pendingInlineDiffs: string[] = []
  persistedToolLabels = new Set<string>()
  persistSpawnTree?: (subagents: SubagentProgress[], sessionId: null | string) => Promise<void>
  protocolWarned = false
  reasoningText = ''
  segmentMessages: Msg[] = []
  pendingSegmentTools: string[] = []
  statusTimer: Timer = null
  toolTokenAcc = 0
  turnTools: string[] = []

  private activeTools: ActiveTool[] = []
  private activityId = 0
  private reasoningStreamingTimer: Timer = null
  private reasoningTimer: Timer = null
  private streamTimer: Timer = null
  private toolProgressTimer: Timer = null

  clearReasoning() {
    this.reasoningTimer = clear(this.reasoningTimer)
    this.reasoningText = ''
    this.toolTokenAcc = 0
    patchTurnState({ reasoning: '', reasoningTokens: 0, toolTokens: 0 })
  }

  clearStatusTimer() {
    this.statusTimer = clear(this.statusTimer)
  }

  endReasoningPhase() {
    this.reasoningStreamingTimer = clear(this.reasoningStreamingTimer)
    patchTurnState({ reasoningActive: false, reasoningStreaming: false })
  }

  idle() {
    this.endReasoningPhase()
    this.activeTools = []
    this.streamTimer = clear(this.streamTimer)
    this.bufRef = ''
    this.pendingInlineDiffs = []
    this.pendingSegmentTools = []
    this.segmentMessages = []

    patchTurnState({
      streamPendingTools: [],
      streamSegments: [],
      streaming: '',
      subagents: [],
      tools: [],
      turnTrail: []
    })
    patchUiState({ busy: false })
    resetFlowOverlays()
  }

  interruptTurn({ appendMessage, gw, sid, sys }: InterruptDeps) {
    this.interrupted = true
    gw.request<SessionInterruptResponse>('session.interrupt', { session_id: sid }).catch(() => {})

    const segments = this.segmentMessages
    const partial = this.bufRef.trimStart()
    const tools = this.pendingSegmentTools

    // Drain streaming/segment state off the nanostore before writing the
    // preserved snapshot to the transcript — otherwise each flushed segment
    // appears in both `turn.streamSegments` and the transcript for one frame.
    this.idle()
    this.clearReasoning()
    this.turnTools = []
    patchTurnState({ activity: [], outcome: '' })

    for (const msg of segments) {
      appendMessage(msg)
    }

    // Always surface an interruption indicator — if there's an in-flight
    // `partial` or pending tools, fold them into a single assistant message;
    // otherwise emit a sys note so the transcript always records that the
    // turn was cancelled, even when only prior `segments` were preserved.
    if (partial || tools.length) {
      appendMessage({
        role: 'assistant',
        text: partial ? `${partial}\n\n*[interrupted]*` : '*[interrupted]*',
        ...(tools.length && { tools })
      })
    } else {
      sys('interrupted')
    }

    patchUiState({ status: 'interrupted' })
    this.clearStatusTimer()

    this.statusTimer = setTimeout(() => {
      this.statusTimer = null
      patchUiState({ status: 'ready' })
    }, INTERRUPT_COOLDOWN_MS)
  }

  pruneTransient() {
    this.turnTools = this.turnTools.filter(line => !isTransientTrailLine(line))
    patchTurnState(state => {
      const next = state.turnTrail.filter(line => !isTransientTrailLine(line))

      return next.length === state.turnTrail.length ? state : { ...state, turnTrail: next }
    })
  }

  flushStreamingSegment() {
    const raw = this.bufRef.trimStart()

    if (!raw) {
      return
    }

    const split = hasReasoningTag(raw) ? splitReasoning(raw) : { reasoning: '', text: raw }

    if (split.reasoning && !this.reasoningText.trim()) {
      this.reasoningText = split.reasoning
      patchTurnState({ reasoning: this.reasoningText, reasoningTokens: estimateTokensRough(this.reasoningText) })
    }

    const text = split.text

    this.streamTimer = clear(this.streamTimer)

    if (text) {
      const tools = this.pendingSegmentTools

      this.segmentMessages = [...this.segmentMessages, { role: 'assistant', text, ...(tools.length && { tools }) }]
      this.pendingSegmentTools = []
    }

    this.bufRef = ''
    patchTurnState({ streamPendingTools: [], streamSegments: this.segmentMessages, streaming: '' })
  }

  pulseReasoningStreaming() {
    this.reasoningStreamingTimer = clear(this.reasoningStreamingTimer)
    patchTurnState({ reasoningActive: true, reasoningStreaming: true })

    this.reasoningStreamingTimer = setTimeout(() => {
      this.reasoningStreamingTimer = null
      patchTurnState({ reasoningStreaming: false })
    }, REASONING_PULSE_MS)
  }

  queueInlineDiff(diffText: string) {
    // Strip CLI chrome the gateway emits before the unified diff (e.g. a
    // leading "┊ review diff" header written by `_emit_inline_diff` for the
    // terminal printer). That header only makes sense as stdout dressing,
    // not inside a markdown ```diff block.
    const text = diffText.replace(/^\s*┊[^\n]*\n?/, '').trim()

    if (!text || this.pendingInlineDiffs.includes(text)) {
      return
    }

    this.pendingInlineDiffs = [...this.pendingInlineDiffs, text]
  }

  pushActivity(text: string, tone: ActivityItem['tone'] = 'info', replaceLabel?: string) {
    patchTurnState(state => {
      const base = replaceLabel
        ? state.activity.filter(item => !sameToolTrailGroup(replaceLabel, item.text))
        : state.activity

      const tail = base.at(-1)

      if (tail?.text === text && tail.tone === tone) {
        return state
      }

      return { ...state, activity: [...base, { id: ++this.activityId, text, tone }].slice(-ACTIVITY_LIMIT) }
    })
  }

  pushTrail(line: string) {
    patchTurnState(state => {
      if (state.turnTrail.at(-1) === line) {
        return state
      }

      const next = [...state.turnTrail.filter(item => !isTransientTrailLine(item)), line].slice(-TRAIL_LIMIT)

      this.turnTools = next

      return { ...state, turnTrail: next }
    })
  }

  recordError() {
    this.idle()
    this.clearReasoning()
    this.clearStatusTimer()
    this.pendingInlineDiffs = []
    this.pendingSegmentTools = []
    this.segmentMessages = []
    this.turnTools = []
    this.persistedToolLabels.clear()
  }

  recordMessageComplete(payload: { rendered?: string; reasoning?: string; text?: string }) {
    const rawText = (payload.rendered ?? payload.text ?? this.bufRef).trimStart()
    const split = splitReasoning(rawText)
    const finalText = split.text
    // Skip appending if the assistant already narrated the diff inside a
    // markdown fence of its own — otherwise we render two stacked diff
    // blocks for the same edit.
    const assistantAlreadyHasDiff = /```(?:diff|patch)\b/i.test(finalText)

    const remainingInlineDiffs = assistantAlreadyHasDiff
      ? []
      : this.pendingInlineDiffs.filter(diff => !finalText.includes(diff))

    const inlineDiffBlock = remainingInlineDiffs.length
      ? `\`\`\`diff\n${remainingInlineDiffs.join('\n\n')}\n\`\`\``
      : ''

    const mergedText = [finalText, inlineDiffBlock].filter(Boolean).join('\n\n')
    const existingReasoning = this.reasoningText.trim() || String(payload.reasoning ?? '').trim()
    const savedReasoning = [existingReasoning, existingReasoning ? '' : split.reasoning].filter(Boolean).join('\n\n')
    const savedReasoningTokens = savedReasoning ? estimateTokensRough(savedReasoning) : 0
    const savedToolTokens = this.toolTokenAcc
    const tools = this.pendingSegmentTools
    const finalMessages = [...this.segmentMessages]

    if (mergedText) {
      finalMessages.push({
        role: 'assistant',
        text: mergedText,
        thinking: savedReasoning || undefined,
        thinkingTokens: savedReasoning ? savedReasoningTokens : undefined,
        toolTokens: savedToolTokens || undefined,
        ...(tools.length && { tools })
      })
    }

    const wasInterrupted = this.interrupted

    // Archive the turn's spawn tree to history BEFORE idle() drops subagents
    // from turnState.  Lets /replay and the overlay's history nav pull up
    // finished fan-outs without a round-trip to disk.
    const finishedSubagents = getTurnState().subagents
    const sessionId = getUiState().sid

    if (finishedSubagents.length > 0) {
      pushSnapshot(finishedSubagents, { sessionId, startedAt: null })
      // Fire-and-forget disk persistence so /replay survives process restarts.
      // The same snapshot lives in memory via spawnHistoryStore for immediate
      // recall — disk is the long-term archive.
      void this.persistSpawnTree?.(finishedSubagents, sessionId)
    }

    this.idle()
    this.clearReasoning()
    this.turnTools = []
    this.persistedToolLabels.clear()
    this.bufRef = ''
    patchTurnState({ activity: [], outcome: '' })

    return { finalMessages, finalText: mergedText, wasInterrupted }
  }

  recordMessageDelta({ rendered, text }: { rendered?: string; text?: string }) {
    this.pruneTransient()
    this.endReasoningPhase()

    if (!text || this.interrupted) {
      return
    }

    this.bufRef = rendered ?? this.bufRef + text

    if (getUiState().streaming) {
      this.scheduleStreaming()
    }
  }

  recordReasoningAvailable(text: string) {
    if (!getUiState().showReasoning) {
      return
    }

    const incoming = text.trim()

    if (!incoming || this.reasoningText.trim()) {
      return
    }

    this.reasoningText = incoming
    this.scheduleReasoning()
    this.pulseReasoningStreaming()
  }

  recordReasoningDelta(text: string) {
    if (!getUiState().showReasoning) {
      return
    }

    this.reasoningText += text
    this.scheduleReasoning()
    this.pulseReasoningStreaming()
  }

  recordToolComplete(toolId: string, fallbackName?: string, error?: string, summary?: string) {
    const done = this.activeTools.find(tool => tool.id === toolId)
    const name = done?.name ?? fallbackName ?? 'tool'
    const label = toolTrailLabel(name)
    const line = buildToolTrailLine(name, done?.context || '', Boolean(error), error || summary || '')

    this.activeTools = this.activeTools.filter(tool => tool.id !== toolId)
    this.pendingSegmentTools = [...this.pendingSegmentTools, line]

    const next = this.turnTools.filter(item => !sameToolTrailGroup(label, item))

    if (!this.activeTools.length) {
      next.push('analyzing tool output…')
    }

    this.turnTools = next.slice(-TRAIL_LIMIT)
    patchTurnState({
      streamPendingTools: this.pendingSegmentTools,
      tools: this.activeTools,
      turnTrail: this.turnTools
    })
  }

  recordToolProgress(toolName: string, preview: string) {
    const index = this.activeTools.findIndex(tool => tool.name === toolName)

    if (index < 0) {
      return
    }

    this.activeTools = this.activeTools.map((tool, i) => (i === index ? { ...tool, context: preview } : tool))

    if (this.toolProgressTimer) {
      return
    }

    this.toolProgressTimer = setTimeout(() => {
      this.toolProgressTimer = null
      patchTurnState({ tools: [...this.activeTools] })
    }, STREAM_BATCH_MS)
  }

  recordToolStart(toolId: string, name: string, context: string) {
    this.flushStreamingSegment()
    this.pruneTransient()
    this.endReasoningPhase()

    const sample = `${name} ${context}`.trim()

    this.toolTokenAcc += sample ? estimateTokensRough(sample) : 0
    this.activeTools = [...this.activeTools, { context, id: toolId, name, startedAt: Date.now() }]

    patchTurnState({ toolTokens: this.toolTokenAcc, tools: this.activeTools })
  }

  reset() {
    this.clearReasoning()
    this.clearStatusTimer()
    this.idle()
    this.bufRef = ''
    this.interrupted = false
    this.lastStatusNote = ''
    this.pendingInlineDiffs = []
    this.pendingSegmentTools = []
    this.protocolWarned = false
    this.segmentMessages = []
    this.turnTools = []
    this.toolTokenAcc = 0
    this.persistedToolLabels.clear()
    patchTurnState({ activity: [], outcome: '' })
  }

  fullReset() {
    this.reset()
    resetTurnState()
  }

  scheduleReasoning() {
    if (this.reasoningTimer) {
      return
    }

    this.reasoningTimer = setTimeout(() => {
      this.reasoningTimer = null
      patchTurnState({
        reasoning: this.reasoningText,
        reasoningTokens: estimateTokensRough(this.reasoningText)
      })
    }, STREAM_BATCH_MS)
  }

  scheduleStreaming() {
    if (this.streamTimer) {
      return
    }

    this.streamTimer = setTimeout(() => {
      this.streamTimer = null
      const raw = this.bufRef.trimStart()
      const visible = hasReasoningTag(raw) ? splitReasoning(raw).text : raw
      patchTurnState({ streaming: visible })
    }, STREAM_BATCH_MS)
  }

  startMessage() {
    this.endReasoningPhase()
    this.clearReasoning()
    this.activeTools = []
    this.pendingInlineDiffs = []
    this.turnTools = []
    this.toolTokenAcc = 0
    this.persistedToolLabels.clear()
    patchUiState({ busy: true })
    patchTurnState({ activity: [], outcome: '', subagents: [], toolTokens: 0, tools: [], turnTrail: [] })
  }

  upsertSubagent(
    p: SubagentEventPayload,
    patch: (current: SubagentProgress) => Partial<SubagentProgress>,
    opts: { createIfMissing?: boolean } = { createIfMissing: true }
  ) {
    // Stable id: prefer the server-issued subagent_id (survives nested
    // grandchildren + cross-tree joins).  Fall back to the composite key
    // for older gateways that omit the field — those produce a flat list.
    const id = p.subagent_id || `sa:${p.task_index}:${p.goal || 'subagent'}`

    patchTurnState(state => {
      const existing = state.subagents.find(item => item.id === id)

      // Late events (subagent.complete/tool/progress arriving after message.complete
      // has already fired idle()) would otherwise resurrect a finished
      // subagent into turn.subagents and block the "finished" title on the
      // /agents overlay.  When `createIfMissing` is false we drop silently.
      if (!existing && !opts.createIfMissing) {
        return state
      }

      const base: SubagentProgress = existing ?? {
        depth: p.depth ?? 0,
        goal: p.goal,
        id,
        index: p.task_index,
        model: p.model,
        notes: [],
        parentId: p.parent_id ?? null,
        startedAt: Date.now(),
        status: 'running',
        taskCount: p.task_count ?? 1,
        thinking: [],
        toolCount: p.tool_count ?? 0,
        tools: [],
        toolsets: p.toolsets
      }

      // Map snake_case payload keys onto camelCase state.  Only overwrite
      // when the event actually carries the field; `??` preserves prior
      // values across streaming events that emit partial payloads.
      const outputTail = p.output_tail
        ? p.output_tail.map(e => ({
            isError: Boolean(e.is_error),
            preview: String(e.preview ?? ''),
            tool: String(e.tool ?? 'tool')
          }))
        : base.outputTail

      const next: SubagentProgress = {
        ...base,
        apiCalls: p.api_calls ?? base.apiCalls,
        costUsd: p.cost_usd ?? base.costUsd,
        depth: p.depth ?? base.depth,
        filesRead: p.files_read ?? base.filesRead,
        filesWritten: p.files_written ?? base.filesWritten,
        goal: p.goal || base.goal,
        inputTokens: p.input_tokens ?? base.inputTokens,
        iteration: p.iteration ?? base.iteration,
        model: p.model ?? base.model,
        outputTail,
        outputTokens: p.output_tokens ?? base.outputTokens,
        parentId: p.parent_id ?? base.parentId,
        reasoningTokens: p.reasoning_tokens ?? base.reasoningTokens,
        taskCount: p.task_count ?? base.taskCount,
        toolCount: p.tool_count ?? base.toolCount,
        toolsets: p.toolsets ?? base.toolsets,
        ...patch(base)
      }

      // Stable order: by spawn (depth, parent, index) rather than insert time.
      // Without it, grandchildren can shuffle relative to siblings when
      // events arrive out of order under high concurrency.
      const subagents = existing
        ? state.subagents.map(item => (item.id === id ? next : item))
        : [...state.subagents, next].sort((a, b) => a.depth - b.depth || a.index - b.index)

      return { ...state, subagents }
    })
  }
}

export const turnController = new TurnController()

export type { TurnController }
