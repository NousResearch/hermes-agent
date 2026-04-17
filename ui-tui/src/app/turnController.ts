import { REASONING_PULSE_MS, STREAM_BATCH_MS } from '../config/timing.js'
import type { SessionInterruptResponse, SubagentEventPayload } from '../gatewayTypes.js'
import {
  buildToolTrailLine,
  estimateTokensRough,
  isTransientTrailLine,
  sameToolTrailGroup,
  toolTrailLabel
} from '../lib/text.js'
import type { ActiveTool, ActivityItem, Msg, SubagentProgress } from '../types.js'

import { resetOverlayState } from './overlayStore.js'
import { patchTurnState, resetTurnState } from './turnStore.js'
import { patchUiState } from './uiStore.js'

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
  persistedToolLabels = new Set<string>()
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
    resetOverlayState()
  }

  interruptTurn({ appendMessage, gw, sid, sys }: InterruptDeps) {
    this.interrupted = true
    gw.request<SessionInterruptResponse>('session.interrupt', { session_id: sid }).catch(() => {})

    const partial = this.bufRef.trimStart()

    partial ? appendMessage({ role: 'assistant', text: `${partial}\n\n*[interrupted]*` }) : sys('interrupted')

    this.idle()
    this.clearReasoning()
    this.turnTools = []
    patchTurnState({ activity: [], outcome: '' })
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
    const text = this.bufRef.trimStart()

    if (!text) {
      return
    }

    const tools = this.pendingSegmentTools

    this.streamTimer = clear(this.streamTimer)
    this.segmentMessages = [...this.segmentMessages, { role: 'assistant', text, ...(tools.length && { tools }) }]
    this.bufRef = ''
    this.pendingSegmentTools = []
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
    this.pendingSegmentTools = []
    this.segmentMessages = []
    this.turnTools = []
    this.persistedToolLabels.clear()
  }

  recordMessageComplete(payload: { rendered?: string; reasoning?: string; text?: string }) {
    const finalText = (payload.rendered ?? payload.text ?? this.bufRef).trimStart()
    const savedReasoning = this.reasoningText.trim() || String(payload.reasoning ?? '').trim()
    const savedReasoningTokens = savedReasoning ? estimateTokensRough(savedReasoning) : 0
    const savedToolTokens = this.toolTokenAcc
    const tools = this.pendingSegmentTools
    const finalMessages = [...this.segmentMessages]

    if (finalText) {
      finalMessages.push({
        role: 'assistant',
        text: finalText,
        thinking: savedReasoning || undefined,
        thinkingTokens: savedReasoning ? savedReasoningTokens : undefined,
        toolTokens: savedToolTokens || undefined,
        ...(tools.length && { tools })
      })
    }

    const wasInterrupted = this.interrupted

    this.idle()
    this.clearReasoning()
    this.turnTools = []
    this.persistedToolLabels.clear()
    this.bufRef = ''
    patchTurnState({ activity: [], outcome: '' })

    return { finalMessages, finalText, wasInterrupted }
  }

  recordMessageDelta({ rendered, text }: { rendered?: string; text?: string }) {
    this.pruneTransient()
    this.endReasoningPhase()

    if (!text || this.interrupted) {
      return
    }

    this.bufRef = rendered ?? this.bufRef + text
    this.scheduleStreaming()
  }

  recordReasoningAvailable(text: string) {
    const incoming = text.trim()

    if (!incoming || this.reasoningText.trim()) {
      return
    }

    this.reasoningText = incoming
    this.scheduleReasoning()
    this.pulseReasoningStreaming()
  }

  recordReasoningDelta(text: string) {
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
      patchTurnState({ streaming: this.bufRef.trimStart() })
    }, STREAM_BATCH_MS)
  }

  startMessage() {
    this.endReasoningPhase()
    this.clearReasoning()
    this.activeTools = []
    this.turnTools = []
    this.toolTokenAcc = 0
    this.persistedToolLabels.clear()
    patchUiState({ busy: true })
    patchTurnState({ activity: [], outcome: '', subagents: [], toolTokens: 0, tools: [], turnTrail: [] })
  }

  upsertSubagent(p: SubagentEventPayload, patch: (current: SubagentProgress) => Partial<SubagentProgress>) {
    const id = `sa:${p.task_index}:${p.goal || 'subagent'}`

    patchTurnState(state => {
      const existing = state.subagents.find(item => item.id === id)

      const base: SubagentProgress = existing ?? {
        goal: p.goal,
        id,
        index: p.task_index,
        notes: [],
        status: 'running',
        taskCount: p.task_count ?? 1,
        thinking: [],
        tools: []
      }

      const next: SubagentProgress = {
        ...base,
        goal: p.goal || base.goal,
        taskCount: p.task_count ?? base.taskCount,
        ...patch(base)
      }

      const subagents = existing
        ? state.subagents.map(item => (item.id === id ? next : item))
        : [...state.subagents, next].sort((a, b) => a.index - b.index)

      return { ...state, subagents }
    })
  }
}

export const turnController = new TurnController()

export type { TurnController }
