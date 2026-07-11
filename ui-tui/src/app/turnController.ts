import {
  REASONING_PULSE_MS,
  STREAM_BATCH_MS,
  STREAM_IDLE_BATCH_MS,
  STREAM_SCROLL_BATCH_MS,
  STREAM_TYPING_BATCH_MS
} from '../config/timing.js'
import type { SessionInterruptResponse, SubagentEventPayload } from '../gatewayTypes.js'
import { appendToolShelfMessage, isToolShelfMessage } from '../lib/liveProgress.js'
import { hasReasoningTag, splitReasoning } from '../lib/reasoning.js'
import {
  boundedLiveRenderText,
  buildToolTrailLine,
  buildVerboseToolTrailLine,
  estimateTokensRough,
  isTransientTrailLine,
  sameToolTrailGroup,
  toolTrailLabel
} from '../lib/text.js'
import type { ActiveTool, ActivityItem, Msg, SubagentProgress, TodoItem } from '../types.js'

import type { Notice } from './interfaces.js'
import { resetFlowOverlays } from './overlayStore.js'
import { pushSnapshot } from './spawnHistoryStore.js'
import { archiveDoneTodos, getTurnState, patchTurnState, resetTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const INTERRUPT_COOLDOWN_MS = 1500
const ACTIVITY_LIMIT = 8
const TRAIL_LIMIT = 8

// Extracts the raw patch from a diff-only segment produced by
// pushInlineDiffSegment. Used at message.complete to dedupe against final
// assistant text that narrates the same patch. Returns null for anything
// else so real assistant narration never gets touched.
const diffSegmentBody = (msg: Msg): null | string => {
  if (msg.kind !== 'diff') {
    return null
  }

  const m = msg.text.match(/^```diff\n([\s\S]*?)\n```$/)

  return m ? m[1]! : null
}

const hasDetails = (msg: Msg): boolean => Boolean(msg.thinking || msg.tools?.length || msg.toolTokens)

const isTodoStatus = (status: unknown): status is TodoItem['status'] =>
  status === 'pending' || status === 'in_progress' || status === 'completed' || status === 'cancelled'

const parseTodos = (value: unknown): null | TodoItem[] => {
  if (!Array.isArray(value)) {
    return null
  }

  return value
    .map(item => {
      if (!item || typeof item !== 'object') {
        return null
      }

      const row = item as Record<string, unknown>
      const status = row.status

      if (!isTodoStatus(status)) {
        return null
      }

      return {
        content: String(row.content ?? '').trim(),
        id: String(row.id ?? '').trim(),
        status
      }
    })
    .filter((item): item is TodoItem => Boolean(item?.id && item.content))
}

const textSegments = (segments: Msg[]) =>
  segments.filter(msg => msg.role === 'assistant' && msg.kind !== 'diff').map(msg => msg.text)

const finalTail = (finalText: string, segments: Msg[]) => {
  let tail = finalText

  for (const text of textSegments(segments)) {
    const trimmed = text.trim()

    if (trimmed && tail.startsWith(trimmed)) {
      tail = tail.slice(trimmed.length).trimStart()
    }
  }

  return tail
}

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
  persistSpawnTree?: (subagents: SubagentProgress[], sessionId: null | string) => Promise<void>
  protocolWarned = false
  reasoningText = ''
  segmentMessages: Msg[] = []
  pendingSegmentTools: string[] = []
  statusTimer: Timer = null
  toolTokenAcc = 0
  turnTools: string[] = []

  private activeTools: ActiveTool[] = []
  private activeReasoningText = ''
  private reasoningSegmentIndex: null | number = null
  private activityId = 0
  private reasoningStreamingTimer: Timer = null
  private reasoningTimer: Timer = null
  private streamTimer: Timer = null
  private streamDelay = STREAM_IDLE_BATCH_MS
  private toolProgressTimer: Timer = null

  // ── Credits notice machinery (Strategy B) ───────────────────────────
  //
  // A notice arriving mid-turn must NOT show (FaceTicker wins while busy);
  // it is held here, latest-wins, and applied at turn end via onTurnEnd().
  // The TTL clock starts only when the notice becomes VISIBLE (on apply),
  // never on arrival, so an 8s "restored" notice shows for its full life.
  // `noticeTimer` is DEDICATED — it is never the shared `statusTimer`.
  private pendingNotice: Notice | null = null
  private noticeTimer: Timer = null
  private noticeIdSeq = 0

  boostStreamingForTyping() {
    this.streamDelay = STREAM_TYPING_BATCH_MS
  }

  boostStreamingForScroll() {
    this.streamDelay = Math.max(this.streamDelay, STREAM_SCROLL_BATCH_MS)
  }

  relaxStreaming() {
    this.streamDelay = STREAM_IDLE_BATCH_MS
  }

  clearReasoning() {
    this.reasoningTimer = clear(this.reasoningTimer)
    this.activeReasoningText = ''
    this.reasoningSegmentIndex = null
    this.reasoningText = ''
    this.toolTokenAcc = 0
    patchTurnState({ reasoning: '', reasoningTokens: 0, toolTokens: 0 })
  }

  clearStatusTimer() {
    this.statusTimer = clear(this.statusTimer)
  }

  // ── Notice: arrival ──────────────────────────────────────────────────
  //
  // A `notification.show` arrived. If a turn is in flight (`busy`), the
  // FaceTicker owns the verb slot, so we hold the notice (latest-wins) and
  // let onTurnEnd() apply it when the turn finishes. If idle, apply it now.
  // The Python side emits STABLE ids per notice kind (e.g. `credits.warn90`,
  // `credits.restored`), NOT unique-per-emission ids.  The id-guard in
  // applyNotice() is a defensive backup; the primary latest-wins mechanism is
  // that applyNotice/clearNotice always cancel the prior timer first.
  showNotice(notice: Notice) {
    const stamped: Notice = { ...notice, id: notice.id || `n${++this.noticeIdSeq}` }

    if (getUiState().busy) {
      this.pendingNotice = stamped

      return
    }

    this.applyNotice(stamped)
  }

  // ── Notice: clear by key (R3-H3 / HIGH-3) ────────────────────────────
  //
  // `notification.clear` only clears the visible notice when its key
  // matches — a stale/late clear must not wipe a NEWER notice. Always drop
  // a matching pending notice so it can't resurface at the next turn end.
  clearNotice(key?: string) {
    if (this.pendingNotice && this.pendingNotice.key === key) {
      this.pendingNotice = null
    }

    if (getUiState().notice?.key === key) {
      this.clearNoticeTimer()
      patchUiState({ notice: null })
    }
  }

  // Apply a notice to the visible UI state and (re)arm its TTL clock.
  // Latest-wins: clear any prior TTL timer FIRST so an older notice's
  // expiry can't wipe this one. A 'ttl' notice with `ttl_ms` self-expires;
  // 'sticky' (default) persists until an explicit clear.
  private applyNotice(notice: Notice) {
    this.clearNoticeTimer()
    patchUiState({ notice })

    if (notice.kind === 'ttl' && typeof notice.ttl_ms === 'number' && notice.ttl_ms > 0) {
      const id = notice.id

      this.noticeTimer = setTimeout(() => {
        this.noticeTimer = null

        // Defensive backup: the prior timer was already cancelled by
        // clearNoticeTimer() when a newer notice was applied, so in
        // practice this guard only fires for the notice that armed it.
        if (getUiState().notice?.id === id) {
          patchUiState({ notice: null })
        }
      }, notice.ttl_ms)
    }
  }

  private clearNoticeTimer() {
    this.noticeTimer = clear(this.noticeTimer)
  }

  // ── Notice: turn-end flush (R3-C1 / R3-H4) ───────────────────────────
  //
  // Invoked ONLY by the three real turn-end sites (recordMessageComplete,
  // interruptTurn, recordError) — NEVER by idle()/reset(), which would leak
  // session A's notice into session B. Applies a pending NEW notice so it
  // appears now that FaceTicker has yielded; the TTL clock starts here, when
  // the notice first becomes visible. With no pending notice this is a
  // no-op, so a standing sticky notice REappears untouched after the turn.
  private flushPendingNotice() {
    if (!this.pendingNotice) {
      return
    }

    const notice = this.pendingNotice
    this.pendingNotice = null
    this.applyNotice(notice)
  }

  // Drop all notice state — pending + timer + visible (R3-H5). Called by
  // reset()/fullReset() so a session A notice can't bleed into session B.
  private clearNoticeState() {
    this.pendingNotice = null
    this.clearNoticeTimer()

    if (getUiState().notice) {
      patchUiState({ notice: null })
    }
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
    resetFlowOverlays()
  }

  // `keepBusy` holds the session busy after interrupting so a queued message
  // drains on the gateway's real settle edge (message.complete, suppressed
  // while `interrupted`) instead of racing the still-unwinding turn — the race
  // duplicated the user bubble, leaked a "queued: …" note, and surfaced the
  // cancelled turn's "[interrupted]" reply.
  interruptTurn({ appendMessage, gw, sid, sys }: InterruptDeps, opts: { keepBusy?: boolean } = {}) {
    this.interrupted = true
    gw.request<SessionInterruptResponse>('session.interrupt', { session_id: sid }).catch(() => {})

    this.closeReasoningSegment()

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

    this.clearStatusTimer()

    if (opts.keepBusy) {
      // `idle()` already cleared busy; re-assert it so the drain waits for settle.
      patchUiState({ busy: true, status: 'interrupting…' })

      return
    }

    patchUiState({ status: 'interrupted' })

    this.statusTimer = setTimeout(() => {
      this.statusTimer = null
      patchUiState({ status: 'ready' })
    }, INTERRUPT_COOLDOWN_MS)

    // Real turn end: surface any notice held back while busy.
    this.flushPendingNotice()
  }

  pruneTransient() {
    this.turnTools = this.turnTools.filter(line => !isTransientTrailLine(line))
    patchTurnState(state => {
      const next = state.turnTrail.filter(line => !isTransientTrailLine(line))

      return next.length === state.turnTrail.length ? state : { ...state, turnTrail: next }
    })
  }

  private syncReasoningSegment() {
    const thinking = this.activeReasoningText.trim()

    if (!thinking) {
      return
    }

    const msg: Msg = {
      kind: 'trail',
      role: 'system',
      text: '',
      thinking,
      thinkingTokens: estimateTokensRough(thinking),
      toolTokens: this.toolTokenAcc || undefined
    }

    if (this.reasoningSegmentIndex === null) {
      this.reasoningSegmentIndex = this.segmentMessages.length
      this.segmentMessages = [...this.segmentMessages, msg]
    } else {
      this.segmentMessages = this.segmentMessages.map((item, i) => (i === this.reasoningSegmentIndex ? msg : item))
    }

    patchTurnState({ streamSegments: this.segmentMessages })
  }

  private closeReasoningSegment() {
    this.syncReasoningSegment()
    this.activeReasoningText = ''
    this.reasoningSegmentIndex = null
  }

  private pushSegment(msg: Msg) {
    this.segmentMessages = appendToolShelfMessage(this.segmentMessages, msg)
  }

  flushStreamingSegment() {
    const raw = this.bufRef.trimStart()

    const split = raw
      ? hasReasoningTag(raw)
        ? splitReasoning(raw)
        : { reasoning: '', text: raw }
      : { reasoning: '', text: '' }

    if (split.reasoning && !this.reasoningText.trim()) {
      this.reasoningText = split.reasoning
      this.activeReasoningText = split.reasoning
      patchTurnState({ reasoning: this.reasoningText, reasoningTokens: estimateTokensRough(this.reasoningText) })
      this.syncReasoningSegment()
    }

    const msg: Msg = {
      role: split.text ? 'assistant' : 'system',
      text: split.text,
      ...(!split.text && { kind: 'trail' as const }),
      ...(this.pendingSegmentTools.length && { tools: this.pendingSegmentTools })
    }

    this.streamTimer = clear(this.streamTimer)

    if (split.text || hasDetails(msg)) {
      this.pushSegment(msg)
    }

    this.pendingSegmentTools = []
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

  recordTodos(value: unknown) {
    if (this.interrupted) {
      return
    }

    const todos = parseTodos(value)

    if (todos !== null) {
      patchTurnState({ todos })
    }
  }

  private flushPendingToolsIntoLastSegment() {
    if (!this.pendingSegmentTools.length) {
      return false
    }

    const next = appendToolShelfMessage(this.segmentMessages, {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: this.pendingSegmentTools
    })

    if (next.length === this.segmentMessages.length + 1) {
      return false
    }

    this.segmentMessages = next
    this.pendingSegmentTools = []
    patchTurnState({ streamPendingTools: [], streamSegments: this.segmentMessages })

    return true
  }

  pushInlineDiffSegment(diffText: string, tools: string[] = []) {
    // Strip CLI chrome the gateway emits before the unified diff (e.g. a
    // leading "┊ review diff" header written by `_emit_inline_diff` for the
    // terminal printer). That header only makes sense as stdout dressing,
    // not inside a markdown ```diff block.
    const stripped = diffText.replace(/^\s*┊[^\n]*\n?/, '').trim()

    if (!stripped) {
      return
    }

    // Flush any in-progress streaming text as its own segment first, so the
    // diff lands BETWEEN the assistant narration that preceded the edit and
    // whatever the agent streams afterwards — not glued onto the final
    // message. This is the whole point of segment-anchored diffs: the diff
    // renders where the edit actually happened.
    this.flushStreamingSegment()

    const block = `\`\`\`diff\n${stripped}\n\`\`\``

    // Skip consecutive duplicates (same tool firing tool.complete twice, or
    // two edits producing the same patch). Keeping this cheap — deeper
    // dedupe against the final assistant text happens at message.complete.
    if (this.segmentMessages.at(-1)?.text === block) {
      return
    }

    this.segmentMessages = [
      ...this.segmentMessages,
      { kind: 'diff', role: 'assistant', text: block, ...(tools.length && { tools }) }
    ]
    patchTurnState({ streamSegments: this.segmentMessages })
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
    if (this.interrupted) {
      return
    }

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

    // Real turn end: surface any notice held back while busy.
    this.flushPendingNotice()
  }

  recordMessageComplete(payload: { rendered?: string; reasoning?: string; text?: string }) {
    this.closeReasoningSegment()

    // Ink renders markdown via <Md>; the gateway's Rich-rendered ANSI
    // (`payload.rendered`) is for terminals that can't.  Prioritising
    // `rendered` here garbles output whenever a user opts into
    // `display.final_response_markdown: render` because raw ANSI escapes
    // pass through into the React tree.  Prefer raw text and fall back
    // only when the gateway elected not to send any (#16391).
    const rawText = (payload.text ?? payload.rendered ?? this.bufRef).trimStart()
    const split = splitReasoning(rawText)
    const finalText = finalTail(split.text, this.segmentMessages)
    const existingReasoning = this.reasoningText.trim() || String(payload.reasoning ?? '').trim()
    const savedReasoning = [existingReasoning, existingReasoning ? '' : split.reasoning].filter(Boolean).join('\n\n')
    const savedToolTokens = this.toolTokenAcc
    let tools = this.pendingSegmentTools
    const last = this.segmentMessages[this.segmentMessages.length - 1]

    if (tools.length && isToolShelfMessage(last)) {
      this.segmentMessages = [
        ...this.segmentMessages.slice(0, -1),
        { ...last, tools: [...(last.tools ?? []), ...tools] }
      ]
      this.pendingSegmentTools = []
      tools = []
    }

    // Drop diff-only segments the agent is about to narrate in the final
    // reply. Without this, a closing "here's the diff …" message would
    // render two stacked copies of the same patch. Only touches segments
    // with `kind: 'diff'` emitted by pushInlineDiffSegment — real
    // assistant narration stays put.
    const finalHasOwnDiffFence = /```(?:diff|patch)\b/i.test(finalText)

    const segments = this.segmentMessages.filter(msg => {
      const body = diffSegmentBody(msg)

      return body === null || (!finalHasOwnDiffFence && !finalText.includes(body))
    })

    const hasReasoningSegment =
      this.reasoningSegmentIndex !== null || segments.some(msg => Boolean(msg.thinking?.trim()))

    const finalThinking = hasReasoningSegment ? '' : savedReasoning.trim()

    const finalDetails: Msg = {
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: finalThinking || undefined,
      thinkingTokens: finalThinking ? estimateTokensRough(finalThinking) : undefined,
      toolTokens: savedToolTokens || undefined,
      ...(tools.length && { tools })
    }

    // Archive prepended so the trail msg anchors under the user prompt,
    // not between thinking/tools and final assistant text.
    const finalMessages: Msg[] = [
      ...archiveDoneTodos(),
      ...segments,
      ...(hasDetails(finalDetails) ? [finalDetails] : [])
    ]

    if (finalText) {
      finalMessages.push({ role: 'assistant', text: finalText })
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
    this.interrupted = false
    patchTurnState({ activity: [], outcome: '' })

    // Real turn end: surface any notice held back while busy. Done after
    // idle() flips busy=false so applyNotice() reaches the visible slot.
    this.flushPendingNotice()

    return { finalMessages, finalText, wasInterrupted }
  }

  recordMessageDelta({ text }: { rendered?: string; text?: string }) {
    if (this.interrupted || !text) {
      return
    }

    this.pruneTransient()
    this.endReasoningPhase()

    // Always accumulate the raw text delta.  The pre-#16391 path replaced
    // the entire buffer with `rendered` (an *incremental* Rich ANSI
    // fragment), which on every tick discarded everything streamed so far
    // — visible as overlapping coloured text and lost prose under
    // `display.final_response_markdown: render`.
    this.bufRef += text

    if (getUiState().streaming) {
      this.scheduleStreaming()
    }
  }

  recordReasoningAvailable(text: string, force = false) {
    if (this.interrupted || (!force && !getUiState().showReasoning)) {
      return
    }

    const incoming = text.trim()

    if (!incoming || this.reasoningText.trim()) {
      return
    }

    this.reasoningText = incoming
    this.activeReasoningText = incoming
    this.scheduleReasoning()
    this.syncReasoningSegment()
    this.pulseReasoningStreaming()
  }

  /**
   * Render one MoA reference model's output as a committed labelled block
   * before the aggregator responds. Unlike reasoning, references are shown
   * regardless of showReasoning (they ARE the mixture-of-agents process the
   * user opted into by selecting a MoA preset). Each becomes its own
   * thinking-style segment tagged with the source model, so a multi-reference
   * preset builds a stack the user can scroll.
   */
  recordMoaReference(label: string, text: string, index?: number, count?: number) {
    if (this.interrupted) {
      return
    }

    // Close any open reasoning segment so the reference block lands as its own
    // committed entry rather than merging into streaming reasoning.
    this.closeReasoningSegment()

    const header =
      index && count ? `◇ Reference ${index}/${count} — ${label}` : `◇ Reference — ${label}`

    const body = text.trim()
    const thinking = body ? `${header}\n${body}` : header

    this.pushSegment({
      kind: 'trail',
      role: 'system',
      text: '',
      thinking,
      thinkingTokens: estimateTokensRough(thinking)
    })
    patchTurnState({ streamSegments: this.segmentMessages })
  }

  recordReasoningDelta(text: string, force = false) {
    if (this.interrupted || (!force && !getUiState().showReasoning)) {
      return
    }

    if (!this.activeReasoningText.trim() && this.pendingSegmentTools.length) {
      this.flushStreamingSegment()
    }

    this.reasoningText += text
    this.activeReasoningText += text

    if (this.reasoningText.length > 80_000) {
      this.reasoningText = this.reasoningText.slice(-60_000)
    }

    this.scheduleReasoning()
    this.syncReasoningSegment()
    this.pulseReasoningStreaming()
  }

  recordToolComplete(
    toolId: string,
    fallbackName?: string,
    error?: string,
    summary?: string,
    duration?: number,
    todos?: unknown,
    resultText?: string
  ) {
    if (this.interrupted) {
      return
    }

    this.recordTodos(todos)
    const line = this.completeTool(toolId, fallbackName, error, summary, duration, resultText)

    this.pendingSegmentTools = [...this.pendingSegmentTools, line]
    this.flushPendingToolsIntoLastSegment()
    this.publishToolState()
  }

  recordInlineDiffToolComplete(
    diffText: string,
    toolId: string,
    fallbackName?: string,
    error?: string,
    duration?: number,
    resultText?: string
  ) {
    if (this.interrupted) {
      return
    }

    this.flushStreamingSegment()
    this.pushInlineDiffSegment(diffText, [this.completeTool(toolId, fallbackName, error, '', duration, resultText)])
    this.publishToolState()
  }

  private completeTool(
    toolId: string,
    fallbackName?: string,
    error?: string,
    summary?: string,
    duration?: number,
    resultText?: string
  ) {
    const done = this.activeTools.find(tool => tool.id === toolId)
    const name = done?.name ?? fallbackName ?? 'tool'
    const label = toolTrailLabel(name)
    const fallbackDuration = done?.startedAt ? (Date.now() - done.startedAt) / 1000 : undefined

    const line =
      done?.verboseArgs || resultText
        ? buildVerboseToolTrailLine(
            name,
            done?.context || '',
            Boolean(error),
            duration ?? fallbackDuration,
            done?.verboseArgs,
            error || resultText || summary || '',
            done?.label
          )
        : buildToolTrailLine(
            name,
            done?.context || '',
            Boolean(error),
            error || summary || '',
            duration ?? fallbackDuration,
            done?.label
          )

    this.activeTools = this.activeTools.filter(tool => tool.id !== toolId)

    const next = this.turnTools.filter(item => !sameToolTrailGroup(label, item))

    if (!this.activeTools.length) {
      next.push('analyzing tool output…')
    }

    this.turnTools = next.slice(-TRAIL_LIMIT)

    return line
  }

  private publishToolState() {
    patchTurnState({
      streamPendingTools: this.pendingSegmentTools,
      tools: this.activeTools,
      turnTrail: this.turnTools
    })
  }

  recordToolProgress(toolName: string, preview: string) {
    if (this.interrupted) {
      return
    }

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

  recordToolStart(toolId: string, name: string, context: string, verboseArgs?: string, label?: string) {
    if (this.interrupted) {
      return
    }

    this.flushStreamingSegment()
    this.closeReasoningSegment()
    this.pruneTransient()
    this.endReasoningPhase()

    const sample = `${name} ${context}`.trim()

    this.toolTokenAcc += sample ? estimateTokensRough(sample) : 0
    this.activeTools = [...this.activeTools, { context, id: toolId, label, name, startedAt: Date.now(), verboseArgs }]

    patchTurnState({ toolTokens: this.toolTokenAcc, tools: this.activeTools })
  }

  reset() {
    this.clearReasoning()
    this.clearStatusTimer()
    this.idle()
    this.bufRef = ''
    this.interrupted = false
    this.lastStatusNote = ''
    this.activeReasoningText = ''
    this.pendingSegmentTools = []
    this.protocolWarned = false
    this.reasoningSegmentIndex = null
    this.segmentMessages = []
    this.turnTools = []
    this.toolTokenAcc = 0
    this.persistedToolLabels.clear()
    // Session boundary: drop notice state so session A's sticky can't bleed
    // into session B (R3-H5). reset()/fullReset() CLEAR — they never flush.
    this.clearNoticeState()
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
      patchTurnState({ streaming: boundedLiveRenderText(visible) })
    }, this.streamDelay)
  }

  hydrateStreamingText(text: string) {
    this.streamTimer = clear(this.streamTimer)
    this.bufRef = text
    const raw = this.bufRef.trimStart()
    const visible = hasReasoningTag(raw) ? splitReasoning(raw).text : raw
    patchTurnState({ streaming: boundedLiveRenderText(visible) })
  }

  startMessage() {
    this.endReasoningPhase()
    this.clearReasoning()
    this.activeTools = []
    this.activeReasoningText = ''
    this.reasoningSegmentIndex = null
    this.turnTools = []
    this.toolTokenAcc = 0
    this.interrupted = false
    this.persistedToolLabels.clear()
    // "Flash and yield" notices clear when a new turn starts: a usage-band heads-up
    // (credits.usage, 50/75/90%) and the one-time "grant spent" transition
    // (credits.grant_spent) should show once, then get out of the way — not camp the
    // bar (e.g. "Grant spent · $990 top-up left" sitting there with plenty of top-up
    // left). Depletion (credits.depleted) and other notices stay — they're explicitly
    // sticky until the policy clears them. The Python `active` latch retains the key,
    // so a yielded notice won't re-fire on the next turn.
    const yieldingNoticeKey = getUiState().notice?.key

    if (yieldingNoticeKey === 'credits.usage' || yieldingNoticeKey === 'credits.grant_spent') {
      this.clearNotice(yieldingNoticeKey)
    }

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
