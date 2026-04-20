import { REASONING_PULSE_MS, STREAM_BATCH_MS } from '../config/timing.js';
import { hasReasoningTag, splitReasoning } from '../lib/reasoning.js';
import { buildToolTrailLine, estimateTokensRough, isTransientTrailLine, sameToolTrailGroup, toolTrailLabel } from '../lib/text.js';
import { resetOverlayState } from './overlayStore.js';
import { patchTurnState, resetTurnState } from './turnStore.js';
import { getUiState, patchUiState } from './uiStore.js';
const INTERRUPT_COOLDOWN_MS = 1500;
const ACTIVITY_LIMIT = 8;
const TRAIL_LIMIT = 8;
const clear = (t) => {
    if (t) {
        clearTimeout(t);
    }
    return null;
};
class TurnController {
    bufRef = '';
    interrupted = false;
    lastStatusNote = '';
    persistedToolLabels = new Set();
    protocolWarned = false;
    reasoningText = '';
    segmentMessages = [];
    pendingSegmentTools = [];
    statusTimer = null;
    toolTokenAcc = 0;
    turnTools = [];
    activeTools = [];
    activityId = 0;
    reasoningStreamingTimer = null;
    reasoningTimer = null;
    streamTimer = null;
    toolProgressTimer = null;
    clearReasoning() {
        this.reasoningTimer = clear(this.reasoningTimer);
        this.reasoningText = '';
        this.toolTokenAcc = 0;
        patchTurnState({ reasoning: '', reasoningTokens: 0, toolTokens: 0 });
    }
    clearStatusTimer() {
        this.statusTimer = clear(this.statusTimer);
    }
    endReasoningPhase() {
        this.reasoningStreamingTimer = clear(this.reasoningStreamingTimer);
        patchTurnState({ reasoningActive: false, reasoningStreaming: false });
    }
    idle() {
        this.endReasoningPhase();
        this.activeTools = [];
        this.streamTimer = clear(this.streamTimer);
        this.bufRef = '';
        this.pendingSegmentTools = [];
        this.segmentMessages = [];
        patchTurnState({
            streamPendingTools: [],
            streamSegments: [],
            streaming: '',
            subagents: [],
            tools: [],
            turnTrail: []
        });
        patchUiState({ busy: false });
        resetOverlayState();
    }
    interruptTurn({ appendMessage, gw, sid, sys }) {
        this.interrupted = true;
        gw.request('session.interrupt', { session_id: sid }).catch(() => { });
        const partial = this.bufRef.trimStart();
        partial ? appendMessage({ role: 'assistant', text: `${partial}\n\n*[interrupted]*` }) : sys('interrupted');
        this.idle();
        this.clearReasoning();
        this.turnTools = [];
        patchTurnState({ activity: [], outcome: '' });
        patchUiState({ status: 'interrupted' });
        this.clearStatusTimer();
        this.statusTimer = setTimeout(() => {
            this.statusTimer = null;
            patchUiState({ status: 'ready' });
        }, INTERRUPT_COOLDOWN_MS);
    }
    pruneTransient() {
        this.turnTools = this.turnTools.filter(line => !isTransientTrailLine(line));
        patchTurnState(state => {
            const next = state.turnTrail.filter(line => !isTransientTrailLine(line));
            return next.length === state.turnTrail.length ? state : { ...state, turnTrail: next };
        });
    }
    flushStreamingSegment() {
        const raw = this.bufRef.trimStart();
        if (!raw) {
            return;
        }
        const split = hasReasoningTag(raw) ? splitReasoning(raw) : { reasoning: '', text: raw };
        if (split.reasoning && !this.reasoningText.trim()) {
            this.reasoningText = split.reasoning;
            patchTurnState({ reasoning: this.reasoningText, reasoningTokens: estimateTokensRough(this.reasoningText) });
        }
        const text = split.text;
        this.streamTimer = clear(this.streamTimer);
        if (text) {
            const tools = this.pendingSegmentTools;
            this.segmentMessages = [...this.segmentMessages, { role: 'assistant', text, ...(tools.length && { tools }) }];
            this.pendingSegmentTools = [];
        }
        this.bufRef = '';
        patchTurnState({ streamPendingTools: [], streamSegments: this.segmentMessages, streaming: '' });
    }
    pulseReasoningStreaming() {
        this.reasoningStreamingTimer = clear(this.reasoningStreamingTimer);
        patchTurnState({ reasoningActive: true, reasoningStreaming: true });
        this.reasoningStreamingTimer = setTimeout(() => {
            this.reasoningStreamingTimer = null;
            patchTurnState({ reasoningStreaming: false });
        }, REASONING_PULSE_MS);
    }
    pushActivity(text, tone = 'info', replaceLabel) {
        patchTurnState(state => {
            const base = replaceLabel
                ? state.activity.filter(item => !sameToolTrailGroup(replaceLabel, item.text))
                : state.activity;
            const tail = base.at(-1);
            if (tail?.text === text && tail.tone === tone) {
                return state;
            }
            return { ...state, activity: [...base, { id: ++this.activityId, text, tone }].slice(-ACTIVITY_LIMIT) };
        });
    }
    pushTrail(line) {
        patchTurnState(state => {
            if (state.turnTrail.at(-1) === line) {
                return state;
            }
            const next = [...state.turnTrail.filter(item => !isTransientTrailLine(item)), line].slice(-TRAIL_LIMIT);
            this.turnTools = next;
            return { ...state, turnTrail: next };
        });
    }
    recordError() {
        this.idle();
        this.clearReasoning();
        this.clearStatusTimer();
        this.pendingSegmentTools = [];
        this.segmentMessages = [];
        this.turnTools = [];
        this.persistedToolLabels.clear();
    }
    recordMessageComplete(payload) {
        const rawText = (payload.rendered ?? payload.text ?? this.bufRef).trimStart();
        const split = splitReasoning(rawText);
        const finalText = split.text;
        const existingReasoning = this.reasoningText.trim() || String(payload.reasoning ?? '').trim();
        const savedReasoning = [existingReasoning, existingReasoning ? '' : split.reasoning].filter(Boolean).join('\n\n');
        const savedReasoningTokens = savedReasoning ? estimateTokensRough(savedReasoning) : 0;
        const savedToolTokens = this.toolTokenAcc;
        const tools = this.pendingSegmentTools;
        const finalMessages = [...this.segmentMessages];
        if (finalText) {
            finalMessages.push({
                role: 'assistant',
                text: finalText,
                thinking: savedReasoning || undefined,
                thinkingTokens: savedReasoning ? savedReasoningTokens : undefined,
                toolTokens: savedToolTokens || undefined,
                ...(tools.length && { tools })
            });
        }
        const wasInterrupted = this.interrupted;
        this.idle();
        this.clearReasoning();
        this.turnTools = [];
        this.persistedToolLabels.clear();
        this.bufRef = '';
        patchTurnState({ activity: [], outcome: '' });
        return { finalMessages, finalText, wasInterrupted };
    }
    recordMessageDelta({ rendered, text }) {
        this.pruneTransient();
        this.endReasoningPhase();
        if (!text || this.interrupted) {
            return;
        }
        this.bufRef = rendered ?? this.bufRef + text;
        if (getUiState().streaming) {
            this.scheduleStreaming();
        }
    }
    recordReasoningAvailable(text) {
        if (!getUiState().showReasoning) {
            return;
        }
        const incoming = text.trim();
        if (!incoming || this.reasoningText.trim()) {
            return;
        }
        this.reasoningText = incoming;
        this.scheduleReasoning();
        this.pulseReasoningStreaming();
    }
    recordReasoningDelta(text) {
        if (!getUiState().showReasoning) {
            return;
        }
        this.reasoningText += text;
        this.scheduleReasoning();
        this.pulseReasoningStreaming();
    }
    recordToolComplete(toolId, fallbackName, error, summary) {
        const done = this.activeTools.find(tool => tool.id === toolId);
        const name = done?.name ?? fallbackName ?? 'tool';
        const label = toolTrailLabel(name);
        const line = buildToolTrailLine(name, done?.context || '', Boolean(error), error || summary || '');
        this.activeTools = this.activeTools.filter(tool => tool.id !== toolId);
        this.pendingSegmentTools = [...this.pendingSegmentTools, line];
        const next = this.turnTools.filter(item => !sameToolTrailGroup(label, item));
        if (!this.activeTools.length) {
            next.push('analyzing tool output…');
        }
        this.turnTools = next.slice(-TRAIL_LIMIT);
        patchTurnState({
            streamPendingTools: this.pendingSegmentTools,
            tools: this.activeTools,
            turnTrail: this.turnTools
        });
    }
    recordToolProgress(toolName, preview) {
        const index = this.activeTools.findIndex(tool => tool.name === toolName);
        if (index < 0) {
            return;
        }
        this.activeTools = this.activeTools.map((tool, i) => (i === index ? { ...tool, context: preview } : tool));
        if (this.toolProgressTimer) {
            return;
        }
        this.toolProgressTimer = setTimeout(() => {
            this.toolProgressTimer = null;
            patchTurnState({ tools: [...this.activeTools] });
        }, STREAM_BATCH_MS);
    }
    recordToolStart(toolId, name, context) {
        this.flushStreamingSegment();
        this.pruneTransient();
        this.endReasoningPhase();
        const sample = `${name} ${context}`.trim();
        this.toolTokenAcc += sample ? estimateTokensRough(sample) : 0;
        this.activeTools = [...this.activeTools, { context, id: toolId, name, startedAt: Date.now() }];
        patchTurnState({ toolTokens: this.toolTokenAcc, tools: this.activeTools });
    }
    reset() {
        this.clearReasoning();
        this.clearStatusTimer();
        this.idle();
        this.bufRef = '';
        this.interrupted = false;
        this.lastStatusNote = '';
        this.pendingSegmentTools = [];
        this.protocolWarned = false;
        this.segmentMessages = [];
        this.turnTools = [];
        this.toolTokenAcc = 0;
        this.persistedToolLabels.clear();
        patchTurnState({ activity: [], outcome: '' });
    }
    fullReset() {
        this.reset();
        resetTurnState();
    }
    scheduleReasoning() {
        if (this.reasoningTimer) {
            return;
        }
        this.reasoningTimer = setTimeout(() => {
            this.reasoningTimer = null;
            patchTurnState({
                reasoning: this.reasoningText,
                reasoningTokens: estimateTokensRough(this.reasoningText)
            });
        }, STREAM_BATCH_MS);
    }
    scheduleStreaming() {
        if (this.streamTimer) {
            return;
        }
        this.streamTimer = setTimeout(() => {
            this.streamTimer = null;
            const raw = this.bufRef.trimStart();
            const visible = hasReasoningTag(raw) ? splitReasoning(raw).text : raw;
            patchTurnState({ streaming: visible });
        }, STREAM_BATCH_MS);
    }
    startMessage() {
        this.endReasoningPhase();
        this.clearReasoning();
        this.activeTools = [];
        this.turnTools = [];
        this.toolTokenAcc = 0;
        this.persistedToolLabels.clear();
        patchUiState({ busy: true });
        patchTurnState({ activity: [], outcome: '', subagents: [], toolTokens: 0, tools: [], turnTrail: [] });
    }
    upsertSubagent(p, patch) {
        const id = `sa:${p.task_index}:${p.goal || 'subagent'}`;
        patchTurnState(state => {
            const existing = state.subagents.find(item => item.id === id);
            const base = existing ?? {
                goal: p.goal,
                id,
                index: p.task_index,
                notes: [],
                status: 'running',
                taskCount: p.task_count ?? 1,
                thinking: [],
                tools: []
            };
            const next = {
                ...base,
                goal: p.goal || base.goal,
                taskCount: p.task_count ?? base.taskCount,
                ...patch(base)
            };
            const subagents = existing
                ? state.subagents.map(item => (item.id === id ? next : item))
                : [...state.subagents, next].sort((a, b) => a.index - b.index);
            return { ...state, subagents };
        });
    }
}
export const turnController = new TurnController();
