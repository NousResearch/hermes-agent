#!/usr/bin/env node
/* global Buffer, console, process, setImmediate */
import inspector from 'node:inspector'
import { performance } from 'node:perf_hooks'

import React from 'react'
import { render } from '@hermes/ink'
import { AppLayout } from '../src/components/appLayout.tsx'
import { resetOverlayState } from '../src/app/overlayStore.ts'
import { resetTurnState } from '../src/app/turnStore.ts'
import { resetUiState } from '../src/app/uiStore.ts'

const session = new inspector.Session()
session.connect()
const post = (method, params = {}) => new Promise((resolve, reject) => {
  session.post(method, params, (err, result) => err ? reject(err) : resolve(result))
})

const historySize = Number(process.env.HISTORY || 500)
const mountedRows = Number(process.env.MOUNTED || 120)

class Sink {
  columns = Number(process.env.COLS || 120)
  rows = Number(process.env.ROWS || 42)
  isTTY = true
  bytes = 0
  writes = 0
  listeners = new Map()
  write(chunk) {
    this.bytes += Buffer.byteLength(String(chunk ?? ''))
    this.writes++
    return true
  }
  on(event, fn) { this.listeners.set(event, fn); return this }
  off(event) { this.listeners.delete(event); return this }
  once(event, fn) { this.listeners.set(event, fn); return this }
  removeListener(event) { this.listeners.delete(event); return this }
}

const theme = {
  brand: { prompt: '›' },
  color: {
    amber: '#d19a66', bronze: '#8b6f47', dim: '#6b7280', error: '#ff5555', gold: '#ffd166', label: '#61afef',
    ok: '#98c379', warn: '#e5c07b', cornsilk: '#fff8dc', prompt: '#c678dd', shellDollar: '#98c379',
    statusCritical: '#ff5555', statusBad: '#e06c75', statusWarn: '#e5c07b', statusGood: '#98c379',
    selectionBg: '#44475a'
  }
}

const noop = () => {}
const historyItems = [
  { kind: 'intro', role: 'system', text: '', info: { model: 'test', tools: {}, skills: {}, version: 'test' } },
  ...Array.from({ length: historySize }, (_, i) => ({
    role: i % 5 === 0 ? 'user' : 'assistant',
    text: `message ${i}\n${'lorem ipsum '.repeat(80)}`
  }))
]
const scrollRef = { current: {
  getScrollTop: () => 0,
  getPendingDelta: () => 0,
  getScrollHeight: () => historySize * 4,
  getViewportHeight: () => 30,
  getViewportTop: () => 0,
  isSticky: () => true,
  subscribe: () => () => {},
  scrollBy: noop,
  scrollTo: noop,
  scrollToBottom: noop,
  setClampBounds: noop,
  getLastManualScrollAt: () => 0
} }

const baseProps = streamingText => ({
  actions: { answerApproval: noop, answerClarify: noop, answerSecret: noop, answerSudo: noop, onModelSelect: noop, resumeById: noop, setStickyPrompt: noop },
  composer: { cols: 120, compIdx: 0, completions: [], empty: false, handleTextPaste: () => null, input: '', inputBuf: [], pagerPageSize: 10, queueEditIdx: null, queuedDisplay: [], submit: noop, updateInput: noop },
  mouseTracking: false,
  progress: {
    activity: [], outcome: '', reasoning: streamingText, reasoningActive: true, reasoningStreaming: true,
    reasoningTokens: Math.ceil(streamingText.length / 4), showProgressArea: true, showStreamingArea: true,
    streamPendingTools: [], streamSegments: [], streaming: streamingText, subagents: [], toolTokens: 0, tools: [], turnTrail: [], todos: []
  },
  status: { cwdLabel: '~/repo', goodVibesTick: 0, sessionStartedAt: Date.now(), showStickyPrompt: false, statusColor: theme.color.ok, stickyPrompt: '', turnStartedAt: Date.now(), voiceLabel: 'voice off' },
  transcript: {
    historyItems,
    scrollRef,
    virtualHistory: { bottomSpacer: 0, end: historyItems.length, measureRef: () => noop, offsets: historyItems.map((_, i) => i * 4), start: Math.max(0, historyItems.length - mountedRows), topSpacer: 0 },
    virtualRows: historyItems.map((msg, index) => ({ index, key: `m${index}`, msg }))
  }
})

async function main() {
  resetUiState()
  resetTurnState()
  resetOverlayState()
  const stdout = new Sink()
  const stdin = { isTTY: true, setRawMode: noop, on: noop, off: noop, resume: noop, pause: noop }
  const text = Array.from({ length: Number(process.env.LINES || 1200) }, (_, i) => `stream line ${i} ${'x'.repeat(90)}`).join('\n')
  const inst = render(React.createElement(AppLayout, baseProps('')), { stdout, stdin, stderr: stdout, debug: false, exitOnCtrlC: false })

  await post('Profiler.enable')
  await post('HeapProfiler.enable')
  await post('Profiler.start')
  const startMem = process.memoryUsage()
  const t0 = performance.now()
  const iterations = Number(process.env.ITERS || 40)
  for (let i = 1; i <= iterations; i++) {
    const prefix = text.slice(0, Math.floor(text.length * i / iterations))
    inst.rerender(React.createElement(AppLayout, baseProps(prefix)))
    await new Promise(r => setImmediate(r))
  }
  const elapsed = performance.now() - t0
  const prof = await post('Profiler.stop')
  const endMem = process.memoryUsage()
  await post('HeapProfiler.collectGarbage')
  const afterGc = process.memoryUsage()
  inst.unmount()
  session.disconnect()
  console.log(JSON.stringify({ elapsedMs: Math.round(elapsed), stdoutBytes: stdout.bytes, stdoutWrites: stdout.writes, startMem, endMem, afterGc, profileNodes: prof.profile.nodes.length }, null, 2))
}

main().catch(err => { console.error(err); process.exit(1) })
