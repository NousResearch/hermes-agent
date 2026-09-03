import type { ChildProcess } from 'node:child_process'

let activeProcess: ChildProcess | null = null
let stopPromise: Promise<void> | null = null
const backpressuredInputs = new WeakSet<NonNullable<ChildProcess['stdin']>>()

export function writeRealtimeVoiceControl(child: ChildProcess, payload: string, required = true): void {
  const input = child.stdin

  if (!input?.writable || (!required && backpressuredInputs.has(input))) {
    return
  }
  if (!input.write(payload)) {
    backpressuredInputs.add(input)
    input.once('drain', () => backpressuredInputs.delete(input))
  }
}

export const registerRealtimeVoiceProcess = (child: ChildProcess): void => {
  if (activeProcess && activeProcess !== child) {
    throw new Error('A native realtime voice process is already registered.')
  }

  activeProcess = child
}

export const unregisterRealtimeVoiceProcess = (child: ChildProcess): void => {
  if (activeProcess === child) {
    activeProcess = null
  }
}

export const stopRegisteredRealtimeVoiceProcess = (graceMs = 1_500): Promise<void> => {
  if (stopPromise) {
    return stopPromise
  }

  const child = activeProcess

  if (!child || child.exitCode !== null || child.signalCode !== null) {
    activeProcess = null
    return Promise.resolve()
  }
  let resolveStop: () => void = () => {}
  const promise = new Promise<void>(resolve => {
    resolveStop = resolve
  })
  let settled = false
  const finish = () => {
    if (settled) {
      return
    }

    settled = true
    clearTimeout(timer)
    child.off('exit', finish)
    resolveStop()
  }
  const timer = setTimeout(() => {
    try {
      child.kill('SIGKILL')
    } finally {
      finish()
    }
  }, graceMs)

  child.once('exit', finish)

  try {
    if (!child.kill('SIGINT')) {
      finish()
    }
  } catch {
    finish()
  }

  const stopping = promise.finally(() => {
    if (activeProcess === child) {
      activeProcess = null
    }
    stopPromise = null
  })
  stopPromise = stopping

  return stopping
}

export type RealtimeVoicePhase = 'composing' | 'listening' | 'solving'
export const MAX_REALTIME_VOICE_FRAME_CHARS = 65_536
export const MAX_REALTIME_VOICE_TEXT_CHARS = 10_000

export interface RealtimeVoiceTranscript {
  final: boolean
  role: 'assistant' | 'user'
  text: string
}

export interface RealtimeVoiceEnvelope {
  protocol_version: 1
  sequence: number
  surface_session_id: string
}
export type RealtimeVoiceEvent = RealtimeVoiceEnvelope &
  (
    | { type: 'delegate'; id: string; request: string }
    | { type: 'error'; message: string }
    | { type: 'warning'; message: string }
    | { type: 'metric'; name: string; value_ms: number }
    | ({ type: 'transcript' } & RealtimeVoiceTranscript)
  )
const EVENT_PREFIX = 'talk: event '

const STATE_PREFIX = 'talk: state '

export const parseRealtimeVoicePhase = (line: string): RealtimeVoicePhase | null => {
  if (!line.startsWith(STATE_PREFIX)) {
    return null
  }

  if (line.length > MAX_REALTIME_VOICE_FRAME_CHARS) {
    return null
  }

  const phase = line.slice(STATE_PREFIX.length).trim()
  return phase === 'listening' || phase === 'solving' || phase === 'composing' ? phase : null
}

export const parseRealtimeVoiceEvent = (line: string): RealtimeVoiceEvent | null => {
  if (line.length > MAX_REALTIME_VOICE_FRAME_CHARS) {
    return null
  }

  if (!line.startsWith(EVENT_PREFIX)) {
    return null
  }

  try {
    const value: unknown = JSON.parse(line.slice(EVENT_PREFIX.length))

    if (!value || typeof value !== 'object') {
      return null
    }

    const event = value as Record<string, unknown>
    if (
      event.protocol_version !== 1 ||
      typeof event.surface_session_id !== 'string' ||
      !event.surface_session_id ||
      !Number.isSafeInteger(event.sequence) ||
      Number(event.sequence) < 1
    ) {
      return null
    }
    const envelope: RealtimeVoiceEnvelope = {
      protocol_version: 1,
      surface_session_id: event.surface_session_id,
      sequence: Number(event.sequence)
    }

    if (
      event.type === 'delegate' &&
      typeof event.id === 'string' &&
      event.id.length > 0 &&
      typeof event.request === 'string' &&
      event.request.trim()
    ) {
      return { ...envelope, type: 'delegate', id: event.id, request: event.request.trim() }
    }

    if (event.type === 'error' && typeof event.message === 'string' && event.message.trim()) {
      return { ...envelope, type: 'error', message: event.message.trim() }
    }

    if (
      event.type === 'metric' &&
      typeof event.name === 'string' &&
      event.name.length > 0 &&
      typeof event.value_ms === 'number' &&
      Number.isFinite(event.value_ms) &&
      event.value_ms >= 0
    ) {
      return { ...envelope, type: 'metric', name: event.name, value_ms: event.value_ms }
    }

    if (event.type === 'warning' && typeof event.message === 'string' && event.message.trim()) {
      return { ...envelope, type: 'warning', message: event.message.trim() }
    }

    if (
      event.type === 'transcript' &&
      (event.role === 'user' || event.role === 'assistant') &&
      typeof event.text === 'string' &&
      typeof event.final === 'boolean'
    ) {
      return {
        ...envelope,
        type: 'transcript',
        role: event.role,
        text: event.text,
        final: event.final
      }
    }
  } catch {
    return null
  }

  return null
}

export const encodeRealtimeVoiceDelegationResult = (id: string, output: string): string =>
  `${JSON.stringify({
    type: 'delegate.result',
    id,
    output: output.slice(0, MAX_REALTIME_VOICE_TEXT_CHARS)
  })}\n`

export const encodeRealtimeVoiceDelegationProgress = (id: string, text: string): string =>
  `${JSON.stringify({
    type: 'delegate.progress',
    id,
    text: text.slice(0, MAX_REALTIME_VOICE_TEXT_CHARS)
  })}\n`
