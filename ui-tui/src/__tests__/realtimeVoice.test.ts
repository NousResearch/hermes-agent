import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  encodeRealtimeVoiceDelegationProgress,
  encodeRealtimeVoiceDelegationResult,
  MAX_REALTIME_VOICE_FRAME_CHARS,
  MAX_REALTIME_VOICE_TEXT_CHARS,
  parseRealtimeVoiceEvent,
  parseRealtimeVoicePhase,
  registerRealtimeVoiceProcess,
  stopRegisteredRealtimeVoiceProcess,
  writeRealtimeVoiceControl
} from '../domain/realtimeVoice.js'

afterEach(() => {
  vi.useRealTimers()
})

const envelope = {
  protocol_version: 1 as const,
  sequence: 1,
  surface_session_id: 'surface-1'
}

const framed = (payload: Record<string, unknown>): string =>
  `talk: event ${JSON.stringify({ ...envelope, ...payload })}`

describe('realtime voice lifecycle', () => {
  it.each(['listening', 'solving', 'composing'] as const)('parses the %s phase', phase => {
    expect(parseRealtimeVoicePhase(`talk: state ${phase}`)).toBe(phase)
  })

  it('ignores transcripts and unknown states', () => {
    expect(parseRealtimeVoicePhase('hello from the model')).toBeNull()
    expect(parseRealtimeVoicePhase('talk: state disconnected')).toBeNull()
  })

  it('parses framed live transcripts without mistaking model text for control data', () => {
    expect(
      parseRealtimeVoiceEvent(framed({ type: 'transcript', role: 'user', text: 'check the weather', final: true }))
    ).toEqual({
      ...envelope,
      type: 'transcript',
      role: 'user',
      text: 'check the weather',
      final: true
    })
    expect(parseRealtimeVoiceEvent(framed({ type: 'error', message: 'connection closed' }))).toEqual({
      ...envelope,
      type: 'error',
      message: 'connection closed'
    })
    expect(parseRealtimeVoiceEvent('{"type":"transcript","role":"user"}')).toBeNull()
    expect(parseRealtimeVoiceEvent('talk: event {"type":"error","message":"missing envelope"}')).toBeNull()
    expect(parseRealtimeVoiceEvent(framed({ type: 'error', message: 'bad sequence', sequence: 0 }))).toBeNull()
  })

  it('parses nonterminal provider warnings', () => {
    expect(parseRealtimeVoiceEvent(framed({ type: 'warning', message: 'input queue overrun' }))).toEqual({
      ...envelope,
      type: 'warning',
      message: 'input queue overrun'
    })
  })

  it('parses monotonic latency observations', () => {
    expect(
      parseRealtimeVoiceEvent(framed({ type: 'metric', name: 'endpoint_to_first_audio_ms', value_ms: 234.5 }))
    ).toEqual({
      ...envelope,
      type: 'metric',
      name: 'endpoint_to_first_audio_ms',
      value_ms: 234.5
    })
  })

  it('round-trips a delegated text-agent result over child stdin', () => {
    expect(parseRealtimeVoiceEvent(framed({ type: 'delegate', id: 'call-1', request: 'inspect the bug' }))).toEqual({
      ...envelope,
      type: 'delegate',
      id: 'call-1',
      request: 'inspect the bug'
    })
    expect(JSON.parse(encodeRealtimeVoiceDelegationResult('call-1', 'fixed'))).toEqual({
      type: 'delegate.result',
      id: 'call-1',
      output: 'fixed'
    })
    expect(JSON.parse(encodeRealtimeVoiceDelegationProgress('call-1', 'checking tests'))).toEqual({
      type: 'delegate.progress',
      id: 'call-1',
      text: 'checking tests'
    })
  })

  it('bounds every child protocol frame and delegated context payload', () => {
    const oversizedEvent = `talk: event ${'x'.repeat(MAX_REALTIME_VOICE_FRAME_CHARS)}`
    const oversizedState = `talk: state ${'x'.repeat(MAX_REALTIME_VOICE_FRAME_CHARS)}`
    const oversizedText = 'x'.repeat(MAX_REALTIME_VOICE_TEXT_CHARS + 100)

    expect(parseRealtimeVoiceEvent(oversizedEvent)).toBeNull()
    expect(parseRealtimeVoicePhase(oversizedState)).toBeNull()
    expect(JSON.parse(encodeRealtimeVoiceDelegationResult('call-1', oversizedText)).output).toHaveLength(
      MAX_REALTIME_VOICE_TEXT_CHARS
    )
    expect(JSON.parse(encodeRealtimeVoiceDelegationProgress('call-1', oversizedText)).text).toHaveLength(
      MAX_REALTIME_VOICE_TEXT_CHARS
    )
  })
})

describe('realtime voice child supervision', () => {
  const childProcess = (exitOnInterrupt: boolean) => {
    const child = new EventEmitter() as EventEmitter & {
      exitCode: null | number
      kill: ReturnType<typeof vi.fn>
      signalCode: NodeJS.Signals | null
    }

    child.exitCode = null
    child.signalCode = null
    child.kill = vi.fn((signal: NodeJS.Signals) => {
      if (signal === 'SIGINT' && exitOnInterrupt) {
        child.signalCode = signal
        child.emit('exit', null, signal)
      }
      return true
    })

    return child as unknown as ChildProcess
  }

  it('drops optional progress while child stdin is backpressured', () => {
    const input = new EventEmitter() as EventEmitter & {
      writable: boolean
      write: ReturnType<typeof vi.fn>
    }
    input.writable = true
    input.write = vi.fn().mockReturnValueOnce(false).mockReturnValue(true)
    const child = { stdin: input } as unknown as ChildProcess

    writeRealtimeVoiceControl(child, 'progress-1', false)
    writeRealtimeVoiceControl(child, 'progress-2', false)
    writeRealtimeVoiceControl(child, 'result', true)
    input.emit('drain')
    writeRealtimeVoiceControl(child, 'progress-3', false)

    expect(input.write.mock.calls.map(([payload]) => payload)).toEqual(['progress-1', 'result', 'progress-3'])
  })

  it('stops the registered child once with SIGINT', async () => {
    const child = childProcess(true)
    registerRealtimeVoiceProcess(child)

    const first = stopRegisteredRealtimeVoiceProcess()
    const second = stopRegisteredRealtimeVoiceProcess()

    expect(first).toBe(second)
    await first
    expect(child.kill).toHaveBeenCalledTimes(1)
    expect(child.kill).toHaveBeenCalledWith('SIGINT')
  })

  it('escalates an unresponsive child to SIGKILL after the grace period', async () => {
    vi.useFakeTimers()
    const child = childProcess(false)
    registerRealtimeVoiceProcess(child)

    const stopped = stopRegisteredRealtimeVoiceProcess(25)
    await vi.advanceTimersByTimeAsync(25)
    await stopped

    expect(child.kill).toHaveBeenNthCalledWith(1, 'SIGINT')
    expect(child.kill).toHaveBeenNthCalledWith(2, 'SIGKILL')
  })
})
