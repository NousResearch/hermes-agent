import { describe, expect, it } from 'vitest'

import {
  renderVoiceVisualization,
  renderVoiceWaveform,
  resolveVoiceMode,
  voiceVisualizationFooter
} from '../components/voiceVisualizer.js'

describe('renderVoiceWaveform', () => {
  it('keeps a fixed seven-row pane and fills the requested width', () => {
    const rows = renderVoiceWaveform(32, 0, true)

    expect(rows).toHaveLength(7)
    expect(rows.every(row => row.length === 32)).toBe(true)
    expect(rows.join('')).toMatch(/[▁▂▃▄▅▆▇█]/u)
  })

  it('animates listening energy without changing pane geometry', () => {
    const initial = renderVoiceWaveform(24, 0, true)
    const advanced = renderVoiceWaveform(24, 8, true)

    expect(advanced).not.toEqual(initial)
    expect(advanced.every(row => row.length === 24)).toBe(true)
  })

  it('renders exactly the selected visualization', () => {
    const orb = renderVoiceVisualization('orb', 19, 3, 'listening')
    const waveform = renderVoiceVisualization('waveform', 32, 3, 'listening')

    expect(orb).toHaveLength(7)
    expect(orb.every(row => row.length === 19)).toBe(true)
    expect(waveform).toHaveLength(7)
    expect(waveform.every(row => row.length === 32)).toBe(true)
  })

  it('distinguishes transport startup from active listening', () => {
    expect(voiceVisualizationFooter('waiting')).toBe('Waiting for realtime voice…')
    expect(voiceVisualizationFooter('listening', 'Ctrl+O')).toBe('Listening · Ctrl+O to stop')
  })

  it('preserves distinct solving and composing phases', () => {
    expect(voiceVisualizationFooter('solving')).toBe('Solving…')
    expect(voiceVisualizationFooter('composing')).toBe('Speaking…')
    expect(renderVoiceVisualization('orb', 19, 3, 'solving')).not.toEqual(
      renderVoiceVisualization('orb', 19, 3, 'composing')
    )
  })
})

describe('resolveVoiceMode', () => {
  const idle = {
    realtimeVoiceConnecting: false,
    realtimeVoicePhase: null,
    voiceProcessing: false,
    voiceRecording: false
  } as const

  it('projects every lifecycle state used by the orb', () => {
    expect(resolveVoiceMode({ ...idle, realtimeVoiceConnecting: true })).toBe('waiting')
    expect(resolveVoiceMode({ ...idle, voiceRecording: true })).toBe('listening')
    expect(resolveVoiceMode({ ...idle, voiceProcessing: true })).toBe('solving')
    expect(resolveVoiceMode({ ...idle, realtimeVoicePhase: 'listening' })).toBe('listening')
    expect(resolveVoiceMode({ ...idle, realtimeVoicePhase: 'solving' })).toBe('solving')
    expect(resolveVoiceMode({ ...idle, realtimeVoicePhase: 'composing' })).toBe('composing')
    expect(resolveVoiceMode(idle)).toBeNull()
  })

  it('keeps connection and explicit realtime phases ahead of legacy flags', () => {
    expect(
      resolveVoiceMode({
        realtimeVoiceConnecting: true,
        realtimeVoicePhase: 'composing',
        voiceProcessing: true,
        voiceRecording: true
      })
    ).toBe('waiting')
    expect(
      resolveVoiceMode({
        realtimeVoiceConnecting: false,
        realtimeVoicePhase: 'composing',
        voiceProcessing: true,
        voiceRecording: true
      })
    ).toBe('composing')
  })
})
