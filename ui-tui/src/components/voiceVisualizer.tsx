import { Box, Text } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'
import { renderTuiOrb } from 'thinking-orbs/tui'

import type { RealtimeVoicePhase, RealtimeVoiceTranscript } from '../domain/realtimeVoice.js'
import type { Theme } from '../theme.js'

export interface VoiceVisualizerProps {
  columns: number
  mode: RealtimeVoicePhase | 'waiting'
  t: Theme
  stopKeyLabel: string
  transcript?: RealtimeVoiceTranscript | null
  visualizer: 'orb' | 'waveform'
}

export interface VoiceModeStatus {
  realtimeVoiceConnecting: boolean
  realtimeVoicePhase: RealtimeVoicePhase | null
  voiceProcessing: boolean
  voiceRecording: boolean
}

export function resolveVoiceMode(status: VoiceModeStatus): VoiceVisualizerProps['mode'] | null {
  if (status.realtimeVoiceConnecting) {
    return 'waiting'
  }
  if (status.realtimeVoicePhase) {
    return status.realtimeVoicePhase
  }
  if (status.voiceRecording) {
    return 'listening'
  }
  return status.voiceProcessing ? 'solving' : null
}

const FRAME_MS = 80
const ORB_COLUMNS = 19
const ORB_ROWS = 7
const WAVE_BLOCKS = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'] as const

export function renderVoiceWaveform(width: number, frame: number, active: boolean): string[] {
  const rows = ORB_ROWS
  const output = Array.from({ length: rows }, () => '')
  const maxHeight = rows * (WAVE_BLOCKS.length - 1)
  const energy = active ? 0.72 + Math.sin(frame * 0.17) * 0.12 : 0.24

  for (let column = 0; column < width; column += 1) {
    const carrier = 0.5 + 0.5 * Math.sin(frame * 0.43 + column * 0.71)
    const shimmer = 0.5 + 0.5 * Math.sin(frame * 0.19 - column * 1.17)
    const height = Math.round(energy * (0.3 + carrier * 0.5 + shimmer * 0.2) * maxHeight)

    for (let row = 0; row < rows; row += 1) {
      const units = Math.max(0, Math.min(WAVE_BLOCKS.length - 1, height - (rows - row - 1) * 8))
      output[row] += WAVE_BLOCKS[units]
    }
  }

  return output
}

export function renderVoiceVisualization(
  visualizer: 'orb' | 'waveform',
  width: number,
  frame: number,
  mode: VoiceVisualizerProps['mode']
): string[] {
  const active = mode === 'listening'

  if (visualizer === 'waveform') {
    return renderVoiceWaveform(width, frame, active)
  }

  const orbState = mode === 'waiting' ? 'connecting' : mode

  return renderTuiOrb(orbState, {
    columns: width,
    rows: ORB_ROWS,
    speed: active ? 0.72 : 0.55,
    threshold: active ? 0.18 : 0.21,
    time: frame * (FRAME_MS / 1000)
  }).lines
}

export function voiceVisualizationFooter(
  mode: VoiceVisualizerProps['mode'],
  stopKeyLabel = 'voice key'
): string {
  if (mode === 'waiting') {
    return 'Waiting for realtime voice…'
  }
  if (mode === 'listening') {
    return `Listening · ${stopKeyLabel} to stop`
  }
  return mode === 'composing' ? 'Speaking…' : 'Solving…'
}

export function VoiceVisualizer({ columns, mode, stopKeyLabel, t, transcript, visualizer }: VoiceVisualizerProps) {
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => setFrame(current => current + 1), FRAME_MS)

    return () => clearInterval(timer)
  }, [])

  const panelWidth = Math.max(24, columns - 2)
  const innerWidth = panelWidth - 2
  const orbColumns = Math.min(ORB_COLUMNS, Math.max(6, innerWidth))
  const active = mode === 'listening'
  const renderWidth = visualizer === 'orb' ? orbColumns : innerWidth
  const lines = useMemo(
    () => renderVoiceVisualization(visualizer, renderWidth, frame, mode),
    [frame, mode, renderWidth, visualizer]
  )
  const footer = voiceVisualizationFooter(mode, stopKeyLabel)

  return (
    <Box borderColor={t.color.border} borderStyle="single" flexDirection="column" width={panelWidth}>
      {lines.map((line, index) => (
        <Text color={visualizer === 'orb' ? t.color.accent : active ? t.color.ok : t.color.warn} key={index}>
          {line}
        </Text>
      ))}
      {transcript?.text.trim() ? (
        <Text color={transcript.role === 'user' ? t.color.text : t.color.accent}>
          {transcript.role === 'user' ? 'you › ' : 'voice › '}
          {transcript.text.trim()}
        </Text>
      ) : null}
      <Text color={t.color.muted}>{footer}</Text>
    </Box>
  )
}
