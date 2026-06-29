export interface VoiceStatusLabelState {
  enabled: boolean
  processing: boolean
  recording: boolean
  tts: boolean
}

export function voiceStatusLabel({ enabled, processing, recording, tts }: VoiceStatusLabelState): string {
  if (recording) {
    return '● REC'
  }

  if (processing) {
    return '◉ STT'
  }

  if (enabled) {
    return `voice on${tts ? ' [tts]' : ''}`
  }

  // Hide the idle voice-off segment entirely while preserving active and TTS indicators.
  if (tts) {
    return 'voice [tts]'
  }

  return ''
}
