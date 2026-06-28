import { describe, expect, it } from 'vitest'

import { voiceStatusLabel } from '../app/voiceStatusLabel.js'

describe('voiceStatusLabel', () => {
  it('preserves active voice indicators and hides idle voice-off', () => {
    expect(voiceStatusLabel({ enabled: false, processing: false, recording: true, tts: false })).toBe('● REC')
    expect(voiceStatusLabel({ enabled: false, processing: true, recording: false, tts: false })).toBe('◉ STT')
    expect(voiceStatusLabel({ enabled: true, processing: false, recording: false, tts: false })).toBe('voice on')
    expect(voiceStatusLabel({ enabled: true, processing: false, recording: false, tts: true })).toBe('voice on [tts]')
    expect(voiceStatusLabel({ enabled: false, processing: false, recording: false, tts: true })).toBe('voice [tts]')
    expect(voiceStatusLabel({ enabled: false, processing: false, recording: false, tts: false })).toBe('')
  })
})
