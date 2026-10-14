import { atom } from 'nanostores'

import type { AnyGatewayEvent } from '../gatewayTypes.js'

// Live STT text so far while the user is still speaking (`voice.partial`, stt.streaming). The
// composer shows it as its placeholder until the capture settles.
export const $voicePartial = atom('')

interface VoiceCaptureSetters {
  setProcessing: (value: boolean) => void
  setRecording: (value: boolean) => void
}

/** Capture-state events (`voice.partial`, `voice.status`); true when fully handled here. A
 *  `voice.transcript` only clears the partial and stays with the main handler. */
export function handleVoiceCapture(ev: AnyGatewayEvent, voice: VoiceCaptureSetters): boolean {
  if (ev.type === 'voice.partial') {
    $voicePartial.set(String(ev.payload?.text ?? ''))

    return true
  }

  if (ev.type === 'voice.transcript') {
    $voicePartial.set('')

    return false
  }

  if (ev.type !== 'voice.status') {
    return false
  }

  // The continuous VAD loop reports listening / transcribing / idle so the status bar needs no polling.
  const state = String(ev.payload?.state ?? '')
  voice.setRecording(state === 'listening')
  voice.setProcessing(state === 'transcribing')

  if (state !== 'listening' && state !== 'transcribing') {
    $voicePartial.set('')
  }

  return true
}
