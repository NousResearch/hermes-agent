import { SCROLLING_IDLE_MS, TYPING_IDLE_MS } from '../config/timing.js'

export type InteractionMode = 'idle' | 'scrolling' | 'typing'

type Timer = null | ReturnType<typeof setTimeout>

let mode: InteractionMode = 'idle'
let scrollingTimer: Timer = null
let typingTimer: Timer = null

const clear = (t: Timer): null => {
  if (t) {
    clearTimeout(t)
  }

  return null
}

export function getInteractionMode(): InteractionMode {
  return mode
}

export function markTyping(): void {
  mode = 'typing'
  typingTimer = clear(typingTimer)
  scrollingTimer = clear(scrollingTimer)
  typingTimer = setTimeout(() => {
    typingTimer = null
    mode = 'idle'
  }, TYPING_IDLE_MS)
}

export function markScrolling(): void {
  if (mode === 'typing') {
    return
  }

  mode = 'scrolling'
  scrollingTimer = clear(scrollingTimer)
  scrollingTimer = setTimeout(() => {
    scrollingTimer = null
    if (mode === 'scrolling') {
      mode = 'idle'
    }
  }, SCROLLING_IDLE_MS)
}

export function resetInteractionMode(): void {
  scrollingTimer = clear(scrollingTimer)
  typingTimer = clear(typingTimer)
  mode = 'idle'
}
