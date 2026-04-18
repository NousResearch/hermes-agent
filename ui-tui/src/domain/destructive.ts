export const CONFIRM_WINDOW_MS = 3_000

export interface DestructiveGate {
  request: (key: string, now?: number) => boolean
  reset: () => void
}

export const createDestructiveGate = (windowMs = CONFIRM_WINDOW_MS): DestructiveGate => {
  let pending: { at: number; key: string } | null = null

  return {
    request: (key, now = Date.now()) => {
      const confirmed = pending?.key === key && now - pending.at < windowMs

      pending = confirmed ? null : { at: now, key }

      return confirmed
    },
    reset: () => {
      pending = null
    }
  }
}
