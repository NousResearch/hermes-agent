export const CONFIRM_WINDOW_MS = 30_000

export interface DestructiveGate {
  armed: () => null | string
  request: (key: string, now?: number) => boolean
  reset: () => void
}

export const createDestructiveGate = (windowMs = CONFIRM_WINDOW_MS): DestructiveGate => {
  let pending: { at: number; key: string } | null = null

  const isFresh = (now: number) => pending != null && now - pending.at < windowMs

  return {
    armed: () => (pending != null && isFresh(Date.now()) ? pending.key : null),
    request: (key, now = Date.now()) => {
      const confirmed = pending?.key === key && isFresh(now)

      pending = confirmed ? null : { at: now, key }

      return confirmed
    },
    reset: () => {
      pending = null
    }
  }
}
