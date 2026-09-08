import type { TelegramOnboardingStartResponse } from './api'

export function isTerminalTelegramOnboardingError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null || !('status' in error)) return false
  return error.status === 404 || error.status === 410
}

export function readTelegramSetup(key: string): TelegramOnboardingStartResponse | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) ?? 'null') as TelegramOnboardingStartResponse | null
    return value &&
      typeof value.pairing_id === 'string' &&
      typeof value.deep_link === 'string' &&
      typeof value.expires_at === 'string'
      ? value
      : null
  } catch {
    return null
  }
}

export function saveTelegramSetup(key: string, value: TelegramOnboardingStartResponse | null): void {
  // Browser storage can be disabled; it must not prevent pairing in this tab.
  try {
    if (value) sessionStorage.setItem(key, JSON.stringify(value))
    else sessionStorage.removeItem(key)
  } catch {
    /* refresh recovery unavailable */
  }
}
