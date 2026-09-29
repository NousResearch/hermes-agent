import type { ModelOptionsResult } from '@hermes/shared/gateway-events'

const CACHE_TTL_MS = 10_000

interface CacheEntry {
  at: number
  result: ModelOptionsResult
}

const cache = new Map<string, CacheEntry>()
const keyFor = (sessionId?: null | string) => sessionId || '__no_session__'

export const rememberModelOptions = (
  sessionId: null | string | undefined,
  result: ModelOptionsResult,
  now = Date.now()
): void => {
  cache.set(keyFor(sessionId), { at: now, result })
}

export const cachedModelOptions = (
  sessionId: null | string | undefined,
  now = Date.now()
): ModelOptionsResult | null => {
  const key = keyFor(sessionId)
  const entry = cache.get(key)

  if (!entry || now - entry.at > CACHE_TTL_MS) {
    cache.delete(key)
    return null
  }

  return entry.result
}

export const invalidateModelOptions = (sessionId?: null | string): void => {
  cache.delete(keyFor(sessionId))
}

export const resetModelOptionsCacheForTests = (): void => {
  cache.clear()
}
