const PTY_ATTACH_TOKEN_PREFIX = 'hermes.pty.token.chat'
const CHAT_CHANNEL_PREFIX = 'hermes.pty.channel.chat'

function browserStorage(): Storage | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    return window.localStorage
  } catch {
    return null
  }
}

function randomHex(bytes = 16): string {
  const cryptoApi = typeof crypto !== 'undefined' ? crypto : null

  if (cryptoApi?.getRandomValues) {
    const values = new Uint8Array(bytes)
    cryptoApi.getRandomValues(values)

    return Array.from(values, b => b.toString(16).padStart(2, '0')).join('')
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function randomChannelId(): string {
  const cryptoApi = typeof crypto !== 'undefined' ? crypto : null

  if (cryptoApi && 'randomUUID' in cryptoApi) {
    return `chat-${cryptoApi.randomUUID()}`
  }

  return `chat-${randomHex(12)}`
}

function scopedKey(prefix: string, scope: string): string {
  return `${prefix}.${scope || 'default'}`
}

function storedIdentityValue(key: string, create: () => string, rotate = false): string {
  const storage = browserStorage()

  if (!rotate && storage) {
    try {
      const existing = storage.getItem(key)

      if (existing) {
        return existing
      }
    } catch {
      // Storage can throw in private mode / blocked third-party contexts.
    }
  }

  const next = create()

  if (storage) {
    try {
      storage.setItem(key, next)
    } catch {
      // Best-effort only; the in-memory caller still gets a usable identity.
    }
  }

  return next
}

/**
 * Scope browser chat identity to the stored session + profile, not just the tab.
 *
 * The dashboard PTY registry keys live processes by `?attach=`. Reusing one
 * global token across `/chat?resume=A` and `/chat?resume=B` can accidentally
 * reattach B's route to A's still-running PTY. Conversely, changing identity on
 * every refresh loses the live PTY/event stream. This scope keeps refreshes
 * stable while isolating different stored sessions and profiles.
 */
export function chatIdentityScope(
  resumeSessionId: string | null | undefined,
  profile: string | null | undefined
): string {
  const session = (resumeSessionId ?? '').trim() || 'new'
  const profileKey = (profile ?? '').trim() || 'default'

  return `${encodeURIComponent(profileKey)}:${encodeURIComponent(session)}`
}

export function getPtyAttachToken(scope: string, rotate = false): string {
  return storedIdentityValue(scopedKey(PTY_ATTACH_TOKEN_PREFIX, scope), () => randomHex(), rotate)
}

export function getChatChannelId(scope: string, rotate = false): string {
  return storedIdentityValue(scopedKey(CHAT_CHANNEL_PREFIX, scope), randomChannelId, rotate)
}
