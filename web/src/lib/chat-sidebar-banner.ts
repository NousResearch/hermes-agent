/**
 * Copy for the ChatSidebar banner: the sidecar (JSON-RPC over /api/ws) and
 * the credential probe the gateway sends in `session.info`.
 *
 * Pure helpers so the wording is testable without React.
 */

/** The side panel could not connect; the chat itself is unaffected. */
export const SIDECAR_DISCONNECTED_MESSAGE =
  'The chat side panel (model and tool activity) could not connect. Chat still works. Click Reconnect side panel to try again.'

/** Transport-level texts the shared JSON-RPC client can throw. All mean "sidecar down". */
const SIDECAR_TRANSPORT_ERRORS = [
  'WebSocket connection failed',
  'WebSocket closed',
  'gateway not connected',
  'Session token not available',
  'heartbeat acknowledgement timed out'
]

/**
 * Map a sidecar connect/RPC error to the banner sentence. Transport jargon
 * collapses to SIDECAR_DISCONNECTED_MESSAGE; anything else (a real gateway
 * error payload, e.g. a bad profile name) is shown as-is.
 */
export function sidecarErrorMessage(raw: string): string {
  return SIDECAR_TRANSPORT_ERRORS.some(needle => raw.includes(needle)) ? SIDECAR_DISCONNECTED_MESSAGE : raw
}

/** Parsed credential warning from the gateway's `_probe_credentials`. */
export interface CredentialWarning {
  provider: string | null
  message: string
}

const NO_KEY_RE = /No API key configured for provider '([^']+)'/

/**
 * Rewrite the gateway's "No API key configured for provider 'x'. First message
 * will fail." into a sentence that names the fix. Unknown warnings pass
 * through unchanged (provider null) so nothing is lost.
 */
export function credentialWarning(raw: string | undefined | null): CredentialWarning | null {
  if (!raw) return null
  const match = raw.match(NO_KEY_RE)
  if (!match) return { provider: null, message: raw }
  const provider = match[1]
  return {
    provider,
    message: `No API key set for ${provider}, so messages will fail. Add a key under Keys, or pick a different provider.`
  }
}
