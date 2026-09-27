/** Build the ``session.create`` params for the sidecar session.
 *
 * Extracted from the effect below so the invariant — close_on_disconnect
 * is set, source is "tool", and the profile is forwarded when present —
 * can be tested without reading component source text. See
 * ``chat-sidebar-session-params.test.ts``.
 */
export function sidecarSessionCreateParams(profile?: string): Record<string, unknown> {
  return {
    close_on_disconnect: true,
    source: 'tool',
    ...(profile ? { profile } : {})
  }
}
