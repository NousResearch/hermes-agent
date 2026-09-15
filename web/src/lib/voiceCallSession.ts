/** Params for the voice card's dedicated session (not the embedded TUI's). */
export function voiceCallSessionCreateParams(
  profile?: string | null,
): Record<string, unknown> {
  // No explicit `source`: the gateway resolves the dashboard's own platform
  // (the same as its TUI sessions). A `"tool"` source would file the session
  // under the hidden sub-agent bucket, so the consult answers could never be
  // opened or resumed after the call.
  return {
    close_on_disconnect: true,
    ...(profile ? { profile } : {}),
  };
}
