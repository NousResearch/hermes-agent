export type NotificationTerminal = 'ghostty' | 'iterm2' | 'kitty' | 'warp'

/** What raised the notification: a finished turn or a prompt waiting on the user. */
export type AttentionKind = 'complete' | 'prompt'

type TerminalEnvironment = Readonly<Record<string, string | undefined>>

// Last Warp release per channel that advertised the CLI-agent protocol but could not render
// structured payloads; mirrors hermes_cli/terminal_notify.py (string comparison, like Warp's script).
const WARP_LAST_BROKEN: Readonly<Record<string, string>> = {
  preview: 'v0.2026.03.25.08.24.preview_05',
  stable: 'v0.2026.03.25.08.24.stable_05'
}


function warpSupported(env: TerminalEnvironment): boolean {
  const client = env.WARP_CLIENT_VERSION ?? ''

  if (env.TERM_PROGRAM !== 'WarpTerminal' || !env.WARP_CLI_AGENT_PROTOCOL_VERSION || !client) {
    return false
  }

  return !Object.entries(WARP_LAST_BROKEN).some(([channel, last]) => client.includes(channel) && client <= last)
}

export function notificationTerminalForEnv(env: TerminalEnvironment): NotificationTerminal | null {
  const term = env.TERM?.toLowerCase() ?? ''
  const termProgram = env.TERM_PROGRAM?.toLowerCase() ?? ''
  const lcTerminal = env.LC_TERMINAL?.toLowerCase() ?? ''

  if (warpSupported(env)) {
    return 'warp'
  }

  if (env.GHOSTTY_BIN_DIR || env.GHOSTTY_RESOURCES_DIR || term.includes('ghostty') || termProgram.includes('ghostty')) {
    return 'ghostty'
  }

  if (env.KITTY_WINDOW_ID || term.includes('kitty') || termProgram.includes('kitty')) {
    return 'kitty'
  }

  if (
    env.WEZTERM_EXECUTABLE ||
    env.WEZTERM_PANE ||
    termProgram.includes('iterm') ||
    lcTerminal.includes('iterm') ||
    termProgram.includes('wezterm')
  ) {
    return 'iterm2'
  }

  return null
}

/**
 * Body of Warp's OSC 777 `warp://cli-agent` notification (same shape the classic CLI sends):
 * `stop` for a finished turn, `permission_request` for a prompt waiting on the user.
 */
export function warpAgentPayload(
  kind: AttentionKind,
  detail: string,
  opts: { cwd: string; protocolVersion?: string; sessionId?: string }
): string {
  const advertised = Number.parseInt(opts.protocolVersion ?? '1', 10)
  const project = opts.cwd.split(/[\\/]/).filter(Boolean).at(-1) ?? ''

  const text = [...detail]
    .filter(ch => ch >= ' ' && ch !== '\x7f')
    .join('')
    .slice(0, 200)

  return JSON.stringify({
    v: Math.min(Number.isNaN(advertised) ? 1 : advertised, 1),
    agent: 'hermes',
    event: kind === 'prompt' ? 'permission_request' : 'stop',
    session_id: opts.sessionId ?? '',
    cwd: opts.cwd,
    project,
    [kind === 'prompt' ? 'summary' : 'response']: text
  })
}
