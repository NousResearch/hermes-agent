import type { TerminalSignals } from './terminalSignals.js'
import { findProfile, findMultiplexer, type TerminalProfile, UNKNOWN_PROFILE } from './terminalRegistry.js'

// Exported for backward compatibility — registry-driven, not hardcoded
export type TerminalFamily = string
export type TransportKind = string
export type ClipboardWritePath = string
export type ClipboardReadPath = 'native' | 'osc52-query' | 'none'
export type KeyboardEncoding = 'legacy' | 'csi-u' | 'kitty' | 'modify-other-keys'

export type TerminalProbeResult = {
  xtversion?: string
  xtversionName?: string
  bracketedPaste?: boolean
  focusReporting?: boolean
  synchronizedOutput?: boolean
  kittyKeyboard?: boolean
  kittyKeyboardFlags?: number
  osc52ReadSupported?: boolean
  osc52Read?: boolean
  osc52WriteHint?: boolean
}

export type TerminalCapabilities = {
  transport: TransportKind
  layers: TransportKind[]
  terminalFamily: TerminalFamily
  terminalVersion?: string
  keyboard: {
    encoding: KeyboardEncoding
    pasteShortcutShapes: string[]
  }
  paste: {
    bracketedPaste: boolean
  }
  copy: {
    writePath: ClipboardWritePath
    readPath: ClipboardReadPath
    copyOnSelect: boolean
  }
  mouse: {
    tracking: boolean
    shiftDragHint: boolean
  }
  diagnostics: string[]
}

// ── Registry-backed capability derivation ──

/** Extract an env-like record from TerminalSignals for registry matching. */
function signalsEnv(s: TerminalSignals): Record<string, string | undefined> {
  return {
    TERM: s.env.TERM,
    TERM_PROGRAM: s.env.TERM_PROGRAM,
    TERM_PROGRAM_VERSION: s.env.TERM_PROGRAM_VERSION,
    COLORTERM: s.env.COLORTERM,
    VTE_VERSION: s.env.VTE_VERSION,
    WT_SESSION: s.env.WT_SESSION,
    KITTY_WINDOW_ID: s.env.KITTY_WINDOW_ID,
    WEZTERM_PANE: s.env.WEZTERM_PANE,
    GHOSTTY_RESOURCES_DIR: s.env.GHOSTTY_RESOURCES_DIR,
    KONSOLE_VERSION: s.env.KONSOLE_VERSION,
    ITERM_SESSION_ID: s.env.ITERM_SESSION_ID,
    LC_TERMINAL: s.env.LC_TERMINAL,
    TERM_SESSION_ID: s.env.TERM_SESSION_ID,
    TMUX: s.multiplexer.tmux ? '1' : undefined,
    STY: s.multiplexer.screen ? '1' : undefined,
    ZELLIJ: s.multiplexer.zellij ? '0' : undefined,
    CY: s.multiplexer.cy ? 'default:1' : undefined,
  }
}

function deriveTerminalVersion(s: TerminalSignals, probe: TerminalProbeResult): string | undefined {
  return (
    probe.xtversion ??
    probe.xtversionName ??
    s.env.TERM_PROGRAM_VERSION ??
    s.env.VTE_VERSION ??
    s.env.KONSOLE_VERSION ??
    undefined
  )
}

function deriveClipboardReadPath(
  transport: TransportKind,
  hasTty: boolean,
  probe: TerminalProbeResult
): ClipboardReadPath {
  if (transport === 'local' && hasTty) {
    return 'native'
  }
  if (probe.osc52ReadSupported ?? probe.osc52Read) {
    return 'osc52-query'
  }
  return 'none'
}

function deriveTransport(layers: TransportKind[]): TransportKind {
  if (layers.length === 0) return 'local'
  if (layers.length > 2) return 'nested'
  return layers[layers.length - 1]!
}

function buildDiagnostics(params: {
  s: TerminalSignals
  transport: TransportKind
  currentLayer: TransportKind | undefined
  outerProfile?: TerminalProfile
}): string[] {
  const { s, transport, currentLayer, outerProfile } = params
  const diagnostics: string[] = []

  if (!s.env.TERM) {
    diagnostics.push('missing TERM')
  }

  if (s.ssh.hasSshConnection && (!s.isStdinTty || !s.isStdoutTty || !s.ssh.hasSshTty)) {
    diagnostics.push('SSH without proper TTY')
  }

  if (transport === 'nested') {
    diagnostics.push(
      `nested transport layers: ${[
        ...(s.ssh.hasSshConnection ? ['ssh'] : []),
        ...(s.multiplexer.tmux ? ['tmux'] : []),
        ...(s.multiplexer.screen ? ['screen'] : []),
        ...(s.multiplexer.zellij ? ['zellij'] : []),
        ...(s.multiplexer.cy ? ['cy'] : []),
      ].join(' > ')}`
    )
  }

  if (currentLayer && currentLayer !== 'ssh' && outerProfile && outerProfile.id !== currentLayer) {
    diagnostics.push(`outer terminal: ${outerProfile.id}`)
  }

  if (s.platform !== 'win32' && s.env.WT_SESSION) {
    diagnostics.push('WT_SESSION on non-Windows platform')
  }

  const hasMacTerminalHint =
    s.env.ITERM_SESSION_ID ||
    s.env.LC_TERMINAL === 'iTerm2' ||
    s.env.TERM_PROGRAM === 'iTerm.app' ||
    s.env.TERM_PROGRAM === 'Apple_Terminal' ||
    !!s.env.TERM_SESSION_ID

  if (s.platform !== 'darwin' && hasMacTerminalHint) {
    diagnostics.push('mac terminal hint on non-mac platform')
  }

  if (currentLayer === 'ssh' && (s.ssh.hasSshClient || s.ssh.hasSshTty) && !s.ssh.hasSshConnection) {
    diagnostics.push('partial SSH signals without SSH_CONNECTION')
  }

  return diagnostics
}

export function deriveTerminalCapabilities(
  s: TerminalSignals,
  probe: TerminalProbeResult = {}
): TerminalCapabilities {
  // Build transport layers from signals
  const layers: TransportKind[] = []

  if (s.ssh.hasSshConnection || s.ssh.hasSshClient || s.ssh.hasSshTty) {
    layers.push('ssh')
  }

  // Add multiplexer layers using registry
  const muxEnv: Record<string, string | undefined> = {}
  if (s.multiplexer.tmux) muxEnv['TMUX'] = '1'
  if (s.multiplexer.screen) muxEnv['STY'] = '1'
  if (s.multiplexer.zellij) muxEnv['ZELLIJ'] = '0'
  if (s.multiplexer.cy) muxEnv['CY'] = 'default:1'

  // Push multiplexer IDs from registry as layers
  for (const [key] of Object.entries(muxEnv)) {
    const mux = findMultiplexer({ [key]: muxEnv[key] })
    if (mux) layers.push(mux.id)
  }

  const currentLayer = layers.length > 0 ? layers[layers.length - 1] : undefined
  const transport = deriveTransport(layers)

  // Find profiles: outer terminal + current layer (for multiplexer capabilities)
  const env = signalsEnv(s)
  for (const key of ['TMUX', 'STY', 'ZELLIJ', 'CY']) {
    delete env[key]
  }
  const outerProfile = findProfile(env)
  // For capability lookup, use multiplexer profile when in a mux layer
  const muxProfile = currentLayer && currentLayer !== 'ssh' && currentLayer !== 'local'
    ? findMultiplexer(Object.fromEntries(
        Object.entries({
          TMUX: s.multiplexer.tmux ? '1' : undefined,
          STY: s.multiplexer.screen ? '1' : undefined,
          ZELLIJ: s.multiplexer.zellij ? '0' : undefined,
          CY: s.multiplexer.cy ? 'default:1' : undefined,
        }).filter(([, v]) => v !== undefined)
      ))
    : undefined
  const profile = muxProfile ?? outerProfile

  const hasTty = s.isStdinTty && s.isStdoutTty
  const localClipboardCapable = transport === 'local' && hasTty

  // Clipboard: multiplexer layer takes priority, then SSH, then native/none
  const isMuxLayer = currentLayer && currentLayer !== 'ssh' && currentLayer !== 'local'
  const writePath: ClipboardWritePath =
    isMuxLayer ? profile.capabilities.clipboard.writePath :
    layers.includes('ssh') ? 'osc52' :
    hasTty ? 'native' : 'none'

  const isDarwin = s.platform === 'darwin'

  // Terminal family: multiplexer layer takes priority, SSH context skips env detection
  let terminalFamily: string
  if (currentLayer && currentLayer !== 'ssh' && currentLayer !== 'local') {
    terminalFamily = currentLayer
  } else if (layers.includes('ssh')) {
    terminalFamily = 'unknown'
  } else {
    terminalFamily = profile.id
  }

  return {
    transport,
    layers,
    terminalFamily,
    terminalVersion: deriveTerminalVersion(s, probe),
    keyboard: {
      encoding: probe.kittyKeyboard ?? ((probe.kittyKeyboardFlags ?? 0) > 0) ? 'kitty' : profile.capabilities.keyboard,
      pasteShortcutShapes: isDarwin
        ? profile.capabilities.pasteShortcuts.darwin
        : profile.capabilities.pasteShortcuts.default,
    },
    paste: {
      bracketedPaste: probe.bracketedPaste ?? true,
    },
    copy: {
      writePath,
      readPath: deriveClipboardReadPath(transport, hasTty, probe),
      copyOnSelect: profile.capabilities.copyOnSelect && isDarwin && localClipboardCapable,
    },
    mouse: {
      tracking: false,
      shiftDragHint: profile.capabilities.shiftDragHint,
    },
    diagnostics: buildDiagnostics({ s, transport, currentLayer, outerProfile }),
  }
}
