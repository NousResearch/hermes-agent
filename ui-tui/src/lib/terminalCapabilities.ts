import { findMultiplexer, findProfile, type TerminalProfile } from './terminalRegistry.js'
import type { TerminalSignals } from './terminalSignals.js'

export type TerminalFamily = string
export type TransportKind = string
export type ClipboardWritePath = string
export type ClipboardReadPath = 'native' | 'none' | 'osc52-query'
export type KeyboardEncoding = 'csi-u' | 'kitty' | 'legacy' | 'modify-other-keys'

export type TerminalProbeResult = {
  bracketedPaste?: boolean
  focusReporting?: boolean
  kittyKeyboard?: boolean
  kittyKeyboardFlags?: number
  osc52Read?: boolean
  osc52ReadSupported?: boolean
  osc52WriteHint?: boolean
  synchronizedOutput?: boolean
  xtversion?: string
  xtversionName?: string
}

export type TerminalCapabilities = {
  copy: {
    copyOnSelect: boolean
    readPath: ClipboardReadPath
    writePath: ClipboardWritePath
  }
  diagnostics: string[]
  keyboard: {
    copyShortcutShapes: string[]
    encoding: KeyboardEncoding
    pasteShortcutShapes: string[]
  }
  layers: TransportKind[]
  mouse: {
    shiftDragHint: boolean
    tracking: boolean
  }
  paste: {
    bracketedPaste: boolean
  }
  terminalFamily: TerminalFamily
  terminalVersion?: string
  transport: TransportKind
}

const DEFAULT_COPY_SHORTCUTS = ['ctrl+shift+c'] as const
const SUPER_COPY_SHORTCUTS = ['super+c', 'super+shift+c', 'ctrl+shift+c'] as const
const SUPER_PASTE_SHORTCUTS = ['super+v', 'super+shift+v'] as const

const normalizeShortcutShape = (shape: string): string => shape.replace(/^cmd\+/, 'super+')

function hasMacTerminalHint(s: TerminalSignals): boolean {
  return Boolean(
    s.platform === 'darwin' ||
      s.env.ITERM_SESSION_ID ||
      s.env.LC_TERMINAL === 'iTerm2' ||
      s.env.TERM_PROGRAM === 'iTerm.app' ||
      s.env.TERM_PROGRAM === 'Apple_Terminal' ||
      s.env.TERM_SESSION_ID
  )
}

function hasRemoteShell(s: TerminalSignals): boolean {
  return s.ssh.hasSshConnection || s.ssh.hasSshClient || s.ssh.hasSshTty
}

function muxEnv(s: TerminalSignals): Record<string, string | undefined> {
  return {
    CY: s.multiplexer.cy ? 'default:1' : undefined,
    STY: s.multiplexer.screen ? '1' : undefined,
    TMUX: s.multiplexer.tmux ? '1' : undefined,
    ZELLIJ: s.multiplexer.zellij ? '0' : undefined
  }
}

function signalsEnv(s: TerminalSignals): Record<string, string | undefined> {
  return {
    COLORTERM: s.env.COLORTERM,
    GHOSTTY_RESOURCES_DIR: s.env.GHOSTTY_RESOURCES_DIR,
    ITERM_SESSION_ID: s.env.ITERM_SESSION_ID,
    KITTY_WINDOW_ID: s.env.KITTY_WINDOW_ID,
    KONSOLE_VERSION: s.env.KONSOLE_VERSION,
    LC_TERMINAL: s.env.LC_TERMINAL,
    TERM: s.env.TERM,
    TERM_PROGRAM: s.env.TERM_PROGRAM,
    TERM_PROGRAM_VERSION: s.env.TERM_PROGRAM_VERSION,
    TERM_SESSION_ID: s.env.TERM_SESSION_ID,
    VTE_VERSION: s.env.VTE_VERSION,
    WEZTERM_PANE: s.env.WEZTERM_PANE,
    WT_SESSION: s.env.WT_SESSION
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

function deriveClipboardReadPath(transport: TransportKind, hasTty: boolean, probe: TerminalProbeResult): ClipboardReadPath {
  if (transport === 'local' && hasTty) {
    return 'native'
  }

  if (probe.osc52ReadSupported ?? probe.osc52Read) {
    return 'osc52-query'
  }

  return 'none'
}

function deriveTransport(layers: TransportKind[]): TransportKind {
  if (layers.length === 0) {return 'local'}

  if (layers.length > 2) {return 'nested'}

  return layers[layers.length - 1]!
}

function buildDiagnostics(params: {
  currentLayer: TransportKind | undefined
  outerProfile: TerminalProfile
  s: TerminalSignals
  transport: TransportKind
}): string[] {
  const { currentLayer, outerProfile, s, transport } = params
  const diagnostics: string[] = []

  if (!s.env.TERM) {
    diagnostics.push('missing TERM')
  }

  if (s.ssh.hasSshConnection && (!s.isStdinTty || !s.isStdoutTty || !s.ssh.hasSshTty)) {
    diagnostics.push('SSH without proper TTY')
  }

  if (transport === 'nested') {
    diagnostics.push(`nested transport layers: ${layersFromSignals(s).join(' > ')}`)
  }

  if (currentLayer && currentLayer !== 'ssh' && outerProfile.id !== currentLayer && outerProfile.id !== 'unknown') {
    diagnostics.push(`outer terminal: ${outerProfile.id}`)
  }

  if (s.platform !== 'win32' && s.env.WT_SESSION) {
    diagnostics.push('WT_SESSION on non-Windows platform')
  }

  if (s.platform !== 'darwin' && hasMacTerminalHint(s)) {
    diagnostics.push('mac terminal hint on non-mac platform')
  }

  if (currentLayer === 'ssh' && (s.ssh.hasSshClient || s.ssh.hasSshTty) && !s.ssh.hasSshConnection) {
    diagnostics.push('partial SSH signals without SSH_CONNECTION')
  }

  return diagnostics
}

function layersFromSignals(s: TerminalSignals): TransportKind[] {
  const layers: TransportKind[] = []

  if (hasRemoteShell(s)) {
    layers.push('ssh')
  }

  const muxProfile = findMultiplexer(muxEnv(s))

  if (muxProfile) {
    layers.push(muxProfile.id)
  }

  return layers
}

function copyShortcutShapes(s: TerminalSignals): string[] {
  return [...(hasMacTerminalHint(s) || hasRemoteShell(s) ? SUPER_COPY_SHORTCUTS : DEFAULT_COPY_SHORTCUTS)]
}

function pasteShortcutShapes(profile: TerminalProfile, isDarwin: boolean, s: TerminalSignals): string[] {
  const shapes = isDarwin ? profile.capabilities.pasteShortcuts.darwin : profile.capabilities.pasteShortcuts.default
  const normalized = shapes.map(normalizeShortcutShape)

  if (hasMacTerminalHint(s) || hasRemoteShell(s)) {
    for (const shape of SUPER_PASTE_SHORTCUTS) {
      if (!normalized.includes(shape)) {
        normalized.unshift(shape)
      }
    }
  }

  return normalized
}

export function deriveTerminalCapabilities(s: TerminalSignals, probe: TerminalProbeResult = {}): TerminalCapabilities {
  const layers = layersFromSignals(s)
  const currentLayer = layers.length > 0 ? layers[layers.length - 1] : undefined
  const transport = deriveTransport(layers)
  const outerProfile = findProfile(signalsEnv(s))
  const muxProfile = currentLayer && currentLayer !== 'ssh' ? findMultiplexer(muxEnv(s)) : undefined
  const profile = muxProfile ?? outerProfile
  const hasTty = s.isStdinTty && s.isStdoutTty
  const isMuxLayer = currentLayer !== undefined && currentLayer !== 'ssh' && currentLayer !== 'local'

  const writePath: ClipboardWritePath = isMuxLayer
    ? profile.capabilities.clipboard.writePath
    : layers.includes('ssh')
      ? 'osc52'
      : hasTty
        ? 'native'
        : 'none'

  const isDarwin = s.platform === 'darwin'
  const localClipboardCapable = transport === 'local' && hasTty
  const kitty = probe.kittyKeyboard ?? (probe.kittyKeyboardFlags ?? 0) > 0
  const terminalFamily = isMuxLayer ? currentLayer : outerProfile.id

  return {
    copy: {
      copyOnSelect: profile.capabilities.copyOnSelect && isDarwin && localClipboardCapable,
      readPath: deriveClipboardReadPath(transport, hasTty, probe),
      writePath
    },
    diagnostics: buildDiagnostics({ currentLayer, outerProfile, s, transport }),
    keyboard: {
      copyShortcutShapes: copyShortcutShapes(s),
      encoding: kitty ? 'kitty' : profile.capabilities.keyboard,
      pasteShortcutShapes: pasteShortcutShapes(profile, isDarwin, s)
    },
    layers,
    mouse: {
      shiftDragHint: profile.capabilities.shiftDragHint,
      tracking: false
    },
    paste: {
      bracketedPaste: probe.bracketedPaste ?? true
    },
    terminalFamily,
    terminalVersion: deriveTerminalVersion(s, probe),
    transport
  }
}
