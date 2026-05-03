import type { TerminalSignals } from './terminalSignals.js'

export type TerminalFamily =
  | 'kitty'
  | 'wezterm'
  | 'ghostty'
  | 'iterm2'
  | 'windows-terminal'
  | 'vscode-xtermjs'
  | 'vte'
  | 'alacritty'
  | 'foot'
  | 'apple-terminal'
  | 'konsole'
  | 'xterm'
  | 'tmux'
  | 'screen'
  | 'unknown'

export type TransportKind = 'local' | 'ssh' | 'tmux' | 'screen' | 'nested'
export type KeyboardEncoding = 'legacy' | 'csi-u' | 'kitty' | 'modify-other-keys'
export type ClipboardWritePath = 'native' | 'osc52' | 'tmux-buffer' | 'screen-passthrough' | 'none'
export type ClipboardReadPath = 'native' | 'osc52-query' | 'none'

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

const MODERN_CSI_U_FAMILIES = new Set<TerminalFamily>(['wezterm', 'ghostty', 'iterm2', 'vscode-xtermjs'])

function isXtermLike(term?: string): boolean {
  return typeof term === 'string' && /^xterm(?:$|[-_])/i.test(term)
}

function detectExplicitTerminalFamily(s: TerminalSignals): TerminalFamily | undefined {
  const termProgram = s.env.TERM_PROGRAM
  const term = s.env.TERM ?? ''

  if (s.env.KITTY_WINDOW_ID || termProgram === 'kitty' || term.toLowerCase().includes('kitty')) return 'kitty'
  if (s.env.WEZTERM_PANE || term === 'wezterm') return 'wezterm'
  if (s.env.GHOSTTY_RESOURCES_DIR || term.toLowerCase().includes('ghostty')) return 'ghostty'
  if (s.env.ITERM_SESSION_ID || termProgram === 'iTerm.app' || s.env.LC_TERMINAL === 'iTerm2') return 'iterm2'
  if (s.env.WT_SESSION) return 'windows-terminal'
  if (termProgram === 'vscode') return 'vscode-xtermjs'
  if (s.env.VTE_VERSION) return 'vte'
  if (termProgram === 'Alacritty' || term === 'alacritty') return 'alacritty'
  if (termProgram === 'foot' || term === 'foot' || term === 'foot-direct') return 'foot'
  if (termProgram === 'Apple_Terminal' || s.env.TERM_SESSION_ID) return 'apple-terminal'
  if (s.env.KONSOLE_VERSION) return 'konsole'

  return undefined
}

function detectOuterTerminalHint(s: TerminalSignals): TerminalFamily | undefined {
  const explicit = detectExplicitTerminalFamily(s)

  if (explicit) {
    return explicit
  }

  if (!s.ssh.hasSshConnection && isXtermLike(s.env.TERM)) {
    return 'xterm'
  }

  return undefined
}

function deriveTransport(layers: TransportKind[]): TransportKind {
  if (layers.length === 0) {
    return 'local'
  }

  if (layers.length > 2) {
    return 'nested'
  }

  return layers[layers.length - 1]!
}

function deriveTerminalFamily(s: TerminalSignals, currentLayer: TransportKind | undefined): TerminalFamily {
  if (currentLayer === 'tmux') {
    return 'tmux'
  }

  if (currentLayer === 'screen') {
    return 'screen'
  }

  const explicit = detectExplicitTerminalFamily(s)

  if (explicit) {
    return explicit
  }

  if (!s.ssh.hasSshConnection && isXtermLike(s.env.TERM)) {
    return 'xterm'
  }

  return 'unknown'
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

function deriveKeyboardEncoding(family: TerminalFamily, probe: TerminalProbeResult): KeyboardEncoding {
  if (family === 'kitty') {
    return 'kitty'
  }

  if (MODERN_CSI_U_FAMILIES.has(family)) {
    return 'csi-u'
  }

  const kittyProbe = probe.kittyKeyboard ?? ((probe.kittyKeyboardFlags ?? 0) > 0)

  return kittyProbe ? 'kitty' : 'legacy'
}

function buildPasteShortcutShapes(platform: NodeJS.Platform, family: TerminalFamily): string[] {
  const shapes = platform === 'darwin' ? ['cmd+v', 'ctrl+shift+v'] : ['ctrl+shift+v', 'alt+v']

  if (family === 'kitty') {
    shapes.push('ctrl+v')
  }

  return shapes
}

function deriveClipboardWritePath(currentLayer: TransportKind | undefined, hasTty: boolean): ClipboardWritePath {
  switch (currentLayer) {
    case 'tmux':
      return 'tmux-buffer'
    case 'screen':
      return 'screen-passthrough'
    case 'ssh':
      return 'osc52'
    default:
      return hasTty ? 'native' : 'none'
  }
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

function buildDiagnostics(params: {
  s: TerminalSignals
  transport: TransportKind
  family: TerminalFamily
  currentLayer: TransportKind | undefined
  outerHint?: TerminalFamily
}): string[] {
  const { s, transport, family, currentLayer, outerHint } = params
  const diagnostics: string[] = []

  if (!s.env.TERM) {
    diagnostics.push('missing TERM')
  }

  if (s.ssh.hasSshConnection && (!s.isStdinTty || !s.isStdoutTty || !s.ssh.hasSshTty)) {
    diagnostics.push('SSH without proper TTY')
  }

  if (transport === 'nested') {
    diagnostics.push(`nested transport layers: ${[...(s.ssh.hasSshConnection ? ['ssh'] : []), ...(s.multiplexer.tmux ? ['tmux'] : []), ...(s.multiplexer.screen ? ['screen'] : [])].join(' > ')}`)
  }

  if ((family === 'tmux' || family === 'screen') && outerHint && outerHint !== family) {
    diagnostics.push(`outer terminal: ${outerHint}`)
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
  const layers: TransportKind[] = []

  if (s.ssh.hasSshConnection) {
    layers.push('ssh')
  }

  if (s.multiplexer.tmux) {
    layers.push('tmux')
  }

  if (s.multiplexer.screen) {
    layers.push('screen')
  }

  const currentLayer = layers.length > 0 ? layers[layers.length - 1] : undefined
  const transport = deriveTransport(layers)
  const terminalFamily = deriveTerminalFamily(s, currentLayer)
  const terminalVersion = deriveTerminalVersion(s, probe)
  const hasTty = s.isStdinTty && s.isStdoutTty
  const localClipboardCapable = transport === 'local' && hasTty
  const outerHint = currentLayer === 'tmux' || currentLayer === 'screen' ? detectOuterTerminalHint(s) : undefined

  return {
    transport,
    layers,
    terminalFamily,
    terminalVersion,
    keyboard: {
      encoding: deriveKeyboardEncoding(terminalFamily, probe),
      pasteShortcutShapes: buildPasteShortcutShapes(s.platform, terminalFamily)
    },
    paste: {
      bracketedPaste: probe.bracketedPaste ?? true
    },
    copy: {
      writePath: deriveClipboardWritePath(currentLayer, hasTty),
      readPath: deriveClipboardReadPath(transport, hasTty, probe),
      copyOnSelect: localClipboardCapable && s.platform === 'darwin'
    },
    mouse: {
      tracking: false,
      shiftDragHint: currentLayer === 'tmux' || currentLayer === 'screen'
    },
    diagnostics: buildDiagnostics({ s, transport, family: terminalFamily, currentLayer, outerHint })
  }
}
