import type { TerminalEnvironment } from '../app/terminalEnvironmentStore.js'
import type { TerminalCapabilities } from './terminalCapabilities.js'
import type { TerminalSignals } from './terminalSignals.js'

const UNKNOWN = 'unknown'
const NOT_DETECTED = 'not detected'

const formatUnknown = (value: string | undefined | null): string => value?.trim() ? value.trim() : UNKNOWN
const formatDetected = (value: string | undefined | null): string => value?.trim() ? value.trim() : NOT_DETECTED
const formatWarning = (line: string): string => (line.startsWith('⚠') ? line : `⚠ ${line}`)

const summarizeTransport = (signals: TerminalSignals): string => {
  if (signals.ssh.hasSshConnection || signals.ssh.hasSshClient || signals.ssh.hasSshTty) {
    return 'ssh'
  }

  if (signals.multiplexer.tmux) {
    return 'tmux'
  }

  if (signals.multiplexer.screen) {
    return 'screen'
  }

  return 'local'
}

const summarizeLayers = (signals: TerminalSignals): string[] => {
  const layers: string[] = []

  if (signals.ssh.hasSshConnection || signals.ssh.hasSshClient || signals.ssh.hasSshTty) {
    layers.push('ssh')
  }

  if (signals.multiplexer.tmux) {
    layers.push('tmux')
  }

  if (signals.multiplexer.screen) {
    layers.push('screen')
  }

  return layers
}

const summarizeTerminal = (signals: TerminalSignals, capabilities: TerminalCapabilities): string => {
  const terminal = formatUnknown(capabilities.terminalFamily)
  const bridge = signals.multiplexer.tmux ? 'tmux' : signals.multiplexer.screen ? 'screen' : ''

  return bridge ? `${terminal} (via ${bridge})` : terminal
}

const summarizeShell = (signals: TerminalSignals): string => {
  const family = formatUnknown(signals.shell.family)
  const version = signals.shell.version?.trim()

  return version ? `${family} ${version}` : family
}

const summarizeKeyboard = (capabilities: TerminalCapabilities): string => formatUnknown(capabilities.keyboard.encoding)

const summarizeBracketedPaste = (capabilities: TerminalCapabilities): string => {
  const value = capabilities.paste?.bracketedPaste

  if (value === true) {
    return 'enabled'
  }

  if (value === false) {
    return 'disabled'
  }

  return UNKNOWN
}

const summarizeCopyOnSelect = (capabilities: TerminalCapabilities): string => {
  const active = capabilities.copy?.copyOnSelect && !capabilities.mouse?.shiftDragHint

  if (capabilities.copy?.copyOnSelect === undefined) {
    return UNKNOWN
  }

  return active ? 'active' : 'inactive'
}

const envRow = (label: string, value: string | undefined | null): string => `${label}: ${formatDetected(value)}`

const redactIp = (value: string | undefined): string | undefined =>
  value?.replace(/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/g, '<redacted>')

const buildEnvironmentRows = (signals: TerminalSignals): string[] => {
  const rows = [
    envRow('TERM', signals.env.TERM),
    envRow('TERM_PROGRAM', signals.env.TERM_PROGRAM),
    envRow('TERM_PROGRAM_VERSION', signals.env.TERM_PROGRAM_VERSION),
    envRow('COLORTERM', signals.env.COLORTERM),
    envRow('VTE_VERSION', signals.env.VTE_VERSION),
    envRow('WT_SESSION', signals.env.WT_SESSION),
    envRow('KITTY_WINDOW_ID', signals.env.KITTY_WINDOW_ID),
    envRow('WEZTERM_PANE', signals.env.WEZTERM_PANE),
    envRow('GHOSTTY_RESOURCES_DIR', signals.env.GHOSTTY_RESOURCES_DIR),
    envRow('KONSOLE_VERSION', signals.env.KONSOLE_VERSION),
    envRow('ITERM_SESSION_ID', signals.env.ITERM_SESSION_ID),
    envRow('LC_TERMINAL', signals.env.LC_TERMINAL),
    envRow('TERM_SESSION_ID', signals.env.TERM_SESSION_ID),
  ]

  if (signals.ssh.hasSshConnection || signals.ssh.hasSshClient || signals.ssh.hasSshTty) {
    rows.push(envRow('SSH_CONNECTION', redactIp(signals.ssh.connection)))
    rows.push(envRow('SSH_TTY', signals.ssh.tty))
  }

  rows.push(envRow('SHELL', signals.shell.executable))

  return rows
}

const buildDiagnostics = (env: TerminalEnvironment): string[] => {
  const { capabilities, signals } = env
  const diagnostics: string[] = []

  if ((signals.ssh.hasSshConnection || signals.ssh.hasSshClient) && !signals.ssh.hasSshTty) {
    diagnostics.push('⚠ SSH detected but SSH_TTY is unset — check ForceCommand or non-interactive SSH')
  }

  if (signals.multiplexer.tmux) {
    diagnostics.push('⚠ tmux detected — clipboard routing uses tmux-buffer')
  }

  if (signals.multiplexer.screen) {
    diagnostics.push('⚠ screen detected — clipboard routing uses screen-passthrough')
  }

  if (capabilities.mouse?.shiftDragHint) {
    diagnostics.push('⚠ Shift-drag for terminal-native selection')
  }

  for (const diagnostic of capabilities.diagnostics ?? []) {
    diagnostics.push(formatWarning(diagnostic))
  }

  for (const hint of signals.shell.startupRiskHints ?? []) {
    diagnostics.push(formatWarning(hint))
  }

  return diagnostics
}

export function formatTerminalDoctor(env: TerminalEnvironment): string {
  const transport = summarizeTransport(env.signals)
  const layers = summarizeLayers(env.signals)
  const diagnostics = buildDiagnostics(env)
  const lines: string[] = [
    '=== Terminal Doctor ===',
    `transport: ${transport}`,
    `layers: ${layers.length ? layers.join(', ') : 'none'}`,
    `terminal: ${summarizeTerminal(env.signals, env.capabilities)}`,
    `shell: ${summarizeShell(env.signals)}`,
    `keyboard: ${summarizeKeyboard(env.capabilities)}`,
    `bracketed paste: ${summarizeBracketedPaste(env.capabilities)}`,
    `clipboard write: ${formatUnknown(env.capabilities.copy.writePath)}`,
    `clipboard read: ${formatUnknown(env.capabilities.copy.readPath)}`,
    `copy-on-select: ${summarizeCopyOnSelect(env.capabilities)}`,
    '',
    '--- environment ---',
    ...buildEnvironmentRows(env.signals)
  ]

  if (diagnostics.length) {
    lines.push('', '--- diagnostics ---', ...diagnostics)
  }

  return lines.join('\n')
}
