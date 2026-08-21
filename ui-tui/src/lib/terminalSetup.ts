import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

export type SupportedTerminal = 'cursor' | 'vscode' | 'windsurf'

export type FileOps = {
  copyFile: typeof copyFile
  mkdir: typeof mkdir
  readFile: typeof readFile
  writeFile: typeof writeFile
}

type Keybinding = {
  args?: { text?: string }
  command?: string
  key?: string
  when?: string
}

export type TerminalSetupResult = {
  message: string
  requiresRestart?: boolean
  success: boolean
}

const DEFAULT_FILE_OPS: FileOps = { copyFile, mkdir, readFile, writeFile }
const COPY_SEQUENCE = '\u001b[99;13u'
// Kitty keyboard protocol CSI u sequences for modified Enter keys.
// Codepoint 13 = Enter; modifier encoding: 1 + (shift?1:0) + (alt?2:0) + (ctrl?4:0) + (super?8:0).
// These are recognized by Ink's parse-keypress CSI u handler and produce
// key.return=true with the correct modifier flags, so textInput.tsx's
// existing k.return && (k.shift || k.ctrl || ...) branch inserts a newline.
const SHIFT_ENTER_SEQUENCE = '\u001b[13;2u' // modifier 2 = shift
const CTRL_ENTER_SEQUENCE = '\u001b[13;5u' // modifier 5 = ctrl
const SUPER_ENTER_SEQUENCE = '\u001b[13;9u' // modifier 9 = super (Cmd on macOS)

// Legacy multiline sequence used before CSI u migration. Old keybindings
// that still send this will be auto-replaced on next terminal setup.
const LEGACY_MULTILINE_SEQUENCE = '\\\r\n'

/**
 * Migrate legacy keybindings that used the old \\\r\n escape sequence
 * for modified Enter keys.  Those sequences arrived at Ink as separate
 * key events (backslash + return), causing unintended submissions.
 * The replacement CSI u sequences are parsed correctly by Ink's
 * parse-keypress handler and produce the proper modifier flags.
 */
function migrateLegacyBindings(keybindings: unknown[]): number {
  let migrated = 0

  const replacements: Map<string, string> = new Map([
    ['shift+enter', SHIFT_ENTER_SEQUENCE],
    ['ctrl+enter', CTRL_ENTER_SEQUENCE],
    ['cmd+enter', SUPER_ENTER_SEQUENCE]
  ])

  for (let i = 0; i < keybindings.length; i++) {
    const entry = keybindings[i]

    if (!isKeybinding(entry)) {
      continue
    }

    const replacement = replacements.get(entry.key ?? '')

    if (
      replacement &&
      entry.command === 'workbench.action.terminal.sendSequence' &&
      entry.when === 'terminalFocus' &&
      entry.args?.text === LEGACY_MULTILINE_SEQUENCE
    ) {
      keybindings[i] = { ...entry, args: { text: replacement } }
      migrated += 1
    }
  }

  return migrated
}

const TERMINAL_META: Record<SupportedTerminal, { appName: string; label: string }> = {
  vscode: { appName: 'Code', label: 'VS Code' },
  cursor: { appName: 'Cursor', label: 'Cursor' },
  windsurf: { appName: 'Windsurf', label: 'Windsurf' }
}

const MAC_COPY_BINDING: Keybinding = {
  key: 'cmd+c',
  command: 'workbench.action.terminal.sendSequence',
  when: 'terminalFocus && terminalTextSelected',
  args: { text: COPY_SEQUENCE }
}

const BASE_BINDINGS: Keybinding[] = [
  {
    key: 'shift+enter',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: SHIFT_ENTER_SEQUENCE }
  },
  {
    key: 'ctrl+enter',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: CTRL_ENTER_SEQUENCE }
  },
  {
    key: 'cmd+enter',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: SUPER_ENTER_SEQUENCE }
  },
  {
    key: 'cmd+z',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: '\u001b[122;9u' }
  },
  {
    key: 'shift+cmd+z',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: '\u001b[122;10u' }
  }
]

const targetBindings = (platform: NodeJS.Platform): Keybinding[] =>
  platform === 'darwin' ? [MAC_COPY_BINDING, ...BASE_BINDINGS] : BASE_BINDINGS

export function detectVSCodeLikeTerminal(env: NodeJS.ProcessEnv = process.env): null | SupportedTerminal {
  const askpass = env['VSCODE_GIT_ASKPASS_MAIN']?.toLowerCase() ?? ''

  if (env['CURSOR_TRACE_ID'] || askpass.includes('cursor')) {
    return 'cursor'
  }

  if (askpass.includes('windsurf')) {
    return 'windsurf'
  }

  if (env['TERM_PROGRAM'] === 'vscode' || env['VSCODE_GIT_IPC_HANDLE']) {
    return 'vscode'
  }

  return null
}

/**
 * Strip JSONC features (// line comments, /* block comments *\/, trailing commas)
 * so the result is valid JSON parseable by JSON.parse().
 * Handles comments inside strings correctly (preserves them).
 */
export function stripJsonComments(content: string): string {
  let result = ''
  let i = 0
  const len = content.length

  while (i < len) {
    const ch = content[i]!

    // String literal — copy as-is, including any comment-like chars inside
    if (ch === '"') {
      let j = i + 1

      while (j < len) {
        if (content[j] === '\\') {
          j += 2 // skip escaped char
        } else if (content[j] === '"') {
          j++

          break
        } else {
          j++
        }
      }

      result += content.slice(i, j)
      i = j

      continue
    }

    // Line comment
    if (ch === '/' && content[i + 1] === '/') {
      const eol = content.indexOf('\n', i)
      i = eol === -1 ? len : eol

      continue
    }

    // Block comment
    if (ch === '/' && content[i + 1] === '*') {
      const end = content.indexOf('*/', i + 2)
      i = end === -1 ? len : end + 2

      continue
    }

    result += ch
    i++
  }

  // Remove trailing commas before ] or }
  return result.replace(/,(\s*[}\]])/g, '$1')
}

export function isRemoteShellSession(env: NodeJS.ProcessEnv): boolean {
  return Boolean(env['SSH_CONNECTION'] || env['SSH_TTY'] || env['SSH_CLIENT'])
}

export function getVSCodeStyleConfigDir(
  appName: string,
  platform: NodeJS.Platform = process.platform,
  env: NodeJS.ProcessEnv = process.env,
  homeDir: string = homedir()
): null | string {
  if (platform === 'darwin') {
    return join(homeDir, 'Library', 'Application Support', appName, 'User')
  }

  if (platform === 'win32') {
    return env['APPDATA'] ? join(env['APPDATA'], appName, 'User') : null
  }

  return join(homeDir, '.config', appName, 'User')
}

function isKeybinding(value: unknown): value is Keybinding {
  return typeof value === 'object' && value !== null
}

function sameBinding(a: Keybinding, b: Keybinding): boolean {
  return a.key === b.key && a.command === b.command && a.when === b.when && a.args?.text === b.args?.text
}

type WhenRequirements = {
  forbidden: Set<string>
  required: Set<string>
}

const WHEN_TOKEN_RE = /!?[A-Za-z_][\w.]*/g

function parseWhenRequirements(when: string): WhenRequirements {
  const required = new Set<string>()
  const forbidden = new Set<string>()

  for (const [token] of when.matchAll(WHEN_TOKEN_RE)) {
    if (token.startsWith('!')) {
      forbidden.add(token.slice(1))
    } else {
      required.add(token)
    }
  }

  return { forbidden, required }
}

function requirementsContradict(a: WhenRequirements, b: WhenRequirements): boolean {
  for (const token of a.required) {
    if (b.forbidden.has(token)) {
      return true
    }
  }

  for (const token of b.required) {
    if (a.forbidden.has(token)) {
      return true
    }
  }

  return false
}

function whensOverlap(a: string, b: string): boolean {
  if (a === b) {
    return true
  }

  // Empty when = global, overlaps every context.
  if (!a || !b) {
    return true
  }

  const left = parseWhenRequirements(a)
  const right = parseWhenRequirements(b)

  if (requirementsContradict(left, right)) {
    return false
  }

  // This intentionally avoids a full VS Code when-clause parser. If two
  // same-key bindings share a positive context token and don't explicitly
  // contradict each other, they can fire together in that context.
  for (const token of left.required) {
    if (right.required.has(token)) {
      return true
    }
  }

  return false
}

// VS Code allows multiple bindings on the same key as long as their `when`
// clauses don't overlap. We flag a conflict when the contexts overlap but
// the bindings differ — e.g. existing `terminalFocus` cmd+c overlaps with
// our `terminalFocus && terminalTextSelected`, so the existing binding
// would shadow ours when text isn't selected.
function bindingsConflict(existing: Keybinding, target: Keybinding): boolean {
  if (existing.key !== target.key) {
    return false
  }

  if (!whensOverlap(existing.when ?? '', target.when ?? '')) {
    return false
  }

  return !sameBinding(existing, target)
}

async function backupFile(filePath: string, ops: FileOps): Promise<void> {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  await ops.copyFile(filePath, `${filePath}.backup.${stamp}`)
}

export async function configureTerminalKeybindings(
  terminal: SupportedTerminal,
  options?: {
    env?: NodeJS.ProcessEnv
    fileOps?: Partial<FileOps>
    homeDir?: string
    platform?: NodeJS.Platform
  }
): Promise<TerminalSetupResult> {
  const env = options?.env ?? process.env
  const platform = options?.platform ?? process.platform
  const homeDir = options?.homeDir ?? homedir()
  const ops: FileOps = { ...DEFAULT_FILE_OPS, ...(options?.fileOps ?? {}) }
  const meta = TERMINAL_META[terminal]

  if (isRemoteShellSession(env)) {
    return {
      success: false,
      message: `${meta.label} terminal setup must be run on the local machine, not inside an SSH session.`
    }
  }

  const configDir = getVSCodeStyleConfigDir(meta.appName, platform, env, homeDir)

  if (!configDir) {
    return {
      success: false,
      message: `Could not determine ${meta.label} settings path on this platform.`
    }
  }

  const keybindingsFile = join(configDir, 'keybindings.json')

  try {
    await ops.mkdir(configDir, { recursive: true })

    let keybindings: unknown[] = []
    let hasExistingFile = false

    try {
      const content = await ops.readFile(keybindingsFile, 'utf8')
      hasExistingFile = true
      const parsed: unknown = JSON.parse(stripJsonComments(content))

      if (!Array.isArray(parsed)) {
        return {
          success: false,
          message: `${meta.label} keybindings.json is not a JSON array: ${keybindingsFile}`
        }
      }

      keybindings = parsed
    } catch (error) {
      const code = (error as NodeJS.ErrnoException | undefined)?.code

      if (code !== 'ENOENT') {
        return {
          success: false,
          message: `Failed to read ${meta.label} keybindings: ${error}`
        }
      }
    }

    const migrated = migrateLegacyBindings(keybindings)
    const targets = targetBindings(platform)

    const conflicts = targets.filter(target =>
      keybindings.some(existing => isKeybinding(existing) && bindingsConflict(existing, target))
    )

    if (conflicts.length) {
      return {
        success: false,
        message:
          `Existing terminal keybindings would conflict in ${keybindingsFile}: ` + conflicts.map(c => c.key).join(', ')
      }
    }

    let added = 0

    for (const target of targets.slice().reverse()) {
      const exists = keybindings.some(existing => isKeybinding(existing) && sameBinding(existing, target))

      if (!exists) {
        keybindings.unshift(target)
        added += 1
      }
    }

    if (!added && !migrated) {
      return {
        success: true,
        message: `${meta.label} terminal keybindings already configured.`
      }
    }

    if (hasExistingFile && (added || migrated)) {
      await backupFile(keybindingsFile, ops)
    }

    await ops.writeFile(keybindingsFile, `${JSON.stringify(keybindings, null, 2)}\n`, 'utf8')

    const parts: string[] = []

    if (added) {
      parts.push(`Added ${added} ${meta.label} terminal keybinding${added === 1 ? '' : 's'}`)
    }

    if (migrated) {
      parts.push(`migrated ${migrated} legacy binding${migrated === 1 ? '' : 's'} to CSI u encoding`)
    }

    return {
      success: true,
      requiresRestart: true,
      message: `${parts.join(', ')} in ${keybindingsFile}`
    }
  } catch (error) {
    return {
      success: false,
      message: `Failed to configure ${meta.label} terminal shortcuts: ${error}`
    }
  }
}

export async function configureDetectedTerminalKeybindings(options?: {
  env?: NodeJS.ProcessEnv
  fileOps?: Partial<FileOps>
  homeDir?: string
  platform?: NodeJS.Platform
}): Promise<TerminalSetupResult> {
  const detected = detectVSCodeLikeTerminal(options?.env ?? process.env)

  if (!detected) {
    return {
      success: false,
      message: 'No supported IDE terminal detected. Supported: VS Code, Cursor, Windsurf.'
    }
  }

  return configureTerminalKeybindings(detected, options)
}

export async function shouldPromptForTerminalSetup(options?: {
  env?: NodeJS.ProcessEnv
  fileOps?: Partial<FileOps>
  homeDir?: string
  platform?: NodeJS.Platform
}): Promise<boolean> {
  const env = options?.env ?? process.env
  const detected = detectVSCodeLikeTerminal(env)

  if (!detected || isRemoteShellSession(env)) {
    return false
  }

  const platform = options?.platform ?? process.platform
  const homeDir = options?.homeDir ?? homedir()
  const ops: FileOps = { ...DEFAULT_FILE_OPS, ...(options?.fileOps ?? {}) }
  const meta = TERMINAL_META[detected]
  const configDir = getVSCodeStyleConfigDir(meta.appName, platform, env, homeDir)

  if (!configDir) {
    return false
  }

  try {
    const content = await ops.readFile(join(configDir, 'keybindings.json'), 'utf8')
    const parsed: unknown = JSON.parse(stripJsonComments(content))

    if (!Array.isArray(parsed)) {
      return true
    }

    return targetBindings(platform).some(
      target => !parsed.some(existing => isKeybinding(existing) && sameBinding(existing, target))
    )
  } catch {
    return true
  }
}
