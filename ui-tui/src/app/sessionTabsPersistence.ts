import { createHash, randomUUID } from 'node:crypto'
import {
  chmodSync,
  closeSync,
  existsSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  realpathSync,
  renameSync,
  unlinkSync,
  writeFileSync
} from 'node:fs'
import { isAbsolute, join, resolve } from 'node:path'

import type { ComposerToken } from './interfaces.js'
import { $sessionTabs, emptyTabDraft, findSessionTab, type SessionTab, tabsInDirectory } from './sessionTabsStore.js'
import { $uiState, getUiState } from './uiStore.js'

export interface TabWorkspace {
  version: 1
  cwd: string
  activeId: string
  tabs: SessionTab[]
}
const MAX_BYTES = 16 * 1024 * 1024
const validId = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z0-9_-]{1,200}$/.test(value)

export const canonicalTabDirectory = (cwd: string) => {
  try {
    return realpathSync(cwd)
  } catch {
    return resolve(cwd)
  }
}

export const tabWorkspaceFile = (home: string, cwd: string) =>
  join(home, 'ui-tui', 'session-tabs', createHash('sha256').update(canonicalTabDirectory(cwd)).digest('hex') + '.json')

function token(value: any): ComposerToken | undefined {
  if (!value || typeof value.label !== 'string') {
    return
  }

  if (value.kind === 'paste' && typeof value.text === 'string') {
    return { kind: 'paste', label: value.label, text: value.text }
  }

  if (value.kind === 'image' && typeof value.path === 'string' && Number.isSafeInteger(value.index)) {
    return {
      kind: 'image',
      label: value.label,
      path: value.path,
      index: value.index,
      ...(typeof value.mime === 'string' ? { mime: value.mime } : {})
    }
  }
}

function savedTab(row: any, directory: string): SessionTab | undefined {
  if (
    !row ||
    !validId(row.id) ||
    !validId(row.storedId) ||
    typeof row.cwd !== 'string' ||
    !isAbsolute(row.cwd) ||
    canonicalTabDirectory(row.cwd) !== directory
  ) {
    return
  }

  const draft = row.draft || emptyTabDraft()

  return {
    id: row.id,
    storedId: row.storedId,
    cwd: directory,
    title: typeof row.title === 'string' ? row.title : '',
    aliases: Array.isArray(row.aliases) ? row.aliases.filter(validId).slice(0, 100) : [],
    running: false,
    draft: {
      input: typeof draft.input === 'string' ? draft.input : '',
      inputBuf: Array.isArray(draft.inputBuf) ? draft.inputBuf.filter((line: unknown) => typeof line === 'string') : [],
      tokens: Array.isArray(draft.tokens)
        ? draft.tokens.map(token).filter((t: ComposerToken | undefined): t is ComposerToken => Boolean(t))
        : []
    }
  }
}

export function readTabWorkspace(home: string, cwd: string): TabWorkspace | null {
  const file = tabWorkspaceFile(home, cwd)

  try {
    const info = lstatSync(file)

    if (!info.isFile() || info.size > MAX_BYTES) {
      return null
    }

    const saved = JSON.parse(readFileSync(file, 'utf8'))
    const directory = canonicalTabDirectory(cwd)

    if (saved.version !== 1 || saved.cwd !== directory || !Array.isArray(saved.tabs) || saved.tabs.length > 200) {
      return null
    }

    const seen = new Set<string>()
    const tabs: SessionTab[] = []

    for (const row of saved.tabs) {
      const tab = savedTab(row, directory)

      if (!tab || seen.has(tab.storedId)) {
        continue
      }

      seen.add(tab.storedId)
      tabs.push(tab)
    }

    const active = tabs.find(tab => [tab.id, tab.storedId, ...tab.aliases].includes(saved.activeId)) || tabs.at(-1)

    return { version: 1, cwd: directory, tabs, activeId: active?.storedId || '' }
  } catch {
    return null
  }
}

export function writeTabWorkspace(home: string, workspace: TabWorkspace): void {
  const file = tabWorkspaceFile(home, workspace.cwd)
  const parent = join(home, 'ui-tui', 'session-tabs')
  mkdirSync(parent, { recursive: true, mode: 0o700 })
  chmodSync(parent, 0o700)
  const data = JSON.stringify(workspace) + '\n'

  if (Buffer.byteLength(data) > MAX_BYTES) {
    throw new Error('Tab drafts exceed the private workspace save limit')
  }

  const temp = file + '.' + randomUUID() + '.new'
  let fd: number | undefined

  try {
    fd = openSync(temp, 'wx', 0o600)
    writeFileSync(fd, data)
    fsyncSync(fd)
    closeSync(fd)
    fd = undefined
    renameSync(temp, file)
  } finally {
    if (fd !== undefined) {
      closeSync(fd)
    }

    if (existsSync(temp)) {
      unlinkSync(temp)
    }
  }
}

let restoredStartupId = ''

export function takeRestoredTabId(): string {
  const id = restoredStartupId
  restoredStartupId = ''

  return id
}

/** Installed only by the real entrypoint, never by importing a React test.
 * Saves private view/draft metadata, not transcripts; restores without sends. */
export function startTabPersistence(cwd: string, home = process.env.HERMES_HOME) {
  if (!home || !isAbsolute(cwd) || process.env.HERMES_TUI_GATEWAY_URL || process.env.HERMES_TUI_DASHBOARD === '1') {
    return () => {}
  }

  const directory = canonicalTabDirectory(cwd)
  const saved = readTabWorkspace(home, directory)

  if (saved) {
    $sessionTabs.set(saved.tabs)
  }

  restoredStartupId = saved?.activeId || ''
  let lastActiveId = saved?.activeId || ''
  let last = ''
  let pending: TabWorkspace | undefined
  let timer: ReturnType<typeof setTimeout> | undefined

  const flush = () => {
    clearTimeout(timer)
    timer = undefined

    if (!pending) {
      return
    }

    const snapshot = pending
    pending = undefined

    try {
      writeTabWorkspace(home, snapshot)
    } catch (error) {
      process.stderr.write(`Hermes tab layout was not saved: ${(error as Error).message}\n`)
    }
  }

  const schedule = () => {
    const ui = getUiState()
    const visible = tabsInDirectory($sessionTabs.get(), directory)
    const active = findSessionTab(ui.sid)

    if (active && visible.some(tab => tab.id === active.id)) {
      lastActiveId = active.storedId
    }

    // No runtime busy state is persisted. On relaunch, the owner supplies it.
    const snapshot: TabWorkspace = {
      version: 1,
      cwd: directory,
      activeId: lastActiveId || visible.at(-1)?.storedId || '',
      tabs: visible.map(tab => ({ ...tab, cwd: directory, running: false, blocked: false }))
    }

    const encoded = JSON.stringify(snapshot)

    if (encoded === last) {
      return
    }

    last = encoded
    pending = snapshot
    clearTimeout(timer)
    timer = setTimeout(flush, 200)
    timer.unref?.()
  }

  const stopTabs = $sessionTabs.listen(schedule)
  const stopUi = $uiState.listen(schedule)
  process.on('exit', flush)

  return () => {
    flush()
    stopTabs()
    stopUi()
    process.off('exit', flush)
    restoredStartupId = ''
  }
}
