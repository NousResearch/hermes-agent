import { mkdirSync, mkdtempSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import {
  readTabWorkspace,
  startTabPersistence,
  tabWorkspaceFile,
  takeRestoredTabId,
  writeTabWorkspace
} from '../app/sessionTabsPersistence.js'
import {
  $sessionTabs,
  openSessionTab,
  removeSessionTab,
  resetSessionTabs,
  saveSessionTabDraft
} from '../app/sessionTabsStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const roots: string[] = []

const fixture = () => {
  const root = mkdtempSync(join(tmpdir(), 'tabs-persist-'))
  roots.push(root)

  const home = join(root, 'home'),
    cwd = join(root, 'project')

  mkdirSync(cwd)
  mkdirSync(home)

  return { root, home, cwd }
}

afterEach(() => {
  resetSessionTabs()
  resetUiState()
  takeRestoredTabId()
  vi.unstubAllEnvs()

  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true })
  }
})

it('restores all opened tabs, order, last selection and private unsent multiline/paste drafts after a process restart', () => {
  const { home, cwd } = fixture()
  let stop = startTabPersistence(cwd, home)
  openSessionTab({ id: 'first', storedId: 'stored-first', cwd, title: 'First' })
  openSessionTab({ id: 'second', cwd, title: 'Second', running: true })
  patchUiState({ sid: 'second' })
  saveSessionTabDraft('first', {
    input: 'unfinished [[paste]]',
    inputBuf: ['earlier line'],
    tokens: [{ kind: 'paste', label: '[[paste]]', text: 'original\ncontents' }]
  })
  stop()
  resetSessionTabs()
  resetUiState()
  stop = startTabPersistence(cwd, home)

  try {
    expect($sessionTabs.get().map(tab => tab.id)).toEqual(['first', 'second'])
    expect(takeRestoredTabId()).toBe('second')
    expect($sessionTabs.get()[0]!.draft).toMatchObject({ input: 'unfinished [[paste]]', inputBuf: ['earlier line'] })
    expect($sessionTabs.get()[0]!.draft.tokens[0]!.text).toBe('original\ncontents')
    expect($sessionTabs.get().every(tab => !tab.running)).toBe(true)
    expect(statSync(tabWorkspaceFile(home, cwd)).mode & 0o777).toBe(0o600)
  } finally {
    stop()
  }
})

it('keeps workspace/profile identities separate and recognizes symlink aliases', () => {
  const { root, home, cwd } = fixture()
  const stop = startTabPersistence(cwd, home)
  openSessionTab({ id: 'own', cwd, title: 'Own' })
  openSessionTab({ id: 'foreign', cwd: cwd + '-other', title: 'Not this directory' })
  stop()
  const alias = join(root, 'alias')
  symlinkSync(cwd, alias)
  expect(readTabWorkspace(home, alias)?.tabs.map(tab => tab.id)).toEqual(['own'])
  expect(readTabWorkspace(home, cwd + '-other')).toBeNull()
  expect(readTabWorkspace(join(root, 'other-home'), cwd)).toBeNull()
})

it('persists an explicit tab close so closed views do not reopen', () => {
  const { home, cwd } = fixture()
  const stop = startTabPersistence(cwd, home)
  openSessionTab({ id: 'keep', cwd })
  openSessionTab({ id: 'close', cwd })
  removeSessionTab('close')
  stop()
  expect(readTabWorkspace(home, cwd)?.tabs.map(tab => tab.id)).toEqual(['keep'])
})

it('ignores corrupt, wrong-directory and invalid-ID manifests without dispatching anything', () => {
  const { home, cwd } = fixture()
  writeTabWorkspace(home, { version: 1, cwd, activeId: '', tabs: [] })
  const file = tabWorkspaceFile(home, cwd)
  writeFileSync(file, '{broken')
  expect(readTabWorkspace(home, cwd)).toBeNull()
  writeFileSync(file, JSON.stringify({ version: 1, cwd: cwd + '-other', tabs: [] }))
  expect(readTabWorkspace(home, cwd)).toBeNull()
  writeFileSync(
    file,
    JSON.stringify({ version: 1, cwd, tabs: [{ id: '--unsafe', storedId: 'not a literal id', cwd }] })
  )
  expect(readTabWorkspace(home, cwd)?.tabs).toEqual([])
})

it('does not restore local histories on explicit remote or hosted dashboard connections', () => {
  const { home, cwd } = fixture()
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', 'ws://remote.invalid/api/ws')
  const stop = startTabPersistence(cwd, home)
  openSessionTab({ id: 'remote', cwd })
  stop()
  expect(readTabWorkspace(home, cwd)).toBeNull()
})
