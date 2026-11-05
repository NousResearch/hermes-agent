import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import type { SessionTreeNode, SessionTreeView } from '../app/interfaces.js'
import { SessionTree, treeRows } from '../components/sessionTree.js'
import { DEFAULT_THEME } from '../theme.js'

const node = (patch: Partial<SessionTreeNode>): SessionTreeNode => ({
  branch_from: null,
  created_at: 1_760_000_000,
  depth: 0,
  folded_segments: 0,
  has_children: false,
  is_current: false,
  kind: 'unknown',
  message_count: 2,
  parent_session_id: null,
  session_id: 'id',
  title: '',
  ...patch
})

const family: SessionTreeNode[] = [
  node({ depth: 0, has_children: true, kind: 'root', session_id: 'root', title: 'side-main' }),
  node({
    depth: 1,
    is_current: true,
    kind: 'branch',
    message_count: 1,
    parent_session_id: 'root',
    session_id: 'branch',
    title: 'side-main #2'
  }),
  node({
    depth: 1,
    folded_segments: 2,
    kind: 'branch',
    parent_session_id: 'root',
    session_id: 'other',
    title: 'side-main #3'
  })
]

const VIEW: SessionTreeView = { folded: [], show_compressions: false, show_subagents: false }

/** One keypress, spaced like a human's: the input tokenizer swallows a second ESC sequence that
 *  arrives in the same tick as the first. */
async function press(stdin: PassThrough, sequence: string) {
  stdin.write(sequence)
  await new Promise(resolve => setTimeout(resolve, 60))
}

function mount(options: { nodes?: SessionTreeNode[]; truncated?: boolean; view?: SessionTreeView } = {}) {
  const stdin = Object.assign(new PassThrough(), {
    isTTY: true,
    setRawMode: vi.fn(),
    ref: vi.fn(),
    unref: vi.fn()
  })

  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 30 })
  let output = ''

  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const onClose = vi.fn()
  const onSelect = vi.fn()
  const onView = vi.fn()

  const tree = (): React.JSX.Element => (
    <SessionTree
      cols={100}
      onClose={onClose}
      overlay={{
        nodes: options.nodes ?? family,
        onSelect,
        onView,
        rootId: 'root',
        truncated: options.truncated ?? false,
        view: options.view ?? VIEW
      }}
      t={DEFAULT_THEME}
    />
  )

  renderSync(tree(), {
    patchConsole: false,
    stderr: new PassThrough() as never,
    stdin: stdin as never,
    stdout: stdout as never
  })

  return {
    onClose,
    onSelect,
    onView,
    output: () => output,
    stdin
  }
}

it('draws each row with its kind, size, time and the current-session mark', () => {
  const { output } = mount()

  expect(output()).toContain('session tree')
  expect(output()).toContain('side-main · root')
  expect(output()).toContain('side-main #2 · branch · 1 msg')
  expect(output()).toContain('current')
  expect(output()).toContain('+2 compressed')
})

it('indents each level and marks a foldable row', () => {
  const { output } = mount()

  expect(output()).toContain('▾ side-main')
  expect(output()).toContain('  · side-main #2')
})

it('resumes the row Enter lands on, and not the session it is already on', async () => {
  const { onSelect, stdin } = mount()

  await press(stdin, '\r') // the cursor starts on the current session: nothing to switch to
  await press(stdin, '\u001B[B') // down to the second branch
  await press(stdin, '\r')

  await vi.waitFor(() => expect(onSelect).toHaveBeenCalledWith('other'))
  expect(onSelect).toHaveBeenCalledTimes(1)
})

it('asks the owner for the same family when the continuation filter is toggled', async () => {
  const { onView, stdin } = mount()

  await press(stdin, 'c')

  await vi.waitFor(() =>
    expect(onView).toHaveBeenCalledWith({ folded: [], show_compressions: true, show_subagents: false })
  )
})

it('asks the owner for the same family when subagent runs are toggled', async () => {
  const { onView, stdin } = mount()

  await press(stdin, 's')

  await vi.waitFor(() =>
    expect(onView).toHaveBeenCalledWith({ folded: [], show_compressions: false, show_subagents: true })
  )
})

it('folds the selected row with the left arrow', async () => {
  const { onView, stdin } = mount()

  await press(stdin, '\u001B[A') // up from the current session to the root
  await press(stdin, '\u001B[D')

  await vi.waitFor(() =>
    expect(onView).toHaveBeenCalledWith({ folded: ['root'], show_compressions: false, show_subagents: false })
  )
})

it('unfolds a folded row with the right arrow', async () => {
  // A folded tree comes back from the owner with the folded row's descendants already gone.
  const { onView, stdin } = mount({ nodes: [family[0]], view: { ...VIEW, folded: ['root'] } })

  await press(stdin, '\u001B[C')

  await vi.waitFor(() =>
    expect(onView).toHaveBeenCalledWith({ folded: [], show_compressions: false, show_subagents: false })
  )
})

it('closes on Esc', async () => {
  const { onClose, stdin } = mount()

  await press(stdin, '\u001B')

  await vi.waitFor(() => expect(onClose).toHaveBeenCalled())
})

it('filters the rows by title, and by id for an untitled row', () => {
  expect(treeRows(family, '').map(node => node.session_id)).toEqual(['root', 'branch', 'other'])
  expect(treeRows(family, '  #3 ').map(node => node.session_id)).toEqual(['other'])
  expect(treeRows(family, 'SIDE-MAIN #2').map(node => node.session_id)).toEqual(['branch'])
  expect(treeRows(family, 'no-such-session')).toEqual([])
})

it('says the family is truncated rather than passing a partial tree off as whole', () => {
  const { output } = mount({ truncated: true })

  expect(output()).toContain('truncated')
})
