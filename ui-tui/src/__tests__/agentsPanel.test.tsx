import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { AgentsPanelView } from '../components/agentsPanel.js'
import type { AgentRow } from '../lib/agentRows.js'
import { DEFAULT_THEME } from '../theme.js'

type ReactNodeLike = React.ReactNode

const textContent = (node: ReactNodeLike): string => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return ''
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return String(node)
  }

  if (Array.isArray(node)) {
    return node.map(textContent).join('')
  }

  if (React.isValidElement(node)) {
    return textContent(node.props.children)
  }

  return ''
}

const findClickableWithText = (node: ReactNodeLike, needle: string): React.ReactElement | null => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return null
  }

  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findClickableWithText(child, needle)

      if (found) {
        return found
      }
    }

    return null
  }

  if (!React.isValidElement(node)) {
    return null
  }

  if (typeof node.props.onClick === 'function' && textContent(node).includes(needle)) {
    return node
  }

  return findClickableWithText(node.props.children, needle)
}

const row = (over: Partial<AgentRow> = {}): AgentRow => ({
  detail: '',
  elapsedSeconds: null,
  goal: 'do a thing',
  key: 'k',
  name: '',
  resultReady: false,
  status: 'running',
  ...over
})

describe('AgentsPanelView', () => {
  const base = { collapsed: false, done: 0, rows: [row()], running: 1, t: DEFAULT_THEME }

  it('renders the header with running/done counts', () => {
    const el = AgentsPanelView({ ...base, done: 1, running: 2 })

    expect(textContent(el)).toContain('agents')
    expect(textContent(el)).toContain('2 running')
    expect(textContent(el)).toContain('1 done')
  })

  it('renders nothing when there are no rows (empty state)', () => {
    const el = AgentsPanelView({ ...base, rows: [] })

    expect(el).toBeNull()
  })

  it('hides row bodies when collapsed but keeps the header', () => {
    const el = AgentsPanelView({ ...base, collapsed: true, rows: [row({ goal: 'secret goal' })] })

    expect(textContent(el)).toContain('agents')
    expect(textContent(el)).not.toContain('secret goal')
  })

  it('appends ⏎ to a result-ready row', () => {
    const el = AgentsPanelView({ ...base, rows: [row({ detail: 'result ready', resultReady: true })] })

    expect(textContent(el)).toContain('result ready ⏎')
  })

  it('exposes a clickable "^a tree" affordance wired to onOpenTree', () => {
    const onOpenTree = vi.fn()
    const el = AgentsPanelView({ ...base, onOpenTree })

    const tree = findClickableWithText(el, '^a tree')
    expect(tree).not.toBeNull()
    tree!.props.onClick()
    expect(onOpenTree).toHaveBeenCalledOnce()
  })

  it('toggles collapse when the header is clicked', () => {
    const onToggle = vi.fn()
    const el = AgentsPanelView({ ...base, onToggle })

    const header = findClickableWithText(el, 'agents')
    expect(header).not.toBeNull()
    header!.props.onClick()
    expect(onToggle).toHaveBeenCalled()
  })
})
