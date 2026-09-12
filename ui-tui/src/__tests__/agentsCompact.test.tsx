import { PassThrough } from 'node:stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it, vi } from 'vitest'

import { renderToScreen } from '../../packages/hermes-ink/src/ink/render-to-screen.js'
import { cellAtIndex } from '../../packages/hermes-ink/src/ink/screen.js'
import { applyAgentSnapshot } from '../app/agentRoster.js'
import { getInputSelection } from '../app/inputSelectionStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { AgentsOverlay } from '../components/agentsOverlay.js'
import { AgentsPanelView } from '../components/agentsPanel.js'
import { TextInput } from '../components/textInput.js'
import type { GatewayClient } from '../gatewayClient.js'
import { buildAgentRows } from '../lib/agentRows.js'
import { DEFAULT_THEME } from '../theme.js'

it('keeps collapsed live chrome to one row without losing count or restore controls', () => {
  const rows = buildAgentRows([], [{ delegation_id: 'work', status: 'running', goal: 'Review the boundary' }], 1000)

  for (const cols of [78, 98]) {
    const view = renderToScreen(<AgentsPanelView collapsed cols={cols} {...rows} t={DEFAULT_THEME} />, cols)
    expect(view.height).toBe(1)
    const text = Array.from({ length: cols }, (_, i) => cellAtIndex(view.screen, i).char).join('')
    expect(text).toContain(`${rows.running} live agents`)
    expect(text).toContain('Ctrl+T expand')
    expect(text).toContain('F7 restore')
    expect(renderToScreen(<AgentsPanelView cols={cols} {...rows} t={DEFAULT_THEME} />, cols).height).toBeGreaterThan(
      view.height
    )
  }
})

it('opens the selected live transcript on Enter while details remain independently accessible', async () => {
  patchUiState({ sid: 'owner' })
  applyAgentSnapshot('owner', {
    subagents: [{ subagent_id: 'child', goal: 'Inspect ownership', status: 'running' }],
    delegations: []
  })

  const request = vi.fn(async (method: string) =>
    method === 'subagent.tail' ? { available: true, text: 'CHILD_TOOL_OUTPUT', truncated: false } : {}
  )

  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''
  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const view = renderSync(
    <Box height={20}>
      <AgentsOverlay gw={{ request } as unknown as GatewayClient} onClose={() => {}} t={DEFAULT_THEME} />
    </Box>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  try {
    await vi.waitFor(() => expect(output).toContain('Inspect ownership'))
    stdin.write('\r')
    await vi.waitFor(() =>
      expect(request).toHaveBeenCalledWith('subagent.tail', { session_id: 'owner', subagent_id: 'child' })
    )
    await vi.waitFor(() => expect(output).toContain('CHILD_TOOL_OUTPUT'))
    output = ''
    stdin.write('d')
    await vi.waitFor(() => expect(output).toContain('Inspect ownership'))
    expect(output).not.toContain('CHILD_TOOL_OUTPUT')
    output = ''
    stdin.write('t')
    await vi.waitFor(() => expect(output).toContain('CHILD_TOOL_OUTPUT'))
    const cursorSnapshotRef = { current: null }
    const onChange = vi.fn()
    view.rerender(<TextInput cursorSnapshotRef={cursorSnapshotRef} onChange={onChange} value="draft" />)
    await vi.waitFor(() => expect(getInputSelection()?.value).toBe('draft'))
    stdin.write('\x1b[D')
    await vi.waitFor(() => expect(getInputSelection()?.start).toBe(4))
    view.rerender(<Box />)
    view.rerender(<TextInput cursorSnapshotRef={cursorSnapshotRef} onChange={onChange} value="draft" />)
    await vi.waitFor(() => expect(getInputSelection()?.start).toBe(4))
    stdin.write('!')
    await vi.waitFor(() => expect(onChange).toHaveBeenCalledWith('draf!t'))
  } finally {
    view.unmount()
    view.cleanup()
    applyAgentSnapshot(null)
    resetUiState()
  }
})
