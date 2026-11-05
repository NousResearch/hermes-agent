import { PassThrough } from 'node:stream'

import { Box, NoSelect, renderSync, Text } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, expect, it, vi } from 'vitest'

import { $copyFeedback, clearCopyFeedback, runCopyWithFeedback } from '../app/copyFeedbackStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { CopyFeedback } from '../components/copyFeedback.js'

afterEach(() => {
  clearCopyFeedback()
  resetUiState()
})

it('renders above the input while busy without replacing an important notice, then clears on session change', async () => {
  patchUiState({ busy: true, sid: 'one', notice: { kind: 'sticky', text: 'Important account notice' } })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 10, isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const view = renderSync(
    <NoSelect flexDirection="column">
      <Box height={1}>
        <Box flexGrow={1} />
        <CopyFeedback />
      </Box>
      <Text>COMPOSER_DRAFT</Text>
    </NoSelect>,
    {
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  try {
    await vi.waitFor(() => expect(output).toContain('COMPOSER_DRAFT'))
    output = ''
    await runCopyWithFeedback(async () => 'terminal')
    await vi.waitFor(() => expect(output).toContain('Copy sent to terminal · paste to verify'))
    expect(output.indexOf('Copy sent to terminal')).toBeLessThan(output.lastIndexOf('COMPOSER_DRAFT'))
    expect(getUiState().busy).toBe(true)
    expect(getUiState().notice?.text).toBe('Important account notice')
    patchUiState({ sid: 'two' })
    await vi.waitFor(() => expect($copyFeedback.get()).toBeNull())
    await runCopyWithFeedback(async () => 'clipboard')
    await vi.waitFor(() => expect(output).toContain('copied to clipboard'))
  } finally {
    view.unmount()
    view.cleanup()
  }

  expect($copyFeedback.get()).toBeNull()
})
