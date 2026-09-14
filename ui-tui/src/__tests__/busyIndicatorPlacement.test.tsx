import { Box, ScrollBox, Text } from '@hermes/ink'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { renderToScreen } from '../../packages/hermes-ink/src/ink/render-to-screen.js'
import { cellAtIndex } from '../../packages/hermes-ink/src/ink/screen.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { StatusRule } from '../components/appChrome.js'
import { StreamingAssistant } from '../components/streamingAssistant.js'
import { DEFAULT_THEME } from '../theme.js'

const screenLines = (node: React.ReactElement, cols = 80): string[] => {
  const view = renderToScreen(node, cols)

  return Array.from({ length: view.height }, (_, row) =>
    Array.from({ length: cols }, (_, col) => cellAtIndex(view.screen, row * cols + col).char)
      .join('')
      .trimEnd()
  )
}

const statusProps: ComponentProps<typeof StatusRule> = {
  bgCount: 0,
  busy: true,
  cols: 80,
  cwdLabel: '~/repo',
  indicatorStyle: 'unicode',
  lastTurnEndedAt: null,
  liveSessionCount: 0,
  model: 'test-model',
  sessionStartedAt: Date.now() - 60_000,
  status: 'running…',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: Date.now() - 1_000,
  usage: {
    calls: 1,
    context_max: 200_000,
    context_percent: 1,
    context_used: 1_000,
    input: 1_000,
    output: 0,
    total: 1_000
  },
  voiceLabel: ''
}

describe('pending-response indicator placement', () => {
  beforeEach(() => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    resetUiState()
    resetTurnState()
    patchUiState({ busy: true, indicatorStyle: 'unicode' })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    resetUiState()
    resetTurnState()
  })

  it('follows the latest message in a short transcript and the viewport bottom after overflow', () => {
    const shortLines = screenLines(
      <Box flexDirection="column">
        <Text>latest user message</Text>
        <StreamingAssistant
          cols={80}
          detailsMode="collapsed"
          detailsModeCommandOverride={false}
          progress={{ showProgressArea: false }}
          statusColor={DEFAULT_THEME.color.ok}
          turnStartedAt={Date.now() - 1_000}
        />
      </Box>
    ).filter(Boolean)

    expect(shortLines[0]).toBe('latest user message')
    expect(shortLines[1]).toMatch(/^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/)

    const longLines = screenLines(
      <ScrollBox height={4} stickyScroll>
        <Box flexDirection="column">
          {Array.from({ length: 8 }, (_, index) => (
            <Text key={index}>message {index + 1}</Text>
          ))}
          <StreamingAssistant
            cols={80}
            detailsMode="collapsed"
            detailsModeCommandOverride={false}
            progress={{ showProgressArea: false }}
            statusColor={DEFAULT_THEME.color.ok}
            turnStartedAt={Date.now() - 1_000}
          />
        </Box>
      </ScrollBox>
    )

    expect(longLines).toHaveLength(4)
    expect(longLines.at(-1)).toMatch(/^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/)
    expect(longLines).not.toContain('message 1')
  })

  it('does not duplicate the running indicator in the fixed status rule', () => {
    const line = screenLines(<StatusRule {...statusProps} />)[0] ?? ''

    expect(line).not.toMatch(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/)
    expect(line).not.toContain('running…')
    expect(line).toContain('test model')
  })
})
