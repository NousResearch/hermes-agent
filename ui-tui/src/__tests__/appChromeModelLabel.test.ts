import { describe, expect, it } from 'vitest'

import { modelLabel } from '../components/appChrome.js'

describe('modelLabel fast-mode rendering', () => {
  it('does NOT double-print "fast" when the model id carries a -fast suffix', () => {
    // regression: selecting "claude-opus-4.8-fast" with fast=true previously rendered
    // "opus 4.8 fast max fast" (the suffix + the badge). Now the suffix is stripped from
    // the model name and the single fast badge remains.
    expect(modelLabel('copilot/claude-opus-4.8-fast', 'max', true)).toBe('opus 4.8 max fast')
  })

  it('renders the same label whether fast came from the -fast id or the /fast toggle', () => {
    const fromSuffix = modelLabel('copilot/claude-opus-4.8-fast', 'max', true)
    const fromToggle = modelLabel('copilot/claude-opus-4.8', 'max', true)
    expect(fromSuffix).toBe(fromToggle)
    expect(fromSuffix).toBe('opus 4.8 max fast')
  })

  it('shows no fast badge when fast is off, even if the id has -fast (defensive)', () => {
    expect(modelLabel('copilot/claude-opus-4.8-fast', 'high', false)).toBe('opus 4.8 high')
  })

  it('keeps the normal (non-fast) label intact', () => {
    expect(modelLabel('copilot/claude-opus-4.8', 'max', false)).toBe('opus 4.8 max')
  })

  it('strips -fast for sonnet/haiku too', () => {
    expect(modelLabel('copilot/claude-sonnet-4.6-fast', 'high', true)).toBe('sonnet 4.6 high fast')
  })
})
