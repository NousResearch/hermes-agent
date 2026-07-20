import { describe, expect, it, vi } from 'vitest'

import { copyLatestAssistantResponse, copyTextForHost } from '../app/slash/commands/core.js'

describe('copyTextForHost', () => {
  it('forces OSC 52 in dashboard mode without writing the Mac clipboard', async () => {
    const writeNative = vi.fn().mockResolvedValue(true)
    const writeOsc52 = vi.fn()

    await expect(copyTextForHost('assistant reply', true, writeNative, writeOsc52)).resolves.toBe('osc52')
    expect(writeNative).not.toHaveBeenCalled()
    expect(writeOsc52).toHaveBeenCalledWith('assistant reply')
  })

  it('keeps native clipboard behavior in a normal terminal', async () => {
    const writeNative = vi.fn().mockResolvedValue(true)
    const writeOsc52 = vi.fn()

    await expect(copyTextForHost('assistant reply', false, writeNative, writeOsc52)).resolves.toBe('native')
    expect(writeNative).toHaveBeenCalledWith('assistant reply')
    expect(writeOsc52).not.toHaveBeenCalled()
  })
})

describe('copyLatestAssistantResponse', () => {
  it('copies the latest assistant response without consulting composer selection', async () => {
    const write = vi.fn().mockResolvedValue('osc52')
    const sys = vi.fn()

    await copyLatestAssistantResponse(
      [
        { role: 'assistant', text: 'older answer' },
        { role: 'user', text: 'selected draft must not win' },
        { role: 'assistant', text: 'latest answer' }
      ],
      sys,
      write
    )

    expect(write).toHaveBeenCalledWith('latest answer')
    expect(sys).toHaveBeenCalledWith('sent OSC52 copy sequence (terminal support required)')
  })
})
