import { describe, expect, it } from 'vitest'

import { formatCompressionSummary } from '../app/slash/commands/session.js'

describe('manual compression localization', () => {
  it('renders structured backend feedback in Simplified Chinese', () => {
    const lines = formatCompressionSummary('zh', {
      summary: {
        after_count: 4,
        after_tokens: 40000,
        before_count: 12,
        before_tokens: 120000,
        dropped_count: 8,
        fallback_used: true,
        failure_reason: '摘要服务返回无效响应',
        noop: false
      }
    })

    expect(lines?.[0]).toBe('已使用降级方案压缩：12 → 4 条消息')
    expect(lines?.join('\n')).toContain('移除了 8 条消息')
    expect(lines?.join('\n')).toContain('原因：摘要服务返回无效响应')
  })

  it('leaves legacy pre-rendered summaries to the compatibility path', () => {
    expect(formatCompressionSummary('zh', { summary: { headline: 'legacy' } })).toBeNull()
  })
})
