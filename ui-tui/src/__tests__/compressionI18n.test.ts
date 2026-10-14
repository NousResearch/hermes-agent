import { afterEach } from 'vitest'

import { resetLocale } from '../i18n/runtime.js'

import { activateZh } from './localeFixture.js'
afterEach(resetLocale)
import { describe, expect, it } from 'vitest'

import { formatCompressionSummary } from '../app/slash/commands/session.js'

describe('manual compression localization', () => {
  it('preserves the refusal reason when the proposed summary would grow the conversation', () => {
    activateZh()

    const lines = formatCompressionSummary({
      summary: {
        before_count: 12,
        after_count: 12,
        before_tokens: 120000,
        after_tokens: 130000,
        noop: true,
        aborted: true,
        refused_would_grow: true
      }
    })

    expect(lines?.[0]).toBe('已拒绝压缩（摘要会使对话变大）：保留了 12 条消息')
    expect(lines?.join('\n')).toContain('生成的摘要比待替换内容更大，未移除任何消息。')
    expect(lines?.[1]).toBe('预计请求大小：约 120000 个 token（未变化）')
    expect(lines?.join('\n')).not.toContain('摘要生成失败')
  })

  it('renders structured backend feedback in Simplified Chinese', () => {
    activateZh()

    const lines = formatCompressionSummary({
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
    expect(formatCompressionSummary({ summary: { headline: 'legacy' } })).toBeNull()
  })
})
