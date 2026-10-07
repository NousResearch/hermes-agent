import { afterEach, describe, expect, it } from 'vitest'

import { imageToken } from '../domain/attachments.js'
import { toTranscriptMessages, userDisplay } from '../domain/messages.js'
import { applyLocale, resetLocale, t } from '../i18n/runtime.js'
import { billingDialogCopy } from '../lib/billingDialog.js'
import { formatSummary } from '../lib/subagentTree.js'
import { boundedLiveRenderText, formatAbandonedClarify } from '../lib/text.js'

// The `libText` namespace is resolved at call time, never at module load, so a
// pack installed after import must be observed by every helper on its next call.
afterEach(() => {
  resetLocale()
})

const pack = (messages: Record<string, string>) => ({ lang: 'xx', messages, surface: 'tui' })

describe('libText catalog swap', () => {
  it('billingDialogCopy re-reads string and function leaves after applyLocale', () => {
    const before = billingDialogCopy({ is_nous: false, provider_label: 'Acme', billing_url: 'https://x' } as never)

    expect(before.title).toBe('Out of credits · Acme')

    applyLocale(
      'xx',
      pack({
        'libText.billingDialog.dismiss': 'XX-dismiss',
        'libText.billingDialog.providerTitle': 'XX-title {0}'
      })
    )

    const after = billingDialogCopy({ is_nous: false, provider_label: 'Acme', billing_url: 'https://x' } as never)

    expect(after.cancelLabel).toBe('XX-dismiss')
    expect(after.title).toBe('XX-title Acme')
    // Untranslated leaves fall back to English.
    expect(after.confirmLabel).toBe('Open billing page')

    resetLocale()
    expect(billingDialogCopy({ is_nous: true } as never).cancelLabel).toBe('Dismiss')
  })

  it('plural leaves are chosen in code, so a pack can override each form', () => {
    applyLocale(
      'xx',
      pack({
        'libText.subagentTree.agentsOne': '{0} XX-agent',
        'libText.subagentTree.agentsOther': '{0} XX-agents',
        'libText.messages.backgroundAgentsFinishedOther': 'XX {0} done'
      })
    )

    const totals = {
      activeCount: 0,
      costUsd: 0,
      descendantCount: 1,
      filesTouched: 0,
      hotness: 0,
      inputTokens: 0,
      maxDepthFromHere: 0,
      outputTokens: 0,
      totalDuration: 0,
      totalTools: 0
    }

    expect(formatSummary(totals)).toContain('1 XX-agent')
    expect(formatSummary({ ...totals, descendantCount: 3 })).toContain('3 XX-agents')

    const [event] = toTranscriptMessages([
      { role: 'user', text: 'x', display_kind: 'async_delegation_complete', display_metadata: { task_count: 3 } }
    ])

    expect(event?.text).toBe('XX 3 done')
  })

  it('composer tokens and transcript trail prose follow the active catalog', () => {
    applyLocale(
      'xx',
      pack({
        'libText.attachments.imageToken': '[[ XX-Bild {0} ]]',
        'libText.text.showingLiveTail': 'XX-tail',
        'libText.text.clarifyHead': 'XX-frage ({0})',
        'libText.messages.longMessage': '{0} XX-lang'
      })
    )

    expect(imageToken(2)).toBe('[[ XX-Bild 2 ]]')
    expect(boundedLiveRenderText('abcdefghij', { maxChars: 4, maxLines: 10 })).toContain('[XX-tail; omitted')
    expect(formatAbandonedClarify([{ qid: 'q0', question: 'Why?' }], {}, 'timed out').split('\n')[0]).toBe(
      'XX-frage (1)'
    )
    expect(userDisplay('word '.repeat(2000))).toContain('XX-lang')
    expect(t('libText.text.argsLabel')).toBe('Args')
  })
})
