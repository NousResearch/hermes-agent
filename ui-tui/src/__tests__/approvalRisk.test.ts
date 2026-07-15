import { describe, expect, it } from 'vitest'

import { approvalRiskText } from '../lib/approval-risk.js'

describe('approvalRiskText', () => {
  it('formats the bilingual banner shown in an approval prompt', () => {
    expect(
      approvalRiskText({
        riskLabel: { en: 'File deletion', zh: '删除文件' },
        riskWarning: {
          en: 'This operation can permanently delete files.',
          zh: '此操作可能永久删除文件。'
        }
      })
    ).toBe('File deletion / 删除文件 — This operation can permanently delete files. / 此操作可能永久删除文件。')
  })

  it('hides incomplete risk copy', () => {
    expect(approvalRiskText({ riskLabel: { en: 'File deletion', zh: '删除文件' } })).toBeNull()
    expect(
      approvalRiskText({
        riskLabel: { en: 'File deletion', zh: '删除文件' },
        riskWarning: { en: 'This operation can permanently delete files.', zh: '' }
      })
    ).toBeNull()
  })
})
