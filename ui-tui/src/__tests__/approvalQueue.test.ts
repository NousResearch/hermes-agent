import { afterEach, describe, expect, it } from 'vitest'

import { dismissApproval, getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { approvalResponseResolved } from '../app/useMainApp.js'
import { approvalOptions, isApprovalExpired } from '../components/prompts.js'
import type { ApprovalReq } from '../types.js'

const approval = (requestId: string): ApprovalReq => ({
  command: `echo ${requestId}`,
  description: 'review',
  requestId
})

afterEach(() => resetOverlayState())

describe('approval queue', () => {
  it('promotes the next exact request after the head resolves', () => {
    const first = approval('request-1')
    const second = approval('request-2')
    patchOverlayState({ approval: first, approvalQueue: [first, second] })

    dismissApproval(first.requestId)

    expect(getOverlayState().approval?.requestId).toBe(second.requestId)
    expect(getOverlayState().approvalQueue?.map(item => item.requestId)).toEqual(['request-2'])
  })

  it('removes a resolved request without dropping other pending requests', () => {
    const first = approval('request-1')
    const second = approval('request-2')
    const third = approval('request-3')
    patchOverlayState({ approval: first, approvalQueue: [first, second, third] })

    dismissApproval(second.requestId)

    expect(getOverlayState().approval?.requestId).toBe(first.requestId)
    expect(getOverlayState().approvalQueue?.map(item => item.requestId)).toEqual(['request-1', 'request-3'])
  })

  it('only treats an approval response as resolved when the server resolves one request', () => {
    expect(approvalResponseResolved({ resolved: 1 })).toBe(true)
    expect(approvalResponseResolved({ resolved: 0 })).toBe(false)
    expect(approvalResponseResolved({ status: 'ok' })).toBe(false)
    expect(approvalResponseResolved(null)).toBe(false)
  })

  it('fails closed for missing capability flags and malformed choices', () => {
    expect(approvalOptions({ ...approval('request-4'), choices: ['session'] })).toEqual(['deny'])
    expect(approvalOptions({ ...approval('request-5'), choices: [], allowSession: 'yes' as unknown as boolean })).toEqual(['deny'])
  })

  it('recognizes an expired approval before accepting input', () => {
    expect(isApprovalExpired({ ...approval('request-6'), expiresAt: 999 }, 1000)).toBe(true)
    expect(isApprovalExpired({ ...approval('request-7'), expiresAt: 1001 }, 1000)).toBe(false)
  })
})
