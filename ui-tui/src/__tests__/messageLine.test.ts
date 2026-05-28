import { describe, expect, it } from 'vitest'

import { splitAssistantLogPrefix } from '../lib/assistantLogPrefix.js'

describe('splitAssistantLogPrefix', () => {
  it('extracts prepared progress prefixes for emphasized rendering', () => {
    expect(splitAssistantLogPrefix('**진행상황:**\n파일 확인 중입니다.')).toEqual({
      body: '파일 확인 중입니다.',
      prefix: '진행상황'
    })
  })

  it('extracts same-line prepared progress prefixes', () => {
    expect(splitAssistantLogPrefix('검증: 테스트를 다시 실행합니다.')).toEqual({
      body: '테스트를 다시 실행합니다.',
      prefix: '검증'
    })
  })

  it('leaves ordinary assistant prose untouched', () => {
    expect(splitAssistantLogPrefix('일반 답변입니다.')).toBeNull()
  })
})
