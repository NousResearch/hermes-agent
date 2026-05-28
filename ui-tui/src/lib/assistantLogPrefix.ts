const ASSISTANT_LOG_PREFIXES = [
  '진행상황',
  '완료',
  '결과',
  '확인',
  '검증',
  '테스트',
  '주의',
  '경고',
  '참고',
  '다음',
  '계획',
  '요약',
  '문제',
  '해결',
  'PLAN',
  'VERIFY'
]

const ASSISTANT_LOG_PREFIX_RE = new RegExp(
  `^\\s*(?:\\*\\*)?(${ASSISTANT_LOG_PREFIXES.join('|')})(?:\\*\\*)?\\s*[:：](?:\\*\\*)?\\s*([\\s\\S]*)$`,
  'i'
)

export interface AssistantLogPrefix {
  body: string
  prefix: string
}

export const splitAssistantLogPrefix = (text: string): null | AssistantLogPrefix => {
  const match = ASSISTANT_LOG_PREFIX_RE.exec(text)

  if (!match) {
    return null
  }

  return {
    body: (match[2] ?? '').replace(/^\s*\n/, '').trimStart(),
    prefix: match[1] ?? ''
  }
}
