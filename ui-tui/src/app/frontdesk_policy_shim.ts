// Frontdesk routing policy — TypeScript shim (Phase 2 substrate).
//
// This module is the TypeScript mirror of `agent/frontdesk_policy.py`.  Python
// is the source of truth; this shim exists so the TUI can classify a user
// fragment *before* it round-trips through the gateway/process boundary, which
// keeps the status bar's "control: ..." line reactive.
//
// Phase 2 contract — read-only and not wired:
//   * No imports from `busyIntegrated`, `useSubmission`, or any other surface
//     module.  This file is a leaf.
//   * No callers in Phase 2.  Phase 5 (`/mode frontdesk` opt-in) is the first
//     phase that calls `classifyFrontdeskFragment` from `useSubmission.ts`.
//   * Behaviour parity with the Python policy is validated in Phase 3 via the
//     shared `tests/agent/data/frontdesk_intents_ko.yaml` corpus.
//
// Hard boundaries this respects (PRD §9.2 / design review §9.2):
//   * No `/mode frontdesk` exposure.
//   * No persona / prompt change.
//   * No `_pending_input` / queue mutation.
//   * No worker dispatch.

// ---------------------------------------------------------------------------
// Enums (string literal unions; vitest-friendly, no `enum` runtime cost)
// ---------------------------------------------------------------------------

export type FrontdeskRecommendation = 'main' | 'worker_lane' | 'steer' | 'control'

export type FrontdeskConfidence = 'low' | 'medium' | 'high'

export type FrontdeskSignal =
  | 'research'
  | 'artifact'
  | 'code_edit'
  | 'long'
  | 'many_tools'
  | 'status'
  | 'stop'
  | 'steer'
  | 'ack'
  | 'duplicate'
  | 'noise'
  | 'korean'
  | 'explicit_worker_req'
  | 'explicit_main_req'

export interface FrontdeskPolicyDecision {
  readonly recommendation: FrontdeskRecommendation
  readonly signals: ReadonlySet<FrontdeskSignal>
  readonly confidence: FrontdeskConfidence
  readonly debugLabel: string
  readonly rawText: string
  readonly hasKorean: boolean
  readonly shouldDelegate: boolean
  readonly isControl: boolean
  readonly isStop: boolean
  readonly isAck: boolean
}

// ---------------------------------------------------------------------------
// Vocabularies — kept in sync with `agent/frontdesk_policy.py`.  Any edit here
// MUST be mirrored on the Python side and re-validated against the corpus.
// ---------------------------------------------------------------------------

const STOP_TOKENS_EN: readonly string[] = [
  'stop',
  'cancel',
  'abort',
  'halt',
  'nevermind',
  'never mind',
  'forget it',
  '/stop',
  '/cancel',
  '/abort',
  '/kill',
]

const STOP_TOKENS_KO: readonly string[] = [
  '그만',
  '그만해',
  '그만둬',
  '중단',
  '취소',
  '됐어',
  '멈춰',
  '잠깐만',
  '잠깐',
  '기다려',
]

const ACK_TOKENS_EN: readonly string[] = [
  'thanks',
  'thanks!',
  'thank you',
  'thank you!',
  'thx',
  'ty',
]

const ACK_TOKENS_KO: readonly string[] = [
  '고마워',
  '고맙습니다',
  '감사',
  '감사합니다',
  '수고',
  '수고했어',
]

const STATUS_ANCHORS_EN: readonly string[] = [
  'status',
  'what are you doing',
  "what's running",
  'show me',
  'list',
  '/tasks',
  '/agents',
  '/mode',
]

const STATUS_ANCHORS_KO: readonly string[] = [
  '상태',
  '어디까지',
  '뭐 했어',
  '뭐해',
  '뭐 해',
  '진행',
  '어떤 lane',
  '어떤 워커',
  '지금 뭐',
]

const ARTIFACT_ANCHORS_EN: readonly string[] = [
  '.md',
  'report',
  'report.md',
  'summary',
  'summary.md',
  'draft',
  'write up',
  'writeup',
  'write a',
  'write the',
  'compose',
  'produce a',
  '.csv',
  '.svg',
  '.png',
  '.pdf',
  '.tsv',
  '.json',
]

const ARTIFACT_ANCHORS_KO: readonly string[] = [
  '리포트',
  '보고서',
  '정리해',
  '정리해줘',
  '작성해',
  '작성해줘',
  '초안',
  '문서로',
]

const RESEARCH_ANCHORS_EN: readonly string[] = [
  'investigate',
  'research',
  'search for',
  'look up',
  'find out',
  'crawl',
  'scrape',
  'audit',
  'deep dive',
  'deep-dive',
  'explore the',
  'study the',
]

const RESEARCH_ANCHORS_KO: readonly string[] = [
  '조사',
  '조사해',
  '조사해줘',
  '찾아봐',
  '리서치',
  '크롤',
  '크롤링',
  '탐색',
  '분석해',
]

const CODE_EDIT_ANCHORS_EN: readonly string[] = [
  'implement',
  'refactor',
  'fix the',
  'patch',
  'rewrite',
  'port the',
  'migrate the',
  'build the',
  'add a tool',
]

const CODE_EDIT_ANCHORS_KO: readonly string[] = [
  '구현해',
  '구현해줘',
  '고쳐',
  '고쳐줘',
  '수정해',
  '수정해줘',
  '리팩',
  '리팩터',
  '재작성',
]

const EXPLICIT_WORKER_ANCHORS_EN: readonly string[] = [
  'in the background',
  'background',
  'delegate this',
  'delegate it',
  'delegate to',
  'give it to a worker',
  'worker lane',
  'kick off a worker',
  'spawn a worker',
]

const EXPLICIT_WORKER_ANCHORS_KO: readonly string[] = [
  '백그라운드',
  '워커에 맡겨',
  '워커에게 맡겨',
  '워커한테 맡겨',
  '클로드한테 맡겨',
  '위임해',
  '위임해줘',
  '맡겨줘',
]

const EXPLICIT_MAIN_ANCHORS_EN: readonly string[] = [
  'do it yourself',
  'do it now',
  'right now',
  'yourself',
  'main thread',
  'in line',
  'inline',
  'directly',
  'no worker',
  'no workers',
]

const EXPLICIT_MAIN_ANCHORS_KO: readonly string[] = [
  '직접 해',
  '직접해',
  '지금 해',
  '지금해',
  '지금 보여줘',
  '바로 해',
  '바로해',
  '여기서 해',
  '메인에서',
]

const STEER_PREFIXES_EN: readonly string[] = [
  'also ',
  'and also ',
  'additionally ',
  'btw ',
  'by the way ',
  'actually ',
  'wait ',
  'oh, ',
  'oh ',
]

const STEER_PREFIXES_KO: readonly string[] = [
  '근데 ',
  '그리고 ',
  '추가로 ',
  '참, ',
  '참 ',
  '아 그리고 ',
  '아, 그리고 ',
]

const LONG_WORDS_EN: readonly string[] = [
  'all the',
  'every',
  'across the codebase',
  'across the repo',
  'the whole',
  'comprehensive',
  'thorough',
  'deep',
  'end-to-end',
  'end to end',
]

const LONG_WORDS_KO: readonly string[] = ['전체', '모든', '꼼꼼', '철저', '전반']

const TOOL_ACTION_KEYWORDS_EN: readonly string[] = [
  'read',
  'grep',
  'search',
  'find',
  'write',
  'edit',
  'patch',
  'fetch',
  'crawl',
  'build',
  'run',
  'test',
  'deploy',
  'list',
  'show',
]

const TOOL_ACTION_KEYWORDS_KO: readonly string[] = ['읽어', '찾아', '써', '수정', '빌드', '테스트', '실행', '보여']

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Hangul detection.  Matches the regex used in the Python policy.
const HANGUL_RE = /[가-힯ᄀ-ᇿ㄰-㆏]/u

const TRAILING_NOISE_RE = /[\s.!?…~^"'`)\-:,;]+$/u
const TRAILING_EMOTE_RE = /(?:[:;][-]?[)dpo](?:[)dpo])*|ㅎ+|ㅋ+)$/iu

function stripTrailingNoise(body: string): string {
  let trimmed = body.trim()
  for (let i = 0; i < 3; i++) {
    const before = trimmed
    trimmed = trimmed.replace(TRAILING_NOISE_RE, '').replace(TRAILING_EMOTE_RE, '').trim()
    if (trimmed === before) break
  }
  return trimmed
}

function isWholeBodyMatch(lowBody: string, vocab: readonly string[]): boolean {
  const core = stripTrailingNoise(lowBody)
  if (!core) return false
  const target = core
  return vocab.some(v => v.toLowerCase() === target)
}

function containsAny(haystack: string, needles: readonly string[]): boolean {
  return needles.some(n => haystack.includes(n))
}

function startsWithAny(body: string, prefixes: readonly string[]): boolean {
  return prefixes.some(p => body.startsWith(p))
}

function looksKorean(text: string): boolean {
  return HANGUL_RE.test(text)
}

function looksLikeSteer(lowBody: string, body: string): boolean {
  if (startsWithAny(lowBody, STEER_PREFIXES_EN)) return true
  if (startsWithAny(body, STEER_PREFIXES_KO)) return true
  for (const sep of [', ', ',\n', ',  ', '， ', '，\n', '，  ']) {
    const cutLow = lowBody.indexOf(sep)
    if (cutLow !== -1) {
      const tailLow = lowBody.slice(cutLow + sep.length).trimStart()
      if (STEER_PREFIXES_EN.some(p => tailLow.startsWith(p))) return true
    }
    const cutRaw = body.indexOf(sep)
    if (cutRaw !== -1) {
      const tailRaw = body.slice(cutRaw + sep.length).trimStart()
      if (STEER_PREFIXES_KO.some(p => tailRaw.startsWith(p))) return true
    }
  }
  return false
}

function estimateToolCalls(lowBody: string, body: string): number {
  const seen = new Set<string>()
  let count = 0
  for (const kw of TOOL_ACTION_KEYWORDS_EN) {
    if (lowBody.includes(kw) && !seen.has(kw)) {
      seen.add(kw)
      count += 1
    }
  }
  for (const kw of TOOL_ACTION_KEYWORDS_KO) {
    if (body.includes(kw) && !seen.has(kw)) {
      seen.add(kw)
      count += 1
    }
  }
  // " and then " always adds; " and " adds half-bucket.
  const andThen = (lowBody.match(/ and then /g) ?? []).length
  const ands = (lowBody.match(/ and /g) ?? []).length
  count += andThen + Math.floor(ands / 2)
  return count
}

function looksLong(body: string, lowBody: string): boolean {
  if (containsAny(lowBody, LONG_WORDS_EN)) return true
  if (containsAny(body, LONG_WORDS_KO)) return true
  if (body.length >= 240) return true
  return false
}

function isStatusQuery(body: string, lowBody: string): boolean {
  if (containsAny(lowBody, STATUS_ANCHORS_EN)) return true
  if (containsAny(body, STATUS_ANCHORS_KO)) return true
  if (body.length <= 12 && body.endsWith('?')) return true
  return false
}

// ---------------------------------------------------------------------------
// Decision builder
// ---------------------------------------------------------------------------

interface DecisionInit {
  recommendation: FrontdeskRecommendation
  signals: ReadonlySet<FrontdeskSignal>
  confidence: FrontdeskConfidence
  debugLabel: string
  rawText: string
}

function buildDecision(init: DecisionInit): FrontdeskPolicyDecision {
  return {
    recommendation: init.recommendation,
    signals: init.signals,
    confidence: init.confidence,
    debugLabel: init.debugLabel,
    rawText: init.rawText,
    hasKorean: init.signals.has('korean'),
    shouldDelegate: init.recommendation === 'worker_lane',
    isControl: init.recommendation === 'control',
    isStop: init.signals.has('stop'),
    isAck: init.signals.has('ack'),
  }
}

// ---------------------------------------------------------------------------
// Public classifier
// ---------------------------------------------------------------------------

export interface ClassifyOptions {
  readonly langHint?: 'ko' | 'en' | null
}

/**
 * Pure mirror of `agent.frontdesk_policy.classify_request`.  Same `(text,
 * langHint)` → same decision.  No side effects, no async, no I/O.
 *
 * Phase 2: function-only.  No surface adapter calls this yet.
 */
export function classifyFrontdeskFragment(
  text: string,
  options: ClassifyOptions = {},
): FrontdeskPolicyDecision {
  const raw = typeof text === 'string' ? text : ''
  const body = raw.trim().normalize('NFC')
  const lowBody = body.toLowerCase()

  const signals = new Set<FrontdeskSignal>()

  // 1. Empty / whitespace
  if (!body) {
    signals.add('noise')
    return buildDecision({
      recommendation: 'control',
      signals,
      confidence: 'high',
      debugLabel: 'noise:empty',
      rawText: raw,
    })
  }

  // 2. Korean detection
  const isKorean = options.langHint === 'ko' || looksKorean(body)
  if (isKorean) signals.add('korean')

  // 3. Whole-body STOP
  if (isWholeBodyMatch(lowBody, STOP_TOKENS_EN) || isWholeBodyMatch(body, STOP_TOKENS_KO)) {
    signals.add('stop')
    return buildDecision({
      recommendation: 'control',
      signals,
      confidence: 'high',
      debugLabel: 'stop',
      rawText: raw,
    })
  }

  // 4. Whole-body ACK
  if (isWholeBodyMatch(lowBody, ACK_TOKENS_EN) || isWholeBodyMatch(body, ACK_TOKENS_KO)) {
    signals.add('ack')
    return buildDecision({
      recommendation: 'control',
      signals,
      confidence: 'high',
      debugLabel: 'ack',
      rawText: raw,
    })
  }

  // 5. Explicit overrides — collected; applied as tie-breakers below.
  if (containsAny(lowBody, EXPLICIT_WORKER_ANCHORS_EN) || containsAny(body, EXPLICIT_WORKER_ANCHORS_KO)) {
    signals.add('explicit_worker_req')
  }
  if (containsAny(lowBody, EXPLICIT_MAIN_ANCHORS_EN) || containsAny(body, EXPLICIT_MAIN_ANCHORS_KO)) {
    signals.add('explicit_main_req')
  }

  // 6. Status query (beats worker anchors so "status of report.md?" stays MAIN).
  if (isStatusQuery(body, lowBody)) {
    signals.add('status')
    return buildDecision({
      recommendation: 'main',
      signals,
      confidence: 'medium',
      debugLabel: 'status',
      rawText: raw,
    })
  }

  // 7. Steer candidate (prefix-based; surface verifies main-in-flight)
  const steerCandidate = looksLikeSteer(lowBody, body)
  if (steerCandidate) signals.add('steer')

  // 8. Worker-candidate signals
  if (containsAny(lowBody, ARTIFACT_ANCHORS_EN) || containsAny(body, ARTIFACT_ANCHORS_KO)) {
    signals.add('artifact')
  }
  if (containsAny(lowBody, RESEARCH_ANCHORS_EN) || containsAny(body, RESEARCH_ANCHORS_KO)) {
    signals.add('research')
  }
  if (containsAny(lowBody, CODE_EDIT_ANCHORS_EN) || containsAny(body, CODE_EDIT_ANCHORS_KO)) {
    signals.add('code_edit')
  }
  if (looksLong(body, lowBody)) signals.add('long')
  if (estimateToolCalls(lowBody, body) >= 3) signals.add('many_tools')

  // 9. Tie-breakers
  if (signals.has('explicit_main_req')) {
    return buildDecision({
      recommendation: 'main',
      signals,
      confidence: 'high',
      debugLabel: 'main:explicit',
      rawText: raw,
    })
  }
  if (signals.has('explicit_worker_req')) {
    return buildDecision({
      recommendation: 'worker_lane',
      signals,
      confidence: 'high',
      debugLabel: 'worker:explicit',
      rawText: raw,
    })
  }

  // 10. Strong worker signals
  const strong = (['artifact', 'research', 'code_edit'] as const).filter(s => signals.has(s))
  if (strong.length > 0) {
    const hasShape = signals.has('long') || signals.has('many_tools')
    const confidence: FrontdeskConfidence = strong.length >= 2 || hasShape ? 'high' : 'medium'
    return buildDecision({
      recommendation: 'worker_lane',
      signals,
      confidence,
      debugLabel: `worker:${strong.sort().join('+')}`,
      rawText: raw,
    })
  }

  // 11. Weak worker shape (LONG + MANY_TOOLS together)
  if (signals.has('long') && signals.has('many_tools')) {
    return buildDecision({
      recommendation: 'worker_lane',
      signals,
      confidence: 'medium',
      debugLabel: 'worker:shape',
      rawText: raw,
    })
  }

  // 12. Steer candidate without worker signal → STEER recommendation
  if (steerCandidate) {
    return buildDecision({
      recommendation: 'steer',
      signals,
      confidence: 'low',
      debugLabel: 'steer:candidate',
      rawText: raw,
    })
  }

  // 13. Default → MAIN
  return buildDecision({
    recommendation: 'main',
    signals,
    confidence: signals.size > 0 ? 'medium' : 'low',
    debugLabel: 'main:default',
    rawText: raw,
  })
}

// Re-export the vocabularies so a parity test (Phase 3) can verify the TS and
// Python sides see the same anchors.  Marked `_` -prefixed because they are
// internal-by-convention; the corpus parity test imports them via the named
// re-exports without surfacing them in the editor's autocomplete for general
// app code.
export const __FRONTDESK_VOCAB__ = {
  STOP_TOKENS_EN,
  STOP_TOKENS_KO,
  ACK_TOKENS_EN,
  ACK_TOKENS_KO,
  STATUS_ANCHORS_EN,
  STATUS_ANCHORS_KO,
  ARTIFACT_ANCHORS_EN,
  ARTIFACT_ANCHORS_KO,
  RESEARCH_ANCHORS_EN,
  RESEARCH_ANCHORS_KO,
  CODE_EDIT_ANCHORS_EN,
  CODE_EDIT_ANCHORS_KO,
  EXPLICIT_WORKER_ANCHORS_EN,
  EXPLICIT_WORKER_ANCHORS_KO,
  EXPLICIT_MAIN_ANCHORS_EN,
  EXPLICIT_MAIN_ANCHORS_KO,
  STEER_PREFIXES_EN,
  STEER_PREFIXES_KO,
} as const
