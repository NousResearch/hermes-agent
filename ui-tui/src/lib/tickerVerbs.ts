import { VERBS } from '../content/verbs.js'

export const getVerbPadLen = (verbs: readonly string[] = VERBS) =>
  verbs.reduce((max, verb) => Math.max(max, verb.length), 0) + 1 // + ellipsis

export const DEFAULT_VERB_PAD_LEN = getVerbPadLen(VERBS)

export const padVerb = (verb: string, padLen = DEFAULT_VERB_PAD_LEN) => `${verb}…`.padEnd(padLen, ' ')

export const normalizeTickerVerbs = (raw: unknown): string[] => {
  if (!Array.isArray(raw)) {
    return VERBS
  }

  const verbs = raw
    .filter((value): value is string => typeof value === 'string')
    .map(value => value.trim())
    .filter(Boolean)

  return verbs.length ? verbs : VERBS
}
