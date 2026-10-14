import { en } from '../i18n/en.js'
import { messages } from '../i18n/runtime.js'

/** Tool name → progress verb, resolved against the active language at call time. */
export const toolVerbs = (): Record<string, string> => messages().content.verbs

export const toolVerb = (name: string): string | undefined => toolVerbs()[name]

export const thinkingVerbs = (): string[] => Object.values(messages().content.thinkingVerbs)

// Legacy transcript cleanup recognizes the backend's English progress tokens.
export const VERBS = Object.values(en.content.thinkingVerbs)
