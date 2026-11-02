import { messages } from '../i18n/runtime.js'

/** Tool name → progress verb, resolved against the active language at call time. */
export const toolVerbs = (): Record<string, string> => messages().content.verbs

export const toolVerb = (name: string): string | undefined => toolVerbs()[name]

export const VERBS = [
  'pondering',
  'contemplating',
  'musing',
  'cogitating',
  'ruminating',
  'deliberating',
  'mulling',
  'reflecting',
  'processing',
  'reasoning',
  'analyzing',
  'computing',
  'synthesizing',
  'formulating',
  'brainstorming'
]
