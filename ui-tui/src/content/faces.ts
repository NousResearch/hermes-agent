import { messages } from '../i18n/runtime.js'

/** Kaomoji spinner frames, in catalog order, resolved against the active language at call time. */
export const faces = (): string[] => Object.values(messages().content.faces)
