import { messages } from '../i18n/runtime.js'
import { pick } from '../lib/text.js'

export const placeholders = (): string[] => Object.values(messages().composer.placeholders)

// Picked once per launch by index so the same slot survives a language switch.
const slot = Math.floor(Math.random() * placeholders().length)

export const placeholder = (): string => placeholders()[slot] ?? pick(placeholders())
