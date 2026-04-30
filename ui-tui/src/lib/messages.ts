import type { Msg, Role } from '../types.js'

import { appendToolShelfMessage } from './liveProgress.js'

export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] => appendToolShelfMessage(prev, msg)

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
