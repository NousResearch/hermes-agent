import type { Msg, Role } from '../types.js'

import { appendToolShelfMessage } from './liveProgress.js'

let msgIdSeq = 0

const nextMsgId = () => `msg-${++msgIdSeq}`

export const ensureMsgId = <T extends Msg>(msg: T): T & { id: string } =>
  msg.id ? (msg as T & { id: string }) : { ...msg, id: nextMsgId() }

export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] => appendToolShelfMessage(prev, ensureMsgId(msg))

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role
    ? [...prev.slice(0, -1), ensureMsgId({ ...prev.at(-1)!, role, text })]
    : [...prev, ensureMsgId({ role, text })]
