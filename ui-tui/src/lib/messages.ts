import type { Msg, Role } from '../types.js'

const isToolShelf = (msg: Msg | undefined) =>
  Boolean(msg?.kind === 'trail' && !msg.text && !msg.thinking?.trim() && msg.tools?.length)

export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] => {
  if (isToolShelf(msg) && isToolShelf(prev.at(-1))) {
    const last = prev.at(-1)!

    return [...prev.slice(0, -1), { ...last, tools: [...(last.tools ?? []), ...(msg.tools ?? [])] }]
  }

  return [...prev, msg]
}

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
