import type { Msg, Role } from '../types.js'

export function upsert(prev: Msg[], role: Role, text: string): Msg[] {
  return prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
}
