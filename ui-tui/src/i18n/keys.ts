// Flatten a catalog into its sorted dotted leaf keys. Shared by the
// `_keys.tui.json` emitter and its test.

import { isRecord } from './merge.js'

export function flattenKeys(tree: unknown, prefix = ''): string[] {
  if (!isRecord(tree)) {
    return prefix ? [prefix] : []
  }

  const out: string[] = []

  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key

    if (isRecord(value)) {
      out.push(...flattenKeys(value, path))
    } else {
      out.push(path)
    }
  }

  return out.sort()
}
