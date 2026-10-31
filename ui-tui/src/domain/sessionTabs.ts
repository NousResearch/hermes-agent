/** Presentation only: stable session IDs and source titles are not modified. */
export function readableSessionTitle(title?: string | null): string {
  let value = (title || '').replace(/\s+/g, ' ').trim()

  if (value.startsWith('[OpenCode] ')) {
    value = value.slice('[OpenCode] '.length).replace(/ \[[A-Za-z0-9]{10}\](?: \(import segment \d+\/\d+\))?$/, '')

    try {
      const parsed = JSON.parse(value)

      if (typeof parsed?.title === 'string') {
        value = parsed.title
      }
    } catch {
      // Most titles are plain text, not an imported JSON title wrapper.
    }
  }

  return value.replace(/\p{Cc}/gu, '').trim()
}

export type TabShortcut =
  { kind: 'new' } | { kind: 'close' } | { kind: 'index'; index: number } | { kind: 'cycle'; direction: 1 | -1 }

/** Leave plain Tab, Ctrl+T (agents), and Ctrl+W (delete word) untouched. */
export function sessionTabShortcut(
  input: string,
  key: {
    meta?: boolean
    ctrl?: boolean
    shift?: boolean
    leftArrow?: boolean
    rightArrow?: boolean
    pageUp?: boolean
    pageDown?: boolean
    tab?: boolean
  }
): TabShortcut | undefined {
  if (key.ctrl && key.tab) {
    return { kind: 'cycle', direction: key.shift ? -1 : 1 }
  }

  if (key.ctrl && (key.pageUp || key.pageDown)) {
    return { kind: 'cycle', direction: key.pageUp ? -1 : 1 }
  }

  if (!key.meta || key.ctrl) {
    return undefined
  }

  // Alt+Left/Right remain the composer's word-navigation keys.

  if (/^[1-9]$/.test(input)) {
    return { kind: 'index', index: Number(input) - 1 }
  }

  if (input.toLowerCase() === 'n') {
    return { kind: 'new' }
  }

  if (input.toLowerCase() === 'w') {
    return { kind: 'close' }
  }

  return undefined
}
