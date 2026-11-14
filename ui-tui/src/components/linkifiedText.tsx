import { Text } from '@hermes/ink'
import type { ComponentProps, ReactNode } from 'react'

import type { Theme } from '../theme.js'

import { renderLink } from './markdown.js'

const BARE_URL_RE = /https?:\/\/[^\s<>"']+/g

/**
 * Plain prompt text with each bare http(s) URL rendered as an OSC 8 link, the same way chat
 * Markdown renders one (openai/codex#51439/#51449/#51458). Trailing prose punctuation stays
 * outside the link, so `see https://x.dev/keys.` does not open `keys.`.
 */
export function linkifyText(text: string, t: Theme): ReactNode[] {
  const parts: ReactNode[] = []
  let last = 0

  for (const m of text.matchAll(BARE_URL_RE)) {
    const url = m[0].replace(/[),.;:!?]+$/g, '')
    const start = m.index ?? 0

    if (start > last) {
      parts.push(text.slice(last, start))
    }

    parts.push(renderLink(parts.length, t, url))
    last = start + url.length
  }

  if (last < text.length) {
    parts.push(text.slice(last))
  }

  return parts
}

/** `<Text>` whose bare URLs are clickable; props other than `text`/`t` pass through. */
export function LinkifiedText({
  text,
  t,
  ...props
}: { text: string; t: Theme } & Omit<ComponentProps<typeof Text>, 'children'>) {
  return <Text {...props}>{linkifyText(text, t)}</Text>
}
