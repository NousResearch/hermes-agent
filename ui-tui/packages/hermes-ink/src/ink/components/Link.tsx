import type { ReactNode } from 'react'
import React from 'react'

import Text from './Text.js'
export type Props = {
  readonly children?: ReactNode
  readonly url: string
  // Kept for backwards-compat: prior versions rendered `fallback` instead of
  // the linked content on terminals where supportsHyperlinks() was false. We
  // now always emit the hyperlink metadata so the in-process click/hover
  // dispatcher can act on it regardless of the terminal's own OSC 8 support
  // (see comment in the function body), so `fallback` is no longer wired up.
  // Leaving the prop on the interface keeps existing call sites compiling.
  readonly fallback?: ReactNode
}

export default function Link({ children, url }: Props): React.ReactNode {
  // Always emit <ink-link>: the renderer stores `hyperlink` per cell in the
  // screen buffer, which the click dispatcher (Ink.getHyperlinkAt →
  // onHyperlinkClick) reads on mouseup to open URLs externally. Gating this
  // on supportsHyperlinks() broke clicks in Apple Terminal / any terminal
  // not on the OSC 8 allowlist — the cell's hyperlink field stayed empty,
  // so the click pipeline had nothing to open.
  //
  // The OSC 8 escape itself is emitted unconditionally by the renderer
  // (wrapWithOsc8Link in render-node-to-output.ts, oscLink in log-update.ts).
  // Terminals that don't understand OSC 8 silently strip it — including
  // Apple Terminal, which is why hover/click affordance has to come from
  // the in-process overlay (applyHyperlinkHoverHighlight) and not from the
  // terminal's own link rendering.
  const content = children ?? url

  return (
    <Text>
      <ink-link href={url}>{content}</ink-link>
    </Text>
  )
}
