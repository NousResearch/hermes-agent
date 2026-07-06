import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it } from 'vitest'

import Box from './components/Box.js'
import Text from './components/Text.js'
import Ink from './ink.js'

// Regression: the bare `/model` slash command (and any other
// FloatingOverlays-based overlay — pager, sessions switcher, pet/skills/
// plugins hub, completions menu) silently failed to paint. React committed
// the overlay correctly (state set, component mounted, RPC resolved), but
// nothing ever appeared on screen.
//
// Root cause: the ui-tui composer wraps its floating overlays in a plain
// <Box position="relative"> that has NO other normal-flow children while an
// overlay is open (the input row is hidden behind it) — its only child is
// the overlay's own <Box position="absolute" bottom="100%">. Yoga excludes
// absolutely-positioned children from a parent's intrinsic height, so that
// wrapper computes height=0. render-node-to-output.ts has an optimization
// that SKIPS rendering (and marks clean) any node with height=0 whose next/
// previous sibling lands on the same row — added to stop a genuine ghosting
// bug (HelpV2's shortcuts column) where a squeezed, genuinely-empty box
// could leave stale characters behind. But it doesn't distinguish "empty"
// from "zero-height parent of an absolute-positioned child with real
// content" — so it discarded the whole overlay subtree and marked the
// wrapper clean, meaning every subsequent frame took the blit-fast-path and
// never descended into it again. The overlay was mounted forever, invisible
// forever.
class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 20
  rows = 6
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const makeInk = () => {
  const stdout = new FakeTty()
  const stdin = new EventEmitter() as unknown as NodeJS.ReadStream
  const stderr = new FakeTty()

  const ink = new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return { ink, stdout }
}

describe('zero-height node with an absolute-positioned child', () => {
  it('still paints the absolute child even when squeezed to h=0 and sharing a row with a sibling', () => {
    const { ink, stdout } = makeInk()

    // Structurally mirrors ComposerPane's <Box position="relative"> wrapper
    // around <FloatingOverlays/>: a column of siblings where the middle box
    // has ONLY an absolute-positioned child (so Yoga gives it h=0), landing
    // it on the same row as the sibling below — the exact trigger for the
    // h===0 && siblingSharesY skip.
    const tree = React.createElement(
      Box,
      { flexDirection: 'column' },
      React.createElement(Text, { key: 'before' }, 'before'),
      // Blank spacer row: the absolute overlay's bottom='100%' extends
      // upward from the h=0 wrapper's top edge by the overlay's own
      // content height (1 line), landing here rather than on 'before'.
      React.createElement(Box, { flexShrink: 0, height: 1, key: 'spacer' }),
      React.createElement(
        Box,
        { key: 'wrapper', position: 'relative' },
        React.createElement(
          Box,
          { bottom: '100%', left: 0, position: 'absolute', right: 0 },
          React.createElement(Text, null, 'OVERLAY-CONTENT')
        )
      ),
      React.createElement(Text, { key: 'after' }, 'after')
    )

    ink.render(tree)
    ink.onRender()

    const out = stdout.chunks.join('')

    expect(out).toContain('before')
    expect(out).toContain('after')
    // This is the assertion that fails without the hasAbsoluteChild guard:
    // the overlay's own text never reaches the terminal at all.
    expect(out).toContain('OVERLAY-CONTENT')

    ink.unmount()
  })
})
