import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { Theme } from '../theme.js'
import { QueuedMessages, STEER_BRAILLE } from '../components/queuedMessages.js'
import { DEFAULT_THEME } from '../theme.js'
import { stripAnsi } from '../lib/text.js'

const mounted: Array<() => void> = []

const mountTree = (tree: React.ReactElement) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  let output = ''

  Object.assign(stdout, { columns: 120, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })

  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(tree, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  mounted.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  // @hermes/ink flushes the rendered frame to the stream at unmount/cleanup,
  // so read the settled frame now (same pattern as the other renderSync tests).
  for (const dispose of mounted.splice(0)) {
    dispose()
  }

  return {
    read: () => stripAnsi(output),
    clear: () => {
      output = ''
    }
  }
}

const t: Theme = DEFAULT_THEME

beforeEach(() => {
  // no per-test store state needed; QueuedMessages is a pure leaf
})

afterEach(() => {
  while (mounted.length) {
    mounted.pop()?.()
  }
})

describe('QueuedMessages pending strip', () => {
  it('renders a pending steer as row 1 with the steer tag and braille marker, above queue rows', () => {
    const view = mountTree(
      <QueuedMessages
        cols={100}
        pendingSteer="fix the path mapping before restarting"
        queueEditIdx={null}
        queued={['after that, run verify-paths.sh', 'then commit']}
        t={t}
      />
    )

    const text = view.read()

    // Header counts steer + queue together.
    expect(text).toContain('pending (3)')

    // Steer row is row 1, tagged, with a braille spinner.
    expect(text).toContain('1. ⠋ [steer] fix the path mapping before restarting')

    // Queue rows render below with their shifted indices.
    expect(text).toContain('2. after that, run verify-paths.sh')
    expect(text).toContain('3. then commit')

    // Row order: steer text appears before the first queue row.
    expect(text.indexOf('fix the path mapping')).toBeLessThan(text.indexOf('run verify-paths.sh'))
  })

  it('keeps the legacy "queued (N)" header and plain rows when no steer is pending', () => {
    const view = mountTree(
      <QueuedMessages cols={100} queueEditIdx={null} queued={['plain follow-up']} t={t} />
    )

    const text = view.read()

    expect(text).toContain('queued (1)')
    expect(text).toContain('1. plain follow-up')
    expect(text).not.toContain('[steer]')
    expect(text).not.toContain(STEER_BRAILLE)
  })

  it('renders nothing when both steer and queue are empty', () => {
    const view = mountTree(<QueuedMessages cols={100} queueEditIdx={null} queued={[]} t={t} />)

    expect(view.read().trim()).toBe('')
  })

  it('shows the editing hint on the steer row when steerEditIdx is active', () => {
    const view = mountTree(
      <QueuedMessages
        cols={100}
        pendingSteer="edit me"
        queueEditIdx={null}
        queued={['queue item']}
        steerEditIdx={0}
        t={t}
      />
    )

    const text = view.read()

    expect(text).toContain('editing 1')
    expect(text).toContain('Ctrl+X delete · Esc cancel')
    // Active steer row gets the ▸ cursor.
    expect(text).toContain('▸ 1.')
  })

  it('advertises edit affordances in the header when a steer is pending', () => {
    const view = mountTree(
      <QueuedMessages cols={100} pendingSteer="pending steer" queueEditIdx={null} queued={[]} t={t} />
    )

    expect(view.read()).toContain('↑↓ edit')
  })
})
