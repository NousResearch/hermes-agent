import { PassThrough } from 'stream'

import { renderSync, Text } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it } from 'vitest'

import { GridStreamsDemo, STREAM_DEFS } from '../components/gridStreamsDemo.js'
import { GridAreas, type GridAreaWidget, WidgetGrid, type WidgetGridWidget } from '../components/widgetGrid.js'
import { stripAnsi } from '../lib/text.js'
import { GRID_STREAM_COUNT, type GridTestState } from '../sdk/apps/gridTestState.js'
import { DEFAULT_THEME } from '../theme.js'

function StatefulCell({ label }: { label: string }) {
  const [value] = useState(label)

  return <Text>{value}</Text>
}

const renderToText = (node: React.ReactElement) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

const renderGrid = (widgets: WidgetGridWidget[]) =>
  renderToText(<WidgetGrid cols={80} columns={2} gap={1} paddingX={0} widgets={widgets} />)

describe('WidgetGrid component composition', () => {
  it('renders stateful direct children and nested grids inside cells', () => {
    const output = renderGrid([
      {
        children: <StatefulCell label="stateful-c1" />,
        id: 'stateful'
      },
      {
        children: (
          <WidgetGrid
            cols={38}
            columns={2}
            gap={1}
            paddingX={0}
            widgets={[
              { children: <StatefulCell label="nested-c1" />, id: 'nested-c1' },
              { render: () => <StatefulCell label="nested-c2" />, id: 'nested-c2' }
            ]}
          />
        ),
        id: 'nested-grid'
      }
    ])

    expect(output).toContain('stateful-c1')
    expect(output).toContain('nested-c1')
    expect(output).toContain('nested-c2')
  })
})

describe('GridAreas component', () => {
  it('renders a rowSpan cell alongside stacked cells at their solved rects', () => {
    const widgets: GridAreaWidget[] = [
      { id: 'tall', render: cell => <Text>{`tall ${cell.width}x${cell.height}`}</Text>, rowSpan: 2 },
      { children: <Text>top-right</Text>, id: 'b' },
      { children: cell => <Text>{`bottom-right y${cell.y}`}</Text>, id: 'c' }
    ]

    const output = renderToText(<GridAreas columns={2} gap={0} height={6} rowGap={0} widgets={widgets} width={40} />)

    // tall spans both rows of the 6-row grid at 20 cells wide.
    expect(output).toContain('tall 20x6')
    expect(output).toContain('top-right')
    expect(output).toContain('bottom-right y3')
  })

  it('gives fixed header/footer rows their size and the fr body the rest', () => {
    const widgets: GridAreaWidget[] = [
      { children: cell => <Text>{`header h${cell.height}`}</Text>, id: 'header' },
      { children: cell => <Text>{`body h${cell.height}`}</Text>, id: 'body' },
      { children: cell => <Text>{`footer h${cell.height}`}</Text>, id: 'footer' }
    ]

    const output = renderToText(
      <GridAreas columns={1} gap={0} height={12} rowGap={0} rows={[1, { fr: 1 }, 1]} widgets={widgets} width={30} />
    )

    expect(output).toContain('header h1')
    expect(output).toContain('body h10')
    expect(output).toContain('footer h1')
  })
})

describe('GridStreamsDemo', () => {
  const streamsState: GridTestState = {
    activeCol: 0,
    activeRow: 0,
    areas: false,
    cols: 4,
    gap: null,
    nested: false,
    paddingX: null,
    rows: 3,
    streamFocus: 1,
    streamMain: 2,
    streams: true,
    zoomed: false
  }

  it('keeps the panel count in lockstep with the input handler focus wrap', () => {
    expect(STREAM_DEFS.length).toBe(GRID_STREAM_COUNT)
  })

  it('renders every stream panel with the promoted panel in the header', () => {
    const output = renderToText(<GridStreamsDemo cols={90} state={streamsState} t={DEFAULT_THEME} />)

    expect(output).toContain('hermes mission control')

    for (const def of STREAM_DEFS) {
      expect(output).toContain(def.title)
    }

    // streamMain: 2 → the memory panel owns the promoted slot.
    expect(output).toContain('main: memory')
  })
})
