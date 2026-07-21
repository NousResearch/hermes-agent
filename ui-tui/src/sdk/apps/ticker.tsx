import { Box, Text } from '@hermes/ink'
import { useEffect, useState } from 'react'

import { Dialog } from '../../components/overlay.js'
import type { Theme } from '../../theme.js'
import { defineWidgetApp } from '../registry.js'
import { isCtrl } from '../types.js'

/**
 * Ticker — the animated ambient reference app: a fake 1-pip chart that
 * random-walks a price and draws a live block sparkline, streams-demo style
 * (the component owns its animation; app state is just the symbol).
 */

const USAGE = 'usage: /ticker [symbol]'
const BLOCKS = '▁▂▃▄▅▆▇█'
const POINTS = 26
const TICK_MS = 250
const PIP = 0.0001

export interface TickerState {
  symbol: string
}

const spark = (series: number[]): string => {
  const min = Math.min(...series)
  const span = Math.max(...series) - min || 1

  return series.map(v => BLOCKS[Math.min(BLOCKS.length - 1, Math.floor(((v - min) / span) * BLOCKS.length))]).join('')
}

function Chart({ symbol, t }: { symbol: string; t: Theme }) {
  const [series, setSeries] = useState<number[]>(() => {
    const seed = 1.1 + Math.random() * 0.4
    const out = [seed]

    while (out.length < POINTS) {
      out.push(out.at(-1)! + (Math.random() - 0.5) * 4 * PIP)
    }

    return out
  })

  useEffect(() => {
    const id = setInterval(
      () => setSeries(prev => [...prev.slice(1), prev.at(-1)! + (Math.random() - 0.5) * 4 * PIP]),
      TICK_MS
    )

    return () => clearInterval(id)
  }, [])

  const price = series.at(-1)!
  const delta = price - series.at(-2)!
  const up = delta >= 0
  const dir = up ? t.color.ok : t.color.error

  return (
    <Box flexDirection="column">
      <Box columnGap={1} flexDirection="row">
        <Text bold color={t.color.label}>
          {symbol}
        </Text>
        <Text color={t.color.text}>{price.toFixed(4)}</Text>
        <Text color={dir}>
          {up ? '▲' : '▼'}
          {Math.abs(delta / PIP).toFixed(1)}p
        </Text>
      </Box>
      <Text color={dir}>{spark(series)}</Text>
    </Box>
  )
}

export const tickerApp = defineWidgetApp<TickerState>({
  id: 'ticker',
  help: 'fake 1-pip chart with a live sparkline',
  mode: 'ambient',
  usage: USAGE,

  init(arg) {
    const symbol = (arg.trim().split(/\s+/)[0] || 'HRMS').toUpperCase().slice(0, 8)

    return { symbol }
  },

  // Never receives input while ambient; contract-complete for modal reuse.
  reduce(state, { ch, key }) {
    return key.escape || ch === 'q' || isCtrl(key, ch, 'c') ? null : state
  },

  render({ state, t }) {
    return (
      <Dialog width={Math.max(32, POINTS + 6)}>
        <Chart symbol={state.symbol} t={t} />
      </Dialog>
    )
  }
})
