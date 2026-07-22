import { Box, Text } from '@hermes/ink'

import { ShimmerRows } from '../../components/loaders.js'
import { Dialog } from '../../components/overlay.js'
import { mix } from '../../lib/color.js'
import type { Theme } from '../../theme.js'
import { updateWidget } from '../host.js'
import { defineWidgetApp } from '../registry.js'
import { isCtrl } from '../types.js'

/**
 * Weather — the data-backed reference app. Demonstrates the async contract:
 * `init` returns a loading state and fires the fetch; the resolution lands
 * through `updateWidget`, which no-ops if the app was closed meanwhile.
 * Everything visual derives from the theme (art tinted by family tones).
 */

const USAGE = 'usage: /weather [location]   (blank = geolocate by IP)'

// Skeleton mirrors the ready layout: art column + four stat lines.
const LOADING_ROWS: readonly (readonly [number, number])[] = [
  [13, 12],
  [13, 16],
  [13, 14],
  [13, 11]
]

type Phase = { kind: 'error'; message: string } | { kind: 'loading' } | { kind: 'ready'; report: Report }

export interface WeatherState {
  location: string
  phase: Phase
}

interface Report {
  area: string
  condition: string
  feelsC: string
  humidity: string
  tempC: string
  weatherCode: number
  windKmph: string
}

// WWO weather codes → art bucket. Table-driven; unknown codes read as cloud.
type Art = 'cloud' | 'fog' | 'rain' | 'snow' | 'sun' | 'thunder'

const ART_BY_CODE: readonly [codes: readonly number[], art: Art][] = [
  [[113], 'sun'],
  [[116, 119, 122], 'cloud'],
  [[143, 248, 260], 'fog'],
  [[176, 263, 266, 293, 296, 299, 302, 305, 308, 353, 356, 359], 'rain'],
  [[179, 182, 185, 227, 230, 320, 323, 326, 329, 332, 335, 338, 350, 368, 371, 374, 377], 'snow'],
  [[200, 386, 389, 392, 395], 'thunder']
]

const artFor = (code: number): Art => ART_BY_CODE.find(([codes]) => codes.includes(code))?.[1] ?? 'cloud'

const ART: Record<Art, readonly string[]> = {
  sun: ['    \\   /    ', '     .-.     ', '  ― (   ) ―  ', "     `-'     ", '    /   \\    '],
  cloud: ['             ', '     .--.    ', '  .-(    ).  ', ' (___.__)__) ', '             '],
  fog: ['             ', ' _ - _ - _ - ', '  _ - _ - _  ', ' _ - _ - _ - ', '             '],
  rain: ['     .-.     ', '    (   ).   ', '   (___(__)  ', '  ‚ʻ‚ʻ‚ʻ‚ʻ   ', '  ‚ʻ‚ʻ‚ʻ‚ʻ   '],
  snow: ['     .-.     ', '    (   ).   ', '   (___(__)  ', '   * * * *   ', '  * * * *    '],
  thunder: ['     .-.     ', '    (   ).   ', '   (___(__)  ', '  ⚡‚ʻ⚡‚ʻ   ', '  ‚ʻ⚡‚ʻ⚡   ']
}

/** Art tint rides the theme family: sun in primary gold, rain in the shell
 *  blue, fog in muted — never hardcoded hexes. */
const artColor = (art: Art, t: Theme): string =>
  ({
    cloud: t.color.muted,
    fog: t.color.muted,
    rain: t.color.shellDollar,
    snow: t.color.text,
    sun: t.color.primary,
    thunder: t.color.warn
  })[art]

async function fetchReport(location: string): Promise<Report> {
  const res = await fetch(`https://wttr.in/${encodeURIComponent(location)}?format=j1`, {
    headers: { 'User-Agent': 'hermes-tui-weather' },
    signal: AbortSignal.timeout(10_000)
  })

  if (!res.ok) {
    throw new Error(`wttr.in answered ${res.status}`)
  }

  const data = (await res.json()) as {
    current_condition?: {
      FeelsLikeC?: string
      humidity?: string
      temp_C?: string
      weatherCode?: string
      weatherDesc?: { value?: string }[]
      windspeedKmph?: string
    }[]
    nearest_area?: { areaName?: { value?: string }[]; country?: { value?: string }[] }[]
  }

  const now = data.current_condition?.[0]
  const area = data.nearest_area?.[0]

  if (!now) {
    throw new Error('no current conditions in reply')
  }

  return {
    area: [area?.areaName?.[0]?.value, area?.country?.[0]?.value].filter(Boolean).join(', ') || location || 'here',
    condition: now.weatherDesc?.[0]?.value ?? 'unknown',
    feelsC: now.FeelsLikeC ?? '?',
    humidity: now.humidity ?? '?',
    tempC: now.temp_C ?? '?',
    weatherCode: Number(now.weatherCode ?? 116),
    windKmph: now.windspeedKmph ?? '?'
  }
}

function load(location: string): void {
  fetchReport(location).then(
    report => updateWidget(weatherApp, state => ({ ...state, phase: { kind: 'ready', report } as Phase })),
    (error: unknown) =>
      updateWidget(weatherApp, state => ({
        ...state,
        phase: { kind: 'error', message: error instanceof Error ? error.message : String(error) } as Phase
      }))
  )
}

export const weatherApp = defineWidgetApp<WeatherState>({
  id: 'weather',
  help: 'current conditions with themed ASCII art (wttr.in)',
  mode: 'ambient',
  usage: USAGE,

  init(arg) {
    const location = arg.trim()

    load(location)

    return { location, phase: { kind: 'loading' } }
  },

  reduce(state, { ch, key }) {
    if (key.escape || key.return || ch === 'q' || isCtrl(key, ch, 'c')) {
      return null
    }

    if (ch === 'r') {
      load(state.location)

      return { ...state, phase: { kind: 'loading' } }
    }

    return state
  },

  // Ambient: renders IN the dock (host owns placement) — a compact card
  // that sits above the status bar while the composer stays live.
  render({ cols, state, t }) {
    const { phase } = state
    const title = phase.kind === 'ready' ? phase.report.area : 'Weather'

    return (
      <Dialog title={title} width={Math.min(42, cols - 4)}>
        {phase.kind === 'loading' && (
          <ShimmerRows
            color={mix(t.color.muted, t.color.completionBg, 0.5)}
            highlight={t.color.label}
            rows={LOADING_ROWS}
          />
        )}
        {phase.kind === 'error' && <Text color={t.color.error}>{phase.message}</Text>}
        {phase.kind === 'ready' && <ReadyBody report={phase.report} t={t} />}
      </Dialog>
    )
  }
})

function ReadyBody({ report, t }: { report: Report; t: Theme }) {
  const art = artFor(report.weatherCode)

  return (
    <Box flexDirection="row" gap={2}>
      <Box flexDirection="column" flexShrink={0}>
        {ART[art].map((line, i) => (
          <Text color={artColor(art, t)} key={i}>
            {line}
          </Text>
        ))}
      </Box>
      <Box flexDirection="column">
        <Text color={t.color.label}>{report.condition}</Text>
        <Text color={t.color.text}>
          {report.tempC}°C <Text color={t.color.muted}>(feels {report.feelsC}°C)</Text>
        </Text>
        <Text color={t.color.muted}>wind {report.windKmph} km/h</Text>
        <Text color={t.color.muted}>humidity {report.humidity}%</Text>
      </Box>
    </Box>
  )
}
