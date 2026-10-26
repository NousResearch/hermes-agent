import { Box, Text } from '@hermes/ink'
import { mix } from '@hermes/shared/color'

import { ShimmerRows } from '../../components/loaders.js'
import { Dialog } from '../../components/overlay.js'
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

// WMO weather codes → art bucket. Table-driven; unknown codes read as cloud.
type Art = 'cloud' | 'fog' | 'rain' | 'snow' | 'sun' | 'thunder'

const ART_BY_CODE: readonly [codes: readonly number[], art: Art][] = [
  [[0], 'sun'],
  [[1, 2, 3], 'cloud'],
  [[45, 48], 'fog'],
  [[51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82], 'rain'],
  [[71, 73, 75, 77, 85, 86], 'snow'],
  [[95, 96, 99], 'thunder']
]

const artFor = (code: number): Art => ART_BY_CODE.find(([codes]) => codes.includes(code))?.[1] ?? 'cloud'

const CONDITION_BY_CODE: Readonly<Record<number, string>> = {
  0: 'Clear sky',
  1: 'Mainly clear',
  2: 'Partly cloudy',
  3: 'Overcast',
  45: 'Fog',
  48: 'Rime fog',
  51: 'Light drizzle',
  53: 'Drizzle',
  55: 'Heavy drizzle',
  56: 'Light freezing drizzle',
  57: 'Freezing drizzle',
  61: 'Light rain',
  63: 'Rain',
  65: 'Heavy rain',
  66: 'Light freezing rain',
  67: 'Freezing rain',
  71: 'Light snow',
  73: 'Snow',
  75: 'Heavy snow',
  77: 'Snow grains',
  80: 'Light rain showers',
  81: 'Rain showers',
  82: 'Heavy rain showers',
  85: 'Light snow showers',
  86: 'Heavy snow showers',
  95: 'Thunderstorm',
  96: 'Thunderstorm with hail',
  99: 'Severe thunderstorm with hail'
}

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

interface Coordinates {
  area: string
  latitude: number
  longitude: number
  timezone: string
}

async function geocodeLocation(location: string): Promise<Coordinates> {
  const params = new URLSearchParams({ count: '1', format: 'json', language: 'en', name: location })

  const res = await fetch(`https://geocoding-api.open-meteo.com/v1/search?${params}`, {
    headers: { 'User-Agent': 'hermes-tui-weather' },
    signal: AbortSignal.timeout(10_000)
  })

  if (!res.ok) {
    throw new Error(`Open-Meteo geocoding answered ${res.status}`)
  }

  const data = (await res.json()) as {
    results?: { country?: string; latitude?: number; longitude?: number; name?: string; timezone?: string }[]
  }

  const place = data.results?.[0]

  if (!place || place.latitude === undefined || place.longitude === undefined) {
    throw new Error(`location not found: ${location}`)
  }

  return {
    area: [place.name, place.country].filter(Boolean).join(', ') || location,
    latitude: place.latitude,
    longitude: place.longitude,
    timezone: place.timezone ?? 'auto'
  }
}

async function geolocateByIp(): Promise<Coordinates> {
  const fields = 'success,city,country,latitude,longitude,timezone'

  const res = await fetch(`https://ipwho.is/?fields=${fields}`, {
    headers: { 'User-Agent': 'hermes-tui-weather' },
    signal: AbortSignal.timeout(10_000)
  })

  if (!res.ok) {
    throw new Error(`IP geolocation answered ${res.status}`)
  }

  const data = (await res.json()) as {
    city?: string
    country?: string
    latitude?: number
    longitude?: number
    success?: boolean
    timezone?: { id?: string }
  }

  if (data.success !== true || data.latitude === undefined || data.longitude === undefined) {
    throw new Error('automatic location could not be resolved')
  }

  return {
    area: [data.city, data.country].filter(Boolean).join(', ') || 'here',
    latitude: data.latitude,
    longitude: data.longitude,
    timezone: data.timezone?.id ?? 'auto'
  }
}

async function fetchReport(location: string): Promise<Report> {
  const place = location ? await geocodeLocation(location) : await geolocateByIp()

  const params = new URLSearchParams({
    current: 'temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code',
    latitude: String(place.latitude),
    longitude: String(place.longitude),
    timezone: place.timezone
  })

  const res = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`, {
    headers: { 'User-Agent': 'hermes-tui-weather' },
    signal: AbortSignal.timeout(10_000)
  })

  if (!res.ok) {
    throw new Error(`Open-Meteo forecast answered ${res.status}`)
  }

  const data = (await res.json()) as {
    current?: {
      apparent_temperature?: number
      relative_humidity_2m?: number
      temperature_2m?: number
      weather_code?: number
      wind_speed_10m?: number
    }
  }

  const now = data.current

  if (!now) {
    throw new Error('no current conditions in reply')
  }

  const weatherCode = now.weather_code ?? -1

  return {
    area: place.area,
    condition: CONDITION_BY_CODE[weatherCode] ?? 'Unknown conditions',
    feelsC: String(now.apparent_temperature ?? '?'),
    humidity: String(now.relative_humidity_2m ?? '?'),
    tempC: String(now.temperature_2m ?? '?'),
    weatherCode,
    windKmph: String(now.wind_speed_10m ?? '?')
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
  help: 'current conditions with themed ASCII art (Open-Meteo)',
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
