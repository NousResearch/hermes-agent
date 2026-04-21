const ESC = '\x1b'
const BEL = '\x07'
const ST = `${ESC}\\`

export const OSC52_CLIPBOARD_QUERY = `${ESC}]52;c;?${BEL}`

type OscResponse = { code: number; data: string; type: 'osc' }

type OscQuerier = {
  flush: () => Promise<void>
  send: <T>(query: { match: (r: unknown) => r is T; request: string }) => Promise<T | undefined>
}

function wrapForMultiplexer(sequence: string): string {
  if (process.env['TMUX']) {
    return `${ESC}Ptmux;${sequence.split(ESC).join(ESC + ESC)}${ST}`
  }

  if (process.env['STY']) {
    return `${ESC}P${sequence}${ST}`
  }

  return sequence
}

export function buildOsc52ClipboardQuery(): string {
  return wrapForMultiplexer(OSC52_CLIPBOARD_QUERY)
}

export function parseOsc52ClipboardData(data: string): null | string {
  const firstSep = data.indexOf(';')

  if (firstSep === -1) {
    return null
  }

  const selection = data.slice(0, firstSep)
  const payload = data.slice(firstSep + 1)

  if ((selection !== 'c' && selection !== 'p') || !payload || payload === '?') {
    return null
  }

  try {
    return Buffer.from(payload, 'base64').toString('utf8')
  } catch {
    return null
  }
}

export async function readOsc52Clipboard(querier: null | OscQuerier): Promise<null | string> {
  if (!querier) {
    return null
  }

  const response = await querier.send<OscResponse>({
    request: buildOsc52ClipboardQuery(),
    match: (r: unknown): r is OscResponse => {
      return !!r && typeof r === 'object' && (r as OscResponse).type === 'osc' && (r as OscResponse).code === 52
    }
  })

  await querier.flush()

  return response ? parseOsc52ClipboardData(response.data) : null
}

export const writeOsc52Clipboard = (s: string) =>
  process.stdout.write(`\x1b]52;c;${Buffer.from(s, 'utf8').toString('base64')}\x07`)
