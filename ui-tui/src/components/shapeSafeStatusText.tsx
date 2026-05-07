import { RawAnsi, stringWidth } from '@hermes/ink'

// Some terminal/font stacks shape `gemini` such that the `n` disappears
// visually in the compact status bar. SGR 25 ("blink off") is a visible
// no-op for Hermes, but it creates a terminal style boundary that prevents
// that shaping run while preserving the visible label and copied text.
const SHAPING_BOUNDARY = '\x1b[25m'

export const stripStatusAnsi = (text: string) => text.replace(/\x1b\[[0-9;]*m/g, '')

export const truncateStatusText = (text: string, maxWidth: number) => {
  if (stringWidth(text) <= maxWidth) {
    return text
  }

  if (maxWidth <= 1) {
    return '…'
  }

  let out = ''

  for (const char of text) {
    const next = `${out}${char}`

    if (stringWidth(`${next}…`) > maxWidth) {
      break
    }

    out = next
  }

  return `${out}…`
}

const ansiForeground = (color: string) => {
  const hex = /^#?([0-9a-f]{6})$/i.exec(color)

  if (hex) {
    const value = parseInt(hex[1]!, 16)
    const red = (value >> 16) & 0xff
    const green = (value >> 8) & 0xff
    const blue = value & 0xff

    return `\x1b[38;2;${red};${green};${blue}m`
  }

  const rgb = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i.exec(color)

  if (rgb) {
    return `\x1b[38;2;${rgb[1]};${rgb[2]};${rgb[3]}m`
  }

  const indexed = /^ansi256\((\d+)\)$/i.exec(color)

  if (indexed) {
    return `\x1b[38;5;${indexed[1]}m`
  }

  return ''
}

export const breakTerminalShapingRuns = (text: string) => text.replace(/\bgemin(?=i\b)/gi, `$&${SHAPING_BOUNDARY}`)

export const statusTextAnsi = (text: string, color: string) =>
  `${ansiForeground(color)}${breakTerminalShapingRuns(text)}\x1b[39m`

export function ShapeSafeStatusText({ color, text }: { color: string; text: string }) {
  return <RawAnsi lines={[statusTextAnsi(text, color)]} width={stringWidth(text)} />
}
