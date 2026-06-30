// Star Map palette — ported from apps/desktop/src/app/starmap/color.ts so the
// TUI overlay derives the same memory ink (complement of the theme primary) and
// the same age fade (rgba(ink, alpha) over the background) as the desktop panel.

interface Rgb {
  b: number
  g: number
  r: number
}

function hexToRgb(hex: string): Rgb {
  let s = hex.trim().replace(/^#/, '')

  if (s.length === 3) {
    s = s
      .split('')
      .map(c => c + c)
      .join('')
  }

  const n = parseInt(s, 16)

  if (Number.isNaN(n) || s.length < 6) {
    return { b: 0, g: 215, r: 255 }
  }

  return { b: n & 255, g: (n >> 8) & 255, r: (n >> 16) & 255 }
}

function rgbToHex(c: Rgb): string {
  const h = (v: number) =>
    Math.max(0, Math.min(255, Math.round(v)))
      .toString(16)
      .padStart(2, '0')

  return `#${h(c.r)}${h(c.g)}${h(c.b)}`
}

function mix(a: Rgb, b: Rgb, t: number): Rgb {
  const p = Math.max(0, Math.min(1, t))

  return { b: a.b + (b.b - a.b) * p, g: a.g + (b.g - a.g) * p, r: a.r + (b.r - a.r) * p }
}

function luminance(c: Rgb): number {
  return (0.2126 * c.r + 0.7152 * c.g + 0.114 * c.b) / 255
}

function rgbToHsl(c: Rgb): [number, number, number] {
  const r = c.r / 255
  const g = c.g / 255
  const b = c.b / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  const d = max - min

  if (!d) {
    return [0, 0, l]
  }

  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
  const h = (max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4) * 60

  return [h, s, l]
}

function hslToRgb(h: number, s: number, l: number): Rgb {
  const hue = ((h % 360) + 360) % 360
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = l - c / 2

  const [r, g, b] =
    hue < 60
      ? [c, x, 0]
      : hue < 120
        ? [x, c, 0]
        : hue < 180
          ? [0, c, x]
          : hue < 240
            ? [0, x, c]
            : hue < 300
              ? [x, 0, c]
              : [c, 0, x]

  return { b: (b + m) * 255, g: (g + m) * 255, r: (r + m) * 255 }
}

function complementaryInk(c: Rgb): Rgb {
  const [h, s, l] = rgbToHsl(c)

  return hslToRgb(h + 165, Math.max(s, 0.5), Math.max(0.5, Math.min(0.7, l)))
}

export interface StarmapPalette {
  bg: Rgb
  dim: Rgb
  label: Rgb
  memory: Rgb
  skill: Rgb
}

/** Derive the Star Map inks from the theme primary + foreground color. */
export function deriveStarmapPalette(primaryHex: string, fgHex: string): StarmapPalette {
  const primary = hexToRgb(primaryHex)
  const dark = luminance(hexToRgb(fgHex)) > 0.55
  const base: Rgb = dark ? { b: 255, g: 255, r: 255 } : { b: 0, g: 0, r: 0 }
  const bg: Rgb = dark ? { b: 12, g: 8, r: 8 } : { b: 250, g: 250, r: 250 }

  return {
    bg,
    dim: mix(base, bg, 0.7),
    label: mix(base, bg, 0.35),
    memory: mix(complementaryInk(primary), bg, 0.45),
    skill: mix(primary, base, dark ? 0.12 : 0.18)
  }
}

/** Fade an explicit hex ink toward the background by alpha (for category bars). */
export function fadeHex(palette: StarmapPalette, hex: string, alpha: number): string {
  const base = hexToRgb(hex)

  return rgbToHex(alpha >= 0.999 ? base : mix(palette.bg, base, alpha))
}

/** Fade a base ink toward the background by alpha (rgba-over-bg), as a hex. */
export function fadeInk(palette: StarmapPalette, style: string, alpha: number): string | undefined {
  const base =
    style === 'skill'
      ? palette.skill
      : style === 'memory'
        ? palette.memory
        : style === 'label'
          ? palette.label
          : style === 'dim'
            ? palette.dim
            : null

  if (!base) {
    return undefined
  }

  return rgbToHex(alpha >= 0.999 ? base : mix(palette.bg, base, alpha))
}
