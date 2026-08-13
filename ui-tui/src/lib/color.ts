/**
 * THE color primitive — every color computation in the TUI goes through this
 * module (the twin of the desktop app's `src/themes/color.ts`). No component
 * or theme code does its own hex parsing or channel math: parse, mix, measure
 * and fix colors here, so tone ladders, contrast floors and one-off UI needs
 * all share one set of semantics.
 *
 * Mixing is an sRGB lerp — deliberately, for byte parity with the desktop's
 * `color-mix(in srgb, ...)` ladder in styles.css.
 */

export type Rgb = readonly [number, number, number]

const HEX6_RE = /^#?([0-9a-f]{6})$/i
const HEX3_RE = /^#?([0-9a-f]{3})$/i
const RGB_FN_RE = /^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})/i

const clampChannel = (v: number) => Math.max(0, Math.min(255, Math.round(v)))

/** Parse `#rgb`, `#rrggbb` or `rgb(r,g,b)` → channels. Null for anything else. */
export function parseColor(input: string): Rgb | null {
  const value = input.trim()

  let m = HEX6_RE.exec(value)

  if (m) {
    const n = parseInt(m[1]!, 16)

    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
  }

  m = HEX3_RE.exec(value)

  if (m) {
    const [r, g, b] = m[1]!

    return [parseInt(r! + r!, 16), parseInt(g! + g!, 16), parseInt(b! + b!, 16)]
  }

  m = RGB_FN_RE.exec(value)

  if (m) {
    return [clampChannel(Number(m[1])), clampChannel(Number(m[2])), clampChannel(Number(m[3]))]
  }

  return null
}

export const toHex = (rgb: Rgb): string => '#' + rgb.map(c => clampChannel(c).toString(16).padStart(2, '0')).join('')

/** sRGB lerp `a → b` by `t` in [0,1]. Unparseable inputs return `a` unchanged. */
export function mix(a: string, b: string, t: number): string {
  const pa = parseColor(a)
  const pb = parseColor(b)

  if (!pa || !pb) {
    return a
  }

  return toHex([pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t])
}

function channelLuminance(value: number): number {
  const normalized = value / 255

  return normalized <= 0.03928 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4
}

/** WCAG relative luminance in [0,1]. Null when unparseable. */
export function relativeLuminance(color: string): null | number {
  const rgb = parseColor(color)

  return rgb
    ? 0.2126 * channelLuminance(rgb[0]) + 0.7152 * channelLuminance(rgb[1]) + 0.0722 * channelLuminance(rgb[2])
    : null
}

/** WCAG contrast ratio between two colors (1–21). Null when unparseable. */
export function contrastRatio(a: string, b: string): null | number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)

  if (la === null || lb === null) {
    return null
  }

  const [hi, lo] = la >= lb ? [la, lb] : [lb, la]

  return (hi + 0.05) / (lo + 0.05)
}

/** The readable ink pole for a given background (desktop `readableOn`). */
export function readableOn(bg: string): '#000000' | '#ffffff' {
  return (relativeLuminance(bg) ?? 0) > 0.5 ? '#000000' : '#ffffff'
}

/**
 * Step-mix `color` toward the readable pole of `bg` until the contrast ratio
 * clears `min` (desktop `ensureContrast`). Each step re-mixes from the
 * ORIGINAL color so hue decays linearly, not exponentially. Returns the
 * original when it already passes or isn't parseable.
 */
export function ensureContrast(color: string, bg: string, min: number): string {
  if (relativeLuminance(bg) === null || parseColor(color) === null) {
    return color
  }

  const pole = readableOn(bg)
  let current = color

  for (let step = 0; step <= 20; step++) {
    const ratio = contrastRatio(current, bg)

    if (ratio === null || ratio >= min) {
      return current
    }

    current = mix(color, pole, Math.min(1, (step + 1) * 0.05))
  }

  return current
}

/**
 * xterm.js's minimum-contrast algorithm (Color.ts reduce/increaseLuminance),
 * ported faithfully: multiplicative 10% channel steps toward the readable
 * pole, which preserves channel RATIOS (hue and perceived chroma) far better
 * than mixing toward black/white. This is the display shim's lift — the same
 * math terminals themselves use, so a palette lifted here matches what hosts
 * like VS Code/Cursor produce when their own minimumContrastRatio kicks in.
 * Colors already at or above `ratio` pass through byte-identical.
 */
export function liftForContrast(color: string, bg: string, ratio: number): string {
  const fg = parseColor(color)
  const bgLum = relativeLuminance(bg)

  if (!fg || bgLum === null) {
    return color
  }

  // Byte-identical passthrough when the color already clears the ratio —
  // the authored value is the design; only failing colors get touched.
  if ((contrastRatio(color, bg) ?? 21) >= ratio) {
    return color
  }

  let [r, g, b] = fg

  if (bgLum > 0.5) {
    // reduceLuminance: darken multiplicatively.
    let cr = contrastRatio(toHex([r, g, b]), bg) ?? 21

    while (cr < ratio && (r > 0 || g > 0 || b > 0)) {
      r -= Math.ceil(r * 0.1)
      g -= Math.ceil(g * 0.1)
      b -= Math.ceil(b * 0.1)
      cr = contrastRatio(toHex([r, g, b]), bg) ?? 21
    }
  } else {
    // increaseLuminance: brighten toward white in 10% remaining-headroom steps.
    let cr = contrastRatio(toHex([r, g, b]), bg) ?? 21

    while (cr < ratio && (r < 255 || g < 255 || b < 255)) {
      r = Math.min(255, r + Math.ceil((255 - r) * 0.1))
      g = Math.min(255, g + Math.ceil((255 - g) * 0.1))
      b = Math.min(255, b + Math.ceil((255 - b) * 0.1))
      cr = contrastRatio(toHex([r, g, b]), bg) ?? 21
    }
  }

  return toHex([r, g, b])
}

/** Recede toward the background pole (opposite of `readableOn`). */
export const lighten = (color: string, t: number) => mix(color, '#ffffff', t)
export const darken = (color: string, t: number) => mix(color, '#000000', t)

/** The luminance-weighted gray of a color (its perceptual brightness). */
export function grayOf(color: string): string {
  const rgb = parseColor(color)

  if (!rgb) {
    return color
  }

  const gray = clampChannel(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])

  return toHex([gray, gray, gray])
}

/** Pull a color toward its own gray by `s` in [0,1] (1 = fully gray). */
export const desaturate = (color: string, s: number) => mix(color, grayOf(color), s)

/** RGB → HSL, channels in [0,1]. */
export function toHsl(rgb: Rgb): [number, number, number] {
  const r = rgb[0] / 255
  const g = rgb[1] / 255
  const b = rgb[2] / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2

  if (max === min) {
    return [0, 0, l]
  }

  const d = max - min
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
  const h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4

  return [h / 6, s, l]
}

const hueChannel = (p: number, q: number, t: number): number => {
  const tt = t < 0 ? t + 1 : t > 1 ? t - 1 : t

  if (tt < 1 / 6) {
    return p + (q - p) * 6 * tt
  }

  if (tt < 1 / 2) {
    return q
  }

  if (tt < 2 / 3) {
    return p + (q - p) * (2 / 3 - tt) * 6
  }

  return p
}

/** HSL → RGB, inputs in [0,1]. */
export function fromHsl(h: number, s: number, l: number): Rgb {
  if (s === 0) {
    const v = Math.round(l * 255)

    return [v, v, v]
  }

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q

  return [
    Math.round(hueChannel(p, q, h + 1 / 3) * 255),
    Math.round(hueChannel(p, q, h) * 255),
    Math.round(hueChannel(p, q, h - 1 / 3) * 255)
  ]
}

/**
 * Re-tone a color while PRESERVING its hue: clamp saturation into
 * [minSaturation, 1] and pin lightness. This is how a light-terminal variant
 * of a pastel dark-terminal accent stays vivid — darkening by mixing toward
 * black muddies pastels (kills saturation with lightness); re-toning keeps
 * the color identity and moves only where it sits.
 */
export function retone(color: string, lightness: number, minSaturation = 0): string {
  const rgb = parseColor(color)

  if (!rgb) {
    return color
  }

  const [h, s] = toHsl(rgb)

  return toHex(fromHsl(h, Math.max(s, minSaturation), lightness))
}

/** Multiply HSL saturation by `factor` (clamped to 1), hue/lightness fixed. */
export function boostSaturation(color: string, factor: number): string {
  const rgb = parseColor(color)

  if (!rgb) {
    return color
  }

  const [h, s, l] = toHsl(rgb)

  if (s === 0) {
    return color
  }

  return toHex(fromHsl(h, Math.min(1, s * factor), l))
}

/**
 * Chainable form for multi-step derivations, e.g.
 * `color(text).mix(bg, 0.35).mix(accent, 0.18).ensureContrast(bg, 2.8).hex()`.
 * Unparseable inputs pass through every operation unchanged.
 */
export function color(value: string): ColorChain {
  return new ColorChain(value)
}

class ColorChain {
  constructor(private readonly value: string) {}

  mix(other: string, t: number): ColorChain {
    return new ColorChain(mix(this.value, other, t))
  }

  lighten(t: number): ColorChain {
    return new ColorChain(lighten(this.value, t))
  }

  darken(t: number): ColorChain {
    return new ColorChain(darken(this.value, t))
  }

  ensureContrast(bg: string, min: number): ColorChain {
    return new ColorChain(ensureContrast(this.value, bg, min))
  }

  luminance(): null | number {
    return relativeLuminance(this.value)
  }

  contrastOn(bg: string): null | number {
    return contrastRatio(this.value, bg)
  }

  hex(): string {
    return this.value
  }
}
