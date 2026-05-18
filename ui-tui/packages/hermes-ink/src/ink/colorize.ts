import chalk from 'chalk'

import type { Color, TextStyles } from './styles.js'

/**
 * xterm.js (VS Code, Cursor, code-server, Coder) has supported truecolor
 * since 2017, but code-server/Coder containers often don't set
 * COLORTERM=truecolor. chalk's supports-color doesn't recognize
 * TERM_PROGRAM=vscode (it only knows iTerm.app/Apple_Terminal), so it falls
 * through to the -256color regex → level 2. At level 2, chalk.rgb()
 * downgrades to the nearest 6×6×6 cube color: rgb(215,119,87) → idx 174
 * rgb(215,135,135) — washed-out salmon.
 *
 * Gated on level === 2 (not < 3) to respect NO_COLOR / FORCE_COLOR=0 —
 * those yield level 0 and are an explicit "no colors" request. Desktop VS
 * Code sets COLORTERM=truecolor itself, so this is a no-op there (already 3).
 *
 * Must run BEFORE the tmux clamp — if tmux is running inside a VS Code
 * terminal, tmux's passthrough limitation wins and we want level 2.
 */
function boostChalkLevelForXtermJs(): boolean {
  if (process.env.TERM_PROGRAM === 'vscode' && chalk.level === 2) {
    chalk.level = 3

    return true
  }

  return false
}

export function shouldUseRichEightBitDowngradeForLegacyAppleTerminal(
  env: NodeJS.ProcessEnv = process.env,
  level = chalk.level
): boolean {
  const termProgram = (env.TERM_PROGRAM ?? '').trim()
  const truecolorOverride = /^(?:1|true|yes|on)$/i.test((env.HERMES_TUI_TRUECOLOR ?? '').trim())
  const advertisesTruecolor = /^(?:truecolor|24bit)$/i.test((env.COLORTERM ?? '').trim())

  return termProgram === 'Apple_Terminal' && !truecolorOverride && !advertisesTruecolor && !('FORCE_COLOR' in env) && level === 2
}

export function richEightBitColorNumber(red: number, green: number, blue: number): number {
  const rn = red / 255
  const gn = green / 255
  const bn = blue / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const lightness = (max + min) / 2
  const saturation = max === min ? 0 : lightness > 0.5 ? (max - min) / (2 - max - min) : (max - min) / (max + min)

  if (saturation < 0.15) {
    const gray = Math.round(lightness * 25)

    return gray === 0 ? 16 : gray === 25 ? 231 : 231 + gray
  }

  const sixRed = red < 95 ? red / 95 : 1 + (red - 95) / 40
  const sixGreen = green < 95 ? green / 95 : 1 + (green - 95) / 40
  const sixBlue = blue < 95 ? blue / 95 : 1 + (blue - 95) / 40

  return 16 + 36 * Math.round(sixRed) + 6 * Math.round(sixGreen) + Math.round(sixBlue)
}

/**
 * tmux parses truecolor SGR (\e[48;2;r;g;bm) into its cell buffer correctly,
 * but its client-side emitter only re-emits truecolor to the outer terminal if
 * the outer terminal advertises Tc/RGB capability (via terminal-overrides).
 * Default tmux config doesn't set this, so tmux emits the cell to iTerm2/etc
 * WITHOUT the bg sequence — outer terminal's buffer has bg=default → black on
 * dark profiles. Clamping to level 2 makes chalk emit 256-color (\e[48;5;Nm),
 * which tmux passes through cleanly. grey93 (255) is visually identical to
 * rgb(240,240,240).
 *
 * Users who HAVE set `terminal-overrides ,*:Tc` get a technically-unnecessary
 * downgrade, but the visual difference is imperceptible. Querying
 * `tmux show -gv terminal-overrides` to detect this would add a subprocess on
 * startup — not worth it.
 *
 * $TMUX is a pty-lifecycle env var set by tmux itself; it never comes from
 * globalSettings.env, so reading it here is correct. chalk is a singleton, so
 * this clamps ALL truecolor output (fg+bg+hex) across the entire app.
 */
function clampChalkLevelForTmux(): boolean {
  if (process.env.TMUX && chalk.level > 2) {
    chalk.level = 2

    return true
  }

  return false
}

// Computed once at module load — terminal/tmux environment doesn't change mid-session.
// Order matters: boost first; then tmux can still clamp RGB to 256.
// Exported for debugging — tree-shaken if unused.
export const CHALK_BOOSTED_FOR_XTERMJS = boostChalkLevelForXtermJs()
export const CHALK_CLAMPED_FOR_TMUX = clampChalkLevelForTmux()
export const CHALK_USES_RICH_EIGHT_BIT_DOWNGRADE = shouldUseRichEightBitDowngradeForLegacyAppleTerminal()

export type ColorType = 'foreground' | 'background'

const RGB_REGEX = /^rgb\(\s?(\d+),\s?(\d+),\s?(\d+)\s?\)$/
const ANSI_REGEX = /^ansi256\(\s?(\d+)\s?\)$/
const HEX_REGEX = /^#[0-9a-fA-F]{6}$/

export const colorize = (str: string, color: string | undefined, type: ColorType): string => {
  if (!color) {
    return str
  }

  if (color.startsWith('ansi:')) {
    const value = color.substring('ansi:'.length)

    switch (value) {
      case 'black':
        return type === 'foreground' ? chalk.black(str) : chalk.bgBlack(str)

      case 'red':
        return type === 'foreground' ? chalk.red(str) : chalk.bgRed(str)

      case 'green':
        return type === 'foreground' ? chalk.green(str) : chalk.bgGreen(str)

      case 'yellow':
        return type === 'foreground' ? chalk.yellow(str) : chalk.bgYellow(str)

      case 'blue':
        return type === 'foreground' ? chalk.blue(str) : chalk.bgBlue(str)

      case 'magenta':
        return type === 'foreground' ? chalk.magenta(str) : chalk.bgMagenta(str)

      case 'cyan':
        return type === 'foreground' ? chalk.cyan(str) : chalk.bgCyan(str)

      case 'white':
        return type === 'foreground' ? chalk.white(str) : chalk.bgWhite(str)

      case 'blackBright':
        return type === 'foreground' ? chalk.blackBright(str) : chalk.bgBlackBright(str)

      case 'redBright':
        return type === 'foreground' ? chalk.redBright(str) : chalk.bgRedBright(str)

      case 'greenBright':
        return type === 'foreground' ? chalk.greenBright(str) : chalk.bgGreenBright(str)

      case 'yellowBright':
        return type === 'foreground' ? chalk.yellowBright(str) : chalk.bgYellowBright(str)

      case 'blueBright':
        return type === 'foreground' ? chalk.blueBright(str) : chalk.bgBlueBright(str)

      case 'magentaBright':
        return type === 'foreground' ? chalk.magentaBright(str) : chalk.bgMagentaBright(str)

      case 'cyanBright':
        return type === 'foreground' ? chalk.cyanBright(str) : chalk.bgCyanBright(str)

      case 'whiteBright':
        return type === 'foreground' ? chalk.whiteBright(str) : chalk.bgWhiteBright(str)
    }
  }

  if (color.startsWith('#')) {
    if (HEX_REGEX.test(color) && CHALK_USES_RICH_EIGHT_BIT_DOWNGRADE) {
      const value = Number.parseInt(color.slice(1), 16)
      const red = (value >> 16) & 0xff
      const green = (value >> 8) & 0xff
      const blue = value & 0xff
      const ansi = richEightBitColorNumber(red, green, blue)

      return type === 'foreground' ? chalk.ansi256(ansi)(str) : chalk.bgAnsi256(ansi)(str)
    }

    return type === 'foreground' ? chalk.hex(color)(str) : chalk.bgHex(color)(str)
  }

  if (color.startsWith('ansi256')) {
    const matches = ANSI_REGEX.exec(color)

    if (!matches) {
      return str
    }

    const value = Number(matches[1])

    return type === 'foreground' ? chalk.ansi256(value)(str) : chalk.bgAnsi256(value)(str)
  }

  if (color.startsWith('rgb')) {
    const matches = RGB_REGEX.exec(color)

    if (!matches) {
      return str
    }

    const firstValue = Number(matches[1])
    const secondValue = Number(matches[2])
    const thirdValue = Number(matches[3])

    if (CHALK_USES_RICH_EIGHT_BIT_DOWNGRADE) {
      const ansi = richEightBitColorNumber(firstValue, secondValue, thirdValue)

      return type === 'foreground' ? chalk.ansi256(ansi)(str) : chalk.bgAnsi256(ansi)(str)
    }

    return type === 'foreground'
      ? chalk.rgb(firstValue, secondValue, thirdValue)(str)
      : chalk.bgRgb(firstValue, secondValue, thirdValue)(str)
  }

  return str
}

/**
 * Apply TextStyles to a string using chalk.
 * This is the inverse of parsing ANSI codes - we generate them from structured styles.
 * Theme resolution happens at component layer, not here.
 */
export function applyTextStyles(text: string, styles: TextStyles): string {
  let result = text

  // Apply styles in reverse order of desired nesting.
  // chalk wraps text so later calls become outer wrappers.
  // Desired order (outermost to innermost):
  //   background > foreground > text modifiers
  // So we apply: text modifiers first, then foreground, then background last.

  if (styles.inverse) {
    result = chalk.inverse(result)
  }

  if (styles.strikethrough) {
    result = chalk.strikethrough(result)
  }

  if (styles.underline) {
    result = chalk.underline(result)
  }

  if (styles.italic) {
    result = chalk.italic(result)
  }

  if (styles.bold) {
    result = chalk.bold(result)
  }

  if (styles.dim) {
    result = chalk.dim(result)
  }

  if (styles.color) {
    // Color is now always a raw color value (theme resolution happens at component layer)
    result = colorize(result, styles.color, 'foreground')
  }

  if (styles.backgroundColor) {
    // backgroundColor is now always a raw color value
    result = colorize(result, styles.backgroundColor, 'background')
  }

  return result
}

/**
 * Apply a raw color value to text.
 * Theme resolution should happen at component layer, not here.
 */
export function applyColor(text: string, color: Color | undefined): string {
  if (!color) {
    return text
  }

  return colorize(text, color, 'foreground')
}
