// Prints the Hermes banner as raw ANSI to stdout before React/Ink load.
// Gives the user instant visual feedback during the ~170ms dynamic-import
// window; `<AlternateScreen>` wipes the normal-screen buffer when Ink
// mounts, so there is no double-banner.
//
// Palette is hardcoded to match DEFAULT_THEME — drifting the theme's
// banner colors here is fine, Ink's real render takes over in ~200ms.

const GOLD = '\x1b[38;2;255;215;0m'
const AMBER = '\x1b[38;2;255;191;0m'
const BRONZE = '\x1b[38;2;205;127;50m'
const DIM = '\x1b[38;2;184;134;11m'
const RESET = '\x1b[0m'

const LOGO = [
  '██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗',
  '██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝',
  '███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ',
  '██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ',
  '██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ',
  '╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   '
]

const GRADIENT = [GOLD, GOLD, AMBER, AMBER, BRONZE, BRONZE]
const LOGO_WIDTH = 98

export function bootBanner(cols: number = process.stdout.columns || 80): string {
  const lines =
    cols >= LOGO_WIDTH
      ? LOGO.map((text, i) => `${GRADIENT[i]}${text}${RESET}`)
      : [`\x1b[1m${GOLD}⚕ NOUS HERMES${RESET}`]

  return (
    '\n' + lines.join('\n') + '\n' + `${DIM}⚕ Nous Research · Messenger of the Digital Gods${RESET}\n\n`
  )
}
