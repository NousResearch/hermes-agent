const GOLD = '\x1b[38;2;255;215;0m';
const AMBER = '\x1b[38;2;255;191;0m';
const BRONZE = '\x1b[38;2;205;127;50m';
const DIM = '\x1b[38;2;184;134;11m';
const RESET = '\x1b[0m';
const LOGO = [
    '██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗',
    '██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝',
    '███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ',
    '██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ',
    '██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ',
    '╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   '
];
const GRADIENT = [GOLD, GOLD, AMBER, AMBER, BRONZE, BRONZE];
const LOGO_WIDTH = 98;
const TAGLINE = `${DIM}⚕ Nous Research · Messenger of the Digital Gods${RESET}`;
const FALLBACK = `\x1b[1m${GOLD}⚕ NOUS HERMES${RESET}`;
export function bootBanner(cols = process.stdout.columns || 80) {
    const body = cols >= LOGO_WIDTH ? LOGO.map((text, i) => `${GRADIENT[i]}${text}${RESET}`).join('\n') : FALLBACK;
    return `\n${body}\n${TAGLINE}\n\n`;
}
