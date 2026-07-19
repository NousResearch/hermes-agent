import type { DashboardTheme, ThemeTypography, ThemeLayout } from "./types";

/**
 * Built-in dashboard themes.
 *
 * Each theme defines its own palette, typography, and layout so switching
 * themes produces visible changes beyond just color — fonts, density, and
 * corner-radius all shift to match the theme's personality.
 *
 * Theme names must stay in sync with the backend's
 * `_BUILTIN_DASHBOARD_THEMES` list in `hermes_cli/web_server.py`.
 */

// ---------------------------------------------------------------------------
// Shared typography / layout presets
// ---------------------------------------------------------------------------

/** Default system stack — neutral, safe fallback for every platform. */
const SYSTEM_SANS =
  'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const SYSTEM_MONO =
  'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace';

const DEFAULT_TYPOGRAPHY: ThemeTypography = {
  fontSans: SYSTEM_SANS,
  fontMono: SYSTEM_MONO,
  baseSize: "15px",
  lineHeight: "1.55",
  letterSpacing: "0",
};

const DEFAULT_LAYOUT: ThemeLayout = {
  radius: "0.5rem",
  density: "comfortable",
};

// ---------------------------------------------------------------------------
// Themes
// ---------------------------------------------------------------------------

export const defaultTheme: DashboardTheme = {
  name: "default",
  label: "Hermes Teal",
  description: "Classic dark teal — the canonical Hermes look",
  palette: {
    background: { hex: "#041c1c", alpha: 1 },
    midground: { hex: "#ffe6cb", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 189, 56, 0.35)",
    noiseOpacity: 1,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: DEFAULT_LAYOUT,
  terminalBackground: "#000000",
};

export const midnightTheme: DashboardTheme = {
  name: "midnight",
  label: "Midnight",
  description: "Deep blue-violet with cool accents",
  palette: {
    background: { hex: "#0a0a1f", alpha: 1 },
    midground: { hex: "#d4c8ff", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(167, 139, 250, 0.32)",
    noiseOpacity: 0.8,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Inter", ${SYSTEM_SANS}`,
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap",
    letterSpacing: "-0.005em",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.75rem",
  },
};

export const emberTheme: DashboardTheme = {
  name: "ember",
  label: "Ember",
  description: "Warm crimson and bronze — forge vibes",
  palette: {
    background: { hex: "#1a0a06", alpha: 1 },
    midground: { hex: "#ffd8b0", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(249, 115, 22, 0.38)",
    noiseOpacity: 1,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Spectral", Georgia, "Times New Roman", serif`,
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;700&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.25rem",
  },
  colorOverrides: {
    destructive: "#c92d0f",
    warning: "#f97316",
  },
};

export const monoTheme: DashboardTheme = {
  name: "mono",
  label: "Mono",
  description: "Clean grayscale — minimal and focused",
  palette: {
    background: { hex: "#0e0e0e", alpha: 1 },
    midground: { hex: "#eaeaea", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 255, 255, 0.1)",
    noiseOpacity: 0.6,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"IBM Plex Sans", ${SYSTEM_SANS}`,
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0",
  },
};

export const cyberpunkTheme: DashboardTheme = {
  name: "cyberpunk",
  label: "Cyberpunk",
  description: "Neon green on black — matrix terminal",
  palette: {
    background: { hex: "#040608", alpha: 1 },
    midground: { hex: "#9bffcf", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(0, 255, 136, 0.22)",
    noiseOpacity: 1.2,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Share Tech Mono", "JetBrains Mono", ${SYSTEM_MONO}`,
    fontMono: `"Share Tech Mono", "JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;700&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0",
  },
  colorOverrides: {
    success: "#00ff88",
    warning: "#ffd700",
    destructive: "#ff0055",
  },
};

export const roseTheme: DashboardTheme = {
  name: "rose",
  label: "Rosé",
  description: "Soft pink and warm ivory — easy on the eyes",
  palette: {
    background: { hex: "#1a0f15", alpha: 1 },
    midground: { hex: "#ffd4e1", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(249, 168, 212, 0.3)",
    noiseOpacity: 0.9,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Fraunces", Georgia, serif`,
    fontMono: `"DM Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=DM+Mono:wght@400;500&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "1rem",
  },
};

/** Light mode — vivid Nous-blue accents on a cream canvas. */
export const nousBlueTheme: DashboardTheme = {
  name: "nous-blue",
  label: "Nous Blue",
  description: "Light mode — vivid Nous-blue accents on cream canvas",
  palette: {
    background: { hex: "#E8F2FD", alpha: 1 },
    midground: { hex: "#0053FD", alpha: 1 },
    foreground: { hex: "#170d02", alpha: 0 },
    warmGlow: "rgba(0, 83, 253, 0.12)",
    noiseOpacity: 0,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: DEFAULT_LAYOUT,
  terminalBackground: "#f5f8fc",
  terminalForeground: "#170d02",
  seriesColors: {
    inputTokenAccent: "#001934",
    outputTokenAccent: "#0053fd",
  },
  swatchColors: ["#170d02", "#0053FD", "#E8F2FD"],
};

/**
 * Same look as ``defaultTheme`` but with a larger root font size, looser
 * line-height, and ``spacious`` density so every rem-based size in the
 * dashboard scales up. For users who find the default 15px UI too dense.
 */
export const defaultLargeTheme: DashboardTheme = {
  name: "default-large",
  label: "Hermes Teal (Large)",
  description: "Hermes Teal with bigger fonts and roomier spacing",
  palette: defaultTheme.palette,
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    baseSize: "18px",
    lineHeight: "1.65",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    density: "spacious",
  },
};

/**
 * Cyberpunk Osaka night — magenta & cyan neon on indigo black, Geist
 * typography, rounded sentence-case chrome — layered with a pixel-art
 * sakura wind: 15 parallax streams of pixel petals drifting in gusts.
 * The petal effect is pure CSS (inline SVG data-URI tiles animated on
 * main::before / main::after), gated behind prefers-reduced-motion, and
 * contained between the SAKURA-PIXELS comment markers in the customCSS.
 */
export const nightOsakaTheme: DashboardTheme = {
  name: "night-osaka",
  label: "Night Osaka",
  description: "Cyberpunk Osaka night with pixel sakura drifting in gusty wind",
  palette: {
    background: { hex: "#0a0816", alpha: 1 }, // night-Osaka indigo black
    midground: { hex: "#e6e4f4", alpha: 1 }, // lavender-white text/chrome
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 45, 149, 0.12)", // magenta city-glow
    noiseOpacity: 0.15, // faint atmospheric grain
  },
  typography: {
    fontSans: `"Geist", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif`,
    fontDisplay: `"Geist", "Inter", system-ui, sans-serif`,
    fontMono: `"Geist Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap",
    baseSize: "14px",
    lineHeight: "1.5",
    letterSpacing: "0em",
  },
  layout: {
    radius: "0.625rem",
    density: "comfortable",
  },
  colorOverrides: {
    primary: "#ff2d95", // neon magenta
    primaryForeground: "#14061e",
    secondary: "#1c1836",
    secondaryForeground: "#e6e4f4",
    accent: "#00e5ff", // electric cyan
    accentForeground: "#06121f",
    muted: "#1c1836",
    mutedForeground: "#8f8ab0",
    card: "#120f26",
    cardForeground: "#e6e4f4",
    popover: "#161233",
    popoverForeground: "#e6e4f4",
    border: "#272052",
    input: "#272052",
    ring: "#00e5ff",
    destructive: "#ff4060",
    warning: "#ffb020",
    success: "#2de0a7",
  },
  customCSS: `

  /* ---- 1. Global type reset: no uppercase, no tracking ---- */
  button, nav a, main a, label, th,
  [class*="tracking-"], [class*="uppercase"] {
    text-transform: none !important;
    letter-spacing: -0.006em !important;
  }
  button {
    font-family: var(--theme-font-sans) !important;
    font-weight: 500 !important;
  }
  main h1, main h2, main h3, main h4 {
    font-family: var(--theme-font-sans) !important;
    text-transform: none !important;
    letter-spacing: -0.02em !important;
    font-weight: 600 !important;
  }
  [class*="font-mono"] {
    font-family: var(--theme-font-mono) !important;
    letter-spacing: 0 !important;
  }
  /* badges/chips (font-courier / font-compressed brand fonts) → readable sans badges */
  [class*="font-courier"], [class*="font-compressed"] {
    font-family: var(--theme-font-sans) !important;
    letter-spacing: 0.02em !important;
  }
  span[class*="font-courier"], span[class*="font-compressed"], [class*="text-[10px]"] {
    font-size: 11px !important;
    line-height: 1.4 !important;
    font-weight: 500 !important;
  }
  /* Mondwest decorative titles → plain sans */
  [class*="font-mondwest"], [class*="font-expanded"] {
    font-family: var(--theme-font-sans) !important;
  }

  /* ---- 2. Rounding: everything bordered gets the radius ---- */
  button, input, select, textarea { border-radius: calc(var(--radius) - 2px) !important; }
  main div[class*="border"], main section[class*="border"] {
    border-radius: var(--radius) !important;
  }
  main span[class*="border"], main [class*="text-[10px]"][class*="border"] {
    border-radius: 9999px !important;
  }
  /* switches: pill track + circular thumb; on = magenta with glow */
  button[role="switch"] { border-radius: 9999px !important; }
  button[role="switch"] > span { border-radius: 9999px !important; }
  button[role="switch"][aria-checked="true"] {
    background: #ff2d95 !important;
    border-color: #ff2d95 !important;
    box-shadow: 0 0 10px -2px rgba(255,45,149,0.6) !important;
  }
  button[role="switch"][aria-checked="true"] > span { background: #14061e !important; }

  /* ---- 3. Night city: indigo base with distant neon haze ---- */
  body {
    background-image:
      radial-gradient(ellipse 90% 60% at 15% -10%, rgba(255,45,149,0.07), transparent 55%),
      radial-gradient(ellipse 80% 70% at 110% 105%, rgba(0,229,255,0.06), transparent 55%),
      radial-gradient(ellipse 60% 45% at 70% 20%, rgba(120,60,255,0.05), transparent 60%);
  }
  #app-sidebar {
    background: rgba(10,8,22,0.96) !important;
    border-right: 1px solid #272052 !important;
    box-shadow: none !important;
  }
  header {
    background: rgba(10,8,22,0.82) !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #272052 !important;
    box-shadow: 0 4px 24px -12px rgba(255,45,149,0.25) !important;
  }

  /* ---- 4. Nav: pill items; active = magenta blade edge + glow ---- */
  nav a {
    margin: 1px 8px !important;
    padding: 8px 12px !important;
    border-radius: calc(var(--radius) - 2px) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #8f8ab0;
    transition: background .12s ease, color .12s ease, box-shadow .12s ease;
  }
  nav a::before, nav a::after { display: none !important; }
  /* the shell's own active indicator: a real child span (w-px white bar + hover flash) */
  nav a span[class*="absolute"][class*="bg-midground"] { display: none !important; }
  nav a:hover { background: rgba(0,229,255,0.07) !important; color: #e6e4f4; }
  nav a[aria-current], nav a[data-active="true"], nav a.active {
    background: rgba(255,45,149,0.12) !important;
    color: #ffffff !important;
    box-shadow: inset 2px 0 0 #ff2d95, 0 0 16px -6px rgba(255,45,149,0.6);
  }

  /* ---- 5. Cards: dark violet glass, cyan edge-glow on hover ---- */
  main div[class*="border"], main section[class*="border"] {
    border-color: #272052 !important;
    transition: border-color .16s ease, box-shadow .16s ease;
  }
  main [class*="border-border"] {
    background: rgba(18,15,38,0.88);
    box-shadow: 0 1px 2px rgba(0,0,0,0.5);
    transition: border-color .25s cubic-bezier(0.16,1,0.3,1),
                box-shadow .25s cubic-bezier(0.16,1,0.3,1),
                transform .25s cubic-bezier(0.16,1,0.3,1);
  }
  main [class*="border-border"]:hover {
    border-color: rgba(0,229,255,0.45) !important;
    box-shadow: 0 0 18px -8px rgba(0,229,255,0.5);
    transform: translateY(-2px);
  }

  /* ---- 6. Headings: neon underline (flowing), faint glow ---- */
  main h3 { position: relative; text-shadow: 0 0 20px rgba(255,45,149,0.30); }
  main h3::after {
    content: ""; display: block; margin-top: 6px; height: 2px; max-width: 180px;
    border-radius: 2px;
    background: linear-gradient(90deg, #ff2d95, #00e5ff, #ff2d95);
    background-size: 200% 100%;
  }

  /* ---- 7. Buttons & inputs ---- */
  button { transition: background .13s ease, color .13s ease, border-color .13s ease, box-shadow .13s ease !important; }
  button:hover:not(:disabled) { box-shadow: 0 0 14px -6px rgba(255,45,149,0.55) !important; }
  button:active:not(:disabled) { transform: scale(0.97) !important; }
  button[class*="border"] { border-color: #272052 !important; }
  button[class*="border"]:hover:not(:disabled) { background: rgba(0,229,255,0.08) !important; border-color: rgba(0,229,255,0.4) !important; }
  input, select, textarea {
    background: rgba(255,255,255,0.03) !important;
    border-color: #272052 !important;
    border-radius: calc(var(--radius) - 2px) !important;
  }
  input:focus, select:focus, textarea:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.18) !important;
  }
  :focus-visible { outline: 3px solid rgba(0,229,255,0.4) !important; outline-offset: 2px; }
  ::selection { background: rgba(255,45,149,0.35); color: #ffffff; }

  /* ---- 8. Tables ---- */
  th {
    font-size: 13px !important; font-weight: 500 !important;
    color: #8f8ab0 !important;
    border-bottom: 1px solid #272052 !important;
  }
  td { border-bottom: 1px solid #1c1740 !important; }
  tr:hover td { background: rgba(255,45,149,0.05); }

  /* ---- 9. Scrollbars ---- */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2b2455; border-radius: 9999px; border: 2px solid #0a0816; }
  ::-webkit-scrollbar-thumb:hover { background: #ff2d95; }

  /* ---- 10. Motion: cinematic neon (slow spring easing, glow pulses) ---- */
  @media (prefers-reduced-motion: no-preference) {
    /* page content rises out of the dark */
    main > * { animation: ns-rise .5s cubic-bezier(0.16,1,0.3,1) both; }
    /* headings flicker on like a neon sign warming up */
    main h3 { animation: ns-flicker 1.4s steps(1, end) 1; }
    /* the gradient underline flows continuously */
    main h3::after { animation: ns-flow 5s linear infinite; }
    /* active nav item breathes */
    nav a[aria-current], nav a[data-active="true"], nav a.active {
      animation: ns-pulse 2.8s ease-in-out infinite;
    }
    button, nav a, input, select, textarea {
      transition-duration: .25s !important;
      transition-timing-function: cubic-bezier(0.16,1,0.3,1) !important;
    }
  }
  @keyframes ns-rise {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: none; }
  }
  @keyframes ns-flicker {
    0% { opacity: .55; } 6% { opacity: 1; } 10% { opacity: .7; }
    15% { opacity: 1; } 22% { opacity: .85; } 30% { opacity: 1; } 100% { opacity: 1; }
  }
  @keyframes ns-flow { to { background-position: 200% 0; } }
  @keyframes ns-pulse {
    0%, 100% { box-shadow: inset 2px 0 0 #ff2d95, 0 0 14px -6px rgba(255,45,149,0.55); }
    50%      { box-shadow: inset 2px 0 0 #ff2d95, 0 0 24px -4px rgba(255,45,149,0.9); }
  }

  /* ================================================================
     SAKURA-PIXELS START — pixel-art sakura petals drifting in gusty
     wind. 15 independent parallax streams (8 near on main::before,
     7 far on main::after); each stream advances exactly one own-tile
     offset per loop, so the drift is always seamless. The entire
     effect lives between the SAKURA-PIXELS START/END markers.
     ================================================================ */
  @media (prefers-reduced-motion: no-preference) {
    main::before, main::after {
      content: "";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: none;
      background-repeat: repeat;
      image-rendering: pixelated;
    }
    /* 8 near/mid streams — one pseudo, 8 bg layers, one clock, 8 speeds */
    main::before {
      background-image:
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'661'%20height%3D'556'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(365,443)%20rotate(0,5,4)%20scale(0.96,0.96)'%20opacity%3D'0.89'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3C/g%3E%3Cg%20transform%3D'translate(511,498)%20rotate(270,5,4)%20scale(0.92,0.92)'%20opacity%3D'0.47'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'846'%20height%3D'592'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(182,169)%20rotate(90,5,4)%20scale(0.95,0.95)'%20opacity%3D'0.59'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3C/g%3E%3Cg%20transform%3D'translate(361,69)%20rotate(0,5,4)%20scale(0.8,0.8)'%20opacity%3D'0.43'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'732'%20height%3D'548'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(594,138)%20rotate(180,5,4)%20scale(-0.81,0.81)'%20opacity%3D'0.75'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'820'%20height%3D'755'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(196,438)%20rotate(0,5,4)%20scale(0.88,0.88)'%20opacity%3D'0.84'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'881'%20height%3D'553'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(79,453)%20rotate(270,5,4)%20scale(-1.0,1.0)'%20opacity%3D'0.78'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'573'%20height%3D'786'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(393,731)%20rotate(270,5,4)%20scale(0.64,0.64)'%20opacity%3D'0.76'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'733'%20height%3D'698'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(267,164)%20rotate(180,5,4)%20scale(0.89,0.89)'%20opacity%3D'0.38'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'886'%20height%3D'683'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(242,250)%20rotate(180,5,4)%20scale(0.63,0.63)'%20opacity%3D'0.32'%3E%3Crect%20x%3D'2'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'6'%20y%3D'0'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'2'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'4'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'6'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'8'%20y%3D'2'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'2'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'4'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'6'%20y%3D'4'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'4'%20y%3D'6'%20width%3D'2'%20height%3D'2'%20fill%3D'%23ff8ac2'/%3E%3C/g%3E%3C/svg%3E");
      background-size: 661px 556px, 846px 592px, 732px 548px, 820px 755px, 881px 553px, 573px 786px, 733px 698px, 886px 683px;
      opacity: 0.4;
      animation: ns-gust-near 38s cubic-bezier(0.45,0,0.55,1) infinite,
                 ns-flutter-a 11s ease-in-out infinite alternate;
    }
    /* 7 far micro streams */
    main::after {
      background-image:
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'700'%20height%3D'548'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(265,172)%20rotate(180,5,4)%20scale(-0.62,0.62)'%20opacity%3D'0.76'%3E%3Crect%20x%3D'1'%20y%3D'0'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'0'%20y%3D'1'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'1'%20y%3D'3'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'564'%20height%3D'560'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(496,43)%20rotate(90,5,4)%20scale(-0.84,0.84)'%20opacity%3D'0.38'%3E%3Crect%20x%3D'1'%20y%3D'0'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'0'%20y%3D'1'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ffe6f2'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'1'%20y%3D'3'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ffb9d9'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'619'%20height%3D'675'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(506,312)%20rotate(0,5,4)%20scale(1.05,1.05)'%20opacity%3D'0.77'%3E%3Crect%20x%3D'1'%20y%3D'0'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'0'%20y%3D'1'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'1'%20y%3D'3'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'817'%20height%3D'503'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(670,463)%20rotate(180,5,4)%20scale(0.84,0.84)'%20opacity%3D'0.69'%3E%3Crect%20x%3D'1'%20y%3D'0'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'0'%20y%3D'1'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'4'%20height%3D'1'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'1'%20y%3D'3'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'742'%20height%3D'597'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(63,88)%20rotate(0,5,4)%20scale(-1.07,1.07)'%20opacity%3D'0.57'%3E%3Crect%20x%3D'1'%20y%3D'0'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'0'%20y%3D'1'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ffb9d9'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'4'%20height%3D'1'%20fill%3D'%23e63d84'/%3E%3Crect%20x%3D'1'%20y%3D'3'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'807'%20height%3D'547'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(528,39)%20rotate(180,5,4)%20scale(0.83,0.83)'%20opacity%3D'0.56'%3E%3Crect%20x%3D'1'%20y%3D'0'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'0'%20y%3D'1'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'1'%20y%3D'3'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3C/g%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg%20xmlns%3D'http://www.w3.org/2000/svg'%20width%3D'555'%20height%3D'534'%20shape-rendering%3D'crispEdges'%3E%3Cg%20transform%3D'translate(146,229)%20rotate(90,5,4)%20scale(0.95,0.95)'%20opacity%3D'0.8'%3E%3Crect%20x%3D'1'%20y%3D'0'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3Crect%20x%3D'0'%20y%3D'1'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ffd2e8'/%3E%3Crect%20x%3D'0'%20y%3D'2'%20width%3D'4'%20height%3D'1'%20fill%3D'%23ff5aa8'/%3E%3Crect%20x%3D'1'%20y%3D'3'%20width%3D'2'%20height%3D'1'%20fill%3D'%23ff8ac2'/%3E%3C/g%3E%3C/svg%3E");
      background-size: 700px 548px, 564px 560px, 619px 675px, 817px 503px, 742px 597px, 807px 547px, 555px 534px;
      opacity: 0.22;
      animation: ns-gust-far 61s cubic-bezier(0.45,0,0.55,1) infinite,
                 ns-flutter-b 17s ease-in-out infinite alternate;
    }
  }
  /* gusty wind: per-stream surge->lull->surge; each stream ends exactly
     one own-tile offset per loop => always seamless */
  @keyframes ns-gust-near {
    0%   { background-position: 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px; }
    30%   { background-position: -298px 161px, -360px 194px, -352px 149px, -321px 209px, -342px 181px, -219px 214px, -302px 221px, -435px 197px; }
    55%   { background-position: -414px 278px, -572px 325px, -403px 309px, -556px 411px, -497px 329px, -345px 432px, -528px 407px, -649px 338px; }
    78%   { background-position: -540px 444px, -651px 460px, -560px 429px, -712px 590px, -687px 394px, -490px 649px, -646px 592px, -724px 530px; }
    100%   { background-position: -661px 556px, -846px 592px, -732px 548px, -820px 755px, -881px 553px, -573px 786px, -733px 698px, -886px 683px; }
  }
  @keyframes ns-gust-far {
    0%   { background-position: 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px, 0px 0px; }
    38%   { background-position: -251px 191px, -248px 231px, -285px 241px, -439px 174px, -365px 234px, -363px 196px, -205px 189px; }
    62%   { background-position: -493px 349px, -344px 341px, -358px 426px, -543px 325px, -490px 385px, -444px 311px, -398px 324px; }
    100%   { background-position: -700px 548px, -564px 560px, -619px 675px, -817px 503px, -742px 597px, -807px 547px, -555px 534px; }
  }
  @keyframes ns-flutter-a { from { opacity: 0.30; } to { opacity: 0.46; } }
  @keyframes ns-flutter-b { from { opacity: 0.14; } to { opacity: 0.27; } }
  /* ================================================================
     SAKURA-PIXELS END
     ================================================================ */
`,
};

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  "nous-blue": nousBlueTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
  "night-osaka": nightOsakaTheme,
};
