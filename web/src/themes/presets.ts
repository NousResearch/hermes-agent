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

/** Shared by the Geist-based themes (Zinc, Neon Samurai). */
const GEIST_TYPOGRAPHY: ThemeTypography = {
  fontSans: `"Geist", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif`,
  fontDisplay: `"Geist", "Inter", system-ui, sans-serif`,
  fontMono: `"Geist Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace`,
  fontUrl:
    "https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap",
  baseSize: "14px",
  lineHeight: "1.5",
  letterSpacing: "0em",
};

/**
 * shadcn/ui-style dark zinc. The customCSS deliberately erases the shell's
 * terminal DNA (uppercase labels, wide tracking, decorative display faces,
 * hard edges) and replaces it with the rounded, sentence-case, tight-tracked
 * look of a modern SaaS dashboard.
 */
export const zincTheme: DashboardTheme = {
  name: "zinc",
  label: "Zinc",
  description: "shadcn/ui-style dark zinc — rounded, soft, Geist",
  palette: {
    background: { hex: "#09090b", alpha: 1 }, // zinc-950
    midground: { hex: "#fafafa", alpha: 1 }, // zinc-50 text/chrome
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 255, 255, 0.03)", // neutral, no warm cast
    noiseOpacity: 0, // pure flat — no grain
  },
  typography: GEIST_TYPOGRAPHY,
  layout: {
    radius: "0.625rem", // shadcn default radius
    density: "comfortable",
  },
  colorOverrides: {
    primary: "#fafafa", // white primary (dark-mode shadcn)
    primaryForeground: "#18181b",
    secondary: "#27272a", // zinc-800
    secondaryForeground: "#fafafa",
    accent: "#27272a",
    accentForeground: "#fafafa",
    muted: "#27272a",
    mutedForeground: "#a1a1aa", // zinc-400
    card: "#131316",
    cardForeground: "#fafafa",
    popover: "#18181b",
    popoverForeground: "#fafafa",
    border: "#27272a",
    input: "#27272a",
    ring: "#d4d4d8",
    destructive: "#ef4444",
    warning: "#f59e0b",
    success: "#22c55e",
  },
  customCSS: `
  /* ---- 1. Global type reset: no uppercase, no tracking, Geist everywhere ---- */
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
  /* code-ish text → Geist Mono */
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
  /* Mondwest (bundled decorative brand font on titles) → plain sans */
  [class*="font-mondwest"], [class*="font-expanded"] {
    font-family: var(--theme-font-sans) !important;
  }

  /* ---- 2. Rounding pass: shadcn radius scale ---- */
  button { border-radius: calc(var(--radius) - 2px) !important; }
  input, select, textarea { border-radius: calc(var(--radius) - 2px) !important; }
  /* ALL bordered containers, not just border-border: rows, alerts, sub-cards */
  main div[class*="border"], main section[class*="border"] {
    border-radius: var(--radius) !important;
  }
  /* small bordered chips/pills → fully round */
  main span[class*="border"], main [class*="text-[10px]"][class*="border"] {
    border-radius: 9999px !important;
  }
  /* switches: pill track + circular thumb (shadcn Switch look) */
  button[role="switch"] { border-radius: 9999px !important; }
  button[role="switch"] > span { border-radius: 9999px !important; }
  button[role="switch"][aria-checked="true"] {
    background: #fafafa !important;
    border-color: #fafafa !important;
  }
  button[role="switch"][aria-checked="true"] > span { background: #18181b !important; }

  /* ---- 3. Page & sidebar: flat zinc, hairline separation ---- */
  body { background-image: none; }
  #app-sidebar {
    background: #09090b !important;
    border-right: 1px solid #27272a !important;
    box-shadow: none !important;
  }
  header {
    background: rgba(9,9,11,0.85) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid #27272a !important;
    box-shadow: none !important;
  }

  /* ---- 4. Nav: soft pill items, no stamps, no bars ---- */
  nav a {
    margin: 1px 8px !important;
    padding: 8px 12px !important;
    border-radius: calc(var(--radius) - 2px) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #a1a1aa;
    transition: background .12s ease, color .12s ease;
  }
  nav a::before, nav a::after { display: none !important; }
  /* the shell's own active indicator: a real child span (w-px white bar + hover flash) */
  nav a span[class*="absolute"][class*="bg-midground"] { display: none !important; }
  nav a:hover { background: rgba(255,255,255,0.06) !important; color: #fafafa; }
  nav a[aria-current], nav a[data-active="true"], nav a.active {
    background: rgba(255,255,255,0.08) !important;
    color: #fafafa !important;
  }

  /* ---- 5. Cards: rounded-xl, soft border, barely-there shadow ---- */
  main [class*="border-border"] {
    background: #131316;
    border-color: #27272a !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.35);
    transition: border-color .15s cubic-bezier(0.2,0,0,1),
                transform .15s cubic-bezier(0.2,0,0,1),
                box-shadow .15s cubic-bezier(0.2,0,0,1);
  }
  main [class*="border-border"]:hover {
    border-color: #3f3f46 !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.45);
  }

  /* ---- 6. Buttons: shadcn variants by feel ---- */
  button { transition: background .13s ease, color .13s ease, border-color .13s ease, opacity .13s ease !important; }
  button:hover:not(:disabled) { opacity: .92; box-shadow: none !important; }
  button:active:not(:disabled) { transform: scale(0.98) !important; opacity: .85; }
  button[class*="border"] { border-color: #27272a !important; }
  button[class*="border"]:hover:not(:disabled) { background: rgba(255,255,255,0.06) !important; opacity: 1; }

  /* ---- 7. Inputs: quiet fields, ring focus ---- */
  input, select, textarea {
    background: rgba(255,255,255,0.03) !important;
    border-color: #27272a !important;
  }
  input:focus, select:focus, textarea:focus {
    border-color: #52525b !important;
    box-shadow: 0 0 0 3px rgba(212,212,216,0.15) !important;
  }
  :focus-visible { outline: 3px solid rgba(161,161,170,0.35) !important; outline-offset: 2px; }
  ::selection { background: rgba(250,250,250,0.16); }

  /* ---- 8. Tables: shadcn ledger — muted medium headers, hairlines ---- */
  th {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #a1a1aa !important;
    border-bottom: 1px solid #27272a !important;
  }
  td { border-bottom: 1px solid #1f1f23 !important; }
  tr:hover td { background: rgba(255,255,255,0.03); }

  /* ---- 9. Scrollbars: thin rounded zinc ---- */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 9999px; border: 2px solid #09090b; }
  ::-webkit-scrollbar-thumb:hover { background: #52525b; }

  /* ---- 10. Motion: quiet shadcn micro-transitions (fast ease-out) ---- */
  @media (prefers-reduced-motion: no-preference) {
    main > * { animation: zn-enter .3s cubic-bezier(0.2,0,0,1) both; }
    button, nav a, input, select, textarea {
      transition-duration: .15s !important;
      transition-timing-function: cubic-bezier(0.2,0,0,1) !important;
    }
  }
  @keyframes zn-enter {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: none; }
  }
`,
};

/**
 * Cyberpunk Osaka night on the same shadcn-style chassis as Zinc — rounded,
 * sentence-case, Geist — but painted magenta & cyan, with a neon motion
 * system (flicker-on headings, flowing gradient underlines, glow pulses).
 * Distinct from the green matrix-terminal `cyberpunk` theme.
 */
export const neonSamuraiTheme: DashboardTheme = {
  name: "neon-samurai",
  label: "Neon Samurai",
  description: "Cyberpunk Osaka night — magenta & cyan neon on indigo black",
  palette: {
    background: { hex: "#0a0816", alpha: 1 }, // night-Osaka indigo black
    midground: { hex: "#e6e4f4", alpha: 1 }, // lavender-white text/chrome
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 45, 149, 0.12)", // magenta city-glow
    noiseOpacity: 0.15, // faint atmospheric grain
  },
  typography: GEIST_TYPOGRAPHY,
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
  zinc: zincTheme,
  "neon-samurai": neonSamuraiTheme,
};
