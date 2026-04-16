/** Dashboard theme definition. Maps 1:1 to CSS custom properties in index.css. */
export interface ThemeColors {
  background: string;
  foreground: string;
  card: string;
  "card-foreground": string;
  primary: string;
  "primary-foreground": string;
  secondary: string;
  "secondary-foreground": string;
  muted: string;
  "muted-foreground": string;
  accent: string;
  "accent-foreground": string;
  destructive: string;
  "destructive-foreground": string;
  success: string;
  warning: string;
  border: string;
  input: string;
  ring: string;
  popover: string;
  "popover-foreground": string;
}

export interface ThemeOverlay {
  noiseOpacity?: number;
  noiseBlendMode?: string;
  warmGlowOpacity?: number;
  warmGlowColor?: string;
}

export interface DashboardTheme {
  name: string;
  label: string;
  description: string;
  colors: ThemeColors;
  overlay?: ThemeOverlay;
}

export interface ThemeListResponse {
  themes: Array<{ name: string; label: string; description: string }>;
  active: string;
}
