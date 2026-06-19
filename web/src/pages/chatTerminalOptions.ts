import type { ITerminalOptions, ITheme } from "@xterm/xterm";

export const TERMINAL_THEME_STATIC = {
  foreground: "#f0e6d2",
  cursor: "#f0e6d2",
  cursorAccent: "#0d2626",
  selectionBackground: "#f0e6d244",
} as const satisfies ITheme;

export type ChatTerminalTheme = ITheme & { background: string };

const TERMINAL_FONT_FAMILY =
  "'JetBrains Mono', 'Cascadia Mono', 'Fira Code', 'MesloLGS NF', 'Source Code Pro', Menlo, Consolas, 'DejaVu Sans Mono', monospace";

export function terminalFontSizeForWidth(layoutWidthPx: number): number {
  if (layoutWidthPx < 300) return 7;
  if (layoutWidthPx < 360) return 8;
  if (layoutWidthPx < 420) return 9;
  if (layoutWidthPx < 520) return 10;
  if (layoutWidthPx < 720) return 11;
  if (layoutWidthPx < 1024) return 12;
  return 14;
}

export function terminalLineHeightForWidth(layoutWidthPx: number): number {
  return layoutWidthPx < 1024 ? 1.02 : 1.15;
}

export function createChatTerminalOptions(
  layoutWidthPx: number,
  theme: ChatTerminalTheme,
): ITerminalOptions {
  return {
    allowProposedApi: true,
    cursorBlink: true,
    fontFamily: TERMINAL_FONT_FAMILY,
    fontSize: terminalFontSizeForWidth(layoutWidthPx),
    lineHeight: terminalLineHeightForWidth(layoutWidthPx),
    letterSpacing: 0,
    fontWeight: "400",
    fontWeightBold: "700",
    // Preserve native macOS Option+key text composition for layouts where
    // printable characters like "@" require Option instead of US-Alt chords.
    macOptionIsMeta: false,
    // Keep the selection bypass for mouse mode even though plain Option+key
    // chords now prioritize text input over terminal meta shortcuts.
    macOptionClickForcesSelection: true,
    rightClickSelectsWord: true,
    scrollback: 5000,
    theme,
  };
}
