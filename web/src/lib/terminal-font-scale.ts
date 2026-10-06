export function terminalFontSizeForWidth(layoutWidthPx: number): number {
  if (layoutWidthPx < 300) return 7;
  if (layoutWidthPx < 360) return 8;
  if (layoutWidthPx < 420) return 9;
  if (layoutWidthPx < 520) return 10;
  if (layoutWidthPx < 720) return 11;
  if (layoutWidthPx < 1024) return 12;
  return 14;
}

export function scaledTerminalFontSize(layoutWidthPx: number, scale: number): number {
  const base = terminalFontSizeForWidth(layoutWidthPx);
  return Math.max(1, Math.round(base * scale));
}
