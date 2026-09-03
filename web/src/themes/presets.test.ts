import { describe, expect, it } from "vitest";

import {
  BUILTIN_THEMES,
  nousBlueContrastTheme,
  nousBlueTheme,
} from "./presets";

function relativeLuminance(hex: string): number {
  const channels = hex
    .slice(1)
    .match(/.{2}/g)!
    .map((pair) => Number.parseInt(pair, 16) / 255)
    .map((channel) =>
      channel <= 0.04045
        ? channel / 12.92
        : ((channel + 0.055) / 1.055) ** 2.4,
    );
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrastRatio(foreground: string, background: string): number {
  const lighter = Math.max(relativeLuminance(foreground), relativeLuminance(background));
  const darker = Math.min(relativeLuminance(foreground), relativeLuminance(background));
  return (lighter + 0.05) / (darker + 0.05);
}

describe("Nous Blue Contrast terminal palette", () => {
  it("is registered as a selectable built-in theme", () => {
    expect(BUILTIN_THEMES["nous-blue-contrast"]).toBe(nousBlueContrastTheme);
  });

  it("does not change the original Nous Blue theme", () => {
    expect(relativeLuminance(nousBlueTheme.terminalBackground!)).toBeGreaterThan(0.8);
    expect(contrastRatio(
      nousBlueTheme.terminalForeground!,
      nousBlueTheme.terminalBackground!,
    )).toBeGreaterThanOrEqual(7);
  });

  it("keeps the ANSI-heavy embedded TUI on a dark terminal canvas", () => {
    expect(nousBlueContrastTheme.terminalBackground).toMatch(/^#[0-9a-f]{6}$/i);
    expect(relativeLuminance(nousBlueContrastTheme.terminalBackground!)).toBeLessThan(0.1);
    expect(contrastRatio(
      nousBlueContrastTheme.terminalForeground!,
      nousBlueContrastTheme.terminalBackground!,
    )).toBeGreaterThanOrEqual(7);
  });

  it("inherits Nous Blue and changes only the terminal treatment", () => {
    expect(nousBlueContrastTheme.palette).toBe(nousBlueTheme.palette);
    expect(nousBlueContrastTheme.typography).toBe(nousBlueTheme.typography);
    expect(nousBlueContrastTheme.layout).toBe(nousBlueTheme.layout);
    expect(nousBlueContrastTheme.terminalBackground).not.toBe(
      nousBlueTheme.terminalBackground,
    );
    expect(nousBlueContrastTheme.terminalForeground).not.toBe(
      nousBlueTheme.terminalForeground,
    );
  });
});
