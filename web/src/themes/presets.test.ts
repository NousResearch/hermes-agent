import { describe, expect, it } from "vitest";
import { BUILTIN_THEMES, clarityTheme, defaultTheme } from "./presets";

describe("dashboard theme typography presets", () => {
  it("includes a readable dark preset with unified display/body typography", () => {
    expect(BUILTIN_THEMES.clarity).toBe(clarityTheme);
    expect(clarityTheme.typography.fontDisplay).toBe(clarityTheme.typography.fontSans);
    expect(clarityTheme.typography.baseSize).toBe("16px");
    expect(clarityTheme.typography.letterSpacing).toBe("0");
    expect(clarityTheme.palette.noiseOpacity).toBe(0);
    expect(clarityTheme.colorOverrides?.mutedForeground).toBeTruthy();
  });

  it("uses CJK-capable system fallbacks for baseline dashboard typography", () => {
    expect(defaultTheme.typography.fontSans).toContain("Malgun Gothic");
    expect(defaultTheme.typography.fontSans).toContain("Noto Sans KR");
    expect(defaultTheme.typography.fontMono).toContain("D2Coding");
  });
});
