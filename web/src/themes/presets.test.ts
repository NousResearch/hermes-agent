import { describe, expect, it } from "vitest";

import { BUILTIN_THEMES, defaultTheme, hadesTheme } from "./presets";

describe("built-in dashboard theme presets", () => {
  it("resolves Hades as a concrete built-in dashboard theme", () => {
    const resolved = BUILTIN_THEMES.hades;

    expect(resolved).toBe(hadesTheme);
    expect(resolved).not.toBe(defaultTheme);
    expect(resolved.name).toBe("hades");
    expect(resolved.label.toLowerCase()).toContain("hades");
    expect(resolved.terminalBackground).not.toBe(defaultTheme.terminalBackground);
    expect(resolved.colorOverrides.primary).toBeTruthy();
    expect(resolved.swatchColors.length).toBeGreaterThanOrEqual(3);
  });
});
