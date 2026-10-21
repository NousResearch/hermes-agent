import { contrastRatio, THEME_PRESET_PALETTES, type ThemePresetPalette } from "@hermes/shared";
import { describe, expect, it } from "vitest";

import { BUILTIN_THEMES, webPresetFromShared } from "./presets";

// Every preset the dashboard shares with the desktop must render the shared
// table's palette, not a private copy — that is the whole point of the table.
// The second assertion keeps the projection honest: whatever slot the mapping
// picks as the dashboard's text/primary colour has to stay legible on the
// canvas it picks, so a future re-mapping cannot silently ship grey-on-grey.
describe("dashboard presets derive from the shared palette table", () => {
  const shared = Object.keys(BUILTIN_THEMES).filter(
    (name): name is keyof typeof THEME_PRESET_PALETTES => name in THEME_PRESET_PALETTES,
  );

  it("covers the presets both surfaces ship", () => {
    expect(shared).toEqual(expect.arrayContaining(["cyberpunk", "ember", "midnight", "mono"]));
  });

  it.each(shared)("%s: canvas equals the shared background and the accent reads on it", (name) => {
    const preset: ThemePresetPalette = THEME_PRESET_PALETTES[name];
    const derived = webPresetFromShared(preset);
    const palette = BUILTIN_THEMES[name].palette;

    expect(palette.background.hex).toBe((preset.darkColors ?? preset.colors).background);
    expect(palette.midground.hex).toBe(derived.midground.hex);
    expect(contrastRatio(palette.midground.hex, palette.background.hex)).toBeGreaterThanOrEqual(3);
  });
});
