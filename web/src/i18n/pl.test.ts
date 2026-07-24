import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { LOCALE_META } from "./context";
import { en } from "./en";
import { pl } from "./pl";

function ownLeafPaths(node: unknown, prefix = ""): string[] {
  if (typeof node === "string") return [prefix];

  if (Array.isArray(node)) {
    return node.flatMap((value, index) => ownLeafPaths(value, `${prefix}[${index}]`));
  }

  if (!node || typeof node !== "object") return [prefix];

  return Object.keys(node).flatMap((key) =>
    ownLeafPaths(
      (node as Record<string, unknown>)[key],
      prefix ? `${prefix}.${key}` : key,
    ),
  );
}

describe("Polish dashboard localization", () => {
  it("registers Polish in the language picker", () => {
    expect(LOCALE_META.pl).toEqual({ name: "Polski" });
  });

  it("has exactly the same own translation paths as English", () => {
    const englishPaths = ownLeafPaths(en);
    const polishPaths = ownLeafPaths(pl);

    expect(polishPaths).toEqual(expect.arrayContaining(englishPaths));
    expect(englishPaths).toEqual(expect.arrayContaining(polishPaths));
    expect(polishPaths).toHaveLength(englishPaths.length);
  });

  it("translates representative visible interface and confirmation copy", () => {
    expect(pl.common.save).toBe("Zapisz");
    expect(pl.common.gateway).toBe("Brama");
    expect(pl.app.nav.sessions).toBe("Sesje");
    expect(pl.config.resetDefaults).toBe("Przywróć domyślne");
    expect(pl.sessions.confirmDeleteMessage).toContain("trwałe usunięcie");
    expect(pl.sessions.confirmDeleteMessage).toContain("nie można cofnąć");
  });

  it("does not contain known literal-translation regressions", () => {
    const source = readFileSync(`${process.cwd()}/src/i18n/pl.ts`, "utf8");
    const forbidden = [
      /modelk/i,
      /Bliźnięt/i,
      /\bBieganie\b/i,
      /\bBiegnij\b/i,
      /\bWłaz\b/i,
      /\bTarło\b/i,
      /Zremis/i,
      /\boddział/i,
      /żeton/i,
      /kompozytor/i,
      /zaplecz/i,
      /Pulpit Hermes/i,
      /Centrum dowodzenia/i,
      /\bbramk/i,
      /\bmonit(?:u|em|ach|ami|y|ów|owi|cie|owanie|owania)?\b/iu,
      /zachęt/i,
      /narzędzi\(a\)|serwera\(ów\)|umiejętność\(i\)/i,
    ];

    for (const pattern of forbidden) {
      expect(source, `forbidden Polish localization pattern: ${pattern}`).not.toMatch(pattern);
    }
  });
});
