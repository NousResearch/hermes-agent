import { describe, expect, it } from "vitest";

import { en } from "./en";
import { zh } from "./zh";

function missingTranslationPaths(
  reference: Record<string, unknown>,
  candidate: Record<string, unknown>,
  prefix = "",
): string[] {
  const missing: string[] = [];

  for (const [key, value] of Object.entries(reference)) {
    const path = prefix ? `${prefix}.${key}` : key;
    const translated = candidate[key];

    if (
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value)
    ) {
      if (
        translated === null ||
        typeof translated !== "object" ||
        Array.isArray(translated)
      ) {
        missing.push(path);
        continue;
      }
      missing.push(
        ...missingTranslationPaths(
          value as Record<string, unknown>,
          translated as Record<string, unknown>,
          path,
        ),
      );
      continue;
    }

    if (translated === undefined) missing.push(path);
  }

  return missing;
}

describe("Simplified Chinese translation coverage", () => {
  it("provides every key in the English reference catalog", () => {
    expect(
      missingTranslationPaths(
        en as unknown as Record<string, unknown>,
        zh as unknown as Record<string, unknown>,
      ),
    ).toEqual([]);
  });
});
