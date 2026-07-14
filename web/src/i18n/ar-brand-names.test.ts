import { describe, expect, it } from "vitest";

import { ar } from "./ar";
import { en } from "./en";

function missingTranslationPaths(
  source: unknown,
  target: unknown,
  prefix = "",
): string[] {
  if (typeof source !== "object" || source === null || Array.isArray(source)) {
    return [];
  }

  const targetRecord =
    typeof target === "object" && target !== null && !Array.isArray(target)
      ? (target as Record<string, unknown>)
      : {};

  return Object.entries(source as Record<string, unknown>).flatMap(
    ([key, value]) => {
      const path = prefix ? `${prefix}.${key}` : key;

      if (!(key in targetRecord)) {
        return [path];
      }

      return missingTranslationPaths(value, targetRecord[key], path);
    },
  );
}

describe("Arabic provider labels", () => {
  it("keeps provider brands in their original script", () => {
    const maps = [ar.env.providerLabels, ar.oauth.providerNames];

    for (const map of maps) {
      for (const label of Object.values(map ?? {})) {
        expect(label).not.toMatch(/\p{Script=Arabic}/u);
      }
    }
  });

  it("provides an Arabic value for every English translation path", () => {
    expect(missingTranslationPaths(en, ar)).toEqual([]);
  });
});
