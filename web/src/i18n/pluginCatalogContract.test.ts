import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { en } from "./en";
import { zh } from "./zh";

const KANBAN_BUNDLE = fileURLToPath(
  new URL("../../../plugins/kanban/dashboard/dist/index.js", import.meta.url),
);
const ACHIEVEMENTS_BUNDLE = fileURLToPath(
  new URL(
    "../../../plugins/hermes-achievements/dashboard/dist/index.js",
    import.meta.url,
  ),
);
const ACHIEVEMENTS_API = fileURLToPath(
  new URL(
    "../../../plugins/hermes-achievements/dashboard/plugin_api.py",
    import.meta.url,
  ),
);

function literalTranslationKeys(source: string): string[] {
  return [
    ...source.matchAll(/\btx\(\s*[^,]+,\s*"([^"]+)"/g),
  ]
    .map((match) => match[1])
    // A trailing dot is the static prefix of a dynamically selected leaf,
    // e.g. ``columnLabels." + status``. The nested dictionaries themselves
    // are covered by the strongly typed English/Simplified catalogs.
    .filter((key) => !key.endsWith("."));
}

function resolvePath(root: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((node, segment) => {
    if (!node || typeof node !== "object") return undefined;
    return (node as Record<string, unknown>)[segment];
  }, root);
}

describe("bundled Dashboard plugin localization contract", () => {
  it.each([
    ["Kanban", KANBAN_BUNDLE, en.kanban, zh.kanban],
    [
      "Achievements",
      ACHIEVEMENTS_BUNDLE,
      en.achievements,
      zh.achievements,
    ],
  ])(
    "keeps every %s translation call backed by English and Simplified Chinese",
    (_name, bundle, englishCatalog, simplifiedCatalog) => {
      const source = readFileSync(bundle, "utf8");
      const keys = [...new Set(literalTranslationKeys(source))].sort();

      expect(keys.length).toBeGreaterThan(0);
      expect(
        keys.filter(
          (key) => typeof resolvePath(englishCatalog, key) !== "string",
        ),
      ).toEqual([]);
      expect(
        keys.filter(
          (key) => typeof resolvePath(simplifiedCatalog, key) !== "string",
        ),
      ).toEqual([]);
    },
  );

  it("keeps every built-in achievement and criterion metric in both complete catalogs", () => {
    const source = readFileSync(ACHIEVEMENTS_API, "utf8");
    const ids = [
      ...new Set([...source.matchAll(/\{"id": "([^"]+)"/g)].map((m) => m[1])),
    ].sort();
    const metrics = [
      ...new Set([
        ...[...source.matchAll(/"threshold_metric": "([^"]+)"/g)].map(
          (m) => m[1],
        ),
        ...[...source.matchAll(/\breq\("([^"]+)"/g)].map((m) => m[1]),
      ]),
    ].sort();

    expect(Object.keys(en.achievements.definitions).sort()).toEqual(ids);
    expect(Object.keys(zh.achievements.definitions).sort()).toEqual(ids);
    expect(Object.keys(en.achievements.metrics).sort()).toEqual(metrics);
    expect(Object.keys(zh.achievements.metrics).sort()).toEqual(metrics);
  });
});
