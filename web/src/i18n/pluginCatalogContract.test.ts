import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { en } from "./en";
import { zh } from "./zh";

const KANBAN_BUNDLE = fileURLToPath(
  new URL("../../../plugins/kanban/dashboard/dist/index.js", import.meta.url),
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
  it("keeps every Kanban translation call backed by English and Simplified Chinese", () => {
    const source = readFileSync(KANBAN_BUNDLE, "utf8");
    const keys = [...new Set(literalTranslationKeys(source))].sort();

    expect(keys.length).toBeGreaterThan(150);
    expect(
      keys.filter((key) => typeof resolvePath(en.kanban, key) !== "string"),
    ).toEqual([]);
    expect(
      keys.filter((key) => typeof resolvePath(zh.kanban, key) !== "string"),
    ).toEqual([]);
  });
});
