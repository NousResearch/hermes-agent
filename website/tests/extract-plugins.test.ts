// @vitest-environment node
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";

test("publishing preserves boolean featured flags in both feeds and defaults missing flags to false", () => {
  const temp = mkdtempSync(join(tmpdir(), "featured-catalog-"));
  const catalog = join(temp, "catalog");
  const output = join(temp, "api");
  mkdirSync(catalog);
  mkdirSync(output);
  const fixtures = [
    { name: "featured", featured: true, tier: "community" },
    { name: "ordinary", featured: false, tier: "official" },
    { name: "legacy", tier: "official" },
  ];
  try {
    for (const entry of fixtures) writeFileSync(join(catalog, `${entry.name}.yaml`), JSON.stringify({
      ...entry, repo: `https://github.com/example/${entry.name}`, sha: "a".repeat(40), category: "tools",
    }));
    writeFileSync(join(output, "plugin-stars.json"), JSON.stringify({ stars: { "example/ordinary": 999, "example/featured": 1 } }));
    execFileSync(process.env.HERMES_PYTHON || "python3", [
      fileURLToPath(new URL("../scripts/extract-plugins.py", import.meta.url)),
      "--catalog-dir", catalog, "--output-dir", output,
    ]);
    const page = JSON.parse(readFileSync(join(output, "plugins.json"), "utf8"));
    const live = JSON.parse(readFileSync(join(output, "plugin-catalog.json"), "utf8")).entries;
    for (const feed of [page, live]) for (const entry of fixtures) {
      expect(feed.find((row: { name: string }) => row.name === entry.name)).toMatchObject({
        name: entry.name, tier: entry.tier, featured: entry.featured === true,
      });
    }
    expect(page.map((row: { name: string }) => row.name)).toEqual(["ordinary", "featured", "legacy"]);
    expect(page[0].stars).toBe(999);
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});
