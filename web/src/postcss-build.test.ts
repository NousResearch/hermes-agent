import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, expect, test } from "vitest";
import { build } from "vite";
import dashboardConfig from "../vite.config";

const fixtures: string[] = [];

afterEach(() => {
  for (const fixture of fixtures.splice(0)) {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test.each(["package.json", ".postcssrc.json"])(
  "dashboard CSS builds without loading an ancestor BOM-marked %s",
  async (filename) => {
    const fixture = mkdtempSync(path.join(tmpdir(), "hermes-web-postcss-"));
    fixtures.push(fixture);
    const root = path.join(fixture, "web");
    mkdirSync(root);
    const ambientConfig = path.join(fixture, filename);
    const content = '\uFEFF{"name":"unrelated-parent","postcss":{}}';
    writeFileSync(ambientConfig, content);
    writeFileSync(path.join(root, "package.json"), '{"type":"module"}');
    writeFileSync(path.join(root, "index.html"), '<script type="module" src="/main.js"></script>');
    writeFileSync(path.join(root, "main.js"), 'import "./style.css";');
    writeFileSync(path.join(root, "style.css"), "body { color: red; }");

    const result = await build({
      ...dashboardConfig,
      root,
      configFile: false,
      publicDir: false,
      logLevel: "silent",
      build: { ...dashboardConfig.build, write: false, minify: false },
    });

    expect(result).toMatchObject({
      output: expect.arrayContaining([
        expect.objectContaining({
          type: "asset",
          fileName: expect.stringMatching(/\.css$/),
          source: expect.stringMatching(/color:\s*red/),
        }),
      ]),
    });
    expect(readFileSync(ambientConfig, "utf8")).toBe(content);
  },
);
