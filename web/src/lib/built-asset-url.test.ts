import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { build, type Rollup } from "vite";
import { afterAll, describe, expect, it } from "vitest";

import viteConfig from "../../vite.config";
import { renderDashboardBuiltUrl } from "./built-asset-url";

/** Evaluate the emitted runtime expression the way a built chunk would. */
function resolveAt(runtime: string, basePath: string | undefined): string {
  const scope = basePath === undefined ? {} : { __HERMES_BASE_PATH__: basePath };
  return new Function("globalThis", `return ${runtime};`)(scope) as string;
}

describe("renderDashboardBuiltUrl", () => {
  it("resolves JS-hosted asset URLs against the runtime base path", () => {
    const out = renderDashboardBuiltUrl("assets/ChatPage-abc123.js", "js");
    expect(out).toBeDefined();
    expect(resolveAt(out!.runtime, "/hermes")).toBe("/hermes/assets/ChatPage-abc123.js");
    expect(resolveAt(out!.runtime, "/a/b")).toBe("/a/b/assets/ChatPage-abc123.js");
  });

  it("keeps the default root-absolute URL when no base path is injected", () => {
    const out = renderDashboardBuiltUrl("assets/xterm-abc123.css", "js");
    expect(resolveAt(out!.runtime, "")).toBe("/assets/xterm-abc123.css");
    expect(resolveAt(out!.runtime, undefined)).toBe("/assets/xterm-abc123.css");
  });

  it("leaves HTML and CSS URLs to Vite's default (the backend prefixes those)", () => {
    expect(renderDashboardBuiltUrl("assets/index-abc123.js", "html")).toBeUndefined();
    expect(renderDashboardBuiltUrl("fonts/x.woff2", "css")).toBeUndefined();
  });

  it("emits the filename as a string literal, not as code", () => {
    const out = renderDashboardBuiltUrl('assets/a"b.js', "js");
    expect(resolveAt(out!.runtime, "/p")).toBe('/p/assets/a"b.js');
  });
});

// Execute the emitted preload helper, rather than matching minified names or
// syntax: a root-absolute CSS preload rejects the lazy route under a proxy (#90068).
describe("dashboard vite config: lazy-chunk preload URLs", () => {
  const root = mkdtempSync(path.join(tmpdir(), "hermes-built-url-"));
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it("loads lazy JS and preloads CSS independently of the document route and build hash", async () => {
    writeFileSync(path.join(root, "package.json"), '{"type":"module"}');
    writeFileSync(path.join(root, "index.html"), '<script type="module" src="./main.js"></script>');
    writeFileSync(path.join(root, "main.js"), 'globalThis.load = () => import("./page.js");\n');
    writeFileSync(path.join(root, "page.css"), ".page{color:red}\n");

    for (const page of [1, 2]) {
      writeFileSync(path.join(root, "page.js"), `import "./page.css";\nexport const page = ${page};\n`);
      const outDir = path.join(root, `dist-${page}`);
      const result = (await build({
        root,
        configFile: false,
        publicDir: false,
        logLevel: "silent",
        experimental: viteConfig.experimental,
        build: { outDir },
      })) as Rollup.RollupOutput;
      const entry = result.output.find(
        (o): o is Rollup.OutputChunk => o.type === "chunk" && o.isEntry,
      )!;
      const css = result.output.find((o) => o.fileName.endsWith(".css"))!;
      const chunk = result.output.find((o) => o.type === "chunk" && o.isDynamicEntry)!;

      for (const prefix of [undefined, "", "/hermes-1", "/hermes-3", "/nested/dashboard"]) {
        // A fresh native ESM loader executes exactly the production output, not
        // Vitest's transforms. Only the DOM/network boundary is simulated.
        const output = execFileSync(process.execPath, ["--input-type=module"], {
          encoding: "utf8",
          input: `
            import { pathToFileURL } from "node:url";
            globalThis.__HERMES_BASE_PATH__ = ${JSON.stringify(prefix)};
            const links = [];
            globalThis.document = {
              createElement: () => ({ relList: { supports: () => true }, addEventListener(event, cb) {
                if (event === "load") queueMicrotask(cb);
              }}),
              querySelector: () => null,
              querySelectorAll: () => [],
              getElementsByTagName: () => [],
              head: { appendChild(link) {
                links.push(new URL(link.href, "https://example.com${prefix || ""}/chat/deep/route").pathname);
              }},
            };
            await import(pathToFileURL(${JSON.stringify(path.join(outDir, entry.fileName))}));
            const loaded = await globalThis.load();
            console.log(JSON.stringify({ page: loaded.page, links }));
          `,
        });
        expect(JSON.parse(output)).toEqual({
          page,
          links: expect.arrayContaining([
            `${prefix || ""}/${css.fileName}`,
            `${prefix || ""}/${chunk.fileName}`,
          ]),
        });
      }
    }
  });
});
