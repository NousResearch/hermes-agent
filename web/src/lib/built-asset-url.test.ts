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

// Guards the integration, not just the helper: if a Vite upgrade drops or
// reshapes `experimental.renderBuiltUrl`, the lazy route's CSS preload goes back
// to a root-absolute "/assets/..." URL and the Chat route blanks under a proxy
// prefix (#90068). Builds a tiny fixture with the dashboard's own setting.
describe("dashboard vite config: lazy-chunk preload URLs", () => {
  const root = mkdtempSync(path.join(tmpdir(), "hermes-built-url-"));
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it("routes lazy-chunk CSS preloads through the runtime base path", async () => {
    writeFileSync(path.join(root, "index.html"), '<script type="module" src="./main.js"></script>');
    writeFileSync(path.join(root, "main.js"), 'globalThis.load = () => import("./page.js");\n');
    writeFileSync(path.join(root, "page.js"), 'import "./page.css";\nexport const page = 1;\n');
    writeFileSync(path.join(root, "page.css"), ".page{color:red}\n");

    const result = (await build({
      root,
      configFile: false,
      logLevel: "silent",
      experimental: viteConfig.experimental,
      build: { write: false, outDir: path.join(root, "dist") },
    })) as Rollup.RollupOutput;

    const entry = result.output.find(
      (o): o is Rollup.OutputChunk => o.type === "chunk" && o.isEntry,
    );
    expect(entry).toBeDefined();
    expect(entry!.code).toMatch(/__HERMES_BASE_PATH__\|\|""\)\+"\/assets\/page-[\w-]+\.css"/);
    expect(entry!.code).not.toMatch(/["'`]assets\/page-[\w-]+\.css/);
  });
});
