import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import { build } from "vite";
import dashboardConfig from "../vite.config";

// Check the shipped CSS, not simulated touch events: node cannot reproduce
// a mobile browser's native pull-to-refresh gesture.
test("dashboard CSS disables vertical root overscroll without disabling touch gestures", async () => {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const result = await build({
    ...dashboardConfig,
    root,
    configFile: false,
    publicDir: false,
    logLevel: "silent",
    build: {
      write: false,
      minify: false,
      cssMinify: false,
      rolldownOptions: { input: path.join(root, "src/index.css") },
    },
  });

  const outputs = Array.isArray(result) ? result : [result];
  const css = outputs.flatMap((output) => "output" in output ? output.output : []).find(
    (output) => output.type === "asset" && output.fileName.endsWith(".css"),
  );
  expect(css).toBeDefined();
  if (!css || css.type !== "asset") throw new Error("Missing dashboard CSS asset");
  const rootRules = [...String(css.source).matchAll(/(?:^|\n)\s*html\s*\{([^}]*)\}/g)]
    .map((match) => match[1]);

  expect(rootRules.some((rule) => /overscroll-behavior-y:\s*none\s*;/.test(rule))).toBe(true);
  expect(rootRules.join("\n")).not.toMatch(/touch-action\s*:|overscroll-behavior(?:-x)?\s*:/);
});
