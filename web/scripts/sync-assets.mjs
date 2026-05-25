#!/usr/bin/env node
// Cross-platform replacement for the previous shell pipeline:
//
//   rm -rf public/fonts public/ds-assets
//   && cp -r node_modules/@nous-research/ui/dist/fonts public/fonts
//   && cp -r node_modules/@nous-research/ui/dist/assets public/ds-assets
//
// `rm -rf` / `cp -r` don't exist on Windows cmd.exe, so `npm run build`
// (invoked from Python via subprocess → cmd.exe) failed before Vite ran.
// Using Node's stdlib fs keeps this dependency-free and platform-neutral.

import { cpSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const uiDist = resolve(webRoot, "node_modules", "@nous-research", "ui", "dist");

const targets = [
  { from: resolve(uiDist, "fonts"), to: resolve(webRoot, "public", "fonts") },
  { from: resolve(uiDist, "assets"), to: resolve(webRoot, "public", "ds-assets") },
];

for (const { from, to } of targets) {
  rmSync(to, { recursive: true, force: true });
  cpSync(from, to, { recursive: true });
}
