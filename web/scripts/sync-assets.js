#!/usr/bin/env node
// Cross-platform sync-assets: replaces `rm -rf public/fonts public/ds-assets && cp -r ...`
// Works on Windows (CMD/PowerShell), macOS, and Linux without Unix utilities.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");

const targets = ["fonts", "ds-assets"];
const destDir = path.join(webRoot, "public");

for (const target of targets) {
  const dest = path.join(destDir, target);
  // Remove existing directory (recursive, works cross-platform)
  if (fs.existsSync(dest)) {
    fs.rmSync(dest, { recursive: true, force: true });
  }
}

// Copy from node_modules
const srcPkg = "@nous-research/ui";
const srcFonts = path.join(webRoot, "node_modules", srcPkg, "dist", "fonts");
const srcAssets = path.join(webRoot, "node_modules", srcPkg, "dist", "assets");
const destFonts = path.join(destDir, "fonts");
const destAssets = path.join(destDir, "ds-assets");

if (fs.existsSync(srcFonts)) {
  fs.cpSync(srcFonts, destFonts, { recursive: true });
}
if (fs.existsSync(srcAssets)) {
  fs.cpSync(srcAssets, destAssets, { recursive: true });
}
