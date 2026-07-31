#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const websiteDir = join(scriptDir, "..");
const packageDir = join(websiteDir, "node_modules", "serve-handler");
const packageJsonPath = join(packageDir, "package.json");
const targetPath = join(packageDir, "src", "index.js");
const expectedVersion = "6.1.7";
const originalImport = "const minimatch = require('minimatch');";
const compatibleImport =
  "const { minimatch } = require('minimatch'); // Hermes: minimatch >=9 compatibility";
const checkOnly = process.argv.includes("--check");

const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
if (packageJson.version !== expectedVersion) {
  throw new Error(
    `expected serve-handler ${expectedVersion}, found ${packageJson.version}`,
  );
}

let source = readFileSync(targetPath, "utf8");
if (!source.includes(compatibleImport)) {
  if (checkOnly) {
    throw new Error("serve-handler minimatch compatibility patch is not installed");
  }

  const matches = source.split(originalImport).length - 1;
  if (matches !== 1) {
    throw new Error(`expected exactly one minimatch import, found ${matches}`);
  }

  source = source.replace(originalImport, compatibleImport);
  writeFileSync(targetPath, source, "utf8");
  console.log(`website: patched serve-handler minimatch import: ${targetPath}`);
}

const requireFromServeHandler = createRequire(targetPath);
const minimatchModule = requireFromServeHandler("minimatch");
if (typeof minimatchModule.minimatch !== "function") {
  throw new Error("patched minimatch module does not expose a callable minimatch export");
}

const cases = [
  ["docs/a.md", "docs/{a,b}.md", true],
  ["docs/c.md", "docs/{a,b}.md", false],
];
for (const [path, pattern, expected] of cases) {
  const actual = minimatchModule.minimatch(path, pattern);
  if (actual !== expected) {
    throw new Error(
      `braced minimatch mismatch for ${path}: expected ${expected}, got ${actual}`,
    );
  }
}

console.log("website: serve-handler braced-minimatch compatibility check passed");
